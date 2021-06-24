import functools
from typing import Iterable
import tensorflow as tf

from dppp.types import *
from dppp.utils import create_resize_fn, inpaint_border


###############################################################################
# DEBLUR NON-BLIND
###############################################################################

# TODO try more steps
def hqs_deblur_nb(degraded, kernel, noise_stddev,
                  denoiser: Denoiser, denoiser_stddev: float,
                  num_steps: int = 8,
                  callbacks: Iterable[Callback] = None):
    # TODO try the iterative approach for deblurring
    return hqs_super_resolve(degraded, 1, denoiser, denoiser_stddev,
                             kernel=kernel, noise_stddev=noise_stddev,
                             num_steps=num_steps,
                             callbacks=callbacks)


###############################################################################
# SUPER RESOLVE
###############################################################################

def hqs_super_resolve(degraded: TensorLike,
                      sr_factor: int,
                      denoiser: Denoiser,
                      denoiser_stddev: float,  # TODO rename this parameter to max stddev??
                      kernel: TensorLike = None,
                      resize_fn: ResizeFnType = None,
                      noise_stddev: float = 0,  # TODO try this in combination with iterative approach
                      num_steps: int = 30,
                      num_steps_data: int = 5,
                      step_size_data: float = 1.5,
                      callbacks: Iterable[Callback] = None
                      ):
    """Run single image super-resolution on the degraded example using a hqs scheme.

    For the data solution either a closed-form solution is used, if the kernel is given, or an
    iterative approach is used if the resize function is given.
    """
    if callbacks is None:
        callbacks = []

    if kernel is None and resize_fn is None:
        resize_fn = create_resize_fn('bicubic', True)

    # Some useful values
    h_lr, w_lr = tf.shape(degraded)[1], tf.shape(degraded)[2]
    h, w = h_lr*sr_factor, w_lr*sr_factor

    # TODO check if this is inline with the iterative approach
    rhos, sigmas = _get_rho_sigma(sigma=tf.maximum(0.001, noise_stddev),
                                  iter_num=num_steps,
                                  model_sigma1=denoiser_stddev,
                                  model_sigma2=tf.maximum(sr_factor / 255.0, noise_stddev))

    # Init the function computing the data solution
    if kernel is not None:
        # Use the closed form data solution
        FB, FBC, F2B, FBFy = _pre_calculate(degraded, kernel, sr_factor)
        data_solution = functools.partial(_data_solution_closed_form,
                                          FB=FB, FBC=FBC, F2B=F2B, FBFy=FBFy, scale_factor=sr_factor)
    else:
        # Use the iterative approach for the data term
        data_solution = functools.partial(_data_solution_iterative,
                                          degraded=degraded, resize_fn=resize_fn,
                                          scale_factor=sr_factor, num_steps=num_steps_data,
                                          step_size=step_size_data)

    # Init the estimate
    # TODO check for wrong sr_factor values?
    if sr_factor == 1:
        x = degraded
    else:
        if resize_fn is not None:
            x = resize_fn(degraded, sr_factor, True)
        else:
            # TODO why is this bicubic upsampling here??
            x = tf.image.resize(degraded, [h, w], method=tf.image.ResizeMethod.BICUBIC)

    # Loop and optimize the estimate
    for step in tf.range(num_steps, dtype=tf.int64):
        x = data_solution(x, rhos[step])
        x = denoiser(x, sigmas[step])

        for callback in callbacks:
            callback(x, step)

    return x


###############################################################################
# INPAINTING
###############################################################################


def hqs_inpaint(degraded: TensorLike,
                mask: TensorLike,
                denoiser: Denoiser,
                denoiser_stddev: float,
                init_fn: InpaintFnType = None,
                num_steps: int = 30,
                callbacks: Iterable[Callback] = None,
                ) -> TensorLike:

    if callbacks is None:
        callbacks = []

    if init_fn is None:
        init_fn = inpaint_border

    rhos, sigmas = _get_rho_sigma(sigma=0.001,  # TODO noise?
                                  iter_num=num_steps,
                                  model_sigma1=denoiser_stddev,
                                  model_sigma2=0.01  # TODO noise? Is this a good default?
                                  )

    x = init_fn(degraded, mask)

    for step in tf.range(num_steps, dtype=tf.int64):

        # TODO see https://github.com/cszn/IRCNN/blob/master/Demo_inpaint.m
        # They use a rho for the data fidelity term (because they (can) have noise in the data)

        # Data fidelity: Only use x where y is undefined
        x = mask * degraded + (1 - mask) * x

        # Regularizer
        x = denoiser(x, sigmas[step])

        for callback in callbacks:
            callback(x, step)

    return x


###############################################################################
# UTILITIES
###############################################################################


def _data_solution_iterative(x, alpha, degraded, resize_fn, scale_factor, num_steps=30, step_size=1.5):
    for _ in range(num_steps):
        data_err = degraded - resize_fn(x, scale_factor, False)
        x = x - step_size * resize_fn(-data_err, scale_factor, True)
    return x


# https://github.com/cszn/DPIR/blob/c4474621c47c2015b521e477a74ec0b895838069/utils/utils_sisr.py#L243
@tf.function()
def _data_solution_closed_form(x, alpha, FB, FBC, F2B, FBFy, scale_factor):
    w = tf.shape(x)[-1]
    # Prepare some inputs
    alpha_complex = tf.cast(alpha, tf.complex64)
    x_channels_first = tf.transpose(x, perm=[0, 3, 1, 2])

    # Compute d = ‾F(k)‾ * F(upsample(x)) + alpha * F(z)
    FR = FBFy + _rfft2d(alpha*x_channels_first)

    # Apply the distinct block downsampler
    # ... on F(k)*d
    FBR = tf.math.reduce_mean(_splits(FB * FR, scale_factor, w), axis=-1, keepdims=False)
    # ... on ‾F(k)‾ * F(k)
    invW = tf.math.reduce_mean(_splits(F2B, scale_factor, w), axis=-1, keepdims=False)

    # Compute the right side of the distinct block processing operator
    # (F(k)d)/(‾F(k)‾F(k) + alpha)
    invWBR = FBR / (invW + alpha_complex)

    # Compute the result of the distinct block processing operator
    # ‾F(k)‾ * (F(k)d)/(‾F(k)‾F(k) + alpha)
#     FCBinvWBR = FBC * tf.tile(invWBR, [1, 1, scale_factor, scale_factor])[...,:tf.shape(FBC)[-1]]
    FCBinvWBR = FBC * tf.tile(invWBR, [1, 1, scale_factor, scale_factor])

    # Fourier transform of the extimation
    FX = (FR - FCBinvWBR)/alpha_complex

    # Compute the extimation
    x_est_channels_first = _irfft2d(FX, like=x_channels_first)

    # Sort channels
    return tf.transpose(x_est_channels_first, perm=[0, 2, 3, 1])


# https://github.com/cszn/DPIR/blob/c4474621c47c2015b521e477a74ec0b895838069/utils/utils_sisr.py#L94
@tf.function()
def _splits(x, scale_factor, width):
    # x2 = two_sided(x, width)
    x2 = x
    b = tf.stack(tf.split(x2, scale_factor, axis=2), axis=-1)
    return tf.concat(tf.split(b, scale_factor, axis=3), axis=-1)


@tf.function()
def _upscale_no_iterpolation(x, scale):
    x_shape = tf.shape(x)
    upscaled_shape = [x_shape[0], x_shape[1] * scale, x_shape[2] * scale, x_shape[3]]
    x_added_dim = x[:, :, None, :, None, :]
    x_padded = tf.pad(x_added_dim, [[0, 0], [0, 0], [0, scale-1], [0, 0], [0, scale-1], [0, 0]])
    return tf.reshape(x_padded, upscaled_shape)


# https://github.com/cszn/DPIR/blob/c4474621c47c2015b521e477a74ec0b895838069/utils/utils_sisr.py#L255
@tf.function(input_signature=[tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[], dtype=tf.int32)])
def _pre_calculate(x, k, scale_factor):
    h, w = tf.shape(x)[1], tf.shape(x)[2]
    k_h, k_w = tf.shape(k)[0], tf.shape(k)[1]

    # Compute the Optical transfer function (FFT of the point spread function (kernel))
    # https://github.com/cszn/DPIR/blob/c4474621c47c2015b521e477a74ec0b895838069/utils/utils_sisr.py#L195
#     k_single_channel = k[:,:,0,0][None,None,...] # TODO adapt for full color kernels
    k_single_channel = k[::-1, ::-1, 0, 0][None, None, ...]  # TODO adapt for full color kernels
    paddings = [[0, 0], [0, 0], [0, (h*scale_factor)-k_h], [0, (w*scale_factor)-k_w]]
    k_padded = tf.pad(k_single_channel, paddings)
    k_padded = tf.roll(k_padded, -tf.cast((k_h / 2), tf.int32), axis=2)
    k_padded = tf.roll(k_padded, -tf.cast((k_w / 2), tf.int32), axis=3)
    FB = _rfft2d(k_padded)  # F(k)

    # Set small imaginary values to 0
    k_shape_float = tf.cast([k_h, k_w], tf.float32)
    n_ops = tf.reduce_sum(k_shape_float * tf.experimental.numpy.log2(k_shape_float))
    FB = tf.where(tf.math.abs(tf.math.imag(FB)) >= n_ops*2.22e-16,
                  FB,
                  tf.cast(tf.math.real(FB), tf.complex64))

    # Compute the complex conjugate of F(k): ‾F(k)‾
    FBC = tf.math.conj(FB)

    # Compute ‾F(k)‾ * F(k)
    F2B = tf.cast(tf.math.real(FB)**2 + tf.math.imag(FB)**2, tf.complex64)

    # Upsample
    Sty = _upscale_no_iterpolation(x, scale_factor)

    # Compute ‾F(k)‾ * F(upsample(x))
    FBFy = FBC * _rfft2d(tf.transpose(Sty, perm=[0, 3, 1, 2]))
    return FB, FBC, F2B, FBFy


# Make a two-sided fft from a one-sided
@tf.function()
def _make_fft_two_sided(x, last_length):
    last_to_copy = tf.cast(tf.math.ceil(tf.cast(last_length, tf.float32) / 2.0), tf.int32)
    to_copy = x[..., last_to_copy-1:0:-1]
    first_row = to_copy[..., 0:1, :]
    other_rows = to_copy[..., :0:-1, :]
    to_copy_flipped = tf.math.conj(tf.concat([first_row, other_rows], axis=-2))
    return tf.concat([x, to_copy_flipped], axis=-1)


@tf.function()
def _rfft2d(x, two_sided=True):
    f = tf.signal.rfft2d(x)
    if two_sided:
        return _make_fft_two_sided(f, tf.shape(x)[-1])
    else:
        return f


@tf.function()
def _irfft2d(x, like, two_sided=True):
    if two_sided:
        x = x[..., :(tf.math.floordiv(tf.shape(x)[-1], 2) + 1)]
    return tf.signal.irfft2d(x, fft_length=tf.shape(like)[-2:])


# See: https://github.com/cszn/DPIR/blob/c4474621c47c2015b521e477a74ec0b895838069/utils/utils_pnp.py#L27
@tf.function()
def _get_rho_sigma(sigma=0.01, iter_num=15, model_sigma1=0.2,
                   model_sigma2=0.01, w=1, lambd=0.23):
    # TODO split into get sigmas and get rhos
    model_sigma_s = tf.cast(tf.experimental.numpy.logspace(tf.experimental.numpy.log10(
        model_sigma1), tf.experimental.numpy.log10(model_sigma2), iter_num), tf.float32)
    model_sigma_s_lin = tf.cast(tf.experimental.numpy.linspace(
        model_sigma1, model_sigma2, iter_num), tf.float32)
    sigmas = (model_sigma_s * w + model_sigma_s_lin*(1-w))
    rhos = lambd*(sigma**2)/(sigmas**2)
    return rhos, sigmas
