import functools
from typing import Iterable
import tensorflow as tf
import tensorflow_datasets_bw as tfdsbw

from dppp.types import *
from dppp.utils import *


###############################################################################
# DEBLUR NON-BLIND
###############################################################################


def hqs_deblur_nb(
    degraded,
    kernel,
    noise_stddev,
    denoiser: Denoiser,
    max_denoiser_stddev: float,
    num_steps: int = 8,
    callbacks: Iterable[Callback] = None,
):
    """Run non-blind deblurring on the degraded example using a hqs scheme.

    Args:
        degraded (TensorLike): A batch of low resolution images. Dimensions [B, H, W, C].
        kernel (TensorLike): The convolution kernel that was used for blurring.
        noise_stddev (int): The stddev of noise in the images.
        denoiser (Denoiser): The gaussian denoiser.
        max_denoiser_stddev (float): The maximum noise level the denoiser can handle.
        num_steps (int, optional): Number of steps for the HQS algorithm. Defaults to 8.
        callbacks (Iterable[Callback], optional): A list of callbacks which will be called after
            each step. Defaults to [].

    Returns:
        The deblurred images.
    """
    return hqs_super_resolve(
        degraded,
        1,
        denoiser,
        max_denoiser_stddev,
        kernel=kernel,
        noise_stddev=noise_stddev,
        num_steps=num_steps,
        callbacks=callbacks,
    )


###############################################################################
# SINGLE IMAGE SUPER RESOLVE
###############################################################################


def hqs_super_resolve(
    degraded: TensorLike,
    sr_factor: int,
    denoiser: Denoiser,
    max_denoiser_stddev: float,
    kernel: TensorLike = None,
    resize_fn: ResizeFnType = None,
    noise_stddev: float = 0,
    num_steps: int = 30,
    num_steps_data: int = 5,
    step_size_data: float = 1.5,
    callbacks: Iterable[Callback] = None,
):
    """Run single image super-resolution on the degraded example using a hqs scheme.

    For the data solution either a closed-form solution is used, if the kernel is given, or an
    iterative approach is used if the resize function is given.

    Args:
        degraded (TensorLike): A batch of low resolution images. Dimensions [B, H, W, C].
        sr_factor (int): The factor by which the image should be upsampled.
        denoiser (Denoiser): The gaussian denoiser.
        max_denoiser_stddev (float): The maximum noise level the denoiser can handle.
        kernel (TensorLike, optional): The convolution kernel that was used for downscaling. If not
            given resize_fn will be used.
        resize_fn (ResizeFnType, optional): Function used to downsample and upsample the image. If
            kernel=None and resize_fn=None, bicubic scaling will be used.
        noise_stddev (int, optional): The stddev of noise in the images. Defaults to 0.
        num_steps (int, optional): Number of steps for the HQS algorithm. Defaults to 30.
        num_steps_data (int, optional): Number of steps for the iterative data solution. Defaults to 5.
        step_size_data (float, optional): Step size for the iterative data solution. Defaults to 1.5.
        callbacks (Iterable[Callback], optional): A list of callbacks which will be called after
            each step. Defaults to [].

    Returns:
        The super resolved images of dimensions [B, H * sr_factor, W * sr_factor, C].
    """
    if callbacks is None:
        callbacks = []

    if kernel is None and resize_fn is None:
        resize_fn = create_resize_fn("bicubic", True)

    # Some useful values
    h_lr, w_lr = tf.shape(degraded)[1], tf.shape(degraded)[2]
    h, w = h_lr * sr_factor, w_lr * sr_factor

    rhos, sigmas = _get_rho_sigma(
        sigma=tf.maximum(0.001, noise_stddev),
        iter_num=num_steps,
        model_sigma1=max_denoiser_stddev,
        model_sigma2=tf.maximum(sr_factor / 255.0, noise_stddev),
    )

    # Init the function computing the data solution
    if kernel is not None:
        # Use the closed form data solution
        FB, FBC, F2B, FBFy = _pre_calculate(degraded, kernel, sr_factor)
        data_solution = functools.partial(
            _data_solution_closed_form,
            FB=FB,
            FBC=FBC,
            F2B=F2B,
            FBFy=FBFy,
            scale_factor=sr_factor,
        )
    else:
        # Use the iterative approach for the data term
        data_solution = functools.partial(
            _data_solution_iterative,
            degraded=degraded,
            resize_fn=resize_fn,
            scale_factor=sr_factor,
            num_steps=num_steps_data,
            step_size=step_size_data,
        )

    # Init the estimate
    if sr_factor == 1:
        x = degraded
    else:
        if resize_fn is not None:
            x = resize_fn(degraded, sr_factor, True)
        else:
            # Use bicubic upsampling for the initial estimate
            x = tf.image.resize(degraded, [h, w], method=tf.image.ResizeMethod.BICUBIC)

    # Loop and optimize the estimate
    for step in tf.range(num_steps, dtype=tf.int64):
        x = data_solution(x, rhos[step])
        x = denoiser(x, sigmas[step])

        for callback in callbacks:
            callback(x, step)

    return x


###############################################################################
# MULTI FRAME SUPER RESOLVE
###############################################################################


def hqs_lf_super_resolve(
    degraded: TensorLike,
    sr_factor: int,
    denoiser: Denoiser,
    max_denoiser_stddev: float,
    kernel: TensorLike = None,
    resize_fn: ResizeFnType = None,
    disparity_lr: TensorLike = None,
    disparity_hr: TensorLike = None,
    **kwargs
):
    """Light field super-resolution of the center frame using the HQS algorithm.

    Args:
        degraded (TensorLike): A batch of low resolution images. Dimensions [B, H, W, C].
        sr_factor (int): The factor by which the center frame should be upsampled.
        denoiser (Denoiser): The gaussian denoiser.
        max_denoiser_stddev (float): The maximum noise level the denoiser can handle.
        kernel (TensorLike, optional): The convolution kernel that was used for downscaling. If not
            given resize_fn will be used.
        resize_fn (ResizeFnType, optional): Function used to downsample and upsample the image. If
            kernel=None and resize_fn=None, bicubic scaling will be used.
        disparity_lr (TensorLike, optional): Low resolution disparity map of shape [H, W]. If none
            is given, this is estimated using `dppp.lf_disparity_lfattnet`.
        disparity_hr (TensorLike, optional): High resolution disparity map of shape
            [H * sr_factor, W * sr_factor]. If none is given, disparity_lr is upscaled.
        **kwargs: Additional arguments that are given to `dppp.hqs_video_super_resolve`.

    Returns:
        The super resolved center frame of dimensions [1, H * sr_factor, W * sr_factor, C].
    """
    lightfield_grid = tf.shape(degraded)[:2]
    degraded_batch = tfdsbw.lf_to_batch(degraded)
    ref_index = tf.math.floordiv(tf.shape(degraded_batch)[0], 2)
    bicubic_resize_fn = create_resize_fn("bicubic", True)

    def flows(disp, lf_batch):
        fw = tfdsbw.lf_to_batch(
            lf_flows(
                disp, grid_height=lightfield_grid[0], grid_width=lightfield_grid[1]
            )
        )
        ref_img = lf_batch[ref_index, None, ...]
        ref_batch = tf.broadcast_to(ref_img, tf.shape(lf_batch))
        return fw, backward_flow(fw, ref_batch, lf_batch)

    if disparity_hr is not None:
        lf_batch_upscaled = bicubic_resize_fn(degraded_batch, sr_factor, True)
        forward_flows, backward_flows = flows(disparity_hr, lf_batch_upscaled)
    else:
        if disparity_lr is None:
            # Estimate the disparity using LFattNet
            disparity_lr = lf_disparity_lfattnet(degraded)
        fw_lr, bw_lr = flows(disparity_lr, degraded_batch)
        forward_flows = resize_flow(
            fw_lr, sr_factor, upscale=True, resize_fn=bicubic_resize_fn
        )
        backward_flows = resize_flow(
            bw_lr, sr_factor, upscale=True, resize_fn=bicubic_resize_fn
        )

    return hqs_video_super_resolve(
        degraded=degraded_batch,
        ref_index=ref_index,
        sr_factor=sr_factor,
        denoiser=denoiser,
        max_denoiser_stddev=max_denoiser_stddev,
        kernel=kernel,
        resize_fn=resize_fn,
        forward_flows=forward_flows,
        backward_flows=backward_flows,
        **kwargs
    )


def hqs_video_super_resolve(
    degraded: TensorLike,
    ref_index: int,
    sr_factor: int,
    denoiser: Denoiser,
    max_denoiser_stddev: float,
    kernel: TensorLike = None,
    resize_fn: ResizeFnType = None,
    forward_flows: TensorLike = None,
    backward_flows: TensorLike = None,
    noise_stddev=0,
    num_steps: int = 30,
    num_steps_data: int = None,
    step_size_data: float = None,
    callbacks: Iterable[Callback] = None,
    conv_mode: str = ConvolutionModes.CONSTANT,
):
    """Video super-resolution using the HQS algorithm.

    Args:
        degraded (TensorLike): A batch of low resolution images. Dimensions [B, H, W, C].
        ref_index (int): The index of the reference frame. This frame will be super resolved.
        sr_factor (int): The factor by which the reference frame should be upsampled.
        denoiser (Denoiser): The gaussian denoiser.
        max_denoiser_stddev (float): The maximum noise level the denoiser can handle.
        kernel (TensorLike, optional): The convolution kernel that was used for downscaling. If not
            given resize_fn will be used.
        resize_fn (ResizeFnType, optional): Function used to downsample and upsample the image. If
            kernel=None and resize_fn=None, bicubic scaling will be used.
        forward_flows (TensorLike, optional): A batch of high resolution optical flows between the
            reference frame and the individial image. By default the flows are estimated using
            `dppp.reference_flows`.
        backward_flows (TensorLike, optional): A batch of high resolution optical flows between an
            individual image and the reference frame. By default the flows are estimated using
            `dppp.reference_flows`.
        noise_stddev (int, optional): The stddev of noise in the images. Defaults to 0.
        num_steps (int, optional): Number of steps for the HQS algorithm. Defaults to 30.
        num_steps_data (int, optional): Number of steps for the data solution. Defaults to 5 if a
            kernel is given and 1 if not.
        step_size_data (float, optional): Step size for the data solution. Defaults to 0.2 if a
            kernel is given and 1.0 if not.
        callbacks (Iterable[Callback], optional): A list of callbacks which will be called after
            each step. Defaults to [].
        conv_mode (str, optional): The border condition for convolutions to use if a kernel is
            given. One of ["constant", "wrap"]. Defaults to "constant".

    Returns:
        The super resolved image of dimensions [1, H * sr_factor, W * sr_factor, C].
    """
    if callbacks is None:
        callbacks = []

    def diff_sign(a, b):
        bigger = tf.cast(a > b, tf.float32)
        smaller = tf.cast(a < b, tf.float32)
        return bigger - smaller

    num_images = tf.shape(degraded)[0]

    # ---
    # Kernel mode vs Bicubic mode

    kernel_mode = kernel is not None

    if kernel_mode:
        resize_fn = create_convolve_resize_fn(kernel, conv_mode)
    elif resize_fn is None:
        resize_fn = create_resize_fn("bicubic", True)

    if num_steps_data is None:
        num_steps_data = 5 if kernel_mode else 1
    if step_size_data is None:
        step_size_data = 0.2 if kernel_mode else 1.0

    # ---
    # Compute the optical flows

    flow_resize_fn = create_resize_fn()

    if forward_flows is None:
        # Estimate flows using pyflow
        forward_flows = reference_flows(degraded, ref_index, backwards=False)
        forward_flows = resize_flow(
            forward_flows, sr_factor, True, resize_fn=flow_resize_fn
        )

    if backward_flows is None:
        # Estimate flows using pyflow
        backward_flows = reference_flows(degraded, ref_index, backwards=True)
        backward_flows = resize_flow(
            backward_flows, sr_factor, True, resize_fn=flow_resize_fn
        )

    # ---
    # Sigmas for the prior

    _, sigmas = _get_rho_sigma(
        sigma=tf.maximum(0.001, noise_stddev),
        iter_num=num_steps,
        model_sigma1=max_denoiser_stddev,
        model_sigma2=tf.maximum(sr_factor / 255.0, noise_stddev),
    )

    # ---
    # Init with bicubic upsampling

    x = create_resize_fn()(degraded[ref_index, None, ...], sr_factor, True)

    # ---
    # Optimization Loop

    for step in tf.range(num_steps, dtype=tf.int64):

        # Data solution
        for _ in tf.range(num_steps_data):
            # Warp x to all frames (backward warping)
            b = tfa.image.dense_image_warp(
                tf.repeat(x, [num_images], axis=0), -backward_flows
            )
            # Downsample x to the size of the degraded images
            c = resize_fn(b, sr_factor, False)
            # Get the difference from the manually degraded x to the observation
            if kernel_mode:
                d = diff_sign(degraded, c)
            else:
                d = degraded - c
            # Upscale the difference
            c = resize_fn(d, sr_factor, True)
            # Warp the difference back to the reference frame (forward warping)
            d = tfa.image.dense_image_warp(c, -forward_flows)
            # Update x
            x += step_size_data * tf.reduce_mean(d, axis=0, keepdims=True)

        # Prior
        x = denoiser(x, sigmas[step])

        for callback in callbacks:
            callback(x, step)

    return x


###############################################################################
# INPAINTING
###############################################################################


def hqs_inpaint(
    degraded: TensorLike,
    mask: TensorLike,
    denoiser: Denoiser,
    mas_denoiser_stddev: float,
    init_fn: InpaintFnType = None,
    num_steps: int = 30,
    callbacks: Iterable[Callback] = None,
) -> TensorLike:

    if callbacks is None:
        callbacks = []

    if init_fn is None:
        init_fn = inpaint_border

    # TODO add a noise parameter
    # This would require a change in the parameters of _get_rho_sigma

    rhos, sigmas = _get_rho_sigma(
        sigma=0.001,
        iter_num=num_steps,
        model_sigma1=denoiser_stddev,
        model_sigma2=0.01,
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


def _data_solution_iterative(
    x, alpha, degraded, resize_fn, scale_factor, num_steps=30, step_size=1.5
):
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
    FR = FBFy + _rfft2d(alpha * x_channels_first)

    # Apply the distinct block downsampler
    # ... on F(k)*d
    FBR = tf.math.reduce_mean(
        _splits(FB * FR, scale_factor, w), axis=-1, keepdims=False
    )
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
    FX = (FR - FCBinvWBR) / alpha_complex

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


# https://github.com/cszn/DPIR/blob/c4474621c47c2015b521e477a74ec0b895838069/utils/utils_sisr.py#L255
@tf.function(
    input_signature=[
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.int32),
    ]
)
def _pre_calculate(x, k, scale_factor):
    h, w = tf.shape(x)[1], tf.shape(x)[2]
    k_h, k_w = tf.shape(k)[0], tf.shape(k)[1]

    # Compute the Optical transfer function (FFT of the point spread function (kernel))
    # https://github.com/cszn/DPIR/blob/c4474621c47c2015b521e477a74ec0b895838069/utils/utils_sisr.py#L195
    k_single_channel = k[::-1, ::-1, 0, 0][
        None, None, ...
    ]  # TODO adapt for full color kernels
    paddings = [
        [0, 0],
        [0, 0],
        [0, (h * scale_factor) - k_h],
        [0, (w * scale_factor) - k_w],
    ]
    k_padded = tf.pad(k_single_channel, paddings)
    k_padded = tf.roll(k_padded, -tf.cast((k_h / 2), tf.int32), axis=2)
    k_padded = tf.roll(k_padded, -tf.cast((k_w / 2), tf.int32), axis=3)
    FB = _rfft2d(k_padded)  # F(k)

    # Set small imaginary values to 0
    k_shape_float = tf.cast([k_h, k_w], tf.float32)
    n_ops = tf.reduce_sum(k_shape_float * tf.experimental.numpy.log2(k_shape_float))
    FB = tf.where(
        tf.math.abs(tf.math.imag(FB)) >= n_ops * 2.22e-16,
        FB,
        tf.cast(tf.math.real(FB), tf.complex64),
    )

    # Compute the complex conjugate of F(k): ‾F(k)‾
    FBC = tf.math.conj(FB)

    # Compute ‾F(k)‾ * F(k)
    F2B = tf.cast(tf.math.real(FB) ** 2 + tf.math.imag(FB) ** 2, tf.complex64)

    # Upsample
    Sty = upscale_no_iterpolation(x, scale_factor)

    # Compute ‾F(k)‾ * F(upsample(x))
    FBFy = FBC * _rfft2d(tf.transpose(Sty, perm=[0, 3, 1, 2]))
    return FB, FBC, F2B, FBFy


# Make a two-sided fft from a one-sided
@tf.function()
def _make_fft_two_sided(x, last_length):
    last_to_copy = tf.cast(
        tf.math.ceil(tf.cast(last_length, tf.float32) / 2.0), tf.int32
    )
    to_copy = x[..., last_to_copy - 1 : 0 : -1]
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
        x = x[..., : (tf.math.floordiv(tf.shape(x)[-1], 2) + 1)]
    return tf.signal.irfft2d(x, fft_length=tf.shape(like)[-2:])


# See: https://github.com/cszn/DPIR/blob/c4474621c47c2015b521e477a74ec0b895838069/utils/utils_pnp.py#L27
@tf.function()
def _get_rho_sigma(
    sigma=0.01, iter_num=15, model_sigma1=0.2, model_sigma2=0.01, w=1, lambd=0.23
):
    # TODO split into get sigmas and get rhos
    model_sigma_s = tf.cast(
        tf.experimental.numpy.logspace(
            tf.experimental.numpy.log10(model_sigma1),
            tf.experimental.numpy.log10(model_sigma2),
            iter_num,
        ),
        tf.float32,
    )
    model_sigma_s_lin = tf.cast(
        tf.experimental.numpy.linspace(model_sigma1, model_sigma2, iter_num), tf.float32
    )
    sigmas = model_sigma_s * w + model_sigma_s_lin * (1 - w)
    rhos = lambd * (sigma ** 2) / (sigmas ** 2)
    return rhos, sigmas
