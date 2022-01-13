from typing import Dict, Iterable, Union

import numpy as np

import tensorflow as tf
from tensorflow.keras import optimizers

from dppp.types import *
from dppp.utils import *


# ###############################################################################
# # DEBLUR NON-BLIND
# ###############################################################################


def dmsp_deblur_nb(
    degraded: TensorLike,
    kernel: TensorLike,
    denoiser: Denoiser,
    denoiser_stddev: float,
    noise_stddev: float = None,
    num_steps: int = 300,
    conv_mode: str = ConvolutionModes.CONSTANT,
    optimizer: Union[Dict, optimizers.Optimizer] = None,
    callbacks: Iterable[Callback] = None,
) -> TensorLike:
    """Run non-blind deblurring or noise-adaptive deblurring on the degraded example using a
    dmsp scheme.

    Args:
        degraded (TensorLike): A batch of low resolution images. Dimensions [B, H, W, C].
        kernel (TensorLike, optional): The convolution kernel that was used for blurring.
        denoiser (Denoiser): The gaussian denoiser.
        denoiser_stddev (float): The noise level to use for the denoiser.
        noise_stddev (int, optional): The stddev of noise in the images.
        num_steps (int, optional): Number of steps for the DMSP algorithm. Defaults to 300.
        conv_mode (str, optional): The border condition for convolutions to use if a kernel is
            given. One of ["constant", "wrap"]. Defaults to "constant".
        optimizer (Union[Dict, Optimizer], optional): Either a `tf.keras.optimizers.Optimizer` or
            a dict containing the arugments for for `tf.keras.optimizers.SGD`. Defaults to
            `tf.keras.optimizer.SGD(learning_rate=1e-4, momentum=0.9)`.
        callbacks (Iterable[Callback], optional): A list of callbacks which will be called after
            each step. Defaults to [].

    Returns:
        The deblurred images.
    """

    if callbacks is None:
        callbacks = []

    # Create the optimizer
    x_optimizer = _optimizer_from_arg(
        optimizer, {"learning_rate": 1e-4, "momentum": 0.9}
    )

    # Variables to optimize
    x = tf.Variable(degraded)

    # Helper variables
    y = degraded
    sig2 = 2 * denoiser_stddev ** 2  # Paper: \sigma^2
    x_size = img_size(x, tf.float32)  # Paper: M
    y_size = img_size(y, tf.float32)  # Paper: N

    # TODO same as in blind: can we combine the code?
    if noise_stddev is None:
        # Noise adaptive
        def data_term():
            # TODO rgb kernel handling
            return (y_size / 2.0) * tf.math.log(
                tf.math.reduce_sum((y - convolve(x, kernel, mode=conv_mode)) ** 2)
                + x_size * sig2 * _kernel_norm2(kernel)
            )

    else:
        # Non-blind
        def data_term():
            return (
                tf.math.reduce_sum((y - convolve(x, kernel, mode=conv_mode)) ** 2)
                + x_size * sig2 * _kernel_norm2(kernel)
            ) / (2 * noise_stddev ** 2) + y_size * tf.math.log(noise_stddev)

    minimize_for_x = _minimize_for_x_fn(
        x_optimizer, x, data_term, denoiser, denoiser_stddev
    )

    # Loop and optimize x and k
    for step in tf.range(num_steps, dtype=tf.int64):

        # Minimize x
        minimize_for_x()

        # Callbacks
        for c in callbacks:
            c(x, step)

    return x


# ###############################################################################
# # DEBLUR BLIND
# ###############################################################################

# TODO get blind deblurring to work


def dmsp_deblur_blind(
    degraded: TensorLike,
    denoiser: Denoiser,
    denoiser_stddev: float,
    kernel_size: int,
    num_steps: int = 300,
    conv_mode: str = ConvolutionModes.CONSTANT,
    image_optimizer: Union[Dict, optimizers.Optimizer] = None,
    kernel_optimizer: Union[Dict, optimizers.Optimizer] = None,
    callbacks: Iterable[Callback] = None,
    callbacks_kernel: Iterable[Callback] = None,
) -> TensorLike:

    if callbacks is None:
        callbacks = []

    if callbacks_kernel is None:
        callbacks_kernel = []

    # TODO try other options to initialize the kernel
    k_init = _init_kernel_box(kernel_size)

    # Create the optimizers
    x_optimizer = _optimizer_from_arg(
        image_optimizer, {"learning_rate": 1e-4, "momentum": 0.9}
    )
    k_optimizer = _optimizer_from_arg(
        kernel_optimizer, {"learning_rate": 1e-11, "momentum": 0.9}
    )

    # Variables to optimize
    x = tf.Variable(degraded)
    k = tf.Variable(k_init)

    # Helper variables
    y = degraded
    sig2 = 2 * denoiser_stddev ** 2  # Paper: \sigma^2
    x_size = img_size(x, tf.float32)  # Paper: M
    y_size = img_size(y, tf.float32)  # Paper: N

    def data_term():
        return (y_size / 2.0) * tf.math.log(
            tf.math.reduce_sum((y - convolve(x, k, mode=conv_mode)) ** 2)
            + x_size * sig2 * _kernel_norm2(k)
        )

    minimize_for_x = _minimize_for_x_fn(
        x_optimizer, x, data_term, denoiser, denoiser_stddev
    )

    # Loop and optimize x and k
    for step in tf.range(num_steps, dtype=tf.int64):

        # Minimize x
        minimize_for_x()

        # Minimize k
        k_optimizer.minimize(data_term, [k])

        # Normalize k
        k.assign(_normalize_kernel(k))

        # Callbacks
        for c in callbacks:
            c(x, step)
        for c in callbacks_kernel:
            c(k, step)

    return x


###############################################################################
# SUPER RESOLVE
###############################################################################


def dmsp_super_resolve(
    degraded: TensorLike,
    sr_factor: int,
    denoiser: Denoiser,
    denoiser_stddev: float,
    kernel: TensorLike = None,
    resize_fn: ResizeFnType = None,
    prior_weight: float = 1e-3,
    num_steps: int = 300,
    learning_rate: float = 1.0,
    momentum: float = 0.9,
    callbacks: Iterable[Callback] = None,
    conv_mode: str = ConvolutionModes.CONSTANT,
) -> TensorLike:
    """Run single image super-resolution on the degraded example using a dmsp scheme.

    Args:
        degraded (TensorLike): A batch of low resolution images. Dimensions [B, H, W, C].
        sr_factor (int): The factor by which the image should be upsampled.
        denoiser (Denoiser): The gaussian denoiser.
        denoiser_stddev (float): The noise level to use for the denoiser.
        kernel (TensorLike, optional): The convolution kernel that was used for downscaling. If not
            given resize_fn will be used.
        resize_fn (ResizeFnType, optional): Function used to downsample and upsample the image. If
            kernel=None and resize_fn=None, bicubic scaling will be used.
        prior_weight (float, optional): The starting weight for the prior. The weight for each step
            `i` is computed by `prior_weight / sqrt(i)`. Defaults to 1e-3.
        num_steps (int, optional): Number of steps for the DMSP algorithm. Defaults to 300.
        learning_rate (float, optional): Learning rate for the stochastic gradient decent. Defaults
            to 1.0.
        momentum (float, optional): Momentum for the stochastic gradient decent. Defaults to 0.9.
        callbacks (Iterable[Callback], optional): A list of callbacks which will be called after
            each step. Defaults to [].
        conv_mode (str, optional): The border condition for convolutions to use if a kernel is
            given. One of ["constant", "wrap"]. Defaults to "constant".

    Returns:
        The super resolved images of dimensions [B, H * sr_factor, W * sr_factor, C].
    """

    # See https://github.com/siavashBigdeli/DMSP-TF2/blob/main/DMSPRestore.py
    # for the kernel based super-resolution

    if resize_fn is None and kernel is None:
        # Both None -> Use bicubic resize_fn
        resize_fn = create_resize_fn("bicubic", True)
        use_kernel = False
    elif resize_fn is not None:
        # Resize fn given -> Use it
        use_kernel = False
    elif kernel is not None:
        # Kernel given -> Use it
        use_kernel = True
        resize_fn = create_resize_fn("bicubic", True)  # For the initialization

        batch, height, width, channel = tf.shape(degraded)

        # Init subsampling mask
        subsampling_mask = np.zeros(
            (batch, height * sr_factor, width * sr_factor, channel)
        )
        subsampling_mask[:, ::sr_factor, ::sr_factor, :] = 1.0
        subsampling_mask = tf.constant(subsampling_mask, tf.float32)
        error_factor = tf.cast(tf.size(subsampling_mask), tf.float32) / tf.reduce_sum(
            subsampling_mask
        )

        # Upscale degraded without interpolation for error calculation
        degraded_upscaled = upscale_no_iterpolation(degraded, sr_factor)

    if callbacks is None:
        callbacks = []

    def grad_fn(x: TensorLike, step: TensorLike) -> TensorType:
        # Data gradient
        if use_kernel:
            map_conv = convolve(x, kernel, mode=conv_mode)
            data_err = error_factor * subsampling_mask * (map_conv - degraded_upscaled)
            data_grad = convolve(data_err, flip_kernel_2d(kernel), mode=conv_mode)
        else:
            x_lr = resize_fn(x, sr_factor, False)
            data_err = x_lr - degraded  # type: ignore
            data_grad = resize_fn(data_err, sr_factor, True)

        # Prior gradient
        prior_grad = dmsp_prior_grad(x, denoiser, denoiser_stddev)

        # Combined gradient
        step_f32 = tf.cast(step, tf.float32)
        current_prior_weight = prior_weight / tf.math.sqrt(step_f32 + 1)  # type: ignore
        return data_grad + current_prior_weight * prior_grad

    # Initialize x
    x = resize_fn(degraded, sr_factor, True)

    velocity = tf.zeros_like(x)

    for step in tf.range(num_steps, dtype=tf.int64):
        # Compute the gradient
        grad = grad_fn(x, step)

        # Apply the gradient with momentum
        velocity = momentum * velocity - learning_rate * grad
        x = x + velocity

        # Clip between 0 and 1
        x = tf.clip_by_value(x, 0, 1)

        # Call the callbacks with the value of the variable
        for callback in callbacks:
            callback(x, step)

    return x


###############################################################################
# INPAINTING
###############################################################################


def dmsp_inpaint(
    degraded: TensorLike,
    mask: TensorLike,
    denoiser: Denoiser,
    denoiser_stddev: float,
    init_fn: InpaintFnType = None,
    prior_weight: float = 0.2,
    num_steps: int = 300,
    learning_rate: float = 1e-2,
    momentum: float = 0.9,
    callbacks: Iterable[Callback] = None,
) -> TensorLike:

    if callbacks is None:
        callbacks = []

    if init_fn is None:
        init_fn = inpaint_border

    def grad_fn(x: TensorLike, step: TensorLike) -> TensorType:
        # Current prior weight
        step_f32 = tf.cast(step, tf.float32)
        current_prior_weight = prior_weight / tf.math.sqrt(step_f32 + 1)  # type: ignore

        # Prior gradient
        prior_grad = dmsp_prior_grad(x, denoiser, denoiser_stddev)

        # Combine with data gradient
        data_grad = (x - degraded) * mask
        return data_grad + current_prior_weight * prior_grad

    x = init_fn(degraded, mask)
    velocity = tf.zeros_like(x)

    for step in tf.range(num_steps, dtype=tf.int64):
        # Compute the gradient
        grad = grad_fn(x, step)

        # Apply the gradient with momentum
        velocity = momentum * velocity - learning_rate * grad
        x = x + velocity

        # Clip between 0 and 1
        x = tf.clip_by_value(x, 0, 1)

        # Call the callbacks with the value of the variable
        for callback in callbacks:
            callback(x, step)

    return degraded * mask + (1 - mask) * x


###############################################################################
# GENERAL UTILITIES
###############################################################################


def dmsp_prior_grad(x: TensorLike, denoiser: Denoiser, noise_stddev: TensorLike):
    """Compute the gradient for the deep mean-shift prior.

    Args:
        x: The images at the current step.
        denoiser: An denoiser that is used to compute the gradient of the prior.

    Returns:
        The gradient of the prior of x.
    """
    noise = tf.random.normal(tf.shape(x), stddev=noise_stddev)
    grad = x - denoiser(x + noise, noise_stddev)  # type: ignore
    return (1 / (noise_stddev ** 2)) * grad  # type: ignore


def img_size(x: TensorLike, dtype: tf.DType = tf.int32) -> tf.Tensor:
    """Total number of elements in all the images in the tensor x.

    Args:
        x (TensorLike): Batch of images. Shape [B, H, W, C].
        dtype (tf.DType): Desired type of the output.

    Returns:
        H * W * C, scalar tensor of type `dtype`.
    """
    return tf.cast(tf.math.reduce_prod(tf.shape(x)[1:]), dtype)


def _normalize_kernel(k: TensorLike) -> tf.Tensor:
    """Normalize the given convolution kernel such that all values are positive
    and sum up to 1 for each output channel.

    Args:
        k (TensorLike): Convolution kernel with the shape [H, W, C_IN, C_OUT]

    Returns:
        Tensor: The normalized convolution kernel
    """
    k = tf.clip_by_value(k, 0, np.inf)
    return k / tf.reduce_sum(k, axis=[0, 1, 2], keepdims=True)


def _optimizer_from_arg(arg, default_args):
    if isinstance(arg, optimizers.Optimizer):
        return arg
    else:
        optimizer_args = default_args.copy()
        if arg is not None:
            optimizer_args.update(arg)
        return optimizers.SGD(**optimizer_args)


def _minimize_for_x_fn(x_optimizer, x, data_term, denoiser, denoiser_stddev):
    # TODO make tf.function
    # @tf.function
    def min():
        with tf.GradientTape() as tape:
            dt = data_term()
        x_data_grad = tape.gradient(dt, x)
        x_prior_grad = dmsp_prior_grad(x, denoiser, denoiser_stddev)
        x_optimizer.apply_gradients([(x_data_grad + x_prior_grad, x)])

        x.assign(tf.clip_by_value(x, 0, 1))

    return min


def _kernel_norm2(k):
    # TODO can we compute a correct gray kernel for the rgb kernel?
    k_gray = tf.math.reduce_mean(k, axis=[2, 3])
    return tf.math.reduce_sum(k_gray ** 2)


def _init_kernel_box(kernel_size: int) -> tf.Tensor:
    k = tf.ones((kernel_size, kernel_size))
    k = conv2D_filter_rgb(k)
    return _normalize_kernel(k)
