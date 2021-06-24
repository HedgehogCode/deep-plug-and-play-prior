from typing import Iterable, Tuple

import tensorflow as tf

from dppp.types import *
from dppp.utils import create_resize_fn, convolve, inpaint_border, ConvolutionModes

# TODO the signs are different here from the paper and report:
# They should be changed without changing the results


###############################################################################
# DEBLUR NON-BLIND
###############################################################################

def dmsp_deblur_nb(degraded, kernel, noise_stddev,
                   denoiser: Denoiser, denoiser_stddev: float,
                   num_steps: int = 300,
                   learning_rate: float = 1e-4,
                   momentum: float = 0.9,
                   mode: str = ConvolutionModes.CONSTANT,
                   callbacks: Iterable[Callback] = None
                   ) -> TensorLike:

    if callbacks is None:
        callbacks = []

    if mode == ConvolutionModes.CONSTANT:
        deblur_nb_grad = _deblur_nb_grad_constant
    elif mode == ConvolutionModes.WRAP:
        deblur_nb_grad = _deblur_nb_grad_wrap
    else:
        raise ValueError(f"'mode' must be either 'constant' or 'wrap'. Got '{mode}'.")

    x = degraded
    velocity = tf.zeros_like(x)

    for step in tf.range(num_steps, dtype=tf.int64):
        # Compute the gradient
        grad = deblur_nb_grad(x, degraded, kernel, noise_stddev) + \
            dmsp_prior_grad(x, denoiser, denoiser_stddev)

        # Apply the gradient with momentum
        velocity = momentum * velocity - learning_rate * grad
        x = x + velocity

        # Clip between 0 and 1
        x = tf.clip_by_value(x, 0, 1)

        # Call the callbacks with the value of the variable
        for callback in callbacks:
            callback(x, step)

    return x


# Gradient functions

@tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[], dtype=tf.float32)))
def _deblur_nb_grad_wrap(x: TensorLike, observation: TensorLike, kernel: TensorLike,
                         noise_stddev: TensorLike) -> TensorType:
    """Compute the gradient for the data term for non-blind deblurring.

    Args:
        x: The images at the current step.
        observation: The original degraded observation of the data with wrap border.
        kernel: The blurring kernel
        noise_stddev: The standard deviation of the degration noise

    Returns:
        The gradient of the data term of x.
    """
    _, data_grad = _compute_deblur_err_grad(x, observation, kernel, mode=ConvolutionModes.WRAP)
    return (1 / (noise_stddev**2)) * data_grad  # type: ignore


@tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[], dtype=tf.float32)))
def _deblur_nb_grad_constant(x: TensorLike, observation: TensorLike, kernel: TensorLike,
                             noise_stddev: TensorLike) -> TensorType:
    """Compute the gradient for the data term for non-blind deblurring.

    Args:
        x: The images at the current step.
        observation: The original degraded observation of the data with constant border.
        kernel: The blurring kernel
        noise_stddev: The standard deviation of the degration noise

    Returns:
        The gradient of the data term of x.
    """
    _, data_grad = _compute_deblur_err_grad(x, observation, kernel, mode=ConvolutionModes.CONSTANT)
    return (1 / (noise_stddev**2)) * data_grad  # type: ignore


@tf.function(input_signature=(tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
                              tf.TensorSpec(shape=[], dtype=tf.float32)))
def _deblur_na_grad(x: TensorLike, observation: TensorLike, kernel: TensorLike,
                    sigma2: TensorLike) -> TensorType:
    data_err, data_grad = _compute_deblur_err_grad(x, observation, kernel)
    lambda_ = _compute_deblur_na_lambda(x, observation, kernel, data_err,
                                        sigma2)
    return lambda_ * data_grad  # type: ignore


# Helper functions

def _compute_deblur_err_grad(x: TensorLike, observation: TensorLike,
                             kernel: TensorLike,
                             mode: str = ConvolutionModes.CONSTANT) -> Tuple[TensorType, TensorType]:
    map_conv = convolve(x, kernel, mode=mode)
    data_err = map_conv - observation
    data_grad = convolve(data_err,
                         _flip_kernel_2d(kernel),
                         mode=mode)
    return data_err, data_grad


def _compute_deblur_na_lambda(x: TensorLike, observation: TensorLike, kernel: TensorLike,
                              data_err: TensorLike, sigma2: TensorLike) -> TensorType:
    x_size = tf.cast(tf.math.reduce_prod(tf.shape(x)[1:]), tf.float32)
    obs_size = tf.cast(tf.math.reduce_prod(tf.shape(observation)[1:]),
                       tf.float32)
    sum_data_err2 = tf.math.reduce_sum(data_err**2)  # type: ignore
    sum_kernel2 = tf.math.reduce_sum(kernel[:, :, 0, 0]**2)  # type: ignore
    return obs_size / (sum_data_err2 + x_size * sigma2 * sum_kernel2)  # type: ignore


def _flip_kernel_2d(kernel):
    """Flip the given convolution kernel (with padding if needed).

    The kernel is padded in all dimensions with uneven length by adding a zero in front of this dimension.

    Args:
        kernel: The convolution kernel of shape [KERNEL_HEIGHT, KERNEL_WIDTH, FEATURES_IN, FEATURES_OUT].

    Returns:
        The flipped convolution kernel.
    """
    pad = tf.cast(tf.math.floormod(tf.shape(kernel), 2) == 0, tf.int32)
    paddings = [[pad[0], 0], [pad[1], 0], [0, 0], [0, 0]]  # type: ignore
    kernel_padded = tf.pad(kernel, paddings, 'CONSTANT')
    return kernel_padded[::-1, ::-1, ...]


###############################################################################
# SUPER RESOLVE
###############################################################################

def dmsp_super_resolve(degraded: TensorLike,
                       sr_factor: int,
                       denoiser: Denoiser,
                       denoiser_stddev: float,
                       resize_fn: ResizeFnType = None,
                       prior_weight: float = 2e-3,
                       num_steps: int = 300,
                       learning_rate: float = 1,
                       momentum: float = 0.9,
                       callbacks: Iterable[Callback] = None
                       ) -> TensorLike:

    if resize_fn is None:
        resize_fn = create_resize_fn('bicubic', True)

    if callbacks is None:
        callbacks = []

    def grad_fn(x: TensorLike, step: TensorLike) -> TensorType:
        # Data gradient
        x_lr = resize_fn(x, sr_factor, False)
        data_err = x_lr - degraded  # type: ignore
        data_grad = resize_fn(data_err, sr_factor, True)

        # Prior gradient
        prior_grad = dmsp_prior_grad(x, denoiser, denoiser_stddev)

        # Combined gradient
        step_f32 = tf.cast(step, tf.float32)
        current_prior_weight = prior_weight / tf.math.sqrt(step_f32 + 1)  # type: ignore
        return data_grad + current_prior_weight * prior_grad

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


def dmsp_inpaint(degraded: TensorLike,
                 mask: TensorLike,
                 denoiser: Denoiser,
                 denoiser_stddev: float,
                 init_fn: InpaintFnType = None,
                 prior_weight: float = 0.2,
                 num_steps: int = 300,
                 learning_rate: float = 1e-2,
                 momentum: float = 0.9,
                 callbacks: Iterable[Callback] = None
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
    return (1 / (noise_stddev**2)) * grad  # type: ignore
