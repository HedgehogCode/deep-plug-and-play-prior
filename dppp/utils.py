import itertools
import math
import re
from typing import Tuple

import tensorflow as tf
import tensorflow_probability as tfp

from dppp.types import *


class ConvolutionModes(object):
    """ Modes for border handling when computing a convolution """
    CONSTANT = "constant"
    WRAP = "wrap"


def conv2D_filter_rgb(kernel):
    """Creates a conv filter that applies the 2D kernel on each channel of a
    RBG image.

    Args:
        kernel: The 2D convolution kernel with the shape [H, W].

    Returns:
        A convolution kernel with the shape [H, W, 3, 3].
    """
    zeros = tf.zeros(tf.shape(kernel))
    filter_r = tf.stack([kernel, zeros, zeros], axis=-1)
    filter_g = tf.stack([zeros, kernel, zeros], axis=-1)
    filter_b = tf.stack([zeros, zeros, kernel], axis=-1)
    return tf.stack([filter_r, filter_g, filter_b], axis=-1)


@tf.function()
def blur(image, kernel, noise_stddev, mode=ConvolutionModes.CONSTANT, clip_final=True):
    """Blur the given image/images with the given kernel and add gaussian noise.

    The given image is blurred with the given convolution kernel and normal distributed noise is added.
    The result is clipped to a range from 0 to 1.

    Args:
        image: The image tensor of shape [BATCH, HEIGHT, WIDTH, CHANNELS]
        kernel: The blur kernel of shape [KERNEL_HEIGHT, KERNEL_WIDTH, IN_CHANNELS, OUT_CHANNELS].
        noise_stddev: The standard deviation of the gaussian distributed noise.
        mode: Defines how the input is extended beyond its boundaries
            (one of {'constant', 'wrap'}, default: 'constant')

    Returns:
        The blurred image/images with added noise.
    """
    # Apply the convolution
    image_filtered = convolve(image, kernel, mode=mode)

    # Add noise
    noise = tf.random.normal(tf.shape(image), stddev=noise_stddev)
    degraded = image_filtered + noise

    # Clip (if clip_final is True) and return
    if clip_final:
        return tf.clip_by_value(degraded, 0, 1)
    return degraded


@tf.function()
def convolve(image, kernel, mode=ConvolutionModes.CONSTANT):
    """Convolve the image/images with the given kernel using the border handling mode.

    Args:
        image: The image tensor of shape [BATCH, HEIGHT, WIDTH, CHANNELS]
        kernel: The blur kernel of shape [KERNEL_HEIGHT, KERNEL_WIDTH, IN_CHANNELS, OUT_CHANNELS].
        mode: Defines how the input is extended beyond its boundaries
            (one of {'constant', 'wrap'}, default: 'constant')

    Returns:
        The convolved image/images
    """
    if mode == ConvolutionModes.CONSTANT:
        padding = 'SAME'
        image_extended = image
    elif mode == ConvolutionModes.WRAP:
        padding = 'VALID'
        image_extended = _wrap_image_for_conv(image, kernel)
    else:
        raise ValueError(
            f"'mode' must be either '{ConvolutionModes.CONSTANT}' or " +
            f"'{ConvolutionModes.WRAP}'. Got '{mode}'.")

    # Apply the convolution
    return tf.nn.conv2d(image_extended, kernel, strides=[1, 1, 1, 1], padding=padding)


def _wrap_image_for_conv(image, kernel):
    im_shape = tf.shape(image)
    k_shape = tf.math.floordiv(tf.shape(kernel), 2)  # Half the kernel height/width
    h, w = im_shape[1], im_shape[2]
    kh, kw = k_shape[0], k_shape[1]

    # Wrap in all directions
    image_wrapped = tf.tile(image, [1, 3, 3, 1])

    # Only take the part that we need
    return image_wrapped[:, (h-kh):(2*h+kh), (w-kw):(2*w+kw), :]


def load_denoiser(path_to_model: str) -> Tuple[Denoiser, Tuple[float, float]]:
    """Load the denoiser from the given h5 file.

    The path must end with the noise standard deviation the model is best suited for. If the model
    is suited for a range of noise standard deviation values the minimum and maximum value must be
    separated by a '-'.
    Examples:
    * "/path/to/model_0.04.h5" -> min_sigma=0.04 max_sigma=0.04
    * "../relative/to/anothermodel-0.01-0.04.h5" -> min_sigma=0.01 max_sigma=0.04

    Args:
        path_to_model: the path to the model as specified above.

    Returns:
        A tuple consisting of
        * The denoiser of type Denoiser. Accepting a tensor (noisy image) and a float (noise_stddev)
          and returing a tensor
        * A tuple containing the min_sigma and max_sigma of the model
    """
    denoiser_input_signature = [
        tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
        tf.TensorSpec(shape=[], dtype=tf.float32)
    ]

    # Match the noise standard deviation in the model name
    groups = re.match(r'.*?(\d\.\d+)?-?(\d\.\d+)\.h5', path_to_model)
    # TODO error handling
    if groups[1]:
        model_noise_stddev = (float(groups[1]), float(groups[2]))
    else:
        model_noise_stddev = (float(groups[2]), float(groups[2]))

    # Load the model
    model_denoiser = tf.keras.models.load_model(path_to_model, compile=False)

    # Noise map input
    if model_denoiser.input.shape[-1] == 4:  # type: ignore
        @tf.function(input_signature=denoiser_input_signature)
        def denoiser_map(x: TensorLike, noise_stddev: TensorLike) -> TensorType:
            noise_map_shape = (tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], 1)
            noise_map = tf.broadcast_to(noise_stddev, noise_map_shape)
            return model_denoiser(tf.concat([x, noise_map], axis=-1))  # type: ignore

        return (denoiser_map, model_noise_stddev)

    # No noise map input
    @tf.function(input_signature=denoiser_input_signature)
    def denoiser(x: TensorLike, _: TensorLike) -> TensorType:
        return model_denoiser(x)  # type: ignore

    return (denoiser, model_noise_stddev)


@tf.function()
def gaussian_kernel(size: int, mean: float, std: float):
    """Makes 2D gaussian Kernel for convolution."""
    d = tfp.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum('i,j->ij', vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


def create_resize_fn(method='bicubic', antialias=True) -> ResizeFnType:
    @tf.function(input_signature=[tf.TensorSpec([None, None, None, None], tf.float32),
                                  tf.TensorSpec([], tf.int32),
                                  tf.TensorSpec([], tf.bool)])
    def apply(images, factor, upscale):
        if upscale:
            scale = tf.cast(factor, tf.float32)
        else:
            scale = tf.cast(1 / factor, tf.float32)
        old_size = tf.cast(tf.shape(images)[1:-1], tf.float32)
        new_size = tf.cast(tf.math.round(old_size * scale), tf.int32)  # type: ignore
        return tf.image.resize(images, new_size, method=method, antialias=antialias)
    return apply


# Simple inpainting functions

def inpaint_border(x: TensorLike,
                   mask: TensorLike,
                   blur_stddev: float = 0,
                   noise_stddev: float = 0
                   ) -> TensorType:
    """Inpaint the image x at the locations where mask is 0 by replicating the nearest pixels.
    """

    def cond(x, x_mask):
        return tf.reduce_any(x_mask == 0)

    def body(x, x_mask):
        def roll_in_all_directions(v):
            list_of_rolled = [tf.roll(v, shift=shift, axis=axis)
                              for shift, axis in itertools.product([1, -1], [1, 2])]
            return tf.stack(list_of_rolled, axis=-1)

        image_neighbors = roll_in_all_directions(x)
        mask_neighbors = roll_in_all_directions(x_mask)

        image_neighbors_sum = tf.reduce_sum(image_neighbors, axis=-1)
        mask_neighbors_sum = tf.reduce_sum(mask_neighbors, axis=-1)
        image_neighbors_mean = tf.math.divide_no_nan(image_neighbors_sum, mask_neighbors_sum)

        return (x * x_mask + image_neighbors_mean * (1 - x_mask),
                tf.clip_by_value(mask_neighbors_sum, clip_value_min=0, clip_value_max=1))

    inpainted = tf.while_loop(cond, body, loop_vars=[x, mask])[0]

    # Apply blur to inpainted region
    if blur_stddev > 0:
        k = gaussian_kernel(math.floor(blur_stddev * 3), 0, blur_stddev)
        blurred = blur(inpainted, conv2D_filter_rgb(k), 0)
        inpainted = mask * inpainted + (1 - mask) * blurred  # type: ignore

    # Apply noise to inpainted region
    if noise_stddev > 0:
        noise = tf.random.normal(tf.shape(inpainted), stddev=noise_stddev)
        noisy = inpainted + noise
        inpainted = mask * inpainted + (1 - mask) * noisy  # type: ignore

    return inpainted


def inpaint_random_normal(x: TensorLike, mask: TensorLike) -> TensorType:
    """Fill the image x at the locations where mask is 0 with random normal noise N(0.5,0.25).
    """
    noise = tf.clip_by_value(tf.random.normal(tf.shape(x), mean=0.5, stddev=0.25), 0, 1)
    return x * mask + noise * (1 - mask)  # type: ignore


def inpaint_random_uniform(x: TensorLike, mask: TensorLike) -> TensorType:
    """Fill the image x at the locations where mask is 0 with random uniform noise between 0 and 1.
    """
    noise = tf.random.uniform(tf.shape(x), minval=0, maxval=1)
    return x * mask + noise * (1 - mask)  # type: ignore


def inpaint_gray(x: TensorLike, mask: TensorLike) -> TensorType:
    """Fill the image x at the locations where mask is 0 with gray values (0.5).
    """
    gray = tf.ones(tf.shape(mask)) / 2
    return x * mask + gray * (1 - mask)  # type: ignore


def inpaint_mean(x: TensorLike, mask: TensorLike) -> TensorType:
    """Fill the image x at the locations where mask is 0 with the mean value of the image.
    """
    mean = tf.ones(tf.shape(mask)) * tf.reduce_mean(x)
    return x * mask + mean * (1 - mask)  # type: ignore
