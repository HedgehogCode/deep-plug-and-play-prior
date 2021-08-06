import os
import functools
import itertools
import math
import re
from typing import Tuple

import h5py
import scipy.io
import numpy as np

import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow_addons as tfa
import tensorflow_datasets_bw as tfdsbw

from dppp.types import *

IMAGE_TENSOR_SPEC = tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32)
CHANNEL_KERNEL_TENSOR_SPEC = tf.TensorSpec(
    shape=[None, None, None, None], dtype=tf.float32
)
SCALAR_TENSOR_SPEC = tf.TensorSpec(shape=[], dtype=tf.float32)

# Kernels by Levin et al. 2009
with h5py.File(
    os.path.normpath(os.path.join(__file__, "..", "kernels", "Levin09.mat")), "r"
) as f:
    NB_DEBLURRING_LEVIN_KERNELS = [
        f[k_ref[0]][()].astype("float32") for k_ref in f["kernels"][()]
    ]

# Isotropic and anisotropic Gaussian kernels by Zhang 2021
ZHANG_GAUSSIAN_KERNELS = np.stack(
    scipy.io.loadmat(
        os.path.normpath(os.path.join(__file__, "..", "kernels", "kernels_12.mat"))
    )["kernels"][0],
    axis=0,
)[:8]

# ==================================================================================================
# DENOISING
# ==================================================================================================


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
        tf.TensorSpec(shape=[], dtype=tf.float32),
    ]

    # Match the noise standard deviation in the model name
    groups = re.match(r".*?(\d\.\d+)?-?(\d\.\d+)\.h5", path_to_model)
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


# ==================================================================================================
# CONVOLUTIONS
# ==================================================================================================


class ConvolutionModes(object):
    """Modes for border handling when computing a convolution"""

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
        padding = "SAME"
        image_extended = image
    elif mode == ConvolutionModes.WRAP:
        padding = "VALID"
        image_extended = _wrap_image_for_conv(image, kernel)
    else:
        raise ValueError(
            f"'mode' must be either '{ConvolutionModes.CONSTANT}' or "
            + f"'{ConvolutionModes.WRAP}'. Got '{mode}'."
        )

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
    return image_wrapped[:, (h - kh) : (2 * h + kh), (w - kw) : (2 * w + kw), :]


@tf.function()
def gaussian_kernel(size: int, mean: float, std: float):
    """Creates a 2D gaussian kernel for convolution."""
    d = tfp.distributions.Normal(mean, std)
    vals = d.prob(tf.range(start=-size, limit=size + 1, dtype=tf.float32))
    gauss_kernel = tf.einsum("i,j->ij", vals, vals)
    return gauss_kernel / tf.reduce_sum(gauss_kernel)


@tf.function()
def gaussian_kernel_multivariate(size, var_ii, var_jj, var_ij):
    loc = [0.0, 0.0]
    covariance_matrix = [[var_ii, var_ij], [var_ij, var_jj]]

    d = tfp.distributions.MultivariateNormalTriL(
        loc=loc,
        scale_tril=tf.linalg.cholesky(covariance_matrix),
    )

    coords_i = tf.range(start=-size[0], limit=size[0] + 1, dtype=tf.float32)
    coords_j = tf.range(start=-size[1], limit=size[1] + 1, dtype=tf.float32)
    grid = tf.stack(tf.meshgrid(coords_i, coords_j, indexing="ij"), axis=-1)

    kernel = d.prob(grid)
    return kernel / tf.reduce_sum(kernel)


def flip_kernel_2d(kernel):
    """Flip the given convolution kernel (with padding if needed).

    The kernel is padded in all dimensions with uneven length by adding a zero in front of this dimension.

    Args:
        kernel: The convolution kernel of shape [KERNEL_HEIGHT, KERNEL_WIDTH, FEATURES_IN, FEATURES_OUT].

    Returns:
        The flipped convolution kernel.
    """
    pad = tf.cast(tf.math.floormod(tf.shape(kernel), 2) == 0, tf.int32)
    paddings = [[pad[0], 0], [pad[1], 0], [0, 0], [0, 0]]  # type: ignore
    kernel_padded = tf.pad(kernel, paddings, "CONSTANT")
    return kernel_padded[::-1, ::-1, ...]


# ==================================================================================================
# RESCALING
# ==================================================================================================


def create_resize_fn(method="bicubic", antialias=True) -> ResizeFnType:
    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None, None, None], tf.float32),
            tf.TensorSpec([], tf.int32),
            tf.TensorSpec([], tf.bool),
        ]
    )
    def apply(images, factor, upscale):
        if upscale:
            scale = tf.cast(factor, tf.float32)
        else:
            scale = tf.cast(1 / factor, tf.float32)
        old_size = tf.cast(tf.shape(images)[1:-1], tf.float32)
        new_size = tf.cast(tf.math.round(old_size * scale), tf.int32)  # type: ignore
        return tf.image.resize(images, new_size, method=method, antialias=antialias)

    return apply


def create_convolve_resize_fn(kernel, mode=ConvolutionModes.CONSTANT):
    @tf.function(
        input_signature=[
            tf.TensorSpec([None, None, None, None], tf.float32),
            tf.TensorSpec([], tf.int32),
            tf.TensorSpec([], tf.bool),
        ]
    )
    def resize_fn(images, factor, upscale):
        if upscale:
            upscaled = upscale_no_iterpolation(images, factor)
            return convolve(upscaled, flip_kernel_2d(kernel), mode=mode)
        else:
            blurred = convolve(images, kernel, mode=mode)
            return blurred[:, ::factor, ::factor, :]

    return resize_fn


@tf.function()
def upscale_no_iterpolation(x, scale):
    x_shape = tf.shape(x)
    upscaled_shape = [x_shape[0], x_shape[1] * scale, x_shape[2] * scale, x_shape[3]]
    x_added_dim = x[:, :, None, :, None, :]
    x_padded = tf.pad(
        x_added_dim, [[0, 0], [0, 0], [0, scale - 1], [0, 0], [0, scale - 1], [0, 0]]
    )
    return tf.reshape(x_padded, upscaled_shape)


# ==================================================================================================
# OPTICAL FLOW
# ==================================================================================================


def resize_flow(
    flow: TensorLike, factor: int, upscale: bool, resize_fn: ResizeFnType = None
) -> tf.Tensor:
    """Resize optical flow.

    Args:
        flow (TensorLike): The optical flow
        factor (int): The factor by which it should be scaled
        upscale (bool): If the flow should be upscaled or downscaled
        resize_fn (ResizeFnType, optional): The resize fn to use for resizing interpolating the
            values. Defaults to 'bicubic' with antialiasing.

    Returns:
        tf.Tensor: The resized flow.
    """
    if resize_fn is None:
        resize_fn = create_resize_fn()

    vector_scale = tf.cast(factor if upscale else (1.0 / factor), tf.float32)
    flow_f32 = tf.cast(flow, tf.float32)
    return resize_fn(flow_f32, factor, upscale) * vector_scale


def reference_flows(
    images: TensorLike,
    reference_idx: int,
    backwards: bool = False,
    flow_fn: FlowFnType = None,
) -> tf.Tensor:
    """Compute the flows from the reference image to the individual images (or backwards).

    Args:
        images (TensorLike): A batch of individual images with the reference image at index
            reference_idx. Shape [N, H, W, C]
        reference_idx (int): The index of the reference frame in the images batch.
        backwards (bool, optional): If the backward flows should be computed. Defaults to False.
        flow_fn (FlowFnType, optional): the function to use to estimate the flow. Defaults to
            `dppp.flow_pyflow`.

    Returns:
        tf.Tensor: The flows from the reference image to the individual images (or backwards).
    """
    if flow_fn is None:
        flow_fn = flow_pyflow

    num_images = tf.shape(images)[0]
    reference_image = images[reference_idx]
    reference_tensor = tf.repeat(reference_image[None, ...], [num_images], axis=0)
    if backwards:
        return flow_fn(images, reference_tensor)
    else:
        return flow_fn(reference_tensor, images)


def flow_pyflow(
    img_a: TensorLike,
    img_b: TensorLike,
    alpha=0.012,
    ratio=0.8,
    min_width=20,
    n_outer_fp_iterations=7,
    n_inner_fp_iterations=1,
    n_sor_iterations=30,
) -> tf.Tensor:
    """Estimate the optical flow using the pyflow library.

    The pyflow library [1] is a Python wrapper for Ce Liu's C++ implementation [2] of Corse2Fine
    Optical Flow.

    Args:
        img_a (TensorLike): Start image for the optical flow. Shape [N, H, W, C]
        img_b (TensorLike): Destination image for the optical flow. Shape [N, H, W, C]
        alpha (float, optional): Regularization weight. Defaults to 0.012.
        ratio (float, optional): Downsample ratio. Defaults to 0.8.
        min_width (int, optional): Minimal witdth of the coarsest level. Defaults to 20.
        n_outer_fp_iterations (int, optional): Number of outer fixed point iterations. Defaults to 7.
        n_inner_fp_iterations (int, optional): Number of inner fixed point iterations. Defaults to 1.
        n_sor_iterations (int, optional): Number of SOR iterations. Defaults to 30.

    Returns:
        tf.Tensor: The optical flow with shape [N, H, W, 2]

    References:
        [1] https://github.com/pathak22/pyflow
        [2] https://people.csail.mit.edu/celiu/OpticalFlow/
    """
    import pyflow

    num_channels = tf.shape(img_a)[-1]
    if num_channels == 3:
        # RGB
        color_type = 0
    elif num_channels == 1:
        # Grayscale
        color_type = 1
    else:
        raise ValueError("Number of channels must be 1 or 3.")

    pyflow_fn = functools.partial(
        pyflow.coarse2fine_flow,
        alpha=alpha,
        ratio=ratio,
        minWidth=min_width,
        nOuterFPIterations=n_outer_fp_iterations,
        nInnerFPIterations=n_inner_fp_iterations,
        nSORIterations=n_sor_iterations,
        colType=color_type,
    )

    def flow_on_stacked(x: TensorLike):
        u, v, _ = tf.numpy_function(pyflow_fn, [x[..., 0], x[..., 1]], tf.float64)
        return tf.stack([v, u], axis=-1)

    # Make sure the input type is float64
    a = tf.cast(img_a, dtype=tf.float64)
    b = tf.cast(img_b, dtype=tf.float64)

    # Map the function on the complete batch
    stacked = tf.stack([a, b], axis=-1)
    return tf.map_fn(flow_on_stacked, stacked)


def backward_flow(
    forward_flow: TensorLike,
    img1: TensorLike = None,
    img2: TensorLike = None,
) -> tf.Tensor:
    """Compute the backward flow from the forward flow.

    If no images are given the negative flow is warped using the flow itself.
    If images are given pyinverseflow is used to compute the inverse flow using
    the strategy "max_image" and fill method "oriented".

    Args:
        forward_flow (TensorLike): The forward flow. Shape [N, H, W, 2]
        img1 (TensorLike, optional): The first image. Shape [N, H, W, 3]
        img2 (TensorLike, optional): The second image. Shape [N, H, W, 3]

    Returns:
        tf.Tensor: The backward flow. Shape [N, H, W, 2]
    """
    # No images given. The best we can do is warping the negative flow
    if img1 is None or img2 is None:
        return tfa.image.dense_image_warp(-forward_flow, forward_flow)

    # Images given. We can use pyinverseflow with strategy "max_image"
    from pyinverseflow import inverse_flow

    def inv_flow_fn(f, i1, i2):
        return inverse_flow(f, i1, i2, strategy="max_image", fill="oriented")

    def inverse_flow_on_concat(x: TensorLike):
        bf, _ = tf.numpy_function(
            inv_flow_fn, [x[..., :2], x[..., 2:5], x[..., 5:8]], tf.float32
        )
        return bf

    concatenated = tf.concat([forward_flow, img1, img2], axis=-1)
    return tf.map_fn(inverse_flow_on_concat, concatenated)


# ==================================================================================================
# LIGHT FIELD
# ==================================================================================================


def lf_flows(
    disparity: TensorLike,
    grid_height: int,
    grid_width: int,
    center_i: int = None,
    center_j: int = None,
):
    """Get the forward flows from the given light field disparity map.

    The forward flow is the flow from the center image to the individual images.

    Args:
        disparity (TensorLike): The disparity map of shape [H, W].
        grid_height (int): The number of lightfield frames in y direction.
        grid_width (int): The number of lightfield frames in x direction.
        center_i (int, optional): The index of the center frame. Defaults to (grid_height-1)/2.
        center_j (int, optional): The index of the center frame. Defaults to (grid_width-1)/2.

    Returns:
        Tensor: The forward flows for all lightfield frames of shape
            [grid_height, grid_width, H, W, 2]
    """

    def center(c, s):
        if c is None:
            return (tf.cast(s, dtype=tf.float32) - 1.0) / 2.0
        return c

    center_i = center(center_i, grid_height)
    center_j = center(center_j, grid_width)

    coords_i = tf.range(grid_height, dtype=tf.float32) - center_i
    coords_j = tf.range(grid_width, dtype=tf.float32) - center_j
    grid = tf.stack(tf.meshgrid(coords_i, coords_j, indexing="ij"), axis=-1)

    return -disparity[None, None, ..., None] * grid[:, :, None, None, :]


def lf_to_video(lf: TensorLike) -> tf.Tensor:
    """Flatten a light field such that in the video a frame always follows a neighboring light
    field frame.

    lf = [
        [ 0  1  2  3]
        [ 4  5  6  7]
        [ 8  9 10 11]
        [12 13 14 15]
        [16 17 18 19]
    ]
    lf_to_video(lf):
        [ 0,  1,  2,  3,  7,  6,  5,  4,  8,  9, 10, 11, 15, 14, 13, 12, 16, 17, 18, 19]

    Args:
        lf (TensorLike): The light field of shape [GH, GW, H, W, C]

    Returns:
        tf.Tensor: The flattened light field of shape [GH * GW, H, W, C]
    """
    gh = tf.shape(lf)[0]
    inner_shape = tf.shape(lf)[2:]

    to_right = lf[::2, :]  # Rows that are not flipped
    to_left = lf[1::2, ::-1]  # Rows that are flipped

    # Stich back together
    to_right_indices = tf.range(gh)[::2]
    to_left_indices = tf.range(gh)[1::2]
    stitched = tf.dynamic_stitch(
        [to_right_indices, to_left_indices], [to_right, to_left]
    )

    # Flatten the first axis
    return tf.reshape(stitched, [-1, *inner_shape])


def resize_lf(
    lf: TensorLike, factor: int, upscale: bool, resize_fn: ResizeFnType = None
):
    """Resize a light field.

    Args:
        lf (TensorLike): The light field with shape [GH, GW, H, W, C]
        factor (int): The factor by which it should be scaled
        upscale (bool): If the light field should be upscaled or downscaled
        resize_fn (ResizeFnType, optional): The resize fn to use for resizing interpolating the
            values. Defaults to 'bicubic' with antialiasing.

    Returns:
        tf.Tensor: The resized light field
    """
    if resize_fn is None:
        resize_fn = create_resize_fn()

    lf_grid = tf.shape(lf)[:2]
    orig_size = tf.shape(lf)[2:]
    lf_batch = tf.reshape(lf, tf.concat([[-1], orig_size], axis=0))
    lf_resized_batch = resize_fn(lf_batch, factor, upscale)
    resized_size = tf.shape(lf_resized_batch)[1:]
    return tf.reshape(lf_resized_batch, tf.concat([lf_grid, resized_size], axis=0))


LFATTNET_MODEL = None


def lf_disparity_lfattnet(lf: TensorLike) -> tf.Tensor:
    """Estimate the disparity of the light field by using LFattNet [1].

    Currently, this function only supports light fields of a 9x9 grid.

    Args:
        lf (TensorLike): The light field of shape [9, 9, H, W, 3]

    Returns:
        tf.Tensor: The disparity estimation of shape [H, W]

    References:
        [1] Y.-J. Tsai, Y.-L. Liu, M. Ouhyoung, and Y.-Y. Chuang, “Attention-Based View
            Selection Networks for Light-Field Disparity Estimation,” AAAI, vol. 34, no. 07,
            pp. 12095–12103, Apr. 2020, doi: 10.1609/aaai.v34i07.6888.
    """

    def load_model():
        model_file = tf.keras.utils.get_file(
            f"LFattNet_9x9.h5",
            "https://github.com/HedgehogCode/deep-plug-and-play-prior/releases/download"
            + f"/thesis/LFattNet_9x9.h5",
        )
        return tf.keras.models.load_model(
            model_file, custom_objects={"tf": tf, "tfa": tfa}, compile=False
        )

    # Load the model if it is not yet loaded
    global LFATTNET_MODEL
    if LFATTNET_MODEL is None:
        LFATTNET_MODEL = load_model()

    # RGB -> Gray
    rgb = [0.299, 0.587, 0.114]
    inp = (rgb[0] * lf[..., 0] + rgb[1] * lf[..., 1] + rgb[2] * lf[..., 2])[..., None]

    # LF tensor to list of inputs
    inp = [f[None, ...] for f in tfdsbw.lf_to_batch(inp)]

    # Run the model
    return LFATTNET_MODEL(inp)[0][0]


# ==================================================================================================
# INPAINTING
# ==================================================================================================


def inpaint_border(
    x: TensorLike, mask: TensorLike, blur_stddev: float = 0, noise_stddev: float = 0
) -> TensorType:
    """Inpaint the image x at the locations where mask is 0 by replicating the nearest pixels."""

    def cond(x, x_mask):
        return tf.reduce_any(x_mask == 0)

    def body(x, x_mask):
        def roll_in_all_directions(v):
            list_of_rolled = [
                tf.roll(v, shift=shift, axis=axis)
                for shift, axis in itertools.product([1, -1], [1, 2])
            ]
            return tf.stack(list_of_rolled, axis=-1)

        image_neighbors = roll_in_all_directions(x)
        mask_neighbors = roll_in_all_directions(x_mask)

        image_neighbors_sum = tf.reduce_sum(image_neighbors, axis=-1)
        mask_neighbors_sum = tf.reduce_sum(mask_neighbors, axis=-1)
        image_neighbors_mean = tf.math.divide_no_nan(
            image_neighbors_sum, mask_neighbors_sum
        )

        return (
            x * x_mask + image_neighbors_mean * (1 - x_mask),
            tf.clip_by_value(mask_neighbors_sum, clip_value_min=0, clip_value_max=1),
        )

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
    """Fill the image x at the locations where mask is 0 with random normal noise N(0.5,0.25)."""
    noise = tf.clip_by_value(tf.random.normal(tf.shape(x), mean=0.5, stddev=0.25), 0, 1)
    return x * mask + noise * (1 - mask)  # type: ignore


def inpaint_random_uniform(x: TensorLike, mask: TensorLike) -> TensorType:
    """Fill the image x at the locations where mask is 0 with random uniform noise between 0 and 1."""
    noise = tf.random.uniform(tf.shape(x), minval=0, maxval=1)
    return x * mask + noise * (1 - mask)  # type: ignore


def inpaint_gray(x: TensorLike, mask: TensorLike) -> TensorType:
    """Fill the image x at the locations where mask is 0 with gray values (0.5)."""
    gray = tf.ones(tf.shape(mask)) / 2
    return x * mask + gray * (1 - mask)  # type: ignore


def inpaint_mean(x: TensorLike, mask: TensorLike) -> TensorType:
    """Fill the image x at the locations where mask is 0 with the mean value of the image."""
    mean = tf.ones(tf.shape(mask)) * tf.reduce_mean(x)
    return x * mask + mean * (1 - mask)  # type: ignore
