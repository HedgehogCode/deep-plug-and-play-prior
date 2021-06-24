import tensorflow as tf
from image_similarity_measures import quality_metrics

from dppp.types import *

def psnr(image_a: TensorLike, image_b: TensorLike) -> TensorType:
    return tf.image.psnr(image_a, image_b, max_val=1)  # type: ignore


def ssim(image_a: TensorLike, image_b: TensorLike) -> TensorType:
    return tf.image.ssim(image_a, image_b, max_val=1)


def fsim(image_a: TensorLike, image_b: TensorLike) -> TensorType:
    def fsim_on_stacked_image(x):
        fsim_val = tf.numpy_function(quality_metrics.fsim, [x[...,0], x[...,1]], tf.float64)
        return tf.cast(fsim_val, tf.float32)
    stacked = tf.stack([image_a, image_b], axis=-1)
    return tf.map_fn(fsim_on_stacked_image, stacked)  # type: ignore