import tensorflow as tf
from image_similarity_measures import quality_metrics

LPIPS_ALEX_MODEL_URL = "https://github.com/HedgehogCode/lpips-tf2/releases/download/0.1.0/lpips_lin_alex.h5"
LPIPS_ALEX_MODEL_NAME = "lpips_lin_alex_0.2.0"
LPIPS_ALEX_MODEL_MD5 = "a35b66a420f518161f715c0675d9bbfb"
lpips_model_alex = None

LPIPS_VGG_MODEL_URL = (
    "https://github.com/HedgehogCode/lpips-tf2/releases/download/0.1.0/lpips_lin_vgg.h5"
)
LPIPS_VGG_MODEL_NAME = "lpips_lin_vgg_0.2.0"
LPIPS_VGG_MODEL_MD5 = "ef185d82115f86ac5736266e02f9222c"
lpips_model_vgg = None


def _handle_unbatched_inputs(metric_fn):
    """Decorator to allow using a function that is defined on batches on single images."""

    def fn(imgs_a, imgs_b):
        if tf.rank(imgs_a) == 3:
            return metric_fn(imgs_a[None, ...], imgs_b[None, ...])[0]
        return metric_fn(imgs_a, imgs_b)

    return fn


@_handle_unbatched_inputs
def psnr(imgs_a, imgs_b):
    return tf.image.psnr(imgs_a, imgs_b, max_val=1)


@_handle_unbatched_inputs
def ssim(imgs_a, imgs_b):
    return tf.image.ssim(imgs_a, imgs_b, max_val=1)


@_handle_unbatched_inputs
def fsim(imgs_a, imgs_b):
    """FSIM: A Feature Similarity Index for Image Quality Assessment

    Lin Zhang, Lei Zhang, Xuanqin Mou, and D. Zhang,
    “FSIM: A Feature Similarity Index for Image Quality Assessment,”
    IEEE Trans. on Image Process., vol. 20, no. 8, pp. 2378–2386, Aug. 2011,
    doi: 10.1109/TIP.2011.2109730.
    """
    # Function that runs FSIM on [H, W, 3, 2]
    # where the last dimension has the two images to compare
    def fsim_on_stacked_image(x):
        fsim_val = tf.numpy_function(
            quality_metrics.fsim, [x[..., 0], x[..., 1]], tf.float64
        )
        return tf.cast(fsim_val, tf.float32)

    # Ensure the type is correct
    a = tf.cast(imgs_a, tf.float32)
    b = tf.cast(imgs_b, tf.float32)

    # Stack the images and map the function over the batch
    stacked = tf.stack([a, b], axis=-1)
    return tf.map_fn(fsim_on_stacked_image, stacked)  # type: ignore


@_handle_unbatched_inputs
def lpips_alex(imgs_a, imgs_b):
    """LPIPS: Learned Perceptual Image Patch Similarity metric

    R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang,
    “The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,”
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, Jun. 2018, pp. 586–595.
    doi: 10.1109/CVPR.2018.00068.
    """
    if lpips_model_alex is None:
        init_lpips_model_alex()
    return lpips_model_alex([imgs_a, imgs_b])


@_handle_unbatched_inputs
def lpips_vgg(imgs_a, imgs_b):
    """LPIPS: Learned Perceptual Image Patch Similarity metric

    R. Zhang, P. Isola, A. A. Efros, E. Shechtman, and O. Wang,
    “The Unreasonable Effectiveness of Deep Features as a Perceptual Metric,”
    in 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition,
    Salt Lake City, UT, Jun. 2018, pp. 586–595.
    doi: 10.1109/CVPR.2018.00068.
    """
    if lpips_model_vgg is None:
        init_lpips_model_vgg()
    return lpips_model_vgg([imgs_a, imgs_b])


def init_lpips_model_alex():
    model_file = tf.keras.utils.get_file(
        LPIPS_ALEX_MODEL_NAME,
        LPIPS_ALEX_MODEL_URL,
        file_hash=LPIPS_ALEX_MODEL_MD5,
        hash_algorithm="md5",
    )
    global lpips_model_alex
    lpips_model_alex = tf.keras.models.load_model(model_file, compile=False)


def init_lpips_model_vgg():
    model_file = tf.keras.utils.get_file(
        LPIPS_VGG_MODEL_NAME,
        LPIPS_VGG_MODEL_URL,
        file_hash=LPIPS_VGG_MODEL_MD5,
        hash_algorithm="md5",
    )
    global lpips_model_vgg
    lpips_model_vgg = tf.keras.models.load_model(model_file, compile=False)
