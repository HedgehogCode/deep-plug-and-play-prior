from typing import Dict, Callable, Tuple
import abc

import tensorflow as tf

import dppp
from dppp.types import *


###############################################################################
# TENSORBOARD
###############################################################################


def callback_tb_image(tb_log_dir: str, summary_name: str, frequency: int) -> Callback:
    tb_writer = tf.summary.create_file_writer(tb_log_dir)

    def call(x: TensorLike, step: TensorLike):
        if step % frequency == 0:  # type: ignore
            with tb_writer.as_default():
                tf.summary.image(summary_name, x, step=step)

    return call


def callback_tb_scalar(
    tb_log_dir: str, summary_name: str, summary_fn: Callable[[TensorLike], TensorLike]
) -> Callback:
    tb_writer = tf.summary.create_file_writer(tb_log_dir)

    def call(x: TensorLike, step: TensorLike):
        with tb_writer.as_default():
            tf.summary.scalar(summary_name, summary_fn(x), step=step)

    return call


def callback_tb_psnr(
    tb_log_dir: str, summary_name: str, gt_image: TensorLike
) -> Callback:
    return callback_tb_scalar(
        tb_log_dir=tb_log_dir,
        summary_name=summary_name,
        summary_fn=lambda x: tf.reduce_mean(dppp.psnr(x, gt_image)),
    )


def callback_tb_ssim(
    tb_log_dir: str, summary_name: str, gt_image: TensorLike
) -> Callback:
    return callback_tb_scalar(
        tb_log_dir=tb_log_dir,
        summary_name=summary_name,
        summary_fn=lambda x: tf.reduce_mean(dppp.ssim(x, gt_image)),
    )


###############################################################################
# TF.PRINT
###############################################################################


def callback_print_scalar(
    name: str, frequency: int, summary_fn: Callable[[TensorLike], TensorLike]
) -> Callback:
    def call(x: TensorLike, step: TensorLike):
        if step % frequency == 0:  # type: ignore
            tf.print("Step:", step, ", ", name, ":", summary_fn(x))

    return call


def callback_print_psnr(name: str, frequency: int, gt_image: TensorLike) -> Callback:
    return callback_print_scalar(
        name=name,
        frequency=frequency,
        summary_fn=lambda x: tf.reduce_mean(dppp.psnr(x, gt_image)),
    )


def callback_print_ssim(name: str, frequency: int, gt_image: TensorLike) -> Callback:
    return callback_print_scalar(
        name=name,
        frequency=frequency,
        summary_fn=lambda x: tf.reduce_mean(dppp.ssim(x, gt_image)),
    )


###############################################################################
# CSV
###############################################################################


def callback_csv_metric(metrics: Dict[str, Tuple[MetricFnType, int]], num_steps: int):
    metric_names = list(metrics.keys())
    csv = tf.Variable("Step,Image_Id," + ",".join(metric_names) + "\n")

    def for_image(image_id: str, gt_image: TensorLike):
        @tf.function(
            input_signature=(
                tf.TensorSpec(shape=[None, None, None, None], dtype=tf.float32),
                tf.TensorSpec(shape=[], dtype=tf.int64),
            )
        )
        def call(x: TensorLike, step: TensorLike):
            def val_for_metric(m):
                metric_fn, per_step = metrics[m]
                if step == num_steps - 1 or (step + 1) % per_step == 0:  # type: ignore
                    val = metric_fn(x, gt_image)[0]  # type: ignore
                else:
                    val = tf.constant(float("nan"))
                return tf.strings.as_string(val)

            values = [val_for_metric(m) for m in metric_names]
            line = tf.strings.join(
                [tf.strings.as_string(step), image_id, *values], separator=","
            )
            csv.assign(tf.strings.join([csv, line, "\n"]))

        return call

    return for_image, csv
