from typing import Callable, Union

import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.python.keras.engine.keras_tensor import KerasTensor

TensorType = Union[tf.Tensor, KerasTensor]
TensorLike = tfa.types.TensorLike

Denoiser = Callable[[TensorLike, TensorLike], TensorType]
Callback = Callable[[TensorLike, TensorLike], None]
MetricFnType = Callable[[TensorLike, TensorLike], TensorType]
ResizeFnType = Callable[[TensorLike, int, bool], TensorType]
FlowFnType = Callable[[TensorLike, TensorLike], TensorType]
InpaintFnType = Callable[[TensorLike, TensorLike], TensorType]
