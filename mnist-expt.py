import logging
from typing import Dict, List

import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import Tensor
from tensorflow.python.data.ops.dataset_ops import DatasetV2

# Set INFO level logging for tensorflow
logger = tf.get_logger()
logger.setLevel(logging.INFO)


def get_dataset() -> (DatasetV2, DatasetV2, List[str]):
	"""Get the Fashion MNIST dataset and it's metadata"""
	dataset: Dict[str, DatasetV2]
	info: tfds.core.DatasetInfo
	dataset, info = tfds.load(name="fashion-mnist", with_info=True, as_supervised=True)
	train, test = dataset["train"], dataset["test"]
	logger.info("Dataset info - %s", str(info))
	class_names: List[str] = info.features["label"].names
	return train, test, class_names


def normalize(images: Tensor, labels: Tensor) -> (Tensor, Tensor):
	"""Normalize each color of the image Tensor to a value between 0 and 1"""
	images = tf.cast(images, tf.float32)
	images /= 255
	return images, labels
