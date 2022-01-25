import logging
import math
import sys
from io import StringIO
from typing import Dict, List, Tuple

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds
from tensorflow.python.data.ops.dataset_ops import DatasetV2


def get_dataset() -> (DatasetV2, DatasetV2, List[str]):
    """Get the Fashion MNIST dataset and it's metadata"""
    dataset: Dict[str, DatasetV2]
    info: tfds.core.DatasetInfo
    dataset, info = tfds.load(name="fashion_mnist", with_info=True, as_supervised=True)
    train, test = dataset["train"], dataset["test"]
    logger.info("Dataset info - %s", str(info))
    class_names: List[str] = info.features["label"].names
    return train, test, class_names


def normalize(images: tf.Tensor, labels: tf.Tensor) -> (tf.Tensor, tf.Tensor):
    """Normalize each color of the image Tensor to a value between 0 and 1"""
    images = tf.cast(images, tf.float32)
    images /= 255
    return images, labels


def generate_model(input_shape: Tuple[int, int, int], output_labels: int,
                   intermediate_nodes: Tuple[int, ...]) -> keras.Model:
    """Generate the model for MNIST dataset"""

    inp_layer: keras.layers.Layer = keras.layers.Flatten(input_shape=input_shape)
    int_layers = []
    for n in intermediate_nodes:
        il: keras.layers.Layer = keras.layers.Dense(n, activation=tf.nn.relu)
        int_layers.append(il)
    op_layer: keras.layers.Layer = keras.layers.Dense(output_labels, activation=tf.nn.softmax)
    layers = [inp_layer]
    for l in int_layers:
        layers.append(l)
    layers.append(op_layer)
    model: keras.Model = tf.keras.Sequential(layers)
    return model


def repeat_data(train_dataset: tf.data.Dataset, test_dataset: tf.data.Dataset, batch_size: int) -> (
        tf.data.Dataset, tf.data.Dataset):
    """Create random repetitions of the data to increase the data size"""
    train_len = len(train_dataset)
    new_train_dataset: tf.data.Dataset = train_dataset.map(normalize).cache().repeat().shuffle(train_len).batch(
        batch_size)
    new_test_dataset: tf.data.Dataset = test_dataset.map(normalize).cache().batch(batch_size)
    return new_train_dataset, new_test_dataset


def model_manager(train_dataset: tf.data.Dataset,
                  test_dataset: tf.data.Dataset,
                  input_shape: Tuple[int, int, int],
                  intermediate_layers: Tuple[int, ...],
                  output_class: int) -> None:
    """Creates and evaluates the model with given format and traint and test datasets"""
    model: keras.Model = generate_model(input_shape=input_shape,
                                        output_labels=output_class,
                                        intermediate_nodes=intermediate_layers)
    model.compile(optimizer="adam", metrics=["accuracy"], loss=keras.losses.SparseCategoricalCrossentropy())

    train_size, test_size, batch_size = len(train_dataset), len(test_dataset), 32
    train_dataset, test_dataset = repeat_data(train_dataset, test_dataset, batch_size)
    model.fit(train_dataset, epochs=10, steps_per_epoch=math.ceil(train_size / batch_size))

    test_loss, test_accuracy = model.evaluate(test_dataset, steps=test_size / batch_size)
    summary = StringIO()
    model.summary(print_fn=lambda message: print(message, file=summary))
    logger.info("Model Shape = %s", summary.getvalue())
    logger.info("Test Loss = %f", test_loss)
    logger.info("Test Accuracy = %.3f %%", test_accuracy * 100.0)


def main() -> None:
    """Main method"""
    train_dataset, test_dataset, classes = get_dataset()
    inp_size = tuple(train_dataset.element_spec[0].shape)
    op_size = len(classes)
    model_manager(train_dataset, test_dataset, inp_size, (32, 64, 128, 256, 512), op_size)


logger = tf.get_logger()
stdoutHandler = logging.StreamHandler(sys.stdout)
logger.handlers = [stdoutHandler]
logger.setLevel(logging.INFO)
logger.propagate = False

if __name__ == '__main__':
    main()
