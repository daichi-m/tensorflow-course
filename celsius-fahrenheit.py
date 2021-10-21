#!/usr/bin/env python
# coding: utf-8

# ##### Copyright 2018 The TensorFlow Authors.


# @title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# Imports and set logging level to error only

import tensorflow as tf
import numpy as np
import logging
from typing import Tuple, List
import random
import matplotlib.pyplot as plt

logger = tf.get_logger()
logger.setLevel(logging.INFO)


def celsius_fahrenheit(count: int) -> Tuple[List, List]:
    random.seed(a=None, version=2)
    cels = []
    fahr = []
    for i in range(0, count):
        c: float = random.uniform(-273, 5000)
        f: float = c * 1.8 + 32
        cels.append(c)
        fahr.append(f)
    return cels, fahr


celsius, fahrenheit = celsius_fahrenheit(10 ** 4)
celsius_q = np.array(celsius, dtype=float)
fahrenheit_a = np.array(fahrenheit, dtype=float)
l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

model = tf.keras.Sequential([l0])
model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

history = model.fit(celsius_q, fahrenheit_a, epochs=500, verbose=True)
print("Finished training the model")

plt.xlabel('Epoch Number')
plt.ylabel("Loss Magnitude")
plt.plot(history.history['loss'])
plt.show()

print(model.predict([100.0, -40]))
print("These are the layer variables: {}".format(l0.get_weights()))

model3 = tf.keras.Sequential([
    tf.keras.layers.Dense(units=4, input_shape=[1]),
    tf.keras.layers.Dense(units=4),
    tf.keras.layers.Dense(units=1)
])
model3.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(0.1))
model3.fit(celsius_q, fahrenheit_a, epochs=500, verbose=False)
print("Finished training the model")
print(model3.predict([100.0, -40]))
print("Model predicts that 100 degrees Celsius is: {} degrees Fahrenheit".format(model3.predict([100.0])))
