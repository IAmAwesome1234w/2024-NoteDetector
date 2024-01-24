import numpy as np
import os

from tflite_model_maker.config import ExportFormat, QuantizationConfig
from tflite_model_maker import model_spec
from tflite_model_maker import object_detector

from tflite_support import metadata

import tensorflow as tf
assert tf.__version__.startswith('2')

tf.get_logger().setLevel('ERROR')
from absl import logging
logging.set_verbosity(logging.ERROR)

train_data = object_detector.DataLoader.from_pascal_voc(
    'train', #name of file of source images
    'train',
    ['note', 'robot'] #ListMap of labels
)

val_data = object_detector.DataLoader.from_pascal_voc(
    'valid',
    'valid',
    ['note', 'robot']
)

test_data = object_detector.DataLoader.from_pascal_voc(
    'test',
    'test',
    ['note', 'robot']
)
#i believe google coral only works with lite0 due to TPU
spec = model_spec.get('efficientdet_lite0')
#change epoch and batch_size depending on dataset size and training time
model = object_detector.create(train_data, model_spec=spec, batch_size=32, train_whole_model=True, epochs=20, validation_data=val_data)

model.export(export_dir='.', tflite_filename='notedetector.tflite') #can change file name

model.evaluate_tflite('notedetector.tflite', test_data)