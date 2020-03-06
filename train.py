"""
Train CorintNet for classification using modelnet40_2048 dataset.

Author: Zhao Yongsheng
Date:   Match, 2020
"""

# Compatibility
from __future__ import print_function, division, absolute_import, unicode_literals

# Add Search Path
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'utils'))
sys.path.append(os.path.join(BASE_DIR, 'models'))

# Tensorflow and Numpy
import tensorflow as tf
import numpy as np

# Customization Models and Utils
from data_utils import download_modelnet40, augmentation, sample
from models import PointNet

import argparse
from datetime import datetime

def custom_loss(labels, logits):
    x = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
    loss = tf.math.reduce_mean(x)
    return loss

# Parsing Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--point_nums', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 2048]')
parser.add_argument('--steps_per_epoch', type=int, default=512, help='Steps per epoch [default: 256]')
parser.add_argument('--epochs', type=int, default=300, help='Epoch to run [default: 250]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 32]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--bn_momentum', type=float, default=0.99, help='Initial batch normalization momentum [default: 0.99]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 200000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.8]')
FLAGS = parser.parse_args()

LOG_DIR = FLAGS.log_dir
BATCH_SIZE = FLAGS.batch_size
POINT_NUMS = FLAGS.point_nums
STEPS_PER_EPOCHS = FLAGS.steps_per_epoch
EPOCHS = FLAGS.epochs
LEARNING_RATE = FLAGS.learning_rate
BN_MOMENTUM = FLAGS.bn_momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate
TIMESTAMP = "{0:%Y-%m-%d/%H-%M-%S/}".format(datetime.now())
CHECKPOINT_DIR= 'checkpoints'
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, "ckpt_{epoch}")

# Prepare Dataset
train_data, train_label, test_data, test_label = download_modelnet40()
train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
train_dataset = train_dataset.map(augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.batch(batch_size=BATCH_SIZE)
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices((test_data, test_label))
test_dataset = test_dataset.map(sample, num_parallel_calls=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size=BATCH_SIZE*2)

model = PointNet(point_nums=POINT_NUMS, bn_momentum=BN_MOMENTUM)
optimizer = tf.keras.optimizers.SGD(learning_rate=LEARNING_RATE)
loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

acc_metric_train = tf.keras.metrics.CategoricalAccuracy()
acc_metric_val = tf.keras.metrics.CategoricalAccuracy()

def train():
    model.summary()
    tf.keras.utils.plot_model(model, to_file=os.path.join(BASE_DIR, 'models/corint_net.png'), show_shapes=True)
    print('Start Training...')
    for index in range(EPOCHS):
        print('Epoch: {}'.format(index))
        for step, (inputs, labels) in enumerate(train_dataset):
            loss = train_step(inputs, labels)
            if step % 100 == 0:
                print('Training loss at step {}: {}, accuracy: {}'.format(step, loss, float(acc_metric_train.result())))
        
        accuracy_train = acc_metric_train.result()
        print('Training accuracy at epoch {}: {}'.format(index, float(accuracy_train)))
        acc_metric_train.reset_states()

        for inputs, labels in test_dataset:
            val_step(inputs, labels)
        accuracy_val = acc_metric_val.result()
        acc_metric_val.reset_states()
        print('Validation accuracy at epoch {}: {}'.format(index, float(accuracy_val)))



@tf.function
def train_step(inputs, labels):
    with tf.GradientTape() as tape:
        logits = model(inputs, training=True)
        loss_1 = loss_fn(y_true=labels, y_pred=logits)
        loss_2 = sum(model.losses)
        loss = loss_1 + loss_2
    
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    acc_metric_train(y_true=labels, y_pred=logits)
    return loss

@tf.function
def val_step(inputs, labels):
    logits = model(inputs, training=False)
    acc_metric_val(y_true=labels, y_pred=logits)


if __name__ == '__main__':
    train()