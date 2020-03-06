"""
Math operation wrappers using Tensorflow and Numpy, including convolution, batch normalization, pooling

Author: Zhao Yongsheng
Date:   Match, 2020
"""

# Make sure compatibility
from __future__ import print_function, division, absolute_import, unicode_literals

# Add working directory to search path
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Tensorflow & Numpy
import tensorflow as tf
import numpy as np

# utils
from data_utils import correlation_transform

def get_weights(name,
                shape,
                initializer=tf.keras.initializers.glorot_normal(),
                dtype=tf.dtypes.float32,
                trainable=True):
    #
    weights = tf.Variable(initializer(shape=shape), trainable=trainable, name=name, dtype=dtype)
    return weights

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           strides=[1, 1],
           padding='VALID',
           activation_fn=tf.nn.relu,
           is_bn=False,
           bn_decay=None,
           is_training=None,
           name=None):
    """ 2D convolution with non-linear operation.

    Args:
        inputs: 4-D tensor variable BxHxWxC
        num_output_channels: int, output channels
        kernel_size: a list of 2 ints, [kernel_height, kernel_width]
        name: a string for name scope
        strides: a list of 2 ints, [stride_height, stride_width]
        padding: 'SAME' or 'VALID'
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        4-D tensor, BxHxWxC, convolution result
    """
    with tf.name_scope(name):
        kernel_h, kernel_w = kernel_size
        stride_h, stride_w = strides
        num_in_channels = inputs.get_shape()[-1]

        # Convolution
        kernel_shape = [kernel_h, kernel_w, num_in_channels, num_output_channels]
        kernel = get_weights('weights',
                             shape=kernel_shape,
                             initializer=tf.keras.initializers.glorot_normal())
        outputs = tf.nn.conv2d(inputs,
                               kernel,
                               strides=[1, stride_h, stride_w, 1],
                               padding=padding,
                               name="conv")
        # Bias
        biases = get_weights('biases',
                             shape = [num_output_channels],
                             initializer=tf.constant_initializer(0.0))
        outputs = tf.nn.bias_add(outputs, biases)

        # Batch Normalization
        if is_bn:
            outputs = batch_norm_conv2d(outputs,
                                        name="batch_norm",
                                        bn_decay=bn_decay,
                                        is_training=is_training)
        # Activation
        if activation_fn is not None:
            outputs = activation_fn(outputs)
        
        # Max Pooling

        return outputs

def batch_norm(inputs, moments_axes, name, bn_decay, is_training=True):
    """ Batch normalization on convolutional maps and beyond...
    
    Args:
        inputs:        Tensor, k-D input, dimension could be BC, or BFC, or BHWC or BDHWC
        moments_axes:  a list of ints, indicating dimensions for moments calculation
        name:          string, for name scope
        bn_decay:      float or float tensor variable, controling moving average weight
        is_training:   boolean, true indicates training phase
    Return:
        Tensor, k-D batch-normalized output
    """
    with tf.name_scope(name):
        # compute mean and variance along axes
        num_channels = inputs.get_shape()[-1]
        var_mean = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[num_channels]), name="bm", trainable=False)
        var_variance = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[num_channels]), name="bv", trainable=False)
        # print(var_mean)
        # print(var_variance)
        batch_mean, batch_variance = tf.nn.moments(inputs, axes=moments_axes, name="moments")
        # print(batch_mean)
        # print(batch_variance)
        var_mean.assign(batch_mean)
        var_variance.assign(batch_variance)
        print(var_mean)
        print(var_variance)

        # compute exponential moving average
        ema = tf.train.ExponentialMovingAverage(decay=0.999)
        # # Operator that maintains moving averages of variables.
        # ema_apply_op = tf.cond(is_training,
        #                        lambda: ema.apply([batch_mean, batch_variance]),
        #                        lambda: tf.no_op())
        
        # # Update moving average and return current batch's avg and var.
        # def mean_var_with_update():
        #     with tf.control_dependencies([ema_apply_op]):
        #         return tf.identity(batch_mean), tf.identity(batch_variance)
        
        # # ema.average returns the Variable holding the average of var.
        # mean, variance = tf.cond(is_training,
        #                          mean_var_with_update,
        #                          lambda: (ema.average(batch_mean), ema.average(batch_variance)))
        if is_training:
            with tf.control_dependencies([ema.apply([var_mean, var_variance])]):
                mean = tf.identity(var_mean, "mean")
                variance = tf.identity(var_variance, "variance")
        else:
            mean = ema.average(var_mean)
            variance = ema.average(var_variance)
        
        # batch normalization
        beta = tf.Variable(tf.constant(0, dtype=tf.float32, shape=[num_channels]), name="beta", trainable=is_training)
        gamma = tf.Variable(tf.constant(1, dtype=tf.float32, shape=[num_channels]), name="gamma", trainable=is_training)
        inputs_normalized = tf.nn.batch_normalization(inputs, mean=mean, variance=variance, offset=beta, scale=gamma, variance_epsilon=1e-3, name="batch_norm")
        return inputs_normalized

def batch_norm_conv2d(inputs, name, bn_decay, is_training=True):
    """ Batch normalization on convolutional maps and beyond...
    
    Args:
        inputs:        Tensor, k-D input, dimension could be BC, or BFC, or BHWC or BDHWC
        name:          string, for name scope
        bn_decay:      float or float tensor variable, controling moving average weight
        is_training:   boolean, true indicates training phase
    Return:
        Tensor, k-D batch-normalized output
    """
    return batch_norm(inputs, moments_axes=[0, 1, 2], name=name, bn_decay=bn_decay, is_training=is_training)

def batch_norm_fc(inputs, name, bn_decay=0.9, is_training=True):
    """ Batch normalization on convolutional maps and beyond...
    
    Args:
        inputs:        Tensor, k-D input, dimension could be BC, or BFC, or BHWC or BDHWC
        name:          string, for name scope
        bn_decay:      float or float tensor variable, controling moving average weight
        is_training:   boolean, true indicates training phase
    Return:
        Tensor, k-D batch-normalized output
    """
    return batch_norm(inputs, moments_axes=[0,], name=name, bn_decay=bn_decay, is_training=is_training)

if __name__ == "__main__":
    """ Test all the functions when run directly
    """
    # weights = get_weights("weights", shape=[1, 6, 3, 4], initializer=tf.keras.initializers.glorot_normal())
    weights = get_weights("weights", shape=[2, 6, 3], initializer=tf.keras.initializers.glorot_normal())
    print(weights)
    # conv_results = conv2d(weights, 6, [1, 3], name="conv_1")
    # print(conv_results)
    mean, variance = tf.nn.moments(weights, axes=[1])
    print(mean)
    print(variance)
    weights_decenter = weights - mean
    print(weights_decenter)
    # bn_decay = tf.Variable(0.9, trainable=False)
    # weights_normalized = batch_norm_conv2d(weights, name="bn", bn_decay=bn_decay)
    # print(weights_normalized)
    # weights_reshaped = tf.reshape(weights, shape=[-1, 4])
    # print(weights_reshaped)
    # mean_reshaped, variance_reshaped = tf.nn.moments(weights_reshaped, axes=[0])
    # print(mean_reshaped)
    # print(variance_reshaped)

    # exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate = 0.9,
    #                                                                    decay_steps = 10,
    #                                                                    decay_rate=0.5,
    #                                                                    staircase=False)
    # for step in range(100):
    #     print("learning_rate = {}".format(exponential_decay(step)))