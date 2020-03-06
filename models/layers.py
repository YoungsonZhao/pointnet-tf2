"""
Custom layers using Tensorflow 2.0 and Numpy, including convolution, batch normalization, pooling

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

# 
import tensorflow as tf
import numpy as np

class Conv2D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, strides=(1, 1), padding='valid', activation=None, use_bn=False, bn_momentum=0.99, use_ct=False, **kwargs):
        super(Conv2D, self).__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.use_bn = use_bn
        self.bn_momentum = bn_momentum
        self.use_ct = use_ct
        if use_bn:
            self.batch_norm = tf.keras.layers.BatchNormalization(momentum=bn_momentum)

    def build(self, input_shape):
        # input_shape = (b, h, w, c)
        # print(input_shape)
        if self.use_ct:
            self.correlate_trans = CorrelateTransform()
            self.kernel_size = (1, 2*input_shape[2])
            self.max_pool = tf.keras.layers.MaxPool2D(pool_size=(input_shape[1], 1))
        self.conv = tf.keras.layers.Conv2D(self.filters, self.kernel_size, strides=self.strides, padding=self.padding, use_bias=not self.use_bn)
        super(Conv2D, self).build(input_shape)

    def call(self, inputs, training=None):
        if self.use_ct:
            inputs = self.correlate_trans(inputs)
        else:
            if len(inputs.shape) != 4:
                inputs = tf.expand_dims(inputs, axis=2)
        x = self.conv(inputs)
        if self.use_bn:
            x = self.batch_norm(x, training=training)
        if self.activation:
            x = self.activation(x)
        if self.use_ct:
            x = self.max_pool(x)
        # x = tf.squeeze(x, axis=[2])
        return x

    def get_config(self):
        config = super(Conv2D, self).get_config()
        config.update({
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'use_bn': self.use_bn,
            'bn_momentum': self.bn_momentum,
            'use_mp': self.use_ct})
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class CorrelateTransform(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(CorrelateTransform, self).__init__(**kwargs)
    
    def build(self, input_shape):
        # print(input_shape)
        super(CorrelateTransform, self).build(input_shape)
        # self.batch_size = input_shape[0]
        self.height = input_shape[1]
        self.width = input_shape[2]
        self.repeats_height = tf.ones([self.height], dtype=tf.int32) * self.height
        # self.repeats_batch  = tf.ones([self.batch_size], dtype=tf.int32) * self.height
        # print(self.repeats_height)
        # print(self.repeats_batch)
        # self.shape = tf.constant([self.batch_size, self.height*self.height, self.width])
        self.reshape_1 = tf.keras.layers.Reshape((self.height*self.width,))
        self.repeat = tf.keras.layers.RepeatVector(self.height)
        self.reshape_2 = tf.keras.layers.Reshape((self.height*self.height, self.width))
        # print(self.shape)

    def call(self, inputs):
        x = tf.repeat(inputs, repeats=self.repeats_height, axis=1)
        y = self.reshape_1(inputs)
        y = self.repeat(y)
        y = self.reshape_2(y)
        xy = tf.concat([x, y], axis=2)
        xy = tf.expand_dims(xy, axis=3)
        return xy

    def get_config(self):
        config = super(CorrelateTransform, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Generate a super large feature space, deprecated
# class CorrelateTransform(tf.keras.layers.Layer):
#     def __init__(self, **kwargs):
#         super(CorrelateTransform, self).__init__(**kwargs)
#     
#     def build(self, input_shape):
#         # print(input_shape)
#         super(CorrelateTransform, self).build(input_shape)
#         # self.batch_size = input_shape[0]
#         self.height = input_shape[1]
#         self.width = input_shape[2]
#         self.repeats_height = tf.ones([self.height], dtype=tf.int32) * self.height
#         # self.repeats_batch  = tf.ones([self.batch_size], dtype=tf.int32) * self.height
#         # print(self.repeats_height)
#         # print(self.repeats_batch)
#         # self.shape = tf.constant([self.batch_size, self.height*self.height, self.width])
#         self.reshape_1 = tf.keras.layers.Reshape((self.height*self.width,))
#         self.repeat = tf.keras.layers.RepeatVector(self.height)
#         self.reshape_2 = tf.keras.layers.Reshape((self.height*self.height, self.width))
#         # print(self.shape)
# 
#     def call(self, inputs):
#         x = tf.repeat(inputs, repeats=self.repeats_height, axis=1)
#         y = self.reshape_1(inputs)
#         y = self.repeat(y)
#         y = self.reshape_2(y)
#         xy = tf.concat([x, y], axis=2)
#         xy = tf.expand_dims(xy, axis=3)
#         return xy
# 
#     def get_config(self):
#         config = super(CorrelateTransform, self).get_config()
#         return config
# 
#     @classmethod
#     def from_config(cls, config):
#         return cls(**config)

class Dense(tf.keras.layers.Layer):
    def __init__(self, units, activation=tf.nn.relu, use_bn=False, bn_momentum=0.99, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.units = units
        self.activation = activation
        self.use_bn = use_bn
        self.bn_momentum = bn_momentum
        self.dense = tf.keras.layers.Dense(units, activation=activation, use_bias=not use_bn)
        if use_bn:
            self.batch_norm = tf.keras.layers.BatchNormalization(momentum=bn_momentum)
    
    def call(self, inputs, training=None):
        x = self.dense(inputs)
        if self.use_bn:
            x = self.batch_norm(x, training=training)
        if self.activation:
            x = self.activation(x)
        return x
    
    def get_config(self):
        config = super(Dense, self).get_config()
        config.update({
            'units': self.units,
            'activation': self.activation,
            'use_bn': self.use_bn,
            'bn_momentum': self.bn_momentum
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

class TransformNet(tf.keras.layers.Layer):
    def __init__(self, add_regularization=True, use_bn=True, bn_momentum=0.99, **kwargs):
        super(TransformNet, self).__init__(**kwargs)
        self.add_regularization = add_regularization
        self.use_bn = use_bn
        self.bn_momentum = bn_momentum
        self.conv_0 = Conv2D(64, (1, 1), activation=tf.nn.relu, use_bn=self.use_bn, bn_momentum=self.bn_momentum)
        self.conv_1 = Conv2D(128, (1, 1), activation=tf.nn.relu, use_bn=self.use_bn, bn_momentum=self.bn_momentum)
        self.conv_2 = Conv2D(1024, (1, 1), activation=tf.nn.relu, use_bn=self.use_bn, bn_momentum=self.bn_momentum)
        self.fc_0 = Dense(512, activation=tf.nn.relu, use_bn=self.use_bn, bn_momentum=self.bn_momentum)
        self.fc_1 = Dense(256, activation=tf.nn.relu, use_bn=self.use_bn, bn_momentum=self.bn_momentum)

    def build(self, input_shape):
        self.K = input_shape[-1]
        self.w = self.add_weight(shape=(256, self.K**2), initializer=tf.zeros_initializer, trainable=True, name='w')
        self.b = self.add_weight(shape=(self.K, self.K), initializer=tf.zeros_initializer, trainable=True, name='b')
        # Initialize bias with identity
        self.eye = tf.constant(np.eye(self.K), dtype=tf.float32)
        self.b.assign(self.eye*0.95)
        super(TransformNet, self).build(input_shape)

    def call(self, inputs, training=None):                              # BxNx3
        # x = tf.expand_dims(inputs, axis=3)
        x = self.conv_0(inputs, training=training)                      # BxNx1x64
        x = self.conv_1(x, training=training)                           # BxNx1x128
        x = self.conv_2(x, training=training)                           # BxNx1x1024
        x = tf.squeeze(x, axis=2)                                       # BxNx1024
        x = tf.reduce_max(x, axis=1)                                    # Bx1x1024
        x = self.fc_0(x, training=training)                             # Bx512
        x = self.fc_1(x, training=training)                             # Bx256

        # Convert to KxK matrix to matmul with input
        x = tf.expand_dims(x, axis=1)
        x = tf.matmul(x, self.w)                                        # BxK^2
        x = tf.reshape(x, (-1, self.K, self.K))                         # BxKxK
        # Add bias term (initialized to identity matrix)
        # x += self.b
        x = tf.math.add(x, self.b)
        # Add regularization
        if self.add_regularization:
            xT_x = tf.matmul(tf.transpose(x, perm=[0, 2, 1]), x)
            reg_loss = tf.nn.l2_loss(xT_x-self.eye)
            self.add_loss(0.0005 * reg_loss)

        if len(inputs.shape) == 4:
            inputs = tf.squeeze(inputs, axis=2)
        return tf.matmul(inputs, x)
    
    def get_config(self):
        config = super(TransformNet, self).get_config()
        config.update({
            'add_regularization': self.add_regularization,
            'bn_momentum': self.bn_momentum
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        return cls(**config)

if __name__ == "__main__":
    """ Test all the layers when run directly
    """
    # inputs = get_weights("weights", shape=[2, 10, 3], initializer=tf.keras.initializers.glorot_normal())
    inputs = tf.Variable(initial_value=tf.keras.initializers.glorot_normal()(shape=(2, 10, 3)), trainable=False, name='inputs', dtype=tf.float32)
    print(inputs)

    # # Test CorrelateTransform
    # ct = CorrelateTransform()
    # inputs_ct = ct(inputs)
    # print(inputs_ct)

    # # Test Conv2D
    # conv = Conv2D(6, (1, 3), activation=tf.nn.relu, use_bn=True, use_ct=False)
    # x = conv(inputs, training=True)
    # print(x)

    # # Test TransorformNet
    # tnet = TransformNet(add_regularization=True, bn_momentum=0.99)
    # inputs_transformed = tnet(inputs)
    # print(inputs_transformed)

    # inputs_correlated = correlation_transform(inputs, "input")
    # print(inputs_correlated)