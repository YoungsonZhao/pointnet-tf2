"""
CorrelateNet, a deep learning frameword for classification and segmentation of point cloud

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

# Tensorflow and Numpy
import tensorflow as tf
import numpy as np

from layers import Conv2D, CorrelateTransform, Dense, TransformNet

# Defind model using keras functional api
# def Model(batch_size, bn_momentum, point_nums, name='PointNet'):
#     points = tf.keras.layers.Input(shape=(point_nums, 3), batch_size=batch_size, name='point_clouds', dtype=tf.float32)
def PointNet(point_nums=None, bn_momentum=None, name='PointNet'):
    inputs = tf.keras.layers.Input(shape=(point_nums, 3), name='point_clouds', dtype=tf.float32)
    inputs_transformed = TransformNet(add_regularization=True, use_bn=True, bn_momentum=bn_momentum)(inputs)
    # inputs = tf.expand_dims(points, axis=3)
    # inputs_transformed = tf.expand_dims(inputs_transformed, axis=3)
    conv_1 = Conv2D(64, (1, 1),
                    strides=(1, 1),
                    padding='valid',
                    activation=tf.nn.relu,
                    use_bn=True,
                    bn_momentum=bn_momentum,
                    use_ct=False)(inputs_transformed)
                    # use_ct=False)(points_transformed)
    conv_2 = Conv2D(64, (1, 1),
                    strides=(1, 1),
                    padding='valid',
                    activation=tf.nn.relu,
                    use_bn=True,
                    bn_momentum=bn_momentum,
                    use_ct=False)(conv_1)
    conv_transformed = TransformNet(add_regularization=True, use_bn=True, bn_momentum=bn_momentum)(conv_2)
    conv_3 = Conv2D(64, (1, 1),
                    strides=(1, 1),
                    padding='valid',
                    activation=tf.nn.relu,
                    use_bn=True,
                    bn_momentum=bn_momentum,
                    # use_ct=False)(conv_2)
                    use_ct=False)(conv_transformed)
    conv_4 = Conv2D(128, (1, 1),
                    strides=(1, 1),
                    padding='valid',
                    activation=tf.nn.relu,
                    use_bn=True,
                    bn_momentum=bn_momentum,
                    use_ct=False)(conv_3)
    conv_5 = Conv2D(1024, (1, 1),
                    strides=(1, 1),
                    padding='valid',
                    activation=tf.nn.relu,
                    use_bn=True,
                    bn_momentum=bn_momentum,
                    use_ct=False)(conv_4)

    # Global feature vector (B x N x 1024 -> B x 1024)
    conv_5 = tf.squeeze(conv_5, axis=[2])
    global_feature = tf.math.reduce_max(conv_5, axis=1)

    # FC layers to output k scores (B x 1024 -> B x 40)
    fc_1 = Dense(512,
                 activation=tf.nn.relu,
                 use_bn=True,
                 bn_momentum=bn_momentum)(global_feature)
    fc_1 = tf.keras.layers.Dropout(rate=0.3)(fc_1)

    fc_2 = Dense(256,
                 activation=tf.nn.relu,
                 use_bn=True,
                 bn_momentum=bn_momentum)(fc_1)
    fc_2 = tf.keras.layers.Dropout(rate=0.3)(fc_2)

    logits = Dense(40,
                 activation=None,
                 use_bn=False)(fc_2)
    # softmax = tf.keras.layers.Activation(activation='softmax')(fc_3)
    model = tf.keras.Model(inputs=inputs, outputs=logits, name=name)
    return model

# def CorintNet(point_nums=None, bn_momentum=None, name='CorintNet'):
#     points = tf.keras.layers.Input(shape=(point_nums, 3), name='point_clouds', dtype=tf.float32)
#     inputs = tf.expand_dims(points, axis=3)
#     # points_transformed = TransformNet(add_regularization=True, bn_momentum=bn_momentum)(points)
#     conv_1 = tf.keras.layers.Conv2D(64, (1, 3), strides=(1, 1), padding='valid', activation='relu')(inputs)
#     conv_2 = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='valid', activation='relu')(conv_1)
#     conv_3 = tf.keras.layers.Conv2D(64, (1, 1), strides=(1, 1), padding='valid', activation='relu')(conv_2)
#     conv_4 = tf.keras.layers.Conv2D(128, (1, 1), strides=(1, 1), padding='valid', activation='relu')(conv_3)
#     conv_5 = tf.keras.layers.Conv2D(1024, (1, 1), strides=(1, 1), padding='valid', activation='relu')(conv_4)
#     # Global feature vector (B x N x 1024 -> B x 1024)
#     conv_5 = tf.squeeze(conv_5, axis=[2])
#     global_feature = tf.math.reduce_max(conv_5, axis=1)
#     fc_1 = tf.keras.layers.Dense(512, activation='relu')(global_feature)
#     fc_1 = tf.keras.layers.Dropout(rate=0.3)(fc_1)
#     fc_2 = tf.keras.layers.Dense(256, activation='relu')(fc_1)
#     fc_2 = tf.keras.layers.Dropout(rate=0.3)(fc_2)
#     fc_3 = tf.keras.layers.Dense(40, activation='relu')(fc_2)
#     softmax = tf.keras.layers.Activation(activation='softmax')(fc_3)
#     model = tf.keras.Model(inputs=points, outputs=softmax, name=name)
#     return model



if __name__ == "__main__":
    """ Test all models when run directly
    """
    model = PointNet(bn_momentum=0.99)
    model.summary()
    tf.keras.utils.plot_model(model, to_file=os.path.join(BASE_DIR, 'corint_net.png'), show_shapes=True)