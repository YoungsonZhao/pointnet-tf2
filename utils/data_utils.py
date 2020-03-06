"""
Data preparation functions using Tensorflow and Numpy

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
PROJ_DIR = os.path.dirname(BASE_DIR)

# Tensorflow & Numpy
import tensorflow as tf
import numpy as np
import h5py

# Visualization
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D     # registers the 3D projection

# Download dataset for point cloud classification
def download_modelnet40():
    DATA_DIR = os.path.join(PROJ_DIR, 'data')
    # print(DATA_DIR)
    MODELNET40_DIR = os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')
    # print(MODELNET40_DIR)
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
        print('Create Dir: {}'.format(DATA_DIR))
    if not os.path.exists(MODELNET40_DIR):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        # os.system('rm %s' % (zipfile))
        
    train_files = getDataFiles(os.path.join(MODELNET40_DIR, 'train_files.txt'))
    test_files  = getDataFiles(os.path.join(MODELNET40_DIR, 'test_files.txt'))
    train_data = []
    train_label = []
    for train_file in train_files:
        data, label = loadDataFile(os.path.join(PROJ_DIR, train_file))
        train_data.append(data[:,0:1024,:])
        train_label.append(label)
    train_data = np.concatenate(train_data, axis=0)
    train_label = np.concatenate(train_label, axis=0)

    test_data = []
    test_label = []
    for test_file in test_files:
        data, label = loadDataFile(os.path.join(PROJ_DIR, test_file))
        test_data.append(data[:,0:1024,:])
        test_label.append(label)
    test_data = np.concatenate(test_data, axis=0)
    test_label = np.concatenate(test_label, axis=0)
    print('Loaded ModelNet40 Dataset:')
    print('---------------Train Data: {}'.format(train_data.shape))
    print('--------------Train Label: {}'.format(train_label.shape))
    print('----------------Test Data: {}'.format(test_data.shape))
    print('---------------Test Label: {}'.format(test_label.shape))
    return (train_data, train_label, test_data, test_label)

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename, 'r')
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)

def augmentation(cloud, label):
    label = tf.one_hot(label, 40, dtype=tf.float32)
    # shuffle along first dimension
    # cloud = tf.random.shuffle(cloud)
    # cloud = random_rotate_and_scale(cloud)
    # cloud = random_jitter_and_shift(cloud)
    return cloud, label

def sample(cloud, label):
    # shuffle along first dimension
    # cloud = tf.random.shuffle(cloud)
    label = tf.one_hot(label, 40, dtype=tf.float32)
    return cloud, label

def random_rotate_and_scale(cloud):
    # Random Rotation Angle and Scale
    alpha, beta, gamma = np.random.uniform(low=0, high=np.pi, size=(3))
    scale = tf.random.uniform(shape=[], minval=0.75, maxval=1.25)

    # Build Eular Angle Matrix
    Rx = tf.constant([[1.0,0.0,0.0],
                      [0.0,np.cos(alpha),-np.sin(alpha)],
                      [0.0,np.sin(alpha), np.cos(alpha)]], dtype=tf.float32)
    Ry = tf.constant([[np.cos(beta),0.0,np.sin(beta)],
                      [0.0,1.0,0.0],
                      [-np.sin(beta),0.0,np.cos(beta)]], dtype=tf.float32)
    Rz = tf.constant([[np.cos(gamma),-np.sin(gamma),0.0],
                      [np.sin(gamma), np.cos(gamma),0.0],
                      [0.0,0.0,1.0]], dtype=tf.float32)
    # Rotation Matrix
    Rxyz = tf.matmul(Rz, tf.matmul(Ry, Rx))
    # Scaled Rotation Matrix
    Rxyz_s = tf.math.scalar_mul(scale, Rxyz)
    cloud = tf.matmul(cloud, Ry)
    return cloud

def random_jitter_and_shift(cloud):
    # Random Jitter Matrix and Shift Vector
    jitter = tf.random.uniform(shape=cloud.shape, minval=-0.005, maxval=0.005)
    shift  = tf.random.uniform(shape=[cloud.shape[-1]], minval=-0.05, maxval=0.05)
    cloud = tf.add(cloud, jitter)
    cloud = tf.nn.bias_add(cloud, shift)
    return cloud

modelnet40 = {0: 'airplane',
              1: 'bathtub',
              2: 'bed',
              3: 'bench',
              4: 'bookshelf',
              5: 'bottle',
              6: 'bowl',
              7: 'car',
              8: 'chair',
              9: 'cone',
             10: 'cup',
             11: 'curtain',
             12: 'desk',
             13: 'door',
             14: 'dresser',
             15: 'flower_pot',
             16: 'glass_box',
             17: 'guitar',
             18: 'keyboard',
             19: 'lamp',
             20: 'laptop',
             21: 'mantel',
             22: 'monitor',
             23: 'night_stand',
             24: 'person',
             25: 'piano',
             26: 'plant',
             27: 'radio',
             28: 'range_hood',
             29: 'sink',
             30: 'sofa',
             31: 'stairs',
             32: 'stool',
             33: 'table',
             34: 'tent',
             35: 'toilet',
             36: 'tv_stand',
             37: 'vase',
             38: 'wardrobe',
             39: 'xbox'}

# Visualize Point Cloud
def plot(data, label):
    # Separate x, y, z coordinates
    xs = data[:, 0]
    ys = data[:, 1]
    zs = data[:, 2]

    # Plot
    fig = plt.figure(figsize=(12.8, 9.6), dpi=160)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs, ys, zs, marker='.', c='b')
    ax.set_title(modelnet40[label], fontsize=12, color='r')
    plt.show()
    plt.close()

# # deprecated, please see the CorrelateTransform Layer
# def correlation_transform(inputs, name):
#     """ Point cloud correlation transform operation, given a point cloud, it returns a correlated point cloud composed of point pairs.
# 
#     Args:
#         inputs: 3-D tensor with dimension as BxHxW (Batch x Point Number x Feature)
#         name: a string for name scope
#     Returns:
#         3-D tensor with dimension as Bx(N^2)x(2W)
#     """
#     with tf.name_scope(name):
#         batch_size = inputs.get_shape()[0]
#         point_nums = inputs.get_shape()[1]
#         feature_length = inputs.get_shape()[2]
#         repeats = [point_nums for index in range(point_nums)]
#         # print(repeats)
#         inputs_repeated_1 = tf.repeat(inputs, repeats=repeats, axis=1, name=name)
#         # print(inputs_repeated_1)
#         repeats = [point_nums for index in range(batch_size)]
#         # print(repeats)
#         inputs_repeated_2 = tf.repeat(inputs, repeats=repeats, axis=0, name=name)
#         inputs_repeated_2 = tf.reshape(inputs_repeated_2, shape=[batch_size, point_nums*point_nums, feature_length], name=name)
#         # print(inputs_repeated_2)
#         outputs = tf.concat([inputs_repeated_1, inputs_repeated_2], axis=2, name=name)
#         return outputs

if __name__ == "__main__":
    """ Test all the functions when run directly
    """
    # inputs = tf.constant(np.arange(60))
    # inputs = tf.reshape(inputs, shape=[2, 10, 3])
    # print(inputs)
    # inputs_correlated = correlation_transform(inputs, "input")
    # print(inputs_correlated)

    train_data, train_label, test_data, test_label = download_modelnet40()
    train_data, train_label, test_data, test_label = download_modelnet40()
    train_dataset = tf.data.Dataset.from_tensor_slices((train_data, train_label))
    # train_dataset = train_dataset.map(augmentation, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    # train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(batch_size=32).repeat()
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    for data in train_dataset.as_numpy_iterator():
        pts, label = data
        print(pts.shape)
        print(label)
        print(train_label[0:32])
        input('Enter for continue.')
    # print('Loaded ModelNet40 Dataset:')
    # print('---------------Train Data: {}'.format(train_data.shape))
    # print('--------------Train Label: {}'.format(train_label.shape))
    # print('----------------Test Data: {}'.format(test_data.shape))
    # print('---------------Test Label: {}'.format(test_label.shape))

    for index in range(train_data.shape[0]):
        plot(train_data[index,:,:], train_label[index,0])
        cloud = tf.constant(train_data[index,:,:])
        label = tf.constant(train_label[index, 0])
        cloud, label = augmentation(cloud, label)
        print(cloud.shape)
        print(label.shape)
        plot(cloud.numpy(), label.numpy())