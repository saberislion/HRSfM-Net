"""
Related papers:
High-Resolution Representations for Labeling Pixels and Regions.higharXiv:1904.04514v1 [cs.CV] 9 Apr 2019
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.training import moving_averages

import tensorflow as tf





def conv(x, kernel_size, out_planes, name,stride=1):
    """3x3 convolution with padding"""
    x_shape = x.get_shape().as_list()
    w = tf.get_variable(name = name+'b',shape=[kernel_size, kernel_size, x_shape[3], out_planes])
    return tf.nn.conv2d(input= x, filter=w, padding='SAME',strides=[1, stride, stride, 1],name= name )

def batch_normal(x, name='BatchNorm'):
    x = tf.layers.batch_normalization(x, momentum=0.99, epsilon=1e-3, training=True, name=name, fused=True,reuse=tf.AUTO_REUSE)
    return x

def batch_norm(x,training):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    axis = list(range(len(x_shape) - 1))
    beta = tf.get_variable(name='beta',
                           shape=params_shape,
                           initializer=tf.zeros_initializer)

    gamma = tf.get_variable(name='gamma',
                            shape=params_shape,
                            initializer=tf.ones_initializer)

    moving_mean = tf.get_variable(name='moving_mean',
                                  shape=params_shape,
                                  initializer=tf.zeros_initializer,
                                  trainable=False)

    moving_variance = tf.get_variable(name='moving_variance',
                                      shape=params_shape,
                                      initializer=tf.ones_initializer,
                                      trainable=False)

    tf.add_to_collection('BN_MEAN_VARIANCE', moving_mean)
    tf.add_to_collection('BN_MEAN_VARIANCE', moving_variance)

    # These ops will only be preformed when training.
    mean, variance = tf.nn.moments(x, axis)
    update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                               mean,
                                                               0.99,
                                                               name='MovingAvgMean')
    update_moving_variance = moving_averages.assign_moving_average(moving_variance,
                                                                   variance,
                                                                   0.99,
                                                                   name='MovingAvgVariance')

    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_mean)
    tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_moving_variance)

    mean, variance = tf.cond(
        pred= training,
        true_fn=lambda: (mean, variance),
        false_fn=lambda: (moving_mean, moving_variance)
    )
    x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, 1e-3)
    return x


def Bottleneck(x, is_training,block_name, outplanes, stride=1, downsample=None):
    residual = x
    with tf.variable_scope(block_name + '11_1'):
        name = block_name + 'conv1'
        out = conv(x, 1, outplanes, name=name, stride=stride)
        out = batch_norm(out,is_training)
        out = tf.nn.relu(out)

    with tf.variable_scope(block_name + '33_2'):
        name = block_name + 'conv2'
        out = conv(x, 3, outplanes, name=name, stride=stride)
        out = batch_norm(out,is_training)
        out = tf.nn.relu(out)

    with tf.variable_scope(block_name + '11_3'):
        name = block_name + 'conv3'
        out = conv(out, 1, outplanes * 4, name=name, stride=stride)
        out = batch_norm(out,is_training)

    if downsample is not None:
        with tf.variable_scope(block_name + 'downsample'):
            residual = downsample(x,1,outplanes * 4, 'stage_dawnSample',stride = stride)
            residual = batch_norm(residual, is_training)

    out = out + residual
    out = tf.nn.relu(out)
    return out


def BasicBlock(x, is_training,block_name, outplanes, stride=1, downsample=None):
    residual = x
    with tf.variable_scope(block_name + '33_1'):
        name = block_name + 'conv1'
        out = conv(x, 3, outplanes, name=name, stride=stride)
        out = batch_norm(out,is_training)
        out = tf.nn.relu(out)
    with tf.variable_scope(block_name + '33_2'):
        name = block_name + 'conv2'
        out = conv(out, 3, outplanes, name=name, stride=stride)
        out = batch_norm(out,is_training)
    if downsample is not None:
        with tf.variable_scope(block_name + 'downsample'):
            residual = downsample(x)
            residual = batch_norm(residual, is_training)

    out = out + residual
    out = tf.nn.relu(out)
    return out


def CBAM(input, reduction):

    _, width, height, channel = input.get_shape()  # (B, W, H, C)

    # channel attention
    x_mean = tf.reduce_mean(input, axis=(1, 2), keepdims=True)   # (B, 1, 1, C)
    x_mean = tf.layers.conv2d(x_mean, channel // reduction, 1, activation=tf.nn.relu)  # (B, 1, 1, C // r)
    x_mean = tf.layers.conv2d(x_mean, channel, 1)   # (B, 1, 1, C)

    x_max = tf.reduce_max(input, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
    x_max = tf.layers.conv2d(x_max, channel // reduction, 1, activation=tf.nn.relu)
    # (B, 1, 1, C // r)
    x_max = tf.layers.conv2d(x_max, channel, 1)  # (B, 1, 1, C)

    x = tf.add(x_mean, x_max)   # (B, 1, 1, C)
    x = tf.nn.sigmoid(x)        # (B, 1, 1, C)
    x = tf.multiply(input, x)   # (B, W, H, C)

    # spatial attention
    y_mean = tf.reduce_mean(x, axis=3, keepdims=True)  # (B, W, H, 1)
    y_max = tf.reduce_max(x, axis=3, keepdims=True)  # (B, W, H, 1)
    y = tf.concat([y_mean, y_max], axis=-1)     # (B, W, H, 2)
    y = tf.layers.conv2d(y, 1, 7, padding='same', activation=tf.nn.sigmoid)    # (B, W, H, 1)
    y = tf.multiply(x, y)  # (B, W, H, C)

    return y


def C_CBAM(input, reduction):

    _, width, height, channel = input.get_shape()  # (B, W, H, C)

    # channel attention
    x_mean = tf.reduce_mean(input, axis=(1, 2), keepdims=True)   # (B, 1, 1, C)
    x_mean = tf.layers.conv2d(x_mean, channel // reduction, 1, activation=tf.nn.relu)  # (B, 1, 1, C // r)
    x_mean = tf.layers.conv2d(x_mean, channel, 1)   # (B, 1, 1, C)

    x_max = tf.reduce_max(input, axis=(1, 2), keepdims=True)  # (B, 1, 1, C)
    x_max = tf.layers.conv2d(x_max, channel // reduction, 1, activation=tf.nn.relu)
    # (B, 1, 1, C // r)
    x_max = tf.layers.conv2d(x_max, channel, 1)  # (B, 1, 1, C)

    x = tf.add(x_mean, x_max)   # (B, 1, 1, C)
    x = tf.nn.sigmoid(x)        # (B, 1, 1, C)
    x = tf.multiply(input, x)   # (B, W, H, C)

    return x

def S_CBAM(input, reduction):

    _, width, height, channel = input.get_shape()  # (B, W, H, C)

    # spatial attention
    y_mean = tf.reduce_mean(input, axis=3, keepdims=True)  # (B, W, H, 1)
    y_max = tf.reduce_max(input, axis=3, keepdims=True)  # (B, W, H, 1)
    y = tf.concat([y_mean, y_max], axis=-1)     # (B, W, H, 2)
    y = tf.layers.conv2d(y, 1, 7, padding='same', activation=tf.nn.sigmoid)    # (B, W, H, 1)
    y = tf.multiply(input, y)  # (B, W, H, C)

    return y







