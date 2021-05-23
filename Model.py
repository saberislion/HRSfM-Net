# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.training import moving_averages

import tensorflow as tf
import  logging
import cv2
import numpy as np
import Models_Block as block_Model
from Models_Block import *
import Models_Config as config
import HighResolutionModule as HM
from nets import *



blocks_dict = {
    'BASIC': block_Model.BasicBlock,
    'BOTTLENECK': block_Model.Bottleneck
}

class HighResolutionNet():
    def __init__(self,num_class):
        self._is_training = tf.cast(True, tf.bool)
        self.layer = []
        self.nua_class = num_class
    def forword(self, x, type):
        input_height = x.get_shape().as_list()[1]
        input_width = x.get_shape().as_list()[2]
        with tf.variable_scope('depth_decode', reuse=tf.AUTO_REUSE):
            self.layer = []
            # x = tf.image.resize_nearest_neighbor(x, [np.int(input_height * 2), np.int(input_width * 2)])

            #..............................stage1..................................#
            with tf.variable_scope('stage1'):
                with tf.variable_scope('stage1_conv1'):
                    temp = block_Model.conv(x, 3, 64, 'stage1_conv1', stride=2)
                    temp = block_Model.batch_norm(temp,self._is_training)
                    temp = tf.nn.relu(temp)
                    temp1 = temp
                with tf.variable_scope('stage1_conv2'):
                    temp = block_Model.conv(temp, 3, 64, 'stage1_conv2', stride=2)
                    temp = block_Model.batch_norm(temp, self._is_training)
                    temp = tf.nn.relu(temp)
                    temp2 = temp
                    
                
                layer1 = self._make_layer(temp,block_Model.Bottleneck, 64, 128, 4)
                
                layer1 = CBAM(layer1, 1)
                self.layer.append(layer1)
                
            #......................................................................#

            # ..............................stage2..................................#
            with tf.variable_scope('stage2'):
                num_channels = config.HIGH_RESOLUTION_NET[1]['NUM_CHANNELS']
                block = blocks_dict[config.HIGH_RESOLUTION_NET[1]['BLOCK']]            
                num_channels = [num_channels[i] * 1 for i in range(len(num_channels))]
                self.layer = self._make_transition_layer('stage2', self.layer, [512], num_channels)
                
                self.layer, pre_stage_channels = self._make_stage('stage2',self.layer,config.HIGH_RESOLUTION_NET[1],
                                                                num_channels)
            # ......................................................................#

            # ..............................stage3..................................#
            with tf.variable_scope('stage3'):
                num_channels = config.HIGH_RESOLUTION_NET[2]['NUM_CHANNELS']
                block = blocks_dict[config.HIGH_RESOLUTION_NET[2]['BLOCK']]
                num_channels = [num_channels[i] * 1 for i in range(len(num_channels))]
                self.layer = self._make_transition_layer('stage3', self.layer, pre_stage_channels, num_channels)
                self.layer, pre_stage_channels = self._make_stage('stage3', self.layer, config.HIGH_RESOLUTION_NET[2],
                                                                num_channels)
            # ......................................................................#

            # ..............................stage4..................................#
            with tf.variable_scope('stage4'):
                num_channels = config.HIGH_RESOLUTION_NET[3]['NUM_CHANNELS']
                block = blocks_dict[config.HIGH_RESOLUTION_NET[3]['BLOCK']]
                num_channels = [num_channels[i] * 1 for i in range(len(num_channels))]
                self.layer = self._make_transition_layer('stage4', self.layer, pre_stage_channels, num_channels)
                self.layer, pre_stage_channels = self._make_stage('stage4', self.layer, config.HIGH_RESOLUTION_NET[3],
                                                                num_channels)
                #self.layer:当前 feature map: pre_stage_channels:不同级别的channel数量
            # ......................................................................#

            # ..............................concat..................................#
            if type == 'disp':
                print("running encoder")
                print("disp")
                self.layer.append(temp1)
                self.layer.append(temp2)
                disp, _ = disp_decoder(self.layer)
                # disp = disp_code(self.layer)
                # disp = build_disp_net(self.layer)

                return disp

            if type == 'pose':
                print(self.layer[3])

                return self.layer[3]
        # else:
        #     with tf.variable_scope('concat'):
        #         for i in range(len(self.layer)):
        #             self.layer[i] = tf.image.resize_bilinear(self.layer[i], [input_height, input_width])
        #         feature_map = tf.concat((self.layer[0], self.layer[1], self.layer[2], self.layer[3]), axis=3)
        #         feature_map = block_Model.conv(feature_map, 1, self.nua_class,'concat_conv', 1)
        #         return feature_map

        # ......................................................................#


    def _make_layer(self, x, block, inplanes, planes, blocks, stride=1):
        '''作用：make layer1
            在stage1中调用'''
        downsample = None
        if stride != 1 or inplanes != planes * 4:
            downsample = block_Model.conv

        x = block(x, is_training=self._is_training, block_name='layer1_Botteleneck0', outplanes=planes,
              stride=stride,downsample=downsample)
        for i in range(1, blocks):
            x = block(x, is_training=self._is_training, block_name='layer1_Botteleneck'+ str(i), outplanes=planes)
        return x

    def _make_transition_layer(self, stage,layer, num_channels_pre_layer, num_channels_cur_layer):

        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                with tf.variable_scope(stage + '_transition_layers_1' + str(i)):
                    if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                        layer[i] = block_Model.conv(layer[i],3,num_channels_cur_layer[i],'transition_layers',1)
                        layer[i] = block_Model.batch_norm(layer[i],self._is_training)
                        layer[i] = tf.nn.relu(layer[i])
                        transition_layers.append(layer[i])
                    else:
                        transition_layers.append(layer[i])
            else:
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    with tf.variable_scope(stage + 'stage1_transition_layers_2'+str(j)):
                        temp = block_Model.conv(layer[i-1], 3, outchannels, 'transition_layers', 2)
                        temp = block_Model.batch_norm(temp,self._is_training)
                        temp = tf.nn.relu(temp)
                        transition_layers.append(temp)
        return transition_layers

    def _make_stage(self, stage,layer,layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']
        is_fuse = layer_config['is_fuse']

        make_stage = []
        channel = []

        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            hm= HM.HighResolutionModule('stage2', self._is_training,num_branches,block,num_blocks,num_inchannels,
                                     num_channels,fuse_method,reset_multi_scale_output, is_fuse)
            make_stage,channel = hm.forword(layer,num_branches,block,num_blocks,num_inchannels,
                                     num_channels,fuse_method,reset_multi_scale_output, is_fuse)
            
        return make_stage,channel

















