# -*- coding: utf-8 -*-
from __future__ import division
import os
import time
import math
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from data_loader import DataLoader
from utils import *
from nets import *

from Model import HighResolutionNet
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


class SfMLearner(object):
    def __init__(self):
        pass

    def build_train_graph(self):
        opt = self.opt
        loader = DataLoader(opt.dataset_dir,
                            opt.batch_size,
                            opt.img_height,
                            opt.img_width,
                            opt.num_source,
                            opt.num_scales)
        with tf.name_scope("data_loading"):

            tgt_image, src_image_stack, tgt_image1, src_image_stack1, tgt_image_aug, src_image_stack_aug, \
            tgt_image1_aug, src_image_stack1_aug, intrinsics = loader.load_train_batch()

            tgt_image = self.preprocess_image(tgt_image)
            src_image_stack = self.preprocess_image(src_image_stack)

            tgt_image_aug = self.preprocess_image(tgt_image_aug)
            src_image_stack_aug = self.preprocess_image(src_image_stack_aug)


        with tf.name_scope("depth_prediction"):

            num_class = 1
            HRmodel = HighResolutionNet(num_class)
            pred_disp = HRmodel.forword(tgt_image_aug, type = 'disp')
            pred_depth = [1. / d for d in pred_disp]

            if opt.inverse:
 
                pred_disp1 = HRmodel.forword(src_image_stack_aug[:,:,:,:3], type = 'disp')
                pred_depth1 = [1. / d for d in pred_disp1]


            

        with tf.name_scope("pose_and_explainability_prediction"):


            
            pred_poses, _ = \
                pose_exp_net(tgt_image_aug,
                             src_image_stack_aug,
                             is_training=True)

        with tf.name_scope("compute_loss"):
            pixel_losses = 0
            ssim_loss = 0
            smooth_loss = 0
            pixel_match_loss = 0
            depth_loss = 0
            epipolar_loss = 0
            tgt_image_all = []
            src_image_stack_all = []

            proj_image_stack_all = []
            proj_error_stack_all = []
            exp_mask_stack_all = []

            for s in range(opt.num_scales):
                print(s)
                # Scale the source and target images for computing loss at the
                # according scale.
                curr_tgt_image = tf.image.resize_area(tgt_image,
                                                      [int(opt.img_height / (2 ** s)), int(opt.img_width / (2 ** s))])
                curr_src_image_stack = tf.image.resize_area(src_image_stack,
                                                            [int(opt.img_height / (2 ** s)),
                                                             int(opt.img_width / (2 ** s))])



                if opt.smooth_weight > 0:
                    smooth_loss += opt.smooth_weight / (2 ** s) * \
                                   self.compute_smooth_loss(pred_disp[s], curr_tgt_image)
                    if opt.inverse:

                        smooth_loss += opt.smooth_weight / (2 ** s) * \
                                    self.compute_smooth_loss(pred_disp1[s], curr_src_image_stack[:,:,:,:3])

                q = 96

                for i in range(opt.num_source):
                    # 2-1 2-3
                    curr_proj_image, _, mask, _= projective_inverse_warp_withdepth(
                        curr_src_image_stack[:, :, :, 3 * i:3 * (i + 1)],
                        tf.squeeze(pred_depth[s], axis=3),
                        pred_poses[:, 6 * i:6 * (i + 1)],
                        intrinsics[:, s, :, :],
                        pred_depth[s])
                    curr_proj_error = tf.abs(curr_proj_image - curr_tgt_image)
                    curr_proj_error1 = tf.abs(curr_src_image_stack[:, :, :, 3 * i:3 * (i + 1)] - curr_tgt_image)
                    if opt.cm_mask:
                        promask = tf.cast(
                            tf.reduce_mean(curr_proj_error, axis=3, keep_dims=True) < tf.reduce_mean(curr_proj_error1,
                                                                                                     axis=3,
                                                                                                     keep_dims=True),
                            'float32')
                        promask = tf.clip_by_value(promask, 0, 1.0) * mask


                        perct_thresh = tf.contrib.distributions.percentile(curr_proj_error, q, axis=[1,2])
                        perct_thresh = tf.expand_dims(tf.expand_dims(perct_thresh, 1), 1)
                        curr_proj_error = tf.clip_by_value(curr_proj_error, 0, perct_thresh)
                        above_perct_thresh_region = tf.reduce_max(tf.cast(tf.equal(curr_proj_error, perct_thresh), 'float32'), axis=3)
                        above_perct_thresh_region = tf.greater_equal(above_perct_thresh_region, 1.0)
                        suppresion_mask = tf.expand_dims(1.0 - tf.cast(above_perct_thresh_region, 'float32'), axis=3)
                        curr_proj_error = tf.multiply(curr_proj_error, suppresion_mask)
                        th_mask = tf.multiply(mask, suppresion_mask)

                        total_mask = promask * th_mask
                        
                        pixel_loss = tf.reduce_mean(curr_proj_error * total_mask)
                        # SSIM loss
                        if opt.ssim_weight > 0:
                            ssim_thmask = slim.avg_pool2d(th_mask, 3, 1, 'SAME')
                            ssim_totalmsk = ssim_thmask * promask
                            ssim_loss = tf.reduce_mean(
                                ssim_totalmsk * self.compute_ssim_loss(curr_proj_image, curr_tgt_image))

                    else:
                        pixel_loss = tf.reduce_mean(curr_proj_error)
                        if opt.ssim_weight > 0:
                            ssim_loss = tf.reduce_mean(self.compute_ssim_loss(curr_proj_image, curr_tgt_image))
                    reprojection_losses = opt.ssim_weight * ssim_loss + (1 - opt.ssim_weight) * pixel_loss
                    pixel_losses += reprojection_losses


                    if i == 0:
                        proj_image_stack = curr_proj_image
                        proj_error_stack = curr_proj_error
                        if opt.cm_mask:
                            exp_mask_stack = total_mask
                    else:
                        # proj_depth_stack = curr_proj_depth
                        proj_image_stack = tf.concat([proj_image_stack,
                                                      curr_proj_image], axis=3)
                        proj_error_stack = tf.concat([proj_error_stack,
                                                      curr_proj_error], axis=3)
                        if opt.cm_mask:
                            exp_mask_stack = tf.concat([exp_mask_stack,
                                                        total_mask], axis=3)
                    
                if opt.inverse:

                    for i in range(1):                 
                        # 1-3
                        curr_depth = pred_depth1[s]
                        curr_pose_first = pose_vec2mat(pred_poses[:,:6])
                        curr_pose_first = tf.matrix_inverse(curr_pose_first)
                        curr_pose_second = pose_vec2mat(pred_poses[:,6:])
                        curr_pose = tf.matmul(curr_pose_second, curr_pose_first)
                        curr_scr_img = curr_src_image_stack[:, :, :, 3:]
                        curr_tgt_img = curr_src_image_stack[:, :, :, :3]
                        
                        curr_proj_image, _, mask, _= projective_inverse_warp_withdepth(
                            curr_scr_img,
                            tf.squeeze(curr_depth, axis=3),
                            curr_pose,
                            intrinsics[:, s, :, :],
                            curr_depth,
                            False)
                        curr_proj_error = tf.abs(curr_proj_image - curr_tgt_img)
                        curr_proj_error1 = tf.abs(curr_scr_img  - curr_tgt_img)
                        if opt.cm_mask:
                            promask = tf.cast(
                            tf.reduce_mean(curr_proj_error, axis=3, keep_dims=True) < tf.reduce_mean(curr_proj_error1,
                                                                                                     axis=3,
                                                                                                     keep_dims=True),
                            'float32')
                            promask = tf.clip_by_value(promask, 0, 1.0) * mask


                            perct_thresh = tf.contrib.distributions.percentile(curr_proj_error, q, axis=[1,2])
                            perct_thresh = tf.expand_dims(tf.expand_dims(perct_thresh, 1), 1)
                            curr_proj_error = tf.clip_by_value(curr_proj_error, 0, perct_thresh)
                            above_perct_thresh_region = tf.reduce_max(tf.cast(tf.equal(curr_proj_error, perct_thresh), 'float32'), axis=3)
                            above_perct_thresh_region = tf.greater_equal(above_perct_thresh_region, 1.0)
                            suppresion_mask = tf.expand_dims(1.0 - tf.cast(above_perct_thresh_region, 'float32'), axis=3)
                            curr_proj_error = tf.multiply(curr_proj_error, suppresion_mask)
                            th_mask = tf.multiply(mask, suppresion_mask)

                            total_mask = promask * th_mask
                            
                            pixel_loss = tf.reduce_mean(curr_proj_error * total_mask)
                            # SSIM loss
                            if opt.ssim_weight > 0:
                                ssim_thmask = slim.avg_pool2d(th_mask, 3, 1, 'SAME')
                                ssim_totalmsk = ssim_thmask * promask
                                ssim_loss = tf.reduce_mean(
                                    ssim_totalmsk * self.compute_ssim_loss(curr_proj_image, curr_tgt_img))

                        else:
                            pixel_loss = tf.reduce_mean(curr_proj_error)
                            if opt.ssim_weight > 0:
                                ssim_loss = tf.reduce_mean(self.compute_ssim_loss(curr_proj_image, curr_tgt_img))
                        reprojection_losses = opt.ssim_weight * ssim_loss + (1 - opt.ssim_weight) * pixel_loss
                        pixel_losses += reprojection_losses 

                tgt_image_all.append(curr_tgt_image)
                src_image_stack_all.append(curr_src_image_stack)
                proj_image_stack_all.append(proj_image_stack)
                proj_error_stack_all.append(proj_error_stack)
                if opt.cm_mask:
                    exp_mask_stack_all.append(exp_mask_stack)

            train_vars = [var for var in tf.trainable_variables()]
            self.total_step = opt.total_epoch * loader.steps_per_epoch
            self.global_step = tf.Variable(0, name='global_step', trainable=False)  # int32

            # match_loss weight
            incr_xs = [1.0, 0.0]
            bound_incrx = [np.int(self.total_step * 2 / 5)]
            self.x = tf.train.piecewise_constant(self.global_step, bound_incrx, incr_xs)

            #lr
            learning_rates = [opt.start_learning_rate, opt.start_learning_rate / 10]
            boundaries = [np.int(self.total_step * 4 / 5)]
            self.learning_rate = tf.train.piecewise_constant(self.global_step, boundaries, learning_rates)

            optimizer = tf.train.AdamOptimizer(self.learning_rate, opt.beta1)
            self.incr_global_step = tf.assign(self.global_step, self.global_step + 1)

            

            pixel_losses /= opt.num_scales

            total_loss = pixel_losses + \
                         smooth_loss + depth_loss + epipolar_loss + pixel_match_loss

            self.train_op = slim.learning.create_train_op(total_loss, optimizer)

        # Collect tensors that are useful later (e.g. tf summary)
        self.pred_depth = pred_depth
        self.steps_per_epoch = loader.steps_per_epoch
        self.pred_poses = pred_poses

        self.total_loss = total_loss
        self.pixel_losses = pixel_losses
        self.ssim_loss = ssim_loss
        self.smooth_loss = smooth_loss
        self.depth_loss = depth_loss
        self.epipolar_loss = epipolar_loss
        self.pixel_match_loss = pixel_match_loss

        self.tgt_image_all = tgt_image_all
        self.src_image_stack_all = src_image_stack_all
        self.proj_image_stack_all = proj_image_stack_all
        self.proj_error_stack_all = proj_error_stack_all
        if opt.cm_mask:
            self.exp_mask_stack_all = exp_mask_stack_all

    def compute_depth_loss(self, proj_depth, computed_depth):
        depth_loss = tf.abs(proj_depth - computed_depth) / (proj_depth + computed_depth)
        return depth_loss

    def compute_ssim_loss(self, x, y):
        """Computes a differentiable structured image similarity measure."""
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2

        x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
        y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

        mu_x = slim.avg_pool2d(x, 3, 1, 'VALID')
        mu_y = slim.avg_pool2d(y, 3, 1, 'VALID')

        sigma_x = slim.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
        sigma_y = slim.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
        sigma_xy = slim.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

        SSIM = SSIM_n / SSIM_d

        return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

    def compute_epipolar_error(self, p1, p2, essential_matrix, mask):
        # r(p1->p2)
        mask = tf.transpose(mask, [0, 3, 1, 2])
        p1 = mask * p1
        p1 = tf.reshape(p1, [self.opt.batch_size, 4, -1])
        p1 = tf.expand_dims(tf.transpose(p1, [0, 2, 1]), axis=-1)

        essential_matrix = tf.tile(
            tf.expand_dims(essential_matrix, axis=1),
            [1, self.opt.img_height * self.opt.img_width, 1, 1])

        ep1 = tf.matmul(essential_matrix, p1[:, :, :-1, :])
        p2 = tf.reshape(p2, [self.opt.batch_size, 4, -1])
        p2 = tf.expand_dims(tf.transpose(p2, [0, 2, 1]), axis=2)
        epipolar_error = tf.matmul(p2[:, :, :, :-1], ep1)
        return tf.reduce_mean(tf.abs(epipolar_error))


    def compute_smooth_loss(self, disp, img):
        norm_disp = disp / (tf.reduce_mean(disp, [1, 2], keep_dims=True) + 1e-7)

        grad_disp_x = tf.abs(norm_disp[:, :-1, :, :] - norm_disp[:, 1:, :, :])
        grad_disp_y = tf.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])

        grad_img_x = tf.abs(img[:, :-1, :, :] - img[:, 1:, :, :])
        grad_img_y = tf.abs(img[:, :, :-1, :] - img[:, :, 1:, :])

        weight_x = tf.exp(-tf.reduce_mean(grad_img_x, 3, keep_dims=True))
        weight_y = tf.exp(-tf.reduce_mean(grad_img_y, 3, keep_dims=True))

        smoothness_x = grad_disp_x * weight_x
        smoothness_y = grad_disp_y * weight_y

        return tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)



    def collect_summaries(self):
        opt = self.opt
        tf.summary.scalar("total_loss", self.total_loss)
        tf.summary.scalar("pixel_losses", self.pixel_losses)
        
        tf.summary.scalar("smooth_loss", self.smooth_loss)
        
        for s in range(opt.num_scales):
            tf.summary.image("scale%d_depth" % s, self.pred_depth[s])
            tf.summary.image('scale%d_disparity_image' % s, 1. / self.pred_depth[s])
            tf.summary.image('scale%d_target_image' % s, \
                             self.deprocess_image(self.tgt_image_all[s]))
            for i in range(1):
                if opt.cm_mask:
                    tf.summary.image(
                        'scale%d_exp_mask_%d' % (s, i),
                        tf.expand_dims(self.exp_mask_stack_all[s][:, :, :, i], -1))
                tf.summary.image(
                    'scale%d_source_image_%d' % (s, i),
                    self.deprocess_image(self.src_image_stack_all[s][:, :, :, i * 3:(i + 1) * 3]))
                tf.summary.image('scale%d_projected_image_%d' % (s, i),
                                 self.deprocess_image(self.proj_image_stack_all[s][:, :, :, i * 3:(i + 1) * 3]))
                tf.summary.image('scale%d_proj_error_%d' % (s, i),
                                 self.deprocess_image(
                                     tf.clip_by_value(self.proj_error_stack_all[s][:, :, :, i * 3:(i + 1) * 3] - 1, -1,
                                                      1)))
        # for var in tf.trainable_variables():
        #     tf.summary.histogram(var.op.name + "/values", var)
        # for grad, var in self.grads_and_vars:
        #     tf.summary.histogram(var.op.name + "/gradients", grad)

    def train(self, opt):
        self.opt = opt
        opt.num_source = opt.seq_length - 1
        # TODO: currently fixed to 4
        #        opt.num_scales = 4

        self.build_train_graph()
        self.collect_summaries()
        with tf.name_scope("parameter_count"):
            parameter_count = tf.reduce_sum([tf.reduce_prod(tf.shape(v)) \
                                             for v in tf.trainable_variables()])
        self.saver = tf.train.Saver([var for var in tf.trainable_variables()] + \
                                    [self.global_step],
                                    max_to_keep=10)
        sv = tf.train.Supervisor(logdir=opt.checkpoint_dir,
                                 save_summaries_secs=0,
                                 saver=None)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        with sv.managed_session(config=config) as sess:
            print('Trainable variables: ')
            for var in tf.trainable_variables():
                print(var.name)
            print("parameter_count =", sess.run(parameter_count))
            if opt.continue_train:
                if opt.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(opt.checkpoint_dir)
                else:
                    checkpoint = opt.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                self.saver.restore(sess, checkpoint)
            start_time = time.time()
            for step in range(1, self.total_step + 1):
                fetches = {
                    "train": self.train_op,
                    "global_step": self.global_step,
                    "incr_global_step": self.incr_global_step
                }
                fetches["x"] = self.x
                # fetches["incr_x"] = self.incr_x
                # fetches["incr_x1"] = self.incr_x1


                if step % opt.summary_freq == 0:
                    fetches["loss"] = self.total_loss
                    fetches["summary"] = sv.summary_op
                    fetches["lr"] = self.learning_rate
                results = sess.run(fetches)
                gs = results["global_step"]

                if step % opt.summary_freq == 0:
                    sv.summary_writer.add_summary(results["summary"], gs)
                    train_epoch = math.ceil(gs / self.steps_per_epoch)
                    train_step = gs - (train_epoch - 1) * self.steps_per_epoch
                    print("Epoch: [%2d] [%5d/%5d] time: %4.4f/it loss: %.3f lr: {%.5f}" \
                          % (train_epoch, train_step, self.steps_per_epoch, \
                             (time.time() - start_time) / opt.summary_freq,
                             results["loss"], results["lr"]))
                    start_time = time.time()

                if step % opt.save_latest_freq == 0:
                    self.save(sess, opt.checkpoint_dir, 'latest')

                #                if step != 0 and step % (self.steps_per_epoch * 2)  == 0:
                if step % self.steps_per_epoch == 0:
                    self.save(sess, opt.checkpoint_dir, gs)

    def build_depth_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
                                                self.img_height, self.img_width, 3], name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        with tf.name_scope("depth_prediction"):
            num_class = 1
            HRmodel = HighResolutionNet(num_class)
            pred_disp = HRmodel.forword(input_mc, type = 'disp')
            pred_depth = [1. / d for d in pred_disp]
            
        #
        #     pred_disp, depth_net_endpoints = disp_net(
        #         input_mc, is_training=False)
        #     pred_depth = [1. / disp for disp in pred_disp]
        pred_depth = pred_depth[0]
        self.inputs = input_uint8
        self.pred_depth = pred_depth


    def build_pose_test_graph(self):
        input_uint8 = tf.placeholder(tf.uint8, [self.batch_size,
                                                self.img_height, self.img_width * self.seq_length, 3],
                                     name='raw_input')
        input_mc = self.preprocess_image(input_uint8)
        loader = DataLoader()
        tgt_image, src_image_stack = \
            loader.batch_unpack_image_sequence(
                input_mc, self.img_height, self.img_width, self.num_source)
        with tf.name_scope("pose_prediction"):
            pred_poses, _, _ = pose_exp_net(
                tgt_image, src_image_stack, do_exp=False, is_training=False)
            self.inputs = input_uint8
            self.pred_poses = pred_poses

    def preprocess_image(self, image):
        # Assuming input image is uint8
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        return image * 2. - 1.

    def deprocess_image(self, image):
        # Assuming input image is float32
        image = (image + 1.) / 2.
        return tf.image.convert_image_dtype(image, dtype=tf.uint8)

    def setup_inference(self,
                        img_height,
                        img_width,
                        mode,
                        seq_length=3,
                        batch_size=1):
        self.img_height = img_height
        self.img_width = img_width
        self.mode = mode
        self.batch_size = batch_size
        if self.mode == 'depth':
            self.build_depth_test_graph()
        if self.mode == 'pose':
            self.seq_length = seq_length
            self.num_source = seq_length - 1
            self.build_pose_test_graph()

    def inference(self, inputs, sess, mode='depth'):
        fetches = {}
        if mode == 'depth':
            fetches['depth'] = self.pred_depth
        if mode == 'pose':
            fetches['pose'] = self.pred_poses
        results = sess.run(fetches, feed_dict={self.inputs: inputs})
        return results

    def save(self, sess, checkpoint_dir, step):
        model_name = 'model'
        print(" [*] Saving checkpoint to %s..." % checkpoint_dir)
        if step == 'latest':
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name + '.latest'))
        else:
            self.saver.save(sess,
                            os.path.join(checkpoint_dir, model_name),
                            global_step=step)
