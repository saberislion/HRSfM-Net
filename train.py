# -*- coding: utf-8 -*-
from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
from SfM import SfMLearner
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "", "Directory name to save the checkpoints")
flags.DEFINE_string("init_checkpoint_file",  "", "Specific checkpoint file to initialize from")
flags.DEFINE_float("start_learning_rate", 0.0001, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_float("smooth_weight", 0.0, "Weight for smoothness")
flags.DEFINE_float("ssim_weight", 0.85, "Weight for SSIM loss")
flags.DEFINE_float("pixel_weight", 1.0, "Weight for pixel_loss")
flags.DEFINE_integer("total_epoch", 20, "The number  of  total_epoch")
flags.DEFINE_integer("batch_size", 4, "The size of  a sample batch")
flags.DEFINE_integer("num_source", 2, "The number of source images")
flags.DEFINE_integer("num_scales", 1, "The size  of  scale")
flags.DEFINE_integer("img_height", 128, "Image height")
flags.DEFINE_integer("img_width", 416, "Image width")
flags.DEFINE_integer("seq_length", 3, "Sequence length for each example")
flags.DEFINE_integer("max_steps", 155000, "Maximum number of training iterations")
flags.DEFINE_integer("summary_freq", 200, "Logging every log_freq iterations")
flags.DEFINE_integer("save_latest_freq", 5000, \
                     "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")
flags.DEFINE_boolean("across", False, "Continue training from previous checkpoint")
flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_boolean("cm_mask", True, "use mask or not")
flags.DEFINE_boolean("inverse", True, "use mask or not")
FLAGS = flags.FLAGS


def main(_):
    seed = 8964
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    pp = pprint.PrettyPrinter()
    pp.pprint(flags.FLAGS.__flags)

    if not os.path.exists(FLAGS.checkpoint_dir):
        os.makedirs(FLAGS.checkpoint_dir)

    sfm = SfMLearner()
    sfm.train(FLAGS)


if __name__ == '__main__':
    tf.app.run()
