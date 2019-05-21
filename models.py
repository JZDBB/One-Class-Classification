from __future__ import division
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import layers
import re
from ops import *
from utils import *
from kh_tools import *
import logging
import matplotlib.pyplot as plt
import read_data

class ALOCC_Model(object):
    def __init__(self, sess, input_height=45, input_width=45, output_height=64, output_width=64,
               batch_size=128, sample_num = 128, attention_label=1, is_training=True, pre=True,
               z_dim=100, gf_dim=16, df_dim=16, gfc_dim=512, dfc_dim=512, c_dim=3,
               dataset_name=None, dataset_address=None, input_fname_pattern=None,
               checkpoint_dir=None, log_dir=None, sample_dir=None, r_alpha = 0.2,
               kb_work_on_patch=True, nd_input_frame_size=(240, 360), nd_patch_size=(10, 10), n_stride=1,
               n_fetch_data=10, n_per_itr_print_results=10):
        self.n_per_itr_print_results = n_per_itr_print_results
        self.nd_input_frame_size = nd_input_frame_size
        self.b_work_on_patch = kb_work_on_patch
        self.sample_dir = sample_dir

        self.sess = sess
        self.is_training = is_training
        self.pre = pre

        self.r_alpha = r_alpha

        self.batch_size = batch_size
        self.sample_num = sample_num

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')
        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')
        self.g_bn5 = batch_norm(name='g_bn5')
        self.g_bn6 = batch_norm(name='g_bn6')

        self.dataset_name = dataset_name
        self.dataset_address = dataset_address
        self.input_fname_pattern = input_fname_pattern
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir

        self.attention_label = attention_label

        if self.is_training:
            logging.basicConfig(filename='ALOCC_loss.log', level=logging.INFO)

        if self.dataset_name == 'mnist':
            mnist = input_data.read_data_sets(self.dataset_address)
            specific_idx = np.where(mnist.train.labels == self.attention_label)[0]
            self.data = mnist.train.images[specific_idx].reshape(-1, 28, 28, 1)
            self.c_dim = 1
        elif self.dataset_name == 'UCSD':
            self.nStride = n_stride
            self.patch_size = nd_patch_size
            self.patch_step = (n_stride, n_stride)
            lst_image_paths = []
            for s_image_dir_path in glob(os.path.join(self.dataset_address, self.input_fname_pattern)):
                for sImageDirFiles in glob(os.path.join(s_image_dir_path + '/*')):
                    lst_image_paths.append(sImageDirFiles)
            self.dataAddress = lst_image_paths
            lst_forced_fetch_data = [self.dataAddress[x] for x in
                                     random.sample(range(0, len(lst_image_paths)), n_fetch_data)]

            self.data = lst_forced_fetch_data
            self.c_dim = 1
        elif self.dataset_name == 'cifar-10':
            self.data = read_data.read_data(self.attention_label)
            self.c_dim = 3
        else:
            assert ('Error in loading dataset')

        self.grayscale = (self.c_dim == 1)

        self.build_model()

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        labels = tf.Variable(tf.ones([1, self.batch_size], tf.int64))
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.sample_inputs = tf.placeholder(tf.float32, [self.sample_num] + image_dims, name='sample_inputs')

        inputs = self.inputs
        sample_inputs = self.sample_inputs

        self.z = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='z')

        self.G, self.feature = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(inputs)

        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        # tesorboard setting
        # self.z_sum = histogram_summary("z", self.z)
        # self.d_sum = histogram_summary("d", self.D)
        # self.d__sum = histogram_summary("d_", self.D_)
        # self.G_sum = image_summary("G", self.G)

        # Simple GAN's losses
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        # Refinement loss
        self.g_r_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=inputs, predictions=self.G))

        # center-loss
        self.center_loss, self.centers, self.centers_update_op = get_center_loss(self.feature, labels, 0.5, 1)

        self.pre_loss = 0.5 * self.center_loss + self.g_r_loss

        self.g_loss = self.g_loss + self.g_r_loss * self.r_alpha
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.d_loss_real_sum = scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = scalar_summary("d_loss_fake", self.d_loss_fake)
        self.g_loss_sum = scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = scalar_summary("d_loss", self.d_loss)
        self.g_r_loss_ = scalar_summary('g_r_loss', self.g_r_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        # =========================================================================================================

    def discriminator(self, image, reuse=False):
        with tf.variable_scope("discriminator") as scope:
            if reuse:
                scope.reuse_variables()
            h0 = lrelu(conv2d(image, self.df_dim, name='d_h0_conv'))
            h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim * 2, name='d_h1_conv')))
            h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim * 4, name='d_h2_conv')))
            h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim * 8, name='d_h3_conv')))
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, 'd_h3_lin')
            h5 = tf.nn.sigmoid(h4, name='d_output')
            return h5, h4

        # =========================================================================================================

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            hae0 = lrelu(self.g_bn4(conv2d(z, self.df_dim * 2, name='g_encoder_h0_conv')))
            hae1 = lrelu(self.g_bn5(conv2d(hae0, self.df_dim * 4, name='g_encoder_h1_conv')))
            hae2 = lrelu(self.g_bn6(conv2d(hae1, self.df_dim * 8, name='g_encoder_h2_conv')))

            h2, self.h2_w, self.h2_b = deconv2d(
                hae2, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_decoder_h1', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_decoder_h0', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_decoder_h00', with_w=True)

            return tf.nn.tanh(h4, name='g_output'), hae2

        # =========================================================================================================

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()

            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            hae0 = lrelu(self.g_bn4(conv2d(z, self.df_dim * 2, name='g_encoder_h0_conv')))
            hae1 = lrelu(self.g_bn5(conv2d(hae0, self.df_dim * 4, name='g_encoder_h1_conv')))
            hae2 = lrelu(self.g_bn6(conv2d(hae1, self.df_dim * 8, name='g_encoder_h2_conv')))

            h2, self.h2_w, self.h2_b = deconv2d(
                hae2, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_decoder_h1', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(
                h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_decoder_h0', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(
                h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_decoder_h00', with_w=True)

            return tf.nn.tanh(h4, name='g_output')
