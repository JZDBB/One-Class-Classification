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
        elif self.dataset_name == 'cifar-10':
            self.data = read_data.read_data(self.attention_label)
            self.c_dim = 3
        else:
            assert ('Error in loading dataset')
        self.grayscale = (self.c_dim == 1)
        self.build_model()

    def build_model(self):
        image_dims = [self.input_height, self.input_width, self.c_dim]
        labels = tf.Variable(tf.zeros([1, self.batch_size], tf.int64))
        self.inputs = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='real_images')
        self.sample_inputs = tf.placeholder(tf.float32, [self.sample_num] + image_dims, name='sample_inputs')
        self.z = tf.placeholder(tf.float32, [self.batch_size] + image_dims, name='z')
        self.G, self.feature = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.inputs)
        self.sampler = self.sampler(self.z)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)

        # Simple GAN's losses
        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        # Refinement loss
        self.g_r_loss = tf.reduce_mean(tf.losses.mean_squared_error(labels=self.inputs, predictions=self.G))

        # center-loss
        self.center_loss, self.centers, self.centers_update_op = get_center_loss(self.feature, labels, 0.5, 1)

        self.pre_loss = 0.5 * self.center_loss + self.g_r_loss

        self.g_loss = self.g_loss + self.g_r_loss * self.r_alpha + self.center_loss * 0.5
        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.pre_loss_sum = scalar_summary("pre_loss_sum", self.pre_loss)
        # self.pre_loss_recon_sum = scalar_summary("pre_loss_recon", self.g_r_loss)
        self.center_loss_sum = scalar_summary("pre_center_loss", self.center_loss)
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
            return h4, h5

        # =========================================================================================================

    def generator(self, z):
        with tf.variable_scope("generator") as scope:
            # s_h, s_w = self.output_height, self.output_width
            # s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            # s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            # s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            # s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            #
            # hae0 = lrelu(self.g_bn4(conv2d(z, self.df_dim * 2, name='g_encoder_h0_conv')))
            # hae1 = lrelu(self.g_bn5(conv2d(hae0, self.df_dim * 4, name='g_encoder_h1_conv')))
            # hae2 = lrelu(self.g_bn6(conv2d(hae1, self.df_dim * 8, name='g_encoder_h2_conv')))
            #
            # flat = tf.contrib.layers.flatten(hae2)
            # feature = tf.layers.dense(flat, units=128, name='g_mean')
            # z_develop = tf.layers.dense(feature, units=4 * 4 * 128, name="g_flat")
            # decode_in = tf.reshape(z_develop, [-1, 4, 4, 128])
            #
            # h2, self.h2_w, self.h2_b = deconv2d(
            #     decode_in, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_decoder_h1', with_w=True)
            # h2 = tf.nn.relu(self.g_bn2(h2))
            #
            # h3, self.h3_w, self.h3_b = deconv2d(
            #     h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_decoder_h0', with_w=True)
            # h3 = tf.nn.relu(self.g_bn3(h3))
            #
            # h4, self.h4_w, self.h4_b = deconv2d(
            #     h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_decoder_h00', with_w=True)
            xavier_initializer = tf.contrib.layers.xavier_initializer()

            conv1 = tf.layers.conv2d(inputs=z, filters=32, kernel_size=4, strides=2, padding='same',
                                     name="g_conv1", kernel_initializer=xavier_initializer, activation=lrelu)

            conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=4, strides=2, padding='same',
                                     name="g_conv2", kernel_initializer=xavier_initializer, activation=lrelu)

            conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=4, strides=2, padding='same',
                                     name="g_conv3", kernel_initializer=xavier_initializer, activation=lrelu)

            flat = tf.contrib.layers.flatten(conv3)
            feature = tf.layers.dense(flat, units=128, name='g_mean')
            z_develop = tf.layers.dense(feature, units=4 * 4 * 128, name="g_flat")
            net = tf.nn.relu(tf.reshape(z_develop, [-1, 4, 4, 128]))

            net = tf.layers.conv2d_transpose(inputs=net, filters=64, kernel_size=4, strides=2, padding='same',
                                             name="g_deconv1", kernel_initializer=xavier_initializer, activation=lrelu)

            net = tf.layers.conv2d_transpose(inputs=net, filters=32, kernel_size=4, strides=2, padding='same',
                                             name="g_deconv2", kernel_initializer=xavier_initializer, activation=lrelu)

            net = tf.layers.conv2d_transpose(inputs=net, filters=3, kernel_size=4, strides=2, padding='same',
                                             name="g_deconv3", kernel_initializer=xavier_initializer)

            # net = tf.nn.sigmoid(net, name="g_sigmoid")
            net = tf.nn.tanh(net, name="g_tanh")
            return net, feature

    def sampler(self, z, y=None):
        with tf.variable_scope("generator") as scope:
            scope.reuse_variables()
            # s_h, s_w = self.output_height, self.output_width
            # s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            # s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            # s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            # s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)
            #
            # hae0 = lrelu(self.g_bn4(conv2d(z, self.df_dim * 2, name='g_encoder_h0_conv')))
            # hae1 = lrelu(self.g_bn5(conv2d(hae0, self.df_dim * 4, name='g_encoder_h1_conv')))
            # hae2 = lrelu(self.g_bn6(conv2d(hae1, self.df_dim * 8, name='g_encoder_h2_conv')))
            #
            # flat = tf.contrib.layers.flatten(hae2)
            # feature = tf.layers.dense(flat, units=128, name='z_mean')
            # z_develop = tf.layers.dense(feature, units=4 * 4 * 128)
            # decode_in = tf.nn.relu(tf.reshape(z_develop, [-1, 4, 4, 128]))
            # h2, self.h2_w, self.h2_b = deconv2d(
            #     hae2, [self.batch_size, s_h4, s_w4, self.gf_dim * 2], name='g_decoder_h1', with_w=True)
            # h2 = tf.nn.relu(self.g_bn2(h2))
            #
            # h3, self.h3_w, self.h3_b = deconv2d(
            #     h2, [self.batch_size, s_h2, s_w2, self.gf_dim * 1], name='g_decoder_h0', with_w=True)
            # h3 = tf.nn.relu(self.g_bn3(h3))
            #
            # h4, self.h4_w, self.h4_b = deconv2d(
            #     h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_decoder_h00', with_w=True)
            #
            # return tf.nn.tanh(h4, name='g_output')
            xavier_initializer = tf.contrib.layers.xavier_initializer()

            conv1 = tf.layers.conv2d(inputs=z, filters=32, kernel_size=4, strides=2, padding='same',
                                     name="g_conv1", kernel_initializer=xavier_initializer, activation=lrelu)

            conv2 = tf.layers.conv2d(inputs=conv1, filters=64, kernel_size=4, strides=2, padding='same',
                                     name="g_conv2", kernel_initializer=xavier_initializer, activation=lrelu)

            conv3 = tf.layers.conv2d(inputs=conv2, filters=128, kernel_size=4, strides=2, padding='same',
                                     name="g_conv3", kernel_initializer=xavier_initializer, activation=lrelu)

            flat = tf.contrib.layers.flatten(conv3)
            feature = tf.layers.dense(flat, units=128, name='g_mean')
            z_develop = tf.layers.dense(feature, units=4 * 4 * 128, name="g_flat")
            net = tf.nn.relu(tf.reshape(z_develop, [-1, 4, 4, 128]))

            net = tf.layers.conv2d_transpose(inputs=net, filters=64, kernel_size=4, strides=2, padding='same',
                                             name="g_deconv1", kernel_initializer=xavier_initializer, activation=lrelu)

            net = tf.layers.conv2d_transpose(inputs=net, filters=32, kernel_size=4, strides=2, padding='same',
                                             name="g_deconv2", kernel_initializer=xavier_initializer, activation=lrelu)

            net = tf.layers.conv2d_transpose(inputs=net, filters=3, kernel_size=4, strides=2, padding='same',
                                             name="g_deconv3", kernel_initializer=xavier_initializer)

            # net = tf.nn.sigmoid(net, name="g_sigmoid")
            net = tf.nn.tanh(net, name="g_tanh")
            return net


    def train(self, config):
        # train = layers.optimize_loss(self.pre_loss, tf.train.get_or_create_global_step(),
        #                              learning_rate=config.learning_rate, optimizer='Adam', update_ops=[])
        pre_optim = tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.pre_loss, var_list=self.g_vars)
        d_optim = tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.d_loss, var_list=self.d_vars)
        g_optim = tf.train.RMSPropOptimizer(config.learning_rate).minimize(self.g_loss, var_list=self.g_vars)

        try:
            tf.global_variables_initializer().run()
        except:
            tf.initialize_all_variables().run()

        self.saver = tf.train.Saver()
        if os.path.exists('cifar-10/checkpoint'):
            self.saver.restore(self.sess, "checkpoint/ALOCC_Model.ckpt")
        elif os.path.exists('cifar-10/pre/checkpoint'):
            self.saver.restore(self.sess, "cifar-10/pre/pre_model.ckpt")

        self.pre_sum = merge_summary([self.center_loss_sum, self.g_r_loss_])
        self.g_sum = merge_summary([self.g_r_loss_, self.g_loss_sum, self.center_loss_sum])
        self.d_sum = merge_summary([self.d_loss_real_sum, self.d_loss_sum])

        log_dir = os.path.join(self.log_dir, self.model_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.writer = SummaryWriter(log_dir, self.sess.graph)
        sample = self.data[0:self.sample_num]
        sample_w_noise = get_noisy_data(self.data)
        # export images
        sample_inputs = np.array(sample).astype(np.float32)
        scipy.misc.imsave('./{}/train_input_samples.jpg'.format(config.sample_dir), montage(sample_inputs))

        # load previous checkpoint
        counter = 1

        for epoch in xrange(config.epoch):
            print('Epoch ({}/{})-------------------------------------------------'.format(epoch, config.epoch))
            batch_idxs = min(len(self.data), config.train_size) // config.batch_size
            # for detecting valuable epoch that we must stop training step
            # sample_input_for_test_each_train_step.npy
            # sample_test = np.load('SIFTETS.npy').reshape([504, 32, 32, 1])[0:128]

            for idx in xrange(0, batch_idxs):
                batch = self.data[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_noise = sample_w_noise[idx * config.batch_size:(idx + 1) * config.batch_size]
                batch_images = np.array(batch).astype(np.float32)
                batch_noise_images = np.array(batch_noise).astype(np.float32)
                batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]).astype(np.float32)

                if self.pre:
                    _, _, summary_str = self.sess.run([d_optim, self.centers_update_op, self.pre_sum],
                        feed_dict={self.inputs: batch_images, self.z: batch_noise_images})

                    self.writer.add_summary(summary_str, counter)
                    counter += 1

                    centerloss = self.center_loss.eval({self.inputs: batch_images, self.z: batch_noise_images})
                    recon = self.g_r_loss.eval({self.inputs: batch_images, self.z: batch_noise_images})
                    msg = "Epoch:[%2d][%4d/%4d]--> centerloss: %.8f, recon-loss: %.8f" % (epoch, idx, batch_idxs, centerloss / self.batch_size, recon / self.batch_size)
                    print(msg)
                    logging.info(msg)
                    if np.mod(counter, self.n_per_itr_print_results) == 0:
                        self.saver.save(self.sess, "cifar-10/pre/pre_model.ckpt")

                else:
                    # Update D network
                    _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                   feed_dict={self.inputs: batch_images, self.z: batch_noise_images})
                    self.writer.add_summary(summary_str, counter)

                    # Update G network
                    _, _, summary_str = self.sess.run([g_optim, self.centers_update_op, self.g_sum],
                                                   feed_dict={self.inputs: batch_images, self.z: batch_noise_images})
                    self.writer.add_summary(summary_str, counter)

                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.inputs: batch_images, self.z: batch_noise_images})
                    self.writer.add_summary(summary_str, counter)

                    errD_fake = self.d_loss_fake.eval({self.z: batch_noise_images})
                    errD_real = self.d_loss_real.eval({self.inputs: batch_images})
                    errG = self.g_loss.eval({self.inputs: batch_images, self.z: batch_noise_images})

                    counter += 1
                    msg = "Epoch:[%2d][%4d/%4d]--> d_loss: %.8f, g_loss: %.8f" % (epoch, idx, batch_idxs, errD_fake + errD_real, errG)
                    print(msg)
                    logging.info(msg)
                    if np.mod(counter, self.n_per_itr_print_results) == 0:
                        self.save(config.checkpoint_dir, epoch)
                # if not self.pre:
                if np.mod(counter, self.n_per_itr_print_results) == 0:
                    samples, d_loss, g_loss = self.sess.run([self.G, self.d_loss, self.g_loss],
                        feed_dict={
                            self.z: sample_inputs,
                            self.inputs: sample_inputs
                        }
                    )
                    manifold_h = int(np.ceil(np.sqrt(samples.shape[0])))
                    manifold_w = int(np.floor(np.sqrt(samples.shape[0])))
                    # samples = samples * 128 + 128
                    save_images(samples, [manifold_h, manifold_w],
                                './{}/train_{:02d}_{:04d}.png'.format(config.sample_dir, epoch, idx))
                    print("[Sample] d_loss: %.8f, g_loss: %.8f" % (d_loss, g_loss))


    @property
    def model_dir(self):
        return "{}_{}_{}_{}".format(self.dataset_name, self.batch_size, self.output_height, self.output_width)

    def save(self, checkpoint_dir, step):
        model_name = "ALOCC_Model.ckpt"
        # checkpoint_dir = os.path.join(checkpoint_dir, self.model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name), global_step=step)