import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from utils import pp, visualize, to_json, show_all_variables
from models import ALOCC_Model
import matplotlib.pyplot as plt
from kh_tools import *
import numpy as np
import scipy.misc
from utils import *
import time
import os
import read_data
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import logging

flags = tf.app.flags
flags.DEFINE_integer("epoch", 1, "Epoch to train [25]")
flags.DEFINE_float("learning_rate", 0, "Learning rate of for adam [0.0002]")
flags.DEFINE_float("beta1", 0.5, "Momentum term of adam [0.5]")
flags.DEFINE_integer("attention_label", 0, "Conditioned label that growth attention of training label [1]")
flags.DEFINE_float("r_alpha", 0.2, "Refinement parameter [0.2]")
flags.DEFINE_float("r_beta", 0.2, "VAE parameter [0.2]")
flags.DEFINE_integer("train_size", 5000, "The size of train images [np.inf]")
flags.DEFINE_integer("batch_size", 128, "The size of batch images [64]")
flags.DEFINE_integer("input_height", 32, "The size of image to use. [45]")
flags.DEFINE_integer("input_width", None, "The size of image to use. If None, same value as input_height [None]")
flags.DEFINE_integer("output_height", 32, "The size of the output images to produce [45]")
flags.DEFINE_integer("output_width", None, "The size of the output images to produce. If None, same value as output_height [None]")
flags.DEFINE_string("dataset", "UCSD", "The name of dataset [UCSD, mnist]")
flags.DEFINE_string("dataset_address", "./dataset/UCSD_Anomaly_Dataset.v1p2/UCSDped2/Test", "The path of dataset")
flags.DEFINE_string("input_fname_pattern", "*", "Glob pattern of filename of input images [*]")
flags.DEFINE_string("checkpoint_dir", "./checkpoint/cifar-10_128_32_32_vae0/", "Directory name to save the checkpoints [checkpoint]")
flags.DEFINE_string("log_dir", "log", "Directory name to save the log [log]")
flags.DEFINE_string("sample_dir", "samples", "Directory name to save the image samples [samples]")
flags.DEFINE_boolean("train", False, "True for training, False for testing [False]")
flags.DEFINE_boolean("pretrain", True, "True for pretrain, False for training")
flags.DEFINE_string("pre_dir", "pretrain", "Directory name to save the pretrain model [pretrain]")

FLAGS = flags.FLAGS

def check_some_assertions():
    """
    to check some assertions in inputs and also check sth else.
    """
    if FLAGS.input_width is None:
        FLAGS.input_width = FLAGS.input_height
    if FLAGS.output_width is None:
        FLAGS.output_width = FLAGS.output_height

    if not os.path.exists(FLAGS.sample_dir):
        os.makedirs(FLAGS.sample_dir)

def main(_):
    print('Program is started at', time.clock())
    pp.pprint(flags.FLAGS.__flags)

    n_per_itr_print_results = 100
    n_fetch_data = 10
    kb_work_on_patch= False
    nd_input_frame_size = (240, 360)
    #nd_patch_size = (45, 45)
    n_stride = 10
    #FLAGS.checkpoint_dir = "./checkpoint/UCSD_128_45_45/"


    check_some_assertions()

    nd_patch_size = (FLAGS.input_width, FLAGS.input_height)
    # FLAGS.nStride = n_stride

    #FLAGS.input_fname_pattern = '*'
    FLAGS.train = False
    FLAGS.epoch = 1
    FLAGS.batch_size = 100

    message = []
    total = []
    acc = 0
    pres = 0
    recall = 0
    f1 = 0

    for i in range(10):
        FLAGS.checkpoint_dir = "result/checkpoint_step2/cifar-10_128_32_32_vae0/"
        FLAGS.attention_label = i
        FLAGS.sample_dir = 'samples/'
        FLAGS.checkpoint_dir = FLAGS.checkpoint_dir.replace("vae0", "vae{}".format(FLAGS.attention_label))


        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.1)
        run_config = tf.ConfigProto(gpu_options=gpu_options)
        run_config.gpu_options.allow_growth=True
        tf.reset_default_graph()

        with tf.Session(config=run_config) as sess:
            tmp_ALOCC_model = ALOCC_Model(
                        sess,
                        input_width=FLAGS.input_width,
                        input_height=FLAGS.input_height,
                        output_width=FLAGS.output_width,
                        output_height=FLAGS.output_height,
                        batch_size=FLAGS.batch_size,
                        sample_num=FLAGS.batch_size,
                        attention_label=FLAGS.attention_label,
                        r_alpha=FLAGS.r_alpha,
                        r_beta=FLAGS.r_beta,
                        is_training=FLAGS.train,
                        pre=FLAGS.pretrain,
                        pre_dir=FLAGS.pre_dir,
                        dataset_name=FLAGS.dataset,
                        dataset_address=FLAGS.dataset_address,
                        input_fname_pattern=FLAGS.input_fname_pattern,
                        checkpoint_dir=FLAGS.checkpoint_dir,
                        sample_dir=FLAGS.sample_dir,
                        nd_patch_size=nd_patch_size,
                        n_stride=n_stride,
                        n_per_itr_print_results=n_per_itr_print_results,
                        kb_work_on_patch=kb_work_on_patch,
                        nd_input_frame_size = nd_input_frame_size,
                        n_fetch_data=n_fetch_data)

            show_all_variables()


            print('--------------------------------------------------')
            print('Load Pretrained Model...')
            tmp_ALOCC_model.f_check_checkpoint()

            if FLAGS.dataset=='mnist':
                mnist = input_data.read_data_sets(FLAGS.dataset_address)

                specific_idx_anomaly = np.where(mnist.train.labels != 6)[0]
                specific_idx = np.where(mnist.train.labels == 6)[0]
                ten_precent_anomaly = [specific_idx_anomaly[x] for x in
                                       random.sample(range(0, len(specific_idx_anomaly)), len(specific_idx) // 40)]

                data = mnist.train.images[specific_idx].reshape(-1, 28, 28, 1)
                tmp_data = mnist.train.images[ten_precent_anomaly].reshape(-1, 28, 28, 1)
                data = np.append(data, tmp_data).reshape(-1, 28, 28, 1)

                lst_prob = tmp_ALOCC_model.f_test_frozen_model(data[0:FLAGS.batch_size])
                print('check is ok')
                exit()
                #generated_data = tmp_ALOCC_model.feed2generator(data[0:FLAGS.batch_size])
            else:
                data, labels = read_data.test_data(FLAGS.attention_label)
                # np.random.shuffle(data)
                lst_prob = tmp_ALOCC_model.f_test_frozen_model(data)
                # maxi = max(lst_prob)
                # mini = min(lst_prob)
                # average = (maxi+mini) / 2.0
                # print(average)
                best_th = np.mean(lst_prob)
                for x in range(len(lst_prob)):
                    if lst_prob[x] >= best_th:
                        lst_prob[x] = 1
                    else:
                        lst_prob[x] = 0
                C = confusion_matrix(labels, lst_prob)
                print(C)
                msg = "class_id: {}, ".format(FLAGS.attention_label) + "threshold: {:.3f}\n".format(best_th) + \
                    'accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1 score: {:.3f}\n'.format(
                        # average,
                        accuracy_score(labels, lst_prob),
                        precision_score(labels, lst_prob, average='binary'),
                        recall_score(labels, lst_prob, average='binary'),
                        f1_score(labels, lst_prob, average='binary')) + str(C)
                acc += accuracy_score(labels, lst_prob) / 10.
                pres += precision_score(labels, lst_prob, average='binary') / 10.
                recall += recall_score(labels, lst_prob, average='binary') / 10.
                f1 += f1_score(labels, lst_prob, average='binary') / 10.

                print(msg)
                message.append(msg)
                print("\n")
                # logging.info(msg)
                # print('check is ok')
                # exit()

    with open("print.txt", "w+") as f:
        for msg in message:
            f.write(msg)
            f.write("\n")

        result = 'accuracy: {:.3f}, precision: {:.3f}, recall: {:.3f}, f1 score: {:.3f}'.format(acc, pres, recall, f1)
        f.write(result)
        f.close()



# ---------------------------------------------------------------------------------------
# ---------------------------------------------------------------------------------------
if __name__ == '__main__':
    tf.app.run()


