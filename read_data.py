from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import random

# mnist = input_data.read_data_sets("./dataset/mnist/")
# specific_idx = np.where(mnist.train.labels == 1)[0]
# data = mnist.train.images[specific_idx].reshape(-1, 28, 28, 1)
# print(np.max(data))
# c_dim = 1

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def read_data(labels):
    dict_train = unpickle('dataset/cifar-10/data_batch_1')
    train_data = dict_train[b'data']
    train_labels = dict_train[b'labels']
    dict_train = unpickle('dataset/cifar-10/data_batch_2')
    train_data = np.append(train_data, dict_train[b'data'], axis=0)
    train_labels.extend(dict_train[b'labels'])
    dict_train = unpickle('dataset/cifar-10/data_batch_3')
    train_data = np.append(train_data, dict_train[b'data'], axis=0)
    train_labels.extend(dict_train[b'labels'])
    dict_train = unpickle('dataset/cifar-10/data_batch_4')
    train_data = np.append(train_data, dict_train[b'data'], axis=0)
    train_labels.extend(dict_train[b'labels'])
    dict_train = unpickle('dataset/cifar-10/data_batch_5')
    train_data = np.append(train_data, dict_train[b'data'], axis=0)
    train_labels.extend(dict_train[b'labels'])
    # dict_test = unpickle('dataset/cifar-10/test_batch')
    # test_data = dict_test[b'data']
    # test_labels = dict_test[b'labels']
    train_labels = np.array(train_labels)
    specific_idx = np.where(train_labels == labels)
    train_data = train_data[specific_idx]
    train_data = np.reshape(train_data, [-1, 32, 32, 3], 'F')
    # train_data = np.dot(train_data[..., :3], [0.299, 0.587, 0.114])
    # train_data = np.reshape(train_data, [-1, 32, 32, 1])
    train_data = train_data / 255.0
    train_data = np.transpose(train_data, [0, 2, 1, 3])
    # train_data = (train_data - 128) /128.0
    return train_data


def test_data(labels):
    dict_test = unpickle('dataset/cifar-10/test_batch')
    test_data = dict_test[b'data']
    test_labels = dict_test[b'labels']
    train_labels = np.array(test_labels)
    specific_idx = np.where(train_labels == labels)[0]
    specific_idx_anomaly = np.where(train_labels != labels)[0]
    ten_precent_anomaly = [specific_idx_anomaly[x] for x in
                                   random.sample(range(0, len(specific_idx_anomaly)), len(specific_idx_anomaly) // 10)]
    ten_precent_anomaly = np.array(ten_precent_anomaly)

    test_data1 = test_data[specific_idx]
    tmp_data = test_data[ten_precent_anomaly]
    test_data = np.append(test_data1, tmp_data, axis=0)
    test_data = np.reshape(test_data, [-1, 32, 32, 3], 'F')

    test_data = np.transpose(test_data, [0, 2, 1, 3])
    # plt.imshow(test_data[0])
    # plt.show()
    test_data = test_data / 255.0

    return test_data

# cifar = read_data(1)
# a = 1