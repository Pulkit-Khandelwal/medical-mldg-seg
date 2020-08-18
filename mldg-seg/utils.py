import numpy as np
import torch.optim as optim
from sklearn.metrics import accuracy_score
import numpy as np
import numpy.random as rnd
import torch
import torch.nn as nn
import torch.utils.data as data_utils
import torch.optim as optim
from torch.autograd import Variable
from dice_loss import *

def unfold_label(labels, classes):
    new_labels = []

    assert len(np.unique(labels)) == classes
    mini = np.min(labels)

    for index in range(len(labels)):
        dump = np.full(shape=[classes], fill_value=0).astype(np.int8)
        _class = int(labels[index]) - mini
        dump[_class] = 1
        new_labels.append(dump)

    return np.array(new_labels)


def shuffle_data(samples, labels):
    num = len(labels)
    shuffle_index = np.random.permutation(np.arange(num))
    shuffled_samples = samples[shuffle_index]
    shuffled_labels = labels[shuffle_index]
    return shuffled_samples, shuffled_labels


def shuffle_list(li):
    np.random.shuffle(li)
    return li


def shuffle_list_with_ind(li):
    shuffle_index = np.random.permutation(np.arange(len(li)))
    li = li[shuffle_index]
    return li, shuffle_index


def num_flat_features(x):
    size = x.size()[1:]  # all dimensions except the batch dimension
    num_features = 1
    for s in size:
        num_features *= s
    return num_features


def crossentropyloss():
    loss_fn = torch.nn.CrossEntropyLoss()
    return loss_fn

def binarycrossentropy():
    loss_fn = torch.nn.BCELoss()
    return loss_fn


def diceloss():
    loss_fn = dice_coeff
    return loss_fn

def mseloss():
    loss_fn = torch.nn.MSELoss()
    return loss_fn


def sgd(model, parameters, lr, weight_decay=0.00005, momentum=0.9):
    opt = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum,
                    weight_decay=weight_decay)
    return opt

def rmsprop(model, parameters, lr, weight_decay=0.00005, momentum=0.99):
    opt = optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, momentum=momentum,
                    weight_decay=weight_decay)
    return opt

def fix_seed():
    # torch.manual_seed(1108)
    # np.random.seed(1108)
    pass


def write_log(log, log_path):
    f = open(log_path, mode='a')
    f.write(str(log))
    f.write('\n')
    f.close()


def compute_accuracy(predictions, labels):

    accuracy = dice_coeff(predictions, labels)
    return accuracy
