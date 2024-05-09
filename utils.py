import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
import torch.nn.functional as F
import torch
from torch import nn
from torch.utils.data.sampler import BatchSampler
import csv

loss_fn = nn.BCELoss()
loss_fn_2 = nn.BCELoss(reduction='none')
loss_fn_3 = nn.CrossEntropyLoss()


def get_random_batch(dataset, batch_size=32, num_classes=10, random_seed=0):
    """
    Extract a random batch from the dataset given in input.

    :param dataset: Input dataset
    :param batch_size: Batch size

    :return: Random batch
    """
    train_dataloader_random = DataLoader(dataset, batch_size=batch_size, shuffle=True, worker_init_fn = lambda id: np.random.seed(random_seed))
    my_testiter = iter(train_dataloader_random)
    random_batch, target = next(my_testiter)
    return random_batch


def obtain_posterior_from_net_out(D, cost_function_v):
    if cost_function_v == 2 or cost_function_v == 5:
        R = (1-D)/D
    elif cost_function_v == 3:
        R = torch.exp(D)  # because linear layer is used, which can be negative. For expressing prob. it needs to be positive
    elif cost_function_v == 7 or cost_function_v == 12:
        R = D
    elif cost_function_v == 9:
        R = torch.pow(D, -1)
    elif cost_function_v == 10:
        R = torch.pow(D, -2)
    return R


def save_dict_lists_csv(path, dictionary):
    with open(path, "w") as outfile:
        writer = csv.writer(outfile)
        writer.writerow(dictionary.keys())
        writer.writerows(zip(*dictionary.values()))


def to_categorical(y, num_classes, t_tensor=False, dtype="uint8"):
    """ 1-hot encodes a tensor """
    if t_tensor:
        return F.one_hot(y, num_classes=num_classes)
    else:
        return np.eye(num_classes, dtype=dtype)[y.astype(int).squeeze()]


def gan_cost_fcn(out_1, out_2, digits, num_classes, device="cpu", t_tensor=True):
    batch_size = out_1.shape[0]
    valid = np.ones((batch_size, num_classes))
    non_valid = np.zeros((batch_size, num_classes))
    loss_1 = loss_fn_2(out_1.squeeze(), torch.Tensor(non_valid).to(device))
    loss_1 = torch.matmul(loss_1, torch.transpose(digits.float(), 0, 1).to(device))
    loss_1 = torch.diagonal(loss_1, 0)
    loss_1 = torch.mean(loss_1)
    loss_2 = loss_fn(out_2.squeeze(), torch.Tensor(valid).to(device))
    loss = loss_1 + loss_2
    return loss


def sl_cost_fcn(out_1, out_2, data_tx, num_classes, alpha):
    loss_1 = sl_first(out_1.squeeze(), data_tx, num_classes)
    loss_2 = sl_sec(out_2.squeeze())
    loss = loss_1 + alpha * loss_2
    return loss


def sl_first(y_pred, data_tx, num_classes, t_tensor=True):
    loss_1 = torch.matmul(y_pred, torch.transpose(data_tx.float(), 0, 1))
    loss_1 = torch.diagonal(loss_1, 0)
    loss_1 = torch.mean(loss_1)
    return loss_1


def sl_sec(y_pred):
    log_pred = torch.log(y_pred) - y_pred
    sum_log_pred = torch.mean(log_pred, dim=1)
    loss = torch.mean(sum_log_pred)
    return -loss


def cross_entropy_sup(out_1, out_2, digits_tx, num_classes, alpha):
    log_pred = torch.log(out_1)
    loss_1 = torch.matmul(log_pred, torch.transpose(digits_tx.float(), 0, 1))  # loss_1 * data_tx
    loss_1 = torch.diagonal(loss_1, 0)
    loss_1 = torch.mean(loss_1)
    loss_2 = torch.mean(torch.sum(out_2, dim=1))
    return -loss_1 + loss_2


def reverse_kl(out_1, out_2, digits_tx, num_classes, alpha):
    first_term = out_1
    loss_1 = torch.matmul(first_term, torch.transpose(digits_tx.float(), 0, 1))
    loss_1 = torch.diagonal(loss_1, 0)
    loss_1 = torch.mean(loss_1)
    second_term = torch.log(out_2 + 1e-5)
    loss_2 = torch.mean(second_term)
    return loss_1 - loss_2


def hellinger_distance(out_1, out_2, digits_tx, num_classes, alpha):
    loss_1 = torch.matmul(out_1, torch.transpose(digits_tx.float(), 0, 1))
    loss_1 = torch.diagonal(loss_1, 0)
    loss_1 = torch.mean(loss_1)
    loss_2 = torch.mean(torch.pow(out_2, -1))
    return loss_1 + loss_2


def pearson_chi2(out_1, out_2, digits_tx, num_classes, alpha):
    loss_1 = torch.matmul(out_1, torch.transpose(digits_tx.float(), 0, 1))
    loss_1 = torch.diagonal(loss_1, 0)
    loss_1 = torch.mean(loss_1)
    loss_2 = torch.pow(out_2, 2)
    loss_2 = torch.mean(loss_2)
    return -2*loss_1 + loss_2


########## COMMUNICATIONS #############
def from_zero_mean_bits_to_digit(x, M):
    #convert to binary representation
    x = (x + (M-1)) / 2 # (x + 1)/2# # bits in (-1,1) gets transformed in (0,1)
    N = np.size(x,0)
    d = np.size(x,1)
    digits = np.zeros((N,1))
    for i in range(d):
        digits[:,0] = digits[:,0] + (2**i)*x[:,d-i-1]
    return digits


def from_digit_to_zero_mean_bits(x,k):
    # convert from digit to zero mean
    # x is a list of numbers that must be converted, k is the length of the binary string
    M = len(x)
    output = np.zeros((M,k))
    for i in x:
        output[i,:] = np.transpose(np.fromstring(np.binary_repr(i, width=k), np.int8) - 48)
    output = 2*output-1  # from bits (0,1) to (-1,1)
    return output


def get_max_idx_loglikelihood(y,x):
    # maxL for AWGN channel
    N = np.size(x,0)
    distances = np.zeros((N,1))
    for i in range(N):
        distances[i] = np.linalg.norm(y[0,:]-x[i,:])
    return np.argmin(distances)


def get_max_idx_loglikelihood_mid(y,x, var_y):
    N = np.size(x,0)
    distances = np.zeros((N,1))
    for i in range(N):
        distances[i] = np.linalg.norm(np.divide(y[0,:]-x[i,:],var_y))
    return np.argmin(distances)


def get_max_idx_logmap(y,x, sigma_N,p_x):
    N = np.size(x,0)
    distances = np.zeros((N,1))
    for i in range(N):
        distances[i] = 0.5*np.square(np.linalg.norm(y[0,:]-x[i,:]))/(sigma_N**2)-np.log(p_x[i])
    return np.argmin(distances)