from torch import nn
import numpy as np
import argparse
import matplotlib.pyplot as plt
import pandas as pd # this module is useful to work with tabular data
import random # this module will be used to select random samples from a collection
import os # this module will be used just to create directories in the local filesystem
import torch
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms, models
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights, resnet152, ResNet152_Weights
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler, random_split
from torch import nn
from utils import *
from NetArchitectures import *


def compute_loss_divergence(cost_function_v, out_1, out_2, data_tx, num_classes, current_batch_size, alpha, device):
    loss_fn = nn.BCELoss()
    loss_fn_2 = nn.BCELoss(reduction='none')
    loss_fn_3 = nn.CrossEntropyLoss()

    data_tx_categorical = torch.Tensor(to_categorical(data_tx, t_tensor=True, num_classes=num_classes))

    if cost_function_v == 2:  # GAN
        loss = gan_cost_fcn(out_1, out_2, data_tx_categorical, num_classes, device=device)
    elif cost_function_v == 3:  # cross-entropy / KL
        loss = loss_fn_3(out_1.squeeze(), data_tx.squeeze().long())
    elif cost_function_v == 5:  # SL
        loss = sl_cost_fcn(out_1, out_2, data_tx_categorical, num_classes, alpha)
    elif cost_function_v == 7:  # KL with softplus
        loss = cross_entropy_sup(out_1, out_2, data_tx_categorical, num_classes, alpha)
    elif cost_function_v == 9: # RKL
        loss = reverse_kl(out_1, out_2, data_tx_categorical, num_classes, alpha)
    elif cost_function_v == 10: # HD
        loss = hellinger_distance(out_1, out_2, data_tx_categorical, num_classes, alpha)
    elif cost_function_v == 12: # P
        loss = pearson_chi2(out_1, out_2, data_tx_categorical, num_classes, alpha)

    return loss


class CombinedArchitectureSingle(nn.Module):
    """
    Class combining two equal neural network architectures.
    """
    def __init__(self, single_architecture, cost_function_v=1):
        super(CombinedArchitectureSingle, self).__init__()
        self.div_to_act_func = {
            2: nn.Sigmoid(),
            3: nn.Identity(),
            5: nn.Sigmoid(),
            7: nn.Softplus(),
            9: nn.Softplus(),
            10: nn.Sigmoid(),
            12: nn.Sigmoid()
        }
        self.cost_function_version = cost_function_v
        self.single_architecture = single_architecture
        self.final_activation = self.div_to_act_func[cost_function_v]

    def forward(self, input_tensor_1):
        intermediate_1 = self.single_architecture(input_tensor_1)
        output_tensor_1 = self.final_activation(intermediate_1)
        return output_tensor_1


############################ COMMUNICATIONS PART #########################


class Discriminator(nn.Module):
    def __init__(self, input_dim, output_dim, vectorize_input=False):
        """
        Initialize the discriminator.

        :param input_dim: Input dimension
        :param output_dim: Output dimension
        """
        super(Discriminator, self).__init__()

        self.vectorize_input = vectorize_input

        self.main = nn.Sequential(
            nn.Linear(input_dim, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Linear(100, output_dim),
        )

    def forward(self, input_tensor):
        if self.vectorize_input:
            input_tensor = input_tensor.reshape(-1, input_tensor.shape[2]**2)
        output_tensor = self.main(input_tensor)
        return output_tensor


class CombinedArchitecture(nn.Module):
    """
    Class combining two equal neural network architectures.
    """
    def __init__(self, single_architecture, cost_function_v=1):
        super(CombinedArchitecture, self).__init__()
        self.div_to_act_func = {
            2: nn.Sigmoid(),
            3: nn.Identity(),
            5: nn.Sigmoid(),
            7: nn.Softplus(),
            9: nn.Softplus(),
            10: nn.Softplus(),
            12: nn.Softplus()
        }
        self.cost_function_version = cost_function_v
        self.single_architecture = single_architecture
        self.final_activation = self.div_to_act_func[cost_function_v]

    def forward(self, input_tensor_1, input_tensor_2):
        intermediate_1 = self.single_architecture(input_tensor_1)
        output_tensor_1 = self.final_activation(intermediate_1)
        intermediate_2 = self.single_architecture(input_tensor_2)
        output_tensor_2 = self.final_activation(intermediate_2)

        return output_tensor_1, output_tensor_2


def train_communication_awgn(model, latent_dim, eps=0.01, lr=0.001, num_epochs=1000, batch_size=40,
                            noise_model="AWGN", cost_function_v=5, alpha=1, random_seed=0):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    M = 2**latent_dim
    h_x_y = np.zeros((batch_size, 1))
    for epoch in range(num_epochs):
        data_tx = 2 * np.random.randint(2, size=(batch_size, latent_dim)) - 1
        noise_rx = eps * np.random.normal(0, 1, (batch_size, latent_dim))
        noise_y = eps * np.random.normal(0, 1, (batch_size, latent_dim))
        if noise_model == 'Middleton':
            K = 5
            noise_rx = noise_rx + np.random.binomial(size=(batch_size, latent_dim), n=1, p=0.05) * (
                np.sqrt(K ** 2 - 1)) * eps * np.random.normal(0, 1, (batch_size, latent_dim))
            noise_y = noise_y + np.random.binomial(size=(batch_size, latent_dim), n=1, p=0.05) * (
                np.sqrt(K ** 2 - 1)) * eps * np.random.normal(0, 1, (batch_size, latent_dim))
        data_rx = data_tx + noise_rx
        data_rx = torch.Tensor(data_rx)
        # Sample from the marginal of the received samples
        data_y = 2 * np.random.randint(2, size=(batch_size, latent_dim)) - 1 + noise_y
        data_y = torch.Tensor(data_y)

        optimizer.zero_grad()
        out_1, out_2 = model(data_rx, data_y)
        digits = torch.Tensor(to_categorical(from_zero_mean_bits_to_digit(data_tx, 2), num_classes=M))

        if cost_function_v==5:
            loss = sl_cost_fcn(out_1, out_2, digits, M, alpha)
        elif cost_function_v==2:
            loss = gan_cost_fcn(out_1, out_2, digits, M, t_tensor=False)
        elif cost_function_v == 7:
            loss = cross_entropy_sup(out_1, out_2, digits, M, alpha)
        elif cost_function_v == 9:
            loss = reverse_kl(out_1, out_2, digits, M, alpha)
        elif cost_function_v == 10:
            loss = hellinger_distance(out_1, out_2, digits, M, alpha)
        elif cost_function_v == 12:
            loss = pearson_chi2(out_1, out_2, digits, M, alpha)

        loss.backward()
        optimizer.step()

        if cost_function_v == 5 or cost_function_v==2:
            D_value_1, _ = model(data_rx, data_rx)
            R = (1 - D_value_1) / D_value_1
        elif cost_function_v == 7 or cost_function_v ==12:
            R, _ = model(data_rx, data_rx)
        elif cost_function_v==9:
            R_inv, _ = model(data_rx, data_rx)
            R = torch.pow(R_inv, -1)
        elif cost_function_v==10:
            R_inv_sqrt, _ = model(data_rx, data_rx)
            R = torch.pow(R_inv_sqrt, -2)

        # To avoid numerical errors, normalize the outputs
        R = R.detach().numpy()
        L1_norm = np.expand_dims(np.sum(R, axis=1), axis=-1) * np.ones((1, np.size(R, axis=1)))
        R = R / L1_norm
        P_x = np.mean(R, axis=0)
        H_x = -P_x.dot(np.log2(P_x))
        for i in range(batch_size):
            R[R == 0] = 1  # to avoid NaN
            h_x_y[i] = -R[i, :].dot(np.log2(R[i, :].T))
        H_x_y = np.mean(h_x_y, axis=0)  # batch conditional entropy estimate
        MI = H_x - H_x_y  # batch mutual information estimate
        P_error = 1 - np.mean(np.max(R, axis=1), axis=0)  # prob. of error

        # Plot the progress
        print(
            "%d [D total loss : %f, Batch source entropy : %f, B. cond. entropy: %f, B. MI: %f, B. prob. error: %f]" % (
            epoch, loss.item(), H_x, H_x_y, MI, P_error))
    return model


def test_communication_awgn(model, latent_dim, test_size=1000, noise_model="AWGN", eps=1, cost_function_v=5, random_seed=0):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    model.eval()
    BER = np.zeros((1, test_size))
    BER_maxL = np.zeros((1, test_size))
    h_x_y = np.zeros((test_size, 1))

    # Produce tx and rx samples
    data_tx = 2 * np.random.randint(2, size=(test_size, latent_dim)) - 1
    noise = eps * np.random.normal(0, 1, (test_size, latent_dim))

    # Add the Bernoulli component to get the truncated Middleton model
    if noise_model == 'Middleton':
        K = 5
        bernoulli_event = np.random.binomial(size=(test_size, latent_dim), n=1, p=0.05)
        bernoulli_noise = bernoulli_event * (np.sqrt(K ** 2 - 1)) * eps * np.random.normal(0, 1,(test_size, latent_dim))
        noise = noise + bernoulli_noise

    data_rx = data_tx + noise

    # Specify the alphabet for the MAP part
    alphabet = range(2 ** latent_dim)
    training_samples = from_digit_to_zero_mean_bits(alphabet, latent_dim)

    # Extract metrics for each transmitted sample
    for i in range(test_size):
        if cost_function_v == 5 or cost_function_v==2:
            D_value_1, _ = model(torch.Tensor(np.expand_dims(data_rx[i, :], axis=0)),
                                         torch.Tensor(np.expand_dims(data_rx[i, :], axis=0)))
            R = (1 - D_value_1) / D_value_1
        elif cost_function_v == 7 or cost_function_v == 12:
            R, _ = model(torch.Tensor(np.expand_dims(data_rx[i, :], axis=0)),
                         torch.Tensor(np.expand_dims(data_rx[i, :], axis=0)))
        elif cost_function_v == 9:
            R_inv, _ = model(torch.Tensor(np.expand_dims(data_rx[i, :], axis=0)),
                         torch.Tensor(np.expand_dims(data_rx[i, :], axis=0)))
            R = torch.pow(R_inv, -1)
        elif cost_function_v == 10:
            R_inv_sqrt, _ = model(torch.Tensor(np.expand_dims(data_rx[i, :], axis=0)),
                         torch.Tensor(np.expand_dims(data_rx[i, :], axis=0)))
            R = torch.pow(R_inv_sqrt, -2)

        R = R.detach().numpy()
        L1_single_norm = np.expand_dims(np.sum(R, axis=1), axis=-1) * np.ones((1, np.size(R, axis=1)))
        R = R / L1_single_norm
        h_x_y[i] = -R[0, :].dot(np.log2(R[0, :].T))  # instantaneous conditional entropy estimate

        # Genie decoder
        if noise_model == 'Middleton':
            variance_rx = eps ** 2 + bernoulli_event[i, :] * (K ** 2 - 1) * (
                        eps ** 2)  # needed for the genie Middleton decoder
            max_idx_genie = get_max_idx_loglikelihood_mid(np.expand_dims(data_rx[i, :], axis=0), training_samples,
                                                          variance_rx)  # map genie criterion for Middleton
        else:
            max_idx_genie = get_max_idx_loglikelihood(np.expand_dims(data_rx[i, :], axis=0),
                                                      training_samples)  # maxL criterion for AWGN

        max_idx = np.argmax(R)  # MAP criterion

        logical_bits = training_samples[max_idx, :] == data_tx[i, :]
        BER[0, i] = 1 - sum(logical_bits) / latent_dim

        logical_bits_genie = training_samples[max_idx_genie, :] == data_tx[i, :]  # comparison in the maxL/genie indices
        BER_maxL[0, i] = 1 - sum(logical_bits_genie) / latent_dim  # instantaneous bit-error-rate with maxL/genie

    data_rx = torch.Tensor(data_rx)

    if cost_function_v == 3:
        R_all, _ = model(data_rx, data_rx)
        R_all = torch.exp(R_all)
    elif cost_function_v == 7 or cost_function_v == 12:
        R_all, _ = model(data_rx, data_rx)
    elif cost_function_v==2 or cost_function_v==5:
        D_all, _ = model(data_rx, data_rx)
        R_all = (1 - D_all) / D_all
    elif cost_function_v == 9:
        R_inv_all, _ = model(data_rx, data_rx)
        R_all = torch.pow(R_inv_all, -1)
    elif cost_function_v == 10:
        R_inv_sqrt_all, _ = model(data_rx, data_rx)
        R_all = torch.pow(R_inv_sqrt_all, -2)

    # To avoid numerical errors, you may want to normalize the outputs
    R_all = R_all.detach().numpy()
    L1_norm = np.expand_dims(np.sum(R_all, axis=1), axis=-1) * np.ones((1, np.size(R_all, axis=1)))
    R_all = R_all / L1_norm
    # Estimate of source entropy, conditional entropy, mutual information, probability of error
    P_x = np.mean(R_all, axis=0)
    H_x = -P_x.dot(np.log2(P_x))  # source entropy estimate
    H_x_y = np.nanmean(h_x_y, axis=0)  # conditional entropy estimate
    MI = H_x - H_x_y  # mutual information estimate
    P_error = 1 - np.mean(np.max(R_all, axis=1), axis=0)  # prob. of error
    return np.sum(BER) / (test_size * latent_dim), \
           np.sum(BER_maxL) / (test_size * latent_dim), H_x, H_x_y, MI, P_error, data_rx


def train_communication_pam_attenuation(model, latent_dim, M, eps=0.01, lr=0.001, num_epochs=1000, batch_size=40,
                       save_training_loss=True, noise_model="AWGN", cost_function_v=5, alpha=1, random_seed=167268734):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    h_x_y = np.zeros((batch_size, 1))

    for epoch in range(num_epochs):
        # Sample noise and generate a batch
        data_tx = 2*np.random.randint(M, size=(batch_size, 1))-(M-1)
        data_rx = np.sign(data_tx)*np.sqrt(np.abs(data_tx)) + eps*np.random.normal(0, 1, (batch_size, 1))

        data_y_new = 2 * np.random.randint(M, size=(batch_size, 1)) - (M - 1)
        data_y = np.sign(data_y_new) * np.sqrt(np.abs(data_y_new)) + eps * np.random.normal(0, 1, (batch_size, 1))

        data_rx = torch.Tensor(data_rx)
        data_y = torch.Tensor(data_y)

        optimizer.zero_grad()
        out_1, out_2 = model(data_rx, data_y)
        digits = torch.Tensor(to_categorical(from_zero_mean_bits_to_digit(data_tx, M), num_classes=M))
        if cost_function_v==5:
            loss = sl_cost_fcn(out_1, out_2, digits, M, alpha)
        elif cost_function_v==2:
            loss = gan_cost_fcn(out_1, out_2, digits, M, t_tensor=False)
        elif cost_function_v == 7:
            loss = cross_entropy_sup(out_1, out_2, digits, M, alpha)
        elif cost_function_v == 9:
            loss = reverse_kl(out_1, out_2, digits, M, alpha)
        elif cost_function_v == 10:
            loss = hellinger_distance(out_1, out_2, digits, M, alpha)
        elif cost_function_v == 12:
            loss = pearson_chi2(out_1, out_2, digits, M, alpha)

        loss.backward()
        optimizer.step()

        if cost_function_v==5 or cost_function_v==2:
            D_value_1, _ = model(data_rx, data_rx)
            R = (1 - D_value_1) / D_value_1
        elif cost_function_v==7 or cost_function_v==12:
            R, _ = model(data_rx, data_rx)
        elif cost_function_v==9:
            R_inv, _ = model(data_rx, data_rx)
            R = torch.pow(R_inv, -1)
        elif cost_function_v==10:
            R_inv_sqrt, _ = model(data_rx, data_rx)
            R = torch.pow(R_inv_sqrt, -2)

        # To avoid numerical errors, you may want to normalize the outputs
        R = R.detach().numpy()
        L1_norm = np.expand_dims(np.sum(R, axis=1), axis=-1) * np.ones((1, np.size(R, axis=1)))
        R = R / L1_norm
        # Real-time estimate of source entropy, conditional entropy, mutual information, probability of error
        P_x = np.mean(R, axis=0)
        H_x = -P_x.dot(np.log2(P_x))  # batch source entropy estimate
        for i in range(batch_size):
            R[R == 0] = 1  # to avoid NaN
            h_x_y[i] = -R[i, :].dot(np.log2(R[i, :].T))  # instantaneous conditional entropy estimate

        H_x_y = np.mean(h_x_y, axis=0)  # batch conditional entropy estimate
        MI = H_x - H_x_y  # batch mutual information estimate
        P_error = 1 - np.mean(np.max(R, axis=1), axis=0)  # prob. of error

        # Plot the progress
        print(
            "%d [D total loss : %f, Batch source entropy : %f, B. cond. entropy: %f, B. MI: %f, B. prob. error: %f]" % (
                epoch, loss.item(), H_x, H_x_y, MI, P_error))
    return model


def test_communication_pam_attenuation(model, latent_dim, M, test_size=1000, noise_model="AWGN", eps=1, cost_function_v=5, random_seed=0):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    model.eval()
    h_x_y = np.zeros((test_size, 1))
    counter = np.zeros((1, test_size))
    counter_LL = np.zeros((1, test_size))
    counter_LL_AWGN = np.zeros((1, test_size))
    fDIME_e = np.zeros((1, test_size))
    R_total = np.zeros((test_size, M))
    data_Y = np.zeros((test_size, latent_dim))

    data_tx = 2*np.random.randint(M, size=(test_size, 1))-(M-1)

    training_samples = np.expand_dims(np.array(range(-M+1,M,2)),axis=-1)

    # Extract metrics for each transmitted sample
    for i in range(test_size):
        data_rx = np.sign(data_tx[i,:])*np.sqrt(np.abs(data_tx[i,:])) + eps * np.random.normal(0, 1, (1, 1))
        data_rx = torch.Tensor(data_rx)
        if cost_function_v==5 or cost_function_v==2:
            D_value_1, _ = model(data_rx, data_rx)
            R = (1 - D_value_1) / D_value_1  # a-posteriori estimates
        elif cost_function_v==7 or cost_function_v==12:
            R, _ = model(data_rx, data_rx)
        elif cost_function_v==9:
            R_inv, _ = model(data_rx, data_rx)
            R = torch.pow(R_inv, -1)
        elif cost_function_v==10:
            R_inv_sqrt, _ = model(data_rx, data_rx)
            R = torch.pow(R_inv_sqrt, -2)

        R = R.detach().numpy()
        L1_single_norm = np.expand_dims(np.sum(R, axis=1), axis=-1) * np.ones((1, np.size(R, axis=1)))
        R = R / L1_single_norm
        h_x_y[i] = -R[0, :].dot(np.log2(R[0, :].T))  # instantaneous conditional entropy estimate

        sum_entropy = 0
        for c in range(M):
            if not np.isnan(R[0, c] * np.log2(R[0, c])):
                sum_entropy = sum_entropy + R[0, c] * np.log2(R[0, c])
        fDIME_e[0, i] = sum_entropy

        R_total[i, :] = R[0, :]
        data_Y[i, :] = data_rx

        max_idx = np.argmax(R)
        max_idx_LL = get_max_idx_loglikelihood(np.sign(data_rx) * np.square(data_rx), training_samples)
        max_idx_LL_AWGN = get_max_idx_loglikelihood(data_rx, training_samples)

        # if (training_samples[max_idx] == data_tx[i, :]).all():
        logical_bits = training_samples[max_idx, :] == data_tx[i, :]
        counter[0, i] = 1 - sum(logical_bits) / 1  # PAM

        logical_bits_LL = training_samples[max_idx_LL, :] == data_tx[i, :]
        counter_LL[0, i] = 1 - sum(logical_bits_LL) / 1

        logical_bits_LL_AWGN = training_samples[max_idx_LL_AWGN, :] == data_tx[i, :]
        counter_LL_AWGN[0, i] = 1 - sum(logical_bits_LL_AWGN) / 1

    return counter, counter_LL, counter_LL_AWGN, fDIME_e, R_total


def train_communication_pam_triangular(model, latent_dim, eps=0.01, lr=0.001, num_epochs=1000, batch_size=40,
                       save_training_loss=True, noise_model="AWGN", cost_function_v=5, alpha=1, random_seed=167268734):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    M = 4
    h_x_y = np.zeros((batch_size, 1))

    delta = 0.1
    for epoch in range(num_epochs):
        # Spike source
        data_tx = 4 * np.random.randint(2, size=(batch_size, 1)) - 3 + 2 * np.random.binomial(size=(batch_size, 1), n=1,
                                                                                              p=0.05)
        data_rx = data_tx + eps * np.random.normal(0, 1, (batch_size, 1))

        data_y_new = 4 * np.random.randint(2, size=(batch_size, 1)) - 3 + 2 * np.random.binomial(size=(batch_size, 1),
                                                                                                 n=1, p=0.05)
        data_y = data_y_new + eps * np.random.normal(0, 1, (batch_size, 1))

        data_rx = torch.Tensor(data_rx)
        data_y = torch.Tensor(data_y)

        optimizer.zero_grad()
        out_1, out_2 = model(data_rx, data_y)
        digits = torch.Tensor(to_categorical(from_zero_mean_bits_to_digit(data_tx, M), num_classes=M))
        if cost_function_v==5:
            loss = sl_cost_fcn(out_1, out_2, digits, M, alpha)
        elif cost_function_v==2:
            loss = gan_cost_fcn(out_1, out_2, digits, M, t_tensor=False)
        elif cost_function_v == 7:
            loss = cross_entropy_sup(out_1, out_2, digits, M, alpha)
        elif cost_function_v == 9:
            loss = reverse_kl(out_1, out_2, digits, M, alpha)
        elif cost_function_v == 10:
            loss = hellinger_distance(out_1, out_2, digits, M, alpha)
        elif cost_function_v == 12:
            loss = pearson_chi2(out_1, out_2, digits, M, alpha)

        loss.backward()
        optimizer.step()

        if cost_function_v==5 or cost_function_v==2:
            D_value_1, _ = model(data_rx, data_rx)
            R = (1 - D_value_1) / D_value_1
        elif cost_function_v==7 or cost_function_v==12:
            R, _ = model(data_rx, data_rx)
        elif cost_function_v==9:
            R_inv, _ = model(data_rx, data_rx)
            R = torch.pow(R_inv, -1)
        elif cost_function_v==10:
            R_inv_sqrt, _ = model(data_rx, data_rx)
            R = torch.pow(R_inv_sqrt, -2)

        # To avoid numerical errors, you may want to normalize the outputs
        R = R.detach().numpy()
        L1_norm = np.expand_dims(np.sum(R, axis=1), axis=-1) * np.ones((1, np.size(R, axis=1)))
        R = R / L1_norm
        # Real-time estimate of source entropy, conditional entropy, mutual information, probability of error
        P_x = np.mean(R, axis=0)
        H_x = -P_x.dot(np.log2(P_x))  # batch source entropy estimate
        for i in range(batch_size):
            R[R == 0] = 1  # to avoid NaN
            h_x_y[i] = -R[i, :].dot(np.log2(R[i, :].T))  # instantaneous conditional entropy estimate

        H_x_y = np.mean(h_x_y, axis=0)  # batch conditional entropy estimate
        MI = H_x - H_x_y  # batch mutual information estimate
        P_error = 1 - np.mean(np.max(R, axis=1), axis=0)  # prob. of error

        # Plot the progress
        print(
            "%d [D total loss : %f, Batch source entropy : %f, B. cond. entropy: %f, B. MI: %f, B. prob. error: %f]" % (
                epoch, loss.item(), H_x, H_x_y, MI, P_error))

    return model


def test_communication_pam_triangular(model, latent_dim, test_size=1000, noise_model="AWGN", eps=1, cost_function_v=5, random_seed=0):
    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)
    model.eval()
    M = 4
    counter = np.zeros((1, test_size))
    counter_LL = np.zeros((1, test_size))
    counter_MAP = np.zeros((1, test_size))
    fDIME_e = np.zeros((1, test_size))
    R_total = np.zeros((test_size, M))

    data_tx = 4 * np.random.randint(2, size=(test_size, 1)) - 3 + 2 * np.random.binomial(size=(test_size, 1), n=1, p=0.05)

    p_x = 0.5*np.array([1-0.05,0.05,1-0.05,0.05])

    training_samples = np.expand_dims(np.array(range(-M + 1, M, 2)), axis=-1)

    # Extract metrics for each transmitted sample
    for i in range(test_size):

        data_rx = data_tx[i,:] + eps * np.random.normal(0, 1, (1, 1))
        data_rx = torch.Tensor(data_rx)
        if cost_function_v==5 or cost_function_v==2:
            D_value_1, _ = model(data_rx, data_rx)
            R = (1 - D_value_1) / D_value_1
        elif cost_function_v == 7 or cost_function_v==12:
            R, _ = model(data_rx, data_rx)
        elif cost_function_v==9:
            R_inv, _ = model(data_rx, data_rx)
            R = torch.pow(R_inv, -1)
        elif cost_function_v==10:
            R_inv_sqrt, _ = model(data_rx, data_rx)
            R = torch.pow(R_inv_sqrt, -2)

        R = R.detach().numpy()
        sum_entropy = 0
        for c in range(M):
            if not np.isnan(R[0, c] * np.log2(R[0, c])):
                sum_entropy = sum_entropy + R[0, c] * np.log2(R[0, c])
        fDIME_e[0, i] = sum_entropy

        R_total[i, :] = R[0, :]

        max_idx = np.argmax(R)
        max_idx_LL = get_max_idx_loglikelihood(data_rx, training_samples)
        max_idx_MAP = get_max_idx_logmap(data_rx, training_samples, eps, p_x)

        logical_bits = training_samples[max_idx, :] == data_tx[i, :]
        counter[0, i] = 1 - sum(logical_bits) / 1

        logical_bits_LL = training_samples[max_idx_LL, :] == data_tx[i, :]
        counter_LL[0, i] = 1 - sum(logical_bits_LL) / 1

        logical_bits_MAP = training_samples[max_idx_MAP, :] == data_tx[i, :]
        counter_MAP[0, i] = 1 - sum(logical_bits_MAP) / 1
    return counter, counter_LL, counter_MAP, fDIME_e, R_total