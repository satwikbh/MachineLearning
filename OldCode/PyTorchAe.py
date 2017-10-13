import math

import hickle as hkl
import matplotlib
import numpy as np
import os
import torch
import torch.nn.functional as nn
import torch.optim as optim

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.autograd import Variable

from sklearn.model_selection import train_test_split as tts
from scipy.sparse import vstack


class PrepareData:
    def __init__(self):
        pass

    @staticmethod
    def nearest_power_of_two(shape):
        value = 1 << (shape - 1).bit_length()
        return int(math.log(value, 2))

    @staticmethod
    def get_files_ends_with_extension(extension, path):
        all_files = list()
        for each_file in os.listdir(path):
            if each_file.endswith(extension):
                all_files.append(os.path.join(path, each_file))
        return all_files

    @staticmethod
    def open_files(list_of_files):
        matrix = list()
        for each in list_of_files:
            fv = hkl.load(each)
            matrix.append(fv)
        return matrix

    def prepare_data(self):
        list_of_files = self.get_files_ends_with_extension(".hkl", "/tmp/Data/pruned_fv_path_1/")

        matrix = self.open_files(list_of_files[:2])
        fv0 = vstack(matrix)

        nearest_repr = self.nearest_power_of_two(fv0.shape[1])
        print("Input matrix dimension : {}\tNearest power of 2 : {}".format(fv0.shape, nearest_repr))

        train_matrix, test_matrix = tts(fv0, test_size=0.25, random_state=1)
        print("Train Matrix shape : {}".format(train_matrix.shape))
        print("Test Matrix shape : {}".format(test_matrix.shape))

        train_matrix = train_matrix.astype('float32')
        test_matrix = test_matrix.astype('float32')
        return train_matrix, test_matrix


class QX:
    def __init__(self, X_dim, h_dim, Z_dim, mb_size):
        self.Wxh, self.bxh, self.Whz_mu, \
        self.bhz_mu, self.Whz_var, self.bhz_var, \
        self.mb_size, self.Z_dim = self.set_params(X_dim, h_dim, Z_dim, mb_size)

    @staticmethod
    def set_params(X_dim, h_dim, Z_dim, mb_size):
        Wxh = PyTorchAe.xavier_init(size=[X_dim, h_dim]).cuda()
        bxh = Variable(torch.zeros(h_dim), requires_grad=True).cuda()

        Whz_mu = PyTorchAe.xavier_init(size=[h_dim, Z_dim]).cuda()
        bhz_mu = Variable(torch.zeros(Z_dim), requires_grad=True).cuda()

        Whz_var = PyTorchAe.xavier_init(size=[h_dim, Z_dim]).cuda()
        bhz_var = Variable(torch.zeros(Z_dim), requires_grad=True).cuda()

        return Wxh, bxh, Whz_mu, bhz_mu, Whz_var, bhz_var, mb_size, Z_dim

    def Q(self, X):
        h = nn.relu(X.mm(self.Wxh) + self.bxh.repeat(X.size(0), 1))
        z_mu = h.mm(self.Whz_mu) + self.bhz_mu.repeat(h.size(0), 1)
        z_var = h.mm(self.Whz_var) + self.bhz_var.repeat(h.size(0), 1)
        return z_mu, z_var

    def sample_z(self, mu, log_var):
        eps = Variable(torch.randn(self.mb_size, self.Z_dim)).cuda()
        return mu + torch.exp(log_var / 2) * eps


class PX:
    def __init__(self, X_dim, h_dim, Z_dim):
        self.Wzh, self.bzh, self.Whx, self.bhx = self.set_params(X_dim, h_dim, Z_dim)

    @staticmethod
    def set_params(X_dim, h_dim, Z_dim):
        Wzh = PyTorchAe.xavier_init(size=[Z_dim, h_dim]).cuda()
        bzh = Variable(torch.zeros(h_dim), requires_grad=True).cuda()

        Whx = PyTorchAe.xavier_init(size=[h_dim, X_dim]).cuda()
        bhx = Variable(torch.zeros(X_dim), requires_grad=True).cuda()
        return Wzh, bzh, Whx, bhx

    def P(self, z):
        h = nn.relu(z.mm(self.Wzh) + self.bzh.repeat(z.size(0), 1))
        X = nn.sigmoid(h.mm(self.Whx) + self.bhx.repeat(h.size(0), 1))
        return X


class PyTorchAe:
    def __init__(self):
        self.data = PrepareData()

        self.X_dim, self.h_dim, self.Z_dim = 0, 0, 0

        self.mb_size = 64
        self.c = 0
        self.lr = 1e-3

    @staticmethod
    def xavier_init(size):
        in_dim = size[0]
        xavier_stddev = 1. / np.sqrt(in_dim / 2.)
        return Variable((torch.randn(*size) * xavier_stddev).cuda(), requires_grad=True)

    def train(self, train_matrix, test_matrix):
        self.qx = QX(X_dim=self.X_dim, h_dim=self.h_dim, Z_dim=self.Z_dim, mb_size=self.mb_size)
        self.px = PX(X_dim=self.X_dim, h_dim=self.h_dim, Z_dim=self.Z_dim)
        params = [self.qx.Wxh, self.qx.bxh, self.qx.Whz_mu, self.qx.bhz_mu, self.qx.Whz_var, self.qx.bhz_var,
                  self.px.Wzh, self.px.bzh, self.px.Whx, self.px.bhx]
        solver = optim.Adam(params, lr=self.lr)

        counter = 0
        it = 0

        while counter + self.mb_size < train_matrix.shape[0]:
            X = train_matrix[counter:counter + self.mb_size].toarray()
            X = Variable(torch.from_numpy(X).cuda(device_id=1))

            z_mu, z_var = self.qx.Q(X)
            z = self.qx.sample_z(z_mu, z_var)
            X_sample = self.px.P(z)

            # Loss
            recon_loss = nn.binary_cross_entropy(X_sample, X, size_average=False) / self.mb_size
            kl_loss = torch.mean(0.5 * torch.sum(torch.exp(z_var) + z_mu ** 2 - 1. - z_var, 1))
            loss = recon_loss + kl_loss

            # Backward
            loss.backward()

            # Update
            solver.step()

            # Housekeeping
            for p in params:
                p.grad.data.zero_()

            # Print and plot every now and then
            if it % 1 == 0:
                print('Iter-{}; Loss: {:.4}'.format(it, loss.data[0]))
                samples = self.px.P(z).data.numpy()[:16]

                fig = plt.figure(figsize=(4, 4))
                gs = gridspec.GridSpec(4, 4)
                gs.update(wspace=0.05, hspace=0.05)

                for i, sample in enumerate(samples):
                    ax = plt.subplot(gs[i])
                    plt.axis('off')
                    ax.set_xticklabels([])
                    ax.set_yticklabels([])
                    ax.set_aspect('equal')
                    plt.imshow(sample.reshape(24995, 1), cmap='Greys_r')

                if not os.path.exists('out/'):
                    os.makedirs('out/')

                plt.savefig('out/{}.png'.format(str(self.c).zfill(3)), bbox_inches='tight')
                self.c += 1
                plt.close(fig)

            counter += self.mb_size
            it += 1

    def main(self):
        train_matrix, test_matrix = self.data.prepare_data()
        self.X_dim = train_matrix.shape[1]
        self.h_dim = train_matrix.shape[1] / 2
        self.Z_dim = train_matrix.shape[1] / 4

        self.train(train_matrix, test_matrix)


if __name__ == '__main__':
    ae = PyTorchAe()
    ae.main()
