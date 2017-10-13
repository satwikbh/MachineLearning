import numpy as np
import sklearn.preprocessing as prep
import tensorflow as tf
from sklearn.model_selection import train_test_split as tts

from HelperFunctions.HelperFunction import HelperFunction
from Utils.ConfigUtil import ConfigUtil
from Utils.LoggerUtil import LoggerUtil


class Autoencoder(object):
    def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer()):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.transfer = transfer_function

        network_weights = self._initialize_weights()
        self.weights = network_weights

        # model
        self.x = tf.placeholder(tf.float32, [None, self.n_input])
        self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
        self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])

        # cost
        self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        self.optimizer = optimizer.minimize(self.cost)

        init = tf.global_variables_initializer()
        self.sess = tf.Session()
        self.sess.run(init)

    def _initialize_weights(self):
        all_weights = dict()
        all_weights['w1'] = tf.Variable(self.xavier_init(self.n_input, self.n_hidden))
        all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
        all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
        all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
        return all_weights

    def partial_fit(self, X):
        cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
        return cost

    def calc_total_cost(self, X):
        return self.sess.run(self.cost, feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.hidden, feed_dict={self.x: X})

    def generate(self, hidden=None):
        if hidden is None:
            hidden = np.random.normal(size=self.weights["b1"])
        return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})

    def reconstruct(self, X):
        return self.sess.run(self.reconstruction, feed_dict={self.x: X})

    def getWeights(self):
        return self.sess.run(self.weights['w1'])

    def getBiases(self):
        return self.sess.run(self.weights['b1'])

    @staticmethod
    def xavier_init(fan_in, fan_out, constant=1):
        low = -constant * np.sqrt(6.0 / (fan_in + fan_out))
        high = constant * np.sqrt(6.0 / (fan_in + fan_out))
        return tf.random_uniform((fan_in, fan_out),
                                 minval=low,
                                 maxval=high,
                                 dtype=tf.float32)


class TensorFlowAutoEncoder:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = HelperFunction()
        self.config = ConfigUtil.get_config_instance()

    @staticmethod
    def standard_scale(train_matrix, test_matrix):
        preprocessor = prep.StandardScaler(with_mean=False).fit(train_matrix)
        train_matrix = preprocessor.transform(train_matrix)
        test_matrix = preprocessor.transform(test_matrix)
        return train_matrix, test_matrix

    @staticmethod
    def get_random_block_from_data(data, batch_size):
        start_index = np.random.randint(0, data.shape[0] - batch_size)
        return data[start_index:(start_index + batch_size)]

    def prepare_data(self):
        pruned_matrix_path = self.config['data']['pruned_feature_vector_path']
        list_of_files = self.helper.get_files_ends_with_extension(path=pruned_matrix_path, extension=".hkl")
        input_matrix = self.helper.stack_matrix(list_of_files)
        self.log.info("Size of the input matrix is : {}".format(input_matrix.shape))

        train_matrix, test_matrix = tts(input_matrix, test_size=0.25, random_state=1)
        self.log.info("Train Matrix size : {}".format(train_matrix.shape))
        self.log.info("Test Matrix size : {}".format(test_matrix.shape))
        return train_matrix, test_matrix

    def training(self, n_rows, batch_size, train_matrix, autoencoder, epoch, display_step):
        avg_cost = 0.
        total_batch = int(n_rows / batch_size)
        # Loop over all batches
        for i in range(total_batch):
            for d in ['/gpu:0', '/gpu:1']:
                with tf.device(d):
                    batch_xs = self.get_random_block_from_data(train_matrix.toarray(), batch_size)

                    # Fit training using batch data
                    cost = autoencoder.partial_fit(batch_xs)
                    # Compute average loss
                    avg_cost += cost / n_rows * batch_size

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost))

    def main(self):
        with tf.device("/cpu:0"):
            train_matrix, test_matrix = self.prepare_data()
            train_matrix, test_matrix = self.standard_scale(train_matrix, test_matrix)
            n_rows = int(train_matrix.shape[0] + test_matrix.shape[0])
            n_cols = train_matrix.shape[1]

            training_epochs = self.config['autoencoder_params']['training_epochs']
            batch_size = self.config['autoencoder_params']['batch_size']
            display_step = self.config['autoencoder_params']['display_step']

            autoencoder = Autoencoder(n_input=n_cols,
                                      n_hidden=(n_cols / 2),
                                      transfer_function=tf.nn.relu,
                                      optimizer=tf.train.AdamOptimizer(learning_rate=0.001))

        for index, epoch in enumerate(range(training_epochs)):
            if index % 2 == 0:
                with tf.device("/gpu:0"):
                    self.training(n_rows, batch_size, train_matrix, autoencoder, epoch, display_step)
            else:
                with tf.device("/gpu:1"):
                    self.training(n_rows, batch_size, train_matrix, autoencoder, epoch, display_step)

        print("Total cost: " + str(autoencoder.calc_total_cost(test_matrix.toarray())))


if __name__ == '__main__':
    ae = TensorFlowAutoEncoder()
    ae.main()
