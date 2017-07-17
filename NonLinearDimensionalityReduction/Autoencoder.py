import pickle as pi

import numpy as np
from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split as tts

from Utils.LoggerUtil import LoggerUtil
from HelperFunctions.HelperFunction import HelperFunction


class Autoencoder:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = HelperFunction()

    def prepare_data(self):
        """
        Method to prepare the data
        :returns train and test matrix
        """
        input_matrix = pi.load(open("input_matrix"))
        self.log.info("Size of the input matrix is : {}".format(input_matrix.shape))

        train_matrix, test_matrix = tts(input_matrix, test_size=0.20, random_state=1)
        self.log.info("Train Matrix size : {}".format(train_matrix.shape))
        self.log.info("Test Matrix size : {}".format(test_matrix.shape))
        return train_matrix, test_matrix

    def prepare_model(self, input_img):
        """
        This method will prepare the model.
        Each layer is a reduction by 2^2 i.e 4.
        :return:
        """
        self.log.info("~~~~~~Model Preparation started~~~~~~")
        encoding_dim = self.helper.nearest_power_of_two(self.train_matrix.shape[1]) - 2
        encoded = Dense(encoding_dim, activation='relu')(input_img)
        decoded = Dense(self.train_matrix.shape[1], activation='sigmoid')(encoded)
        autoencoder = Model(input=input_img, output=decoded)

        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')

        self.log.info("~~~~~~Model Preparation completed~~~~~~")
        return autoencoder

    def train_model(self, autoencoder):
        """
        This method will train the model.
        :return:
        """
        x_train = self.train_matrix.astype('float32') / 255
        x_test = self.test_matrix.astype('float32') / 255
        x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
        x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

        self.log.info("~~~~~~Model Training started~~~~~~")

        autoencoder.fit(x_train, x_train,
                        nb_epoch=2,
                        batch_size=150,
                        shuffle=True,
                        validation_data=(x_test, x_test))

        self.log.info("~~~~~~Model Training completed~~~~~~")

    def main(self):
        """
        The main method
        :return:
        """
        self.train_matrix, self.test_matrix = self.prepare_data()
        input_img = Input(shape=(103299,))
        autoencoder = self.prepare_model(input_img)
        self.train_model(autoencoder)


if __name__ == '__main__':
    sae = Autoencoder()
    sae.main()
