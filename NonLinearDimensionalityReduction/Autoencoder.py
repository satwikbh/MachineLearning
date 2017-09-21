from keras.layers import Input, Dense
from keras.models import Model
from sklearn.model_selection import train_test_split as tts

from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil
from HelperFunctions.HelperFunction import HelperFunction


class Autoencoder:
    def __init__(self):
        self.log = LoggerUtil(self.__class__.__name__).get()
        self.helper = HelperFunction()
        self.config = ConfigUtil().get_config_instance()

    def prepare_autoencoder(self, input_layer, output_layer):
        self.log.info("~~~~~~Model Preparation started~~~~~~")
        autoencoder = Model(input=input_layer, output=output_layer)
        autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
        self.log.info("~~~~~~Model Preparation completed~~~~~~")
        self.log.info("~~~~~~Model Summary ~~~~~~")
        self.log.info(autoencoder.summary())
        return autoencoder

    @staticmethod
    def prepare_decoding_model(layers, last_encoded_layer):
        """
        Decoding layers
        :param layers:
        :param last_encoded_layer:
        :return:
        """
        list_of_layers = list()
        for index, each_layer in enumerate(layers):
            decoder_hidden_layer_dims = each_layer
            if index == 0:
                decoder_hidden_layer = Dense(decoder_hidden_layer_dims, activation="relu", trainable=True,
                                             name="dec_layer" + str(index))(last_encoded_layer)
            else:
                decoder_hidden_layer = Dense(decoder_hidden_layer_dims, activation="relu", trainable=True,
                                             name="dec_layer" + str(index))(decoder_hidden_layer)
            list_of_layers.append(decoder_hidden_layer)
        return list_of_layers

    @staticmethod
    def prepare_encoding_model(layers, input_layer):
        """
        Encoding layers
        :param layers:
        :param input_layer:
        :return:
        """
        list_of_layers = list()
        for index, each_layer in enumerate(layers):
            encoder_hidden_layer_dims = each_layer
            if index == 0:
                encoder_hidden_layer = Dense(encoder_hidden_layer_dims, activation="relu", trainable=True,
                                             name="layer" + str(index))(input_layer)
            else:
                encoder_hidden_layer = Dense(encoder_hidden_layer_dims, activation="relu", trainable=True,
                                             name="layer" + str(index))(encoder_hidden_layer)
            list_of_layers.append(encoder_hidden_layer)
        return list_of_layers

    def prepare_model(self, train_matrix, test_matrix, layers):
        """
        This method will prepare the model.
        Each layer is a reduction by 2^2 i.e 4.
        :return:
        """
        input_matrix = Input(shape=(train_matrix.shape[1],))

        encoding_layers = self.prepare_encoding_model(layers, input_layer=input_matrix)
        decoding_layers = self.prepare_decoding_model(layers[::-1], last_encoded_layer=encoding_layers[-1])

        autoencoder_model = self.prepare_autoencoder(input_matrix, decoding_layers[-1])
        return autoencoder_model

    def train_model(self, autoencoder, train_matrix, test_matrix):
        """
        This method will train the model.
        :return:
        """
        self.log.info("~~~~~~Model Training started~~~~~~")

        history = autoencoder.fit(train_matrix, train_matrix,
                                  nb_epoch=2,
                                  batch_size=150,
                                  shuffle=True,
                                  validation_data=(test_matrix, test_matrix))

        self.log.info("~~~~~~Model Training completed~~~~~~")
        return history

    @staticmethod
    def prepare_layers(train_matrix):
        """
        Recursively reduce the nodes in a layer by 50%.
        :param train_matrix: the training matrix
        :return layers: List containing the number of nodes in each layer/
        """
        layers = list()
        shape = train_matrix.shape[1]
        while shape > 1024:
            layers.append(shape)
            shape /= 2
        return layers

    def prepare_data(self):
        """
        Method to prepare the data
        :returns train and test matrix
        """
        pruned_matrix_path = self.config['data']['pruned_feature_vector_path']
        list_of_files = self.helper.get_files_ends_with_extension(path=pruned_matrix_path, extension=".hkl")
        input_matrix = self.helper.stack_matrix(list_of_files)
        self.log.info("Size of the input matrix is : {}".format(input_matrix.shape))

        train_matrix, test_matrix = tts(input_matrix, test_size=0.25, random_state=1)
        self.log.info("Train Matrix size : {}".format(train_matrix.shape))
        self.log.info("Test Matrix size : {}".format(test_matrix.shape))
        return train_matrix, test_matrix

    def main(self):
        """
        The main method
        :return:
        """
        train_matrix, test_matrix = self.prepare_data()
        layers = self.prepare_layers(train_matrix)
        autoencoder_model = self.prepare_model(train_matrix, test_matrix, layers)
        # history = self.train_model(autoencoder_model, train_matrix, test_matrix)
        # return history


if __name__ == '__main__':
    sae = Autoencoder()
    sae.main()
