import numpy as np
import pickle as pi

from keras.layers import Dense, Input, Dropout, BatchNormalization, Flatten, Concatenate
from keras.layers import Embedding, Conv1D, MaxPooling1D
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras.optimizers import adam
from keras.models import Model
from keras.utils import np_utils

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from scipy.sparse import vstack, load_npz, save_npz

from PrepareData.DataGeneratorKeras import Script
from Utils.LoggerUtil import LoggerUtil
from Utils.ConfigUtil import ConfigUtil


class OneDCnn:
    def __init__(self, **kwargs):
        self.vocab_size = kwargs['vocab_size']
        self.embedding_dims = kwargs['embedding_dims']
        self.max_sequence_length = kwargs['max_sequence_length']
        self.dropout = kwargs['dropout']
        self.learning_rate = kwargs['learning_rate']
        self.n_classes = kwargs['n_classes']
        self.epochs = kwargs['epochs']
        self.num_rows = kwargs['num_rows']
        self.batch_size = kwargs['batch_size']
        self.n_samples = kwargs['n_samples']
        self.n_features = kwargs['n_features']
        self.filters = kwargs['filters']
        self.kernel_size = kwargs['kernel_size']
        self.strides = kwargs['strides']

        self.log = LoggerUtil(self.__class__.__name__).get()
        self.config = ConfigUtil.get_config_instance()
        self.script = Script(num_rows=self.num_rows, batch_size=self.batch_size, test_size=0.25,
                             n_classes=self.n_classes, n_samples=self.n_samples, n_features=self.n_features)

    def sequential_cnn(self, sequence_input, embedded_sequences):
        conv1_1 = Conv1D(filters=self.filters[0], kernel_size=self.kernel_size[0], strides=self.strides[0],
                         activation='relu', name="conv1_1")(embedded_sequences)
        conv1_2 = Conv1D(filters=self.filters[0], kernel_size=self.kernel_size[0], strides=self.strides[0],
                         activation='relu', name="conv1_2")(conv1_1)
        pool_1 = MaxPooling1D(3, name="maxpool_1")(conv1_2)

        conv2_1 = Conv1D(filters=self.filters[1], kernel_size=self.kernel_size[1], strides=self.strides[1],
                         activation='relu', name="conv2_1")(pool_1)
        conv2_2 = Conv1D(filters=self.filters[1], kernel_size=self.kernel_size[1], strides=self.strides[1],
                         activation='relu', name="conv2_2")(conv2_1)
        pool_2 = MaxPooling1D(3, name="maxpool_2")(conv2_2)

        conv3_1 = Conv1D(filters=self.filters[2], kernel_size=self.kernel_size[2], strides=self.strides[2],
                         activation='relu', name="conv3_1")(pool_2)
        conv3_2 = Conv1D(filters=self.filters[2], kernel_size=self.kernel_size[2], strides=self.strides[2],
                         activation='relu', name="conv3_2")(conv3_1)
        conv3_3 = Conv1D(filters=self.filters[2], kernel_size=self.kernel_size[2], strides=self.strides[2],
                         activation='relu', name="conv3_3")(conv3_2)
        pool_3 = MaxPooling1D(3, name="maxpool_3")(conv3_3)

        conv4_1 = Conv1D(filters=self.filters[3], kernel_size=self.kernel_size[3], strides=self.strides[3],
                         activation='relu', name="conv4_1")(pool_3)
        conv4_2 = Conv1D(filters=self.filters[3], kernel_size=self.kernel_size[3], strides=self.strides[3],
                         activation='relu', name="conv4_2")(conv4_1)
        conv4_3 = Conv1D(filters=self.filters[3], kernel_size=self.kernel_size[3], strides=self.strides[3],
                         activation='relu', name="conv4_3")(conv4_2)
        pool_4 = MaxPooling1D(3, name="maxpool_4")(conv4_3)

        conv5_1 = Conv1D(filters=self.filters[4], kernel_size=self.kernel_size[4], strides=self.strides[4],
                         activation='relu', name="conv5_1")(pool_4)
        conv5_2 = Conv1D(filters=self.filters[4], kernel_size=self.kernel_size[4], strides=self.strides[4],
                         activation='relu', name="conv5_2")(conv5_1)
        conv5_3 = Conv1D(filters=self.filters[4], kernel_size=self.kernel_size[4], strides=self.strides[4],
                         activation='relu', name="conv5_3")(conv5_2)
        pool_5 = MaxPooling1D(3, name="maxpool_5")(conv5_3)

        l_flat = Flatten(name="flatten")(pool_5)

        dense_1 = Dense(512, activation='relu', name="dense_1")(l_flat)

        dense_2 = Dense(128, activation='relu', name="dense_2")(dense_1)

        preds = Dense(self.n_classes, activation='softmax', name="output_softmax")(dense_2)

        model = Model(sequence_input, preds)
        self.log.info("Summary : \n{}".format(model.summary()))
        return model

    def parallel_cnn(self, sequence_input, embedded_sequences):
        conv1_1 = Conv1D(filters=self.filters[0], kernel_size=3, strides=3, activation='relu', name="conv1_1")(
            embedded_sequences)
        conv1_2 = Conv1D(filters=self.filters[0], kernel_size=3, strides=3, activation='relu', name="conv1_2")(conv1_1)
        pool_1 = MaxPooling1D(3, name="maxpool_1")(conv1_2)
        batch_norm_1 = BatchNormalization(name="batch_norm_1")(pool_1)
        dropout_1 = Dropout(self.dropout)(batch_norm_1)

        conv2_1 = Conv1D(filters=self.filters[1], kernel_size=3, strides=3, activation='relu', name="conv2_1")(
            embedded_sequences)
        conv2_2 = Conv1D(filters=self.filters[1], kernel_size=3, strides=3, activation='relu', name="conv2_2")(conv2_1)
        pool_2 = MaxPooling1D(3, name="maxpool_2")(conv2_2)
        batch_norm_2 = BatchNormalization(name="batch_norm_2")(pool_2)
        dropout_2 = Dropout(self.dropout)(batch_norm_2)

        conv3_1 = Conv1D(filters=self.filters[2], kernel_size=3, strides=3, activation='relu', name="conv3_1")(
            embedded_sequences)
        conv3_2 = Conv1D(filters=self.filters[2], kernel_size=3, strides=3, activation='relu', name="conv3_2")(conv3_1)
        pool_3 = MaxPooling1D(3, name="maxpool_3")(conv3_2)
        batch_norm_3 = BatchNormalization(name="batch_norm_3")(pool_3)
        dropout_3 = Dropout(self.dropout)(batch_norm_3)

        convs = [dropout_1, dropout_2, dropout_3]

        l_merge = Concatenate(name="l_concatenate")(convs)
        l_flat = Flatten(name="flatten")(l_merge)
        l_dense = Dense(128, activation='relu', name="dense_1")(l_flat)
        preds = Dense(self.n_classes, activation='softmax', name="output_softmax")(l_dense)

        model = Model(sequence_input, preds)
        self.log.info("Summary : \n{}".format(model.summary()))
        return model

    def build_model(self, sequential, parallel):
        embedding_layer = Embedding(self.vocab_size, self.embedding_dims, input_length=self.max_sequence_length)

        sequence_input = Input(shape=(self.vocab_size,), dtype='int32')
        embedded_sequences = embedding_layer(sequence_input)
        if sequential:
            return self.sequential_cnn(sequence_input=sequence_input, embedded_sequences=embedded_sequences)
        elif parallel:
            return self.parallel_cnn(sequence_input=sequence_input, embedded_sequences=embedded_sequences)
        else:
            raise Exception("One of Sequential or Parallel must be True")

    def compile_model(self, model, model_path):
        optimizer = adam(lr=self.learning_rate)
        model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

        filepath = model_path + "/" + "weights-improvement-{epoch:03d}-{val_acc:.4f}.hdf5"
        checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

        partition, training_generator, validation_generator = self.script.main()
        return training_generator, validation_generator, partition, checkpoint

    def run_model(self, model, training_generator, validation_generator, partition, checkpoint):
        model.fit_generator(
            generator=training_generator,
            steps_per_epoch=len(partition['train']),
            epochs=self.epochs,
            validation_data=validation_generator,
            validation_steps=len(partition['validation']),
            verbose=1,
            callbacks=[checkpoint])

    def cross_validate(self, model, x_train, y_train, nb_classes):
        kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=7)
        cvscores = []
        X = x_train.toarray()
        Y = y_train

        for index, (train, test) in enumerate(kfold.split(X, Y)):
            # create model
            self.log.info("kfold iter : #{}".format(index))
            model.fit(X[train], np_utils.to_categorical(Y[train], num_classes=len(nb_classes)),
                      epochs=10, batch_size=64, verbose=1,
                      callbacks=[TensorBoard(log_dir='/tmp/temp/tensorboard_logs/')]
                      )

            # evaluate the model
            scores = model.evaluate(X[test], Y[test], verbose=0)
            self.log.info("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
            cvscores.append(scores[1] * 100)

        self.log.info("%.2f%% (+/- %.2f%%)" % (np.mean(cvscores), np.std(cvscores)))
        return cvscores

    def get_data(self, num_rows):
        labels_path = self.config["data"]["labels_path"]
        base_data_path = self.config["data"]["feature_selection_path"]

        labels = np.asarray(pi.load(open(labels_path + "/" + "labels.pkl")))
        nb_classes = np.unique(labels)

        fv = []
        for x in xrange(num_rows):
            fv.append(load_npz(base_data_path + "feature_selection_" + str(x) + ".npz"))
        matrix = vstack(fv)
        del fv

        return matrix, labels, nb_classes

    def main(self, cross_validate, num_rows):
        model_path = self.config['models']['one_d_cnn']['results_path']
        model = self.build_model(sequential=False, parallel=True)
        if cross_validate:
            matrix, labels, nb_classes = self.get_data(num_rows=num_rows)
            cv_scores = self.cross_validate(model=model, x_train=matrix, y_train=labels, nb_classes=nb_classes)
            pi.dump(cv_scores, open(model_path + "/" + "cross_validated_scores.pkl", "w"))
        else:
            training_generator, validation_generator, partition, checkpoint = self.compile_model(model=model,
                                                                                                 model_path=model_path)
            self.run_model(model=model, partition=partition, training_generator=training_generator,
                           validation_generator=validation_generator, checkpoint=checkpoint)


def get_params():
    param_dict = dict()

    # The below param changes as per the type of data being loaded and in malware these 3 are same.
    param_dict['n_features'] = 10357
    param_dict['vocab_size'] = 10357
    param_dict['max_sequence_length'] = 10357

    # The smaller the embedding_dims the faster the model will be.
    param_dict['embedding_dims'] = 2 ** 7

    # Model compile params
    param_dict['learning_rate'] = 0.001
    param_dict['epochs'] = 20
    param_dict['dropout'] = 0.5

    param_dict['n_classes'] = 4098
    param_dict['num_rows'] = 407
    param_dict['batch_size'] = 64
    param_dict['n_samples'] = 406762

    # CNN params
    param_dict['filters'] = [2 ** 3, 2 ** 4, 2 ** 5]
    param_dict['kernel_size'] = [3, 3, 3, 3, 3]
    param_dict['strides'] = [1, 1, 1, 1, 1]

    return param_dict


if __name__ == '__main__':
    params = get_params()
    one_d_cnn = OneDCnn(**params)
    one_d_cnn.main(cross_validate=True, num_rows=25000)