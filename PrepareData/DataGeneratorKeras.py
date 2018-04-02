import numpy as np
import pickle as pi

from scipy.sparse import vstack, load_npz
from keras.utils import np_utils


class DataGeneratorKeras:
    def __init__(self, num_rows, n_classes, dim_x=32, dim_y=32, batch_size=1, shuffle=True):
        self.num_rows = num_rows
        self.n_samples = dim_x
        self.n_features = dim_y
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.flag = False

    def __get_exploration_order(self, list_ids):
        """
        Generates order of exploration
        :param list_ids:
        :return:
        """
        # Find exploration order
        indexes = np.arange(len(list_ids))
        if self.shuffle:
            np.random.shuffle(indexes)
        return indexes

    def __data_generation(self, list_ids_temp, n_classes):
        """
        Generates data of batch_size samples
        :param list_ids_temp:
        :param n_classes:
        :return:
        """
        index = list_ids_temp[0]

        fv = load_npz("/tmp/DataComplete/pruned_fv_path/pruned_mat_part_" + str(index) + ".npz")
        matrix = vstack(fv)

        labels_complete = pi.load(open("/tmp/DataComplete/labels.pkl"))
        partial_labels = labels_complete[index]
        del labels_complete

        y = self.sparsify(partial_labels, n_classes)
        return matrix, y

    @staticmethod
    def sparsify(y, n_classes):
        """
        :return:
        """
        label_encoder = np_utils.to_categorical(y, n_classes)
        return label_encoder

    def generate(self, list_ids):
        """
        Generates batches of samples
        :param list_ids:
        :return:
        """
        # Infinite loop
        while 1:
            # Generate order of exploration of dataset
            indexes = self.__get_exploration_order(list_ids)

            # Generate batches
            imax = int(len(indexes) / self.batch_size)
            for i in range(imax):
                # Find list of IDs
                # list_ids_temp = list()
                # for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]:
                #     list_ids_temp.append(list_ids[k])
                list_ids_temp = [list_ids[k] for k in indexes[i * self.batch_size:(i + 1) * self.batch_size]]

                # Generate data
                x, y = self.__data_generation(list_ids_temp, self.n_classes)

                yield x.toarray(), y


class Script:
    def __init__(self, num_rows, batch_size, test_size, n_classes, n_samples, n_features):
        self.batch_size = batch_size
        self.num_rows = num_rows
        self.test_size = test_size
        self.n_classes = n_classes
        self.n_samples = n_samples
        self.n_features = n_features

    def main(self):
        validation = int(self.test_size * self.num_rows)
        train = self.num_rows - validation

        params = {
            'num_rows': self.num_rows,
            'dim_x': self.n_samples,
            'dim_y': self.n_features,
            'n_classes': self.n_classes,
            'batch_size': self.batch_size,
            'shuffle': True
        }

        partition = {'train': range(train), 'validation': range(train, self.num_rows)}

        # Generators
        training_generator = DataGeneratorKeras(**params).generate(partition['train'])
        validation_generator = DataGeneratorKeras(**params).generate(partition['validation'])

        # Model Run
        # model.fit_generator(
        #     generator = training_generator,
        #     steps_per_epoch = len(partition['train']),
        #     epochs = epochs,
        #     validation_data = validation_generator,
        #     validation_steps = len(partition['validation']),
        #     verbose = 1,
        #     callbacks = [plot_losses]
        # )
        return partition, training_generator, validation_generator


if __name__ == '__main__':
    script = Script(num_rows=407, test_size=0.25, n_classes=4098,
                    n_samples=406762, n_features=26421, batch_size=1)
    script.main()
