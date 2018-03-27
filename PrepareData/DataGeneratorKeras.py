import numpy as np

from scipy.sparse import vstack, load_npz


class DataGeneratorKeras:
    def __init__(self, num_rows, dim_x=32, dim_y=32, batch_size=1, shuffle=True):
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_rows = num_rows

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

    def __data_generation(self, labels, list_ids_temp):
        """
        Generates data of batch_size samples
        :param labels: 
        :param list_ids_temp: 
        :return: 
        """
        index = list_ids_temp[0]
        fv = load_npz("/tmp/pruned_fv_path/pruned_fv_part_" + str(index) + ".npz")
        matrix = vstack(fv)
        if self.num_rows - 1 == index:
            partial_labels = labels[index * 1000:]
        else:
            partial_labels = labels[index * 1000: (index + 1) * 1000]
        return matrix, self.sparsify(partial_labels)

    @staticmethod
    def sparsify(y):
        """
        :return:
        """
        n_classes = 0
        return np.array([[1 if y[i] == j else 0 for j in range(n_classes)]
                         for i in range(y.shape[0])])

    def generate(self, labels, list_ids):
        """
        Generates batches of samples
        :param labels:
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
                x, y = self.__data_generation(labels, list_ids_temp)

                yield x, y


class Script:
    def __init__(self):
        self.batch_size = 1

    def main(self):
        params = {'dim_x': 100,
                  'dim_y': 50,
                  'batch_size': 1,
                  'shuffle': True}

        # Datasets
        num_rows = 406762
        test_size = 0.25

        validation = int(test_size * num_rows)
        train = num_rows - validation

        partition = {'train': range(train), 'validation': range(train, num_rows)}
        labels = {'train': range(train), 'validation': range(train, num_rows)}

        # Generators
        training_generator = DataGeneratorKeras(**params).generate(labels, partition['train'])
        validation_generator = DataGeneratorKeras(**params).generate(labels, partition['validation'])

        while training_generator.next() is not None:
            training_generator.next()
            validation_generator.next()

            # # Design model
            # model = Sequential()
            #
            # # Train model on dataset
            # model.fit_generator(generator=training_generator,
            #                     steps_per_epoch=len(partition['train']) // self.batch_size,
            #                     validation_data=validation_generator,
            #                     validation_steps=len(partition['validation']) // self.batch_size)


if __name__ == '__main__':
    script = Script()
    script.main()
