import pickle
import numpy as np
import os


class Dataset:
    def __init__(self, root):
        classes = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
        img_rows, img_cols = 32, 32
        input_shape = (img_rows, img_cols, 3)
        self.root = root

    def load_pickle(self, f):
        return pickle.load(f, encoding='bytes')

    def load_CIFAR_batch(self, filename):
        """ load single batch of cifar """
        with open(filename, 'rb') as f:
            datadict = self.load_pickle(f)
            X = datadict[b'data']
            Y = datadict[b'labels']
            X = X.reshape(10000, 3072)
            Y = np.array(Y)
            return X, Y

    def load_CIFAR10(self):
        """ load all of cifar """
        xs = []
        ys = []
        for b in range(1, 6):
            f = os.path.join(self.root, 'data_batch_%d' % (b, ))
            X, Y = self.load_CIFAR_batch(f)
            xs.append(X)
            ys.append(Y)
        Xtr = np.concatenate(xs)
        Ytr = np.concatenate(ys)
        del X, Y
        Xte, Yte = self.load_CIFAR_batch(os.path.join(self.root, 'test_batch'))
        return Xtr, Ytr, Xte, Yte

    def get_CIFAR10_data(self, num_training=49000, num_validation=1000, num_test=10000):
        # Load the raw CIFAR-10 data
        X_train, y_train, X_test, y_test = self.load_CIFAR10()

        x_train = X_train.astype('float32')
        x_test = X_test.astype('float32')

        x_train /= 255
        x_test /= 255

        return x_train, y_train, x_test, y_test


# Invoke the above function to get our data.
if __name__ == "__main__":
    dataset = Dataset('cifar-10-batches-py')
    x_train, y_train, x_test, y_test = dataset.get_CIFAR10_data()
