import h5py
import sys
import numpy
from high_order_statistics import cumulants, wavelet_transform

class DataLoader(object):
    def __init__(self, file_name):
        f_in = h5py.File(file_name, 'r+')
        self.data_len = len(f_in['X'])
        self.features = numpy.array(f_in['X'])
        self.labels = numpy.array(f_in['Y'])
        self.snr = numpy.array(f_in['Z'])
        idxes = numpy.arange(self.data_len)
        numpy.random.shuffle(idxes)
        self.train_len = self.data_len // 10 * 9 
        print("training data: %d, testing data: %d" % (self.train_len, self.data_len - self.train_len))
        self.train_idxes = numpy.copy(idxes[:self.train_len])
        self.test_idxes = numpy.copy(idxes[self.train_len:])

    def shuffle_train_idx(self):
        numpy.random.shuffle(self.train_idxes)

    def get_batch(self, batch_size):
        cur_idx = 0
        self.shuffle_train_idx()
        print(self.train_idxes)
        while cur_idx < self.train_len:
            sel_idxes = self.train_idxes[cur_idx:(cur_idx + batch_size)]
            cur_idx += batch_size
            yield self.features[sel_idxes], self.labels[sel_idxes], self.snr[sel_idxes]

class FeaturesExtractor(object):
    def __init__(self):
        pass

if __name__=='__main__':
    dataset = DataLoader(sys.argv[1])
    for X,Y,snr in dataset.get_batch(4):
        cum = cumulants(X)
        print(cum)
        print(Y)
