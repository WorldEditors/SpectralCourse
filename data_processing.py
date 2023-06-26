import h5py
import sys
import numpy
import matplotlib
import time
import argparse
from high_order_statistics import cumulants, wavelet_transform
from plot_tools import plot3D

class DataLoader(object):
    Modulation_Types = ["OOK", "ASK4", "ASK8", "BPSK", 
            "QPSK", "PSK8", "PSK16", "PSK32", 
            "APSK16", "APSK32", "APSK64", "APSK128", 
            "QAM16", "QAM32", "QAM64", "QAM128", 
            "QAM256", "AM_SSB_WC", "AM_SSB_SC", "AM_DSB_WC", 
            "AM_DSB_SC", "FM", "GMSK", "OQPS"]
    def __init__(self, file_name, 
            file_type='Raw'):
        if(file_type == 'Raw'):
            f_in = h5py.File(file_name, 'r+')
            self.data_len = len(f_in['X'])
            self.features = numpy.array(f_in['X'])
            self.labels = numpy.array(f_in['Y'])
            self.snr = numpy.array(f_in['Z'])
        elif(file_type == 'NPZ'):
            data = numpy.load(file_name)
            self.features = data['features']
            self.labels = data['labels']
            self.snr = data['snr']
            self.data_len = self.labels.shape[0]
        else:
            raise Exception("Can not support file type : %s" % file_type)

    def split_file(self, ratio=0.90):
        idxes = numpy.arange(self.data_len)
        numpy.random.shuffle(idxes)
        train_len = int(self.data_len * ratio)
        print("training data: %d, testing data: %d" % (train_len, self.data_len - train_len))
        train_idxes = numpy.copy(idxes[:train_len])
        test_idxes = numpy.copy(idxes[train_len:])
        return train_idxes, test_idxes

    def save_file(self, file_name, features, labels, snr):
        numpy.savez(file_name, features=features, labels=labels, snr=snr)

    def get_batch(self, batch_size,
            idxes = None,
            shuffle = True):
        cur_idx = 0
        if(idxes is None):
            idxes = numpy.arange(self.data_len)
            if(shuffle):
                numpy.random.shuffle(idxes)
        while cur_idx < self.data_len:
            sel_idxes = idxes[cur_idx:(cur_idx + batch_size)]
            cur_idx += batch_size
            yield cur_idx / self.data_len, self.features[sel_idxes], self.labels[sel_idxes], self.snr[sel_idxes]

class CumulantsPDF(object):
    def __init__(self, n, min_val, max_val):
        deta_val = (max_val - min_val) / n
        self.val_thres = numpy.arange(min_val + deta_val, max_val - 1.0e-6, deta_val)
        self.val_thres = numpy.concatenate([[-1.0e+10], self.val_thres, [+1.0e+10]], axis=0)
        self.val_pdf = numpy.zeros((n,))
        self.deta_val = deta_val

    def update(self, val):
        # 输入val： [batch]
        idx = (val > self.val_thres[:-1]) & (val < self.val_thres[1:])
        self.val_pdf += idx.astype("float32")

    @property
    def pdf(self):
        return self.val_pdf / numpy.sum(self.val_pdf)

    def dist_c(self):
        val_thres = self.val_thres[:-1]
        val_thres[0] = 2.0 * val_thres[1] - val_thres[2]
        val_thres += 0.5 * (val_thres[1] - val_thres[0])
        return val_thres, self.deta_val

def statistics(dataset):
    c_20_pdfs = [CumulantsPDF(100, 0.0, 1.0) for _ in range(24)]
    c_40_pdfs = [CumulantsPDF(100, 0.0, 2.0) for _ in range(24)]
    c_41_pdfs = [CumulantsPDF(100, 0.0, 2.0) for _ in range(24)]
    c_42_pdfs = [CumulantsPDF(100, 1.5, 4.5) for _ in range(24)]
    prt_ratio = 1.0
    cur_ratio = prt_ratio
    for r,X,Y,snr in dataset.get_batch(256):
        c20, c40, c41, c42 = cumulants(X)
        ys = numpy.argmax(Y, axis=-1)
        for i,y in enumerate(ys.tolist()):
            c_20_pdfs[y].update(c20[i])
            c_40_pdfs[y].update(c40[i])
            c_41_pdfs[y].update(c41[i])
            c_42_pdfs[y].update(c42[i])
        if(100.0 * r > cur_ratio):
            print("Finished data percentage: %.1f" % (100.0 * r))
            cur_ratio += prt_ratio
    ret = dict()
    ret["c_20"] = [x.pdf for x in c_20_pdfs]
    ret["c_40"] = [x.pdf for x in c_40_pdfs]
    ret["c_41"] = [x.pdf for x in c_41_pdfs]
    ret["c_42"] = [x.pdf for x in c_42_pdfs]
    dist_c = dict()
    dist_c["c_20"] = c_20_pdfs[0].dist_c()
    dist_c["c_40"] = c_40_pdfs[0].dist_c()
    dist_c["c_41"] = c_41_pdfs[0].dist_c()
    dist_c["c_42"] = c_42_pdfs[0].dist_c()
    return ret, dist_c

def data_preprocessing(dataset, gen_dir):
    prt_ratio = 1.0
    cur_ratio = prt_ratio
    post_features = numpy.zeros((dataset.data_len, 4))
    print("start_time", time.time())
    for r,X,Y,snr in dataset.get_batch(dataset.data_len, shuffle=False):
        c20, c40, c41, c42 = cumulants(X)
        post_features[:, 0] = c20
        post_features[:, 1] = c40
        post_features[:, 2] = c41
        post_features[:, 3] = c42
    print("end_time", time.time())
    sys.exit(1)
    train_idx, test_idx = dataset.split_file()
    dataset.save_file("%s/train.raw.dat"%gen_dir, dataset.features[train_idx], dataset.labels[train_idx], dataset.snr[train_idx])
    dataset.save_file("%s/test.raw.dat"%gen_dir, dataset.features[test_idx], dataset.labels[test_idx], dataset.snr[test_idx])
    dataset.save_file("%s/train.cum.dat"%gen_dir, post_features[train_idx], dataset.labels[train_idx], dataset.snr[train_idx])
    dataset.save_file("%s/test.cum.dat"%gen_dir, post_features[test_idx], dataset.labels[test_idx], dataset.snr[test_idx])

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Preprocessing the training and testing data')

    parser.add_argument('-raw_data', type=str)
    parser.add_argument('-task', type=str)
    parser.add_argument('--generate_dir', type=str, default='./data')

    args = parser.parse_args()
    print(args.raw_data, args.task)
    dataset = DataLoader(args.raw_data)
    if(args.task == "statistics"):
        res, dist_c = statistics(dataset)
        for name in res:
            dist_center, dist_width = dist_c[name]
            plot3D(res[name], DataLoader.Modulation_Types, dist_center, dist_width, name, "statistics_%s.png"%name)
    elif(args.task == "data_preprocessing"):
        data_preprocessing(dataset, args.generate_dir)
