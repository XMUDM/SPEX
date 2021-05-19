import numpy as np

class Data():
    def __init__(self, data, n_node, shuffle=False, test=False):
        self.test = test
        inputs = data[0]
        self.n_node = n_node  # for padding
        inputs, mask, len_max = self.data_masks(inputs, [self.n_node]) # padding
        self.inputs = np.asarray(inputs)
        self.mask = np.asarray(mask)
        self.len_max = len_max
        self.targets = np.asarray(data[1])
        if self.test:
            self.neg = np.asarray(data[2])
        self.length = len(inputs)
        self.shuffle = shuffle

    def generate_batch(self, batch_size):
        if self.shuffle:
            shuffled_arg = np.arange(self.length)
            np.random.shuffle(shuffled_arg)
            self.inputs = self.inputs[shuffled_arg]
            self.mask = self.mask[shuffled_arg]
            self.targets = self.targets[shuffled_arg]
            if self.test:
                self.neg = self.neg[shuffled_arg]
        n_batch = int(self.length / batch_size)
        if self.length % batch_size != 0:
            n_batch += 1
        slices = np.split(np.arange(n_batch * batch_size), n_batch)
        slices[-1] = slices[-1][:(self.length - batch_size * (n_batch - 1))]
        return slices

    def get_slice(self, i):
        inputs = self.inputs[i]
        mask = self.mask[i]
        targets = self.targets[i]
        if self.test:
            negs = self.neg[i]
            return inputs, mask, targets, negs
        else:
            return inputs, mask, targets

    def data_masks(self, all_usr_pois, item_tail):  # inputs, [self.n_node]
        us_lens = [len(upois) for upois in all_usr_pois]
        len_max = max(us_lens)
        us_pois = [upois + item_tail * (len_max - le) for upois, le in zip(all_usr_pois, us_lens)]
        us_msks = [[1] * le + [0] * (len_max - le) for le in us_lens]
        return us_pois, us_msks, len_max  # inputs, mask, len_max
