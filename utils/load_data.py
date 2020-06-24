# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 21:38:32 2020

@author: LSH
"""

import cv2
import numpy as np
import copy

class load_data():
    train_set = []
    valid_set = []
    test_set = []

    def __init__(self):
        return

    def __get_4byte__(self, data_list, idx, endian="little"):
        if endian == "big":
            enu = range(idx, idx + 4)
        elif endian == "little":
            enu = range(idx + 3, idx - 1, -1)

        num = 0
        for i in enu:
            num *= 256
            num += data_list[i]

        return num

    def __load_im_data__(self, f):
        magic_num = self.__get_4byte__(list(map(int, f.read(4))), 0, endian="big")
        num_im = self.__get_4byte__(list(map(int, f.read(4))), 0, endian="big")
        num_rows = self.__get_4byte__(list(map(int, f.read(4))), 0, endian="big")
        num_cols = self.__get_4byte__(list(map(int, f.read(4))), 0, endian="big")

        im = np.array(list(map(int, f.read())), dtype=np.uint8)

        im = im.reshape((num_im, num_rows, num_cols, 1))

        return im

    def __load_num__(self, f):
        magic_num = self.__get_4byte__(list(map(int, f.read(4))), 0, endian="big")
        num_item = self.__get_4byte__(list(map(int, f.read(4))), 0, endian="big")

        num = np.array(list(map(int, f.read())), dtype=np.uint8)

        return num

    def __load__(self, d_type='all'):
        if not self.train_set:
            if d_type == "train" or d_type == "all":
                with open("./data/train-images.idx3-ubyte", 'rb') as f:
                    d = self.__load_im_data__(f)

                with open("./data/train-labels.idx1-ubyte", 'rb') as f:
                    n = self.__load_num__(f)

                self.train_set = [d, n]

        if not self.test_set:
            if d_type == "test" or d_type == "all":
                with open("./data/t10k-images.idx3-ubyte", 'rb') as f:
                    d = self.__load_im_data__(f)

                with open("./data/t10k-labels.idx1-ubyte", 'rb') as f:
                    n = self.__load_num__(f)

                self.valid_set = [d[:d.shape[0] // 2], n[:d.shape[0] // 2]]
                self.test_set = [d[d.shape[0] // 2:], n[d.shape[0] // 2:]]

    def get_data(self, d_type=['all']):
        ret_dic = dict()
        if 'all' in d_type or 'train' in d_type:
            self.__load__('train')

            ret_dic.update({'train':copy.deepcopy(self.train_set)})

        if 'all' in d_type or 'test' in d_type:
            self.__load__('test')

            ret_dic.update({'test':copy.deepcopy(self.test_set)})

        if 'all' in d_type or 'valid' in d_type:
            self.__load__('test')

            ret_dic.update({'valid':copy.deepcopy(self.valid_set)})

        return ret_dic


if __name__ == '__main__':
    ld = load_data()

    data = ld.get_data()

