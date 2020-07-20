# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 16:26:42 2020

@author: LSH
"""

import tensorflow as tf
from tensorflow.keras import regularizers, activations, optimizers
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, LeakyReLU
from tensorflow.keras.models import Sequential, load_model, save_model
import tensorflow.keras.backend as K
import utils.load_data as ld
import numpy as np
import utils.annotator
import plate_detector
import cv2 as cv
import copy as cp
import time
from collections import Counter
from operator import itemgetter

class car_number_detector(utils.annotator.annotator):
    def __init__(self, seed=0, pre_model=False):
        super().__init__(seed)
        self.num_dict = {'None': 0, '0': 1, '1': 2, '2': 3, '3': 4,
                         '4': 5, '5': 6, '6': 7, '7': 8, '8': 9,
                         '9': 10, '가': 11, '거': 12, '경': 13, '고': 14,
                         '구': 15, '기': 16, '나': 17, '너': 18, '노': 19,
                         '누': 20, '다': 21, '더': 22, '도': 23, '두': 24,
                         '라': 25, '러': 26, '로': 27, '루': 28, '마': 29,
                         '머': 30, '모': 31, '무': 32, '미': 33, '바': 34,
                         '버': 35, '보': 36, '부': 37, '북': 38, '사': 39,
                         '서': 40, '소': 41, '수': 42, '어': 42, '오': 44,
                         '우': 45, '울': 46, '자': 47, '저': 48, '전': 49,
                         '조': 50, '주': 51, '하': 52, '허': 53, '호': 54,
                         '히': 55}

        if pre_model:
            self.model = self.__load_preset_model__()

        return

    def __data_augment_color_reverse__(self, data, target):
        aug_data = cp.deepcopy(255 - data)
        aug_target = cp.deepcopy(target)

        return aug_data, aug_target

    def __data_augment_brightness__(self, data, target, c=1.5):
        aug_data = cp.deepcopy(c * data.astype(np.uint32))
        aug_data[aug_data > 255] = 255
        aug_data = aug_data.astype(np.uint8)

        aug_target = cp.deepcopy(target)

        return aug_data, aug_target

    def __data_processing__(self, data, target, augment=True):
        ret_data = []
        ret_target = []

        t_size = [256, 128]
        div = 8
        for n, (d, t) in enumerate(zip(data, target)):
            for plate_cord, car_num, char_cord in zip(t[0], t[1], t[2]):
                temp_im = d[plate_cord[1] - plate_cord[3] // 2: plate_cord[1] + plate_cord[3] // 2,
                            plate_cord[0] - plate_cord[2] // 2: plate_cord[0] + plate_cord[2] // 2]

                temp_t = np.zeros((t_size[1] // div, t_size[0] // div, len(self.num_dict)))
                matched = np.zeros((t_size[1] // div, t_size[0] // div))

                h, w = temp_im.shape[0], temp_im.shape[1]

                temp_im = cv.resize(temp_im, (t_size[0], t_size[1]))
                temp_im = temp_im.reshape((t_size[1], t_size[0], 1))
                im_y, im_x, _ = temp_im.shape

                for num, cord in zip(car_num, char_cord):
                    s_x, s_y, c_w, c_h = super().__xywh2ppwh__(cord)
                    s_x = (s_x * im_x) // (div * w)
                    s_y = (s_y * im_y) // (div * h)
                    c_w = int((c_w * im_x) / (div * w) + 0.5)
                    c_h = int((c_h * im_y) / (div * h) + 0.5)

                    temp_t[s_y: s_y + c_h + 1, s_x: s_x + c_w + 1, self.num_dict[num]] = 1
# =============================================================================
#                     matched[s_y: s_y + c_h, s_x: s_x + c_w] = 1
# =============================================================================

                ret_data.append(temp_im)

                temp_t[np.sum(temp_t, axis=2) == 0, self.num_dict['None']] = 1
                temp_t[np.sum(temp_t, axis=2) > 1] = 0
                #temp_t /= np.sum(temp_t, axis=2, keepdims=True)

                ret_target.append(temp_t)

        ret_data = np.array(ret_data, dtype=np.uint8)
        ret_target = np.array(ret_target)

        if augment:
            aug_data1, aug_target1 = self.__data_augment_brightness__(ret_data, ret_target, 1.5)
            aug_data2, aug_target2 = self.__data_augment_brightness__(ret_data, ret_target, 0.5)

            ret_data = np.vstack([ret_data, aug_data1, aug_data2])
            ret_target = np.vstack([ret_target, aug_target1, aug_target2])

            aug_data3, aug_target3 = self.__data_augment_color_reverse__(ret_data, ret_target)

            ret_data = np.vstack([ret_data, aug_data3])
            ret_target = np.vstack([ret_target, aug_target3])

        return ret_data, ret_target

    def get_data(self, root_path, augment=True):
        tr_data, tr_target, val_data, val_target, te_data, te_target = super().get_data_car_number(root_path)

        tr_data, tr_target = self.__data_processing__(tr_data, tr_target, augment=augment)
        val_data, val_target = self.__data_processing__(val_data, val_target, augment=False)
        te_data, te_target = self.__data_processing__(te_data, te_target, augment=False)

        self.tr_data, self.tr_target, self.val_data, self.val_target, self.te_data, self.te_target = \
            tr_data, tr_target, val_data, val_target, te_data, te_target

        return tr_data, tr_target, val_data, val_target, te_data, te_target

    def create_model(self, input_size):
        max_pool = 0
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_size,
                         kernel_regularizer=regularizers.l2(5*10**(-4)), bias_regularizer=regularizers.l2(5*10**(-4))))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_size,
                         kernel_regularizer=regularizers.l2(5*10**(-4)), bias_regularizer=regularizers.l2(5*10**(-4))))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(5*10**(-4)), bias_regularizer=regularizers.l2(5*10**(-4))))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(5*10**(-4)), bias_regularizer=regularizers.l2(5*10**(-4))))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(5*10**(-4)), bias_regularizer=regularizers.l2(5*10**(-4))))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(5*10**(-4)), bias_regularizer=regularizers.l2(5*10**(-4))))
        model.add(MaxPooling2D((2, 2)))

        model.add(Conv2D(1024, (5, 5), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(5*10**(-4)), bias_regularizer=regularizers.l2(5*10**(-4))))
        model.add(Dropout(0.5))
        model.add(Conv2D(1024, (1, 1), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(5*10**(-4)), bias_regularizer=regularizers.l2(5*10**(-4))))
        model.add(Dropout(0.5))
        model.add(Conv2D(len(self.num_dict), (1, 1), padding='valid', activation='softmax',
                         kernel_regularizer=regularizers.l2(5*10**(-4)), bias_regularizer=regularizers.l2(5*10**(-4))))

        self.model = model

        return model

    def save_model(self):
        save_model(self.model, './model/car_detect_model.h5')

    def __load_preset_model__(self):
        return load_model('./model/car_detect_model.h5', custom_objects={'loss': self.loss})

    def train_step(self, data=[], target=[], val_data=[], val_target=[], lr=0.0001, epoch=300):
        if len(data) == 0:
            data = self.tr_data
            target = self.tr_target

        if len(val_data) == 0:
            val_data = self.val_data
            val_target = self.val_target

        if self.model == None:
            self.create_model(data[0].shape)

        self.model.compile(optimizer=optimizers.Adam(lr), loss=self.loss)

        self.model.fit(data, target, validation_data=(val_data, val_target), batch_size=32, epochs=epoch)

    def loss(self, y_true, y_pred):
        c1 = 1
        c2 = 1
        eps = 10 ** (-6)

        term1 = -K.sum(y_true[:, :, :, 1:] * K.log(y_pred[:, :, :, 1:] + eps)) # number class
        term2 = -K.sum(y_true[:, :, :, 0:1] * K.log(y_pred[:, :, :, 0:1] + eps)) # None class

        return c1*term1 + c2*term2

    def predict_car_plate(self, d_type="train"):
        start = time.time()
        if d_type == "train":
            y_pred = self.model.predict(self.tr_data)
        elif d_type == "val":
            y_pred = self.model.predict(self.val_data)
        elif d_type == "test":
            y_pred = self.model.predict(self.te_data)
        print("%d ms for 1 sample" %(1000 * (time.time() - start) / self.val_data.shape[0]))

        return y_pred

    def most_common(self, src):
        c = Counter(list(src.flatten().astype(int)))
        c = sorted(c.items(), key=lambda x: x[1], reverse=True)

        return c[0][0]

if __name__ == "__main__":
    main_class = car_number_detector(pre_model=True)
    tr_data, tr_target, val_data, val_target, te_data, te_target = main_class.get_data('./data', augment=True)
# =============================================================================
#     main_class.create_model(tr_data[0].shape)
#     main_class.train_step(tr_data, tr_target, lr=0.0001, epoch=20)
#     main_class.train_step(tr_data, tr_target, lr=0.00001, epoch=20)
#     #main_class.train_step(tr_data, tr_target, lr=0.00001, epoch=30)
#
#     main_class.save_model()
# =============================================================================

    t = main_class.predict_car_plate(d_type="val")

    result_array = []
    for i, (im, y_pred, y_true) in enumerate(zip(val_data, t, val_target)):
        result = np.zeros((val_target[0].shape[0], val_target[0].shape[1]))
        xx, yy = np.meshgrid(np.arange(val_target[0].shape[0]), np.arange(val_target[0].shape[1]))
        for y, x in zip(xx.flatten(), yy.flatten()):
            temp_y = y_pred[y][x]
            y_t = np.argmax(y_true[y][x])
            idx = np.argwhere(temp_y >= 0.17)

            if idx.shape[0] > 0:
                idx = idx[0]

# =============================================================================
#             if y_t in idx:
#                 result[y][x] = y_t
#             else:
#                 result[y][x] = np.argmax(temp_y)
# =============================================================================

            result[y][x] = np.argmax(temp_y)

        k_size = 3
        temp_result = np.zeros(result.shape)
        for y, x in zip(xx.flatten(), yy.flatten()):
            temp_result[y][x] = main_class.most_common(result[ \
                    y - k_size // 2 if y > k_size // 2 else 0: \
                    y + k_size // 2 + 1 if y + k_size // 2 + 1 < val_target[0].shape[0] else val_target[0].shape[0], \
                    x - k_size // 2 if x > k_size // 2 else 0: \
                    x + k_size // 2 + 1 if x + k_size // 2 + 1 < val_target[0].shape[1] else val_target[0].shape[1]])

        result = temp_result.astype(np.uint8)
# =============================================================================
#         cv.medianBlur(result, ksize=3)
# =============================================================================
        result_array.append(result)
        result1 = cp.deepcopy(result)
        result1[result1 > 25] = 25
        result1 *= 10
        im1 = cv.resize(result1, (256, 128))
        dst = cv.addWeighted(im, 0.3, im1, 0.7, 0)
        cv.imshow(str(i), dst)

    result_array = np.array(result_array, dtype=np.uint8)

# =============================================================================
#     t_size = [256, 128]
#     for i in range(10):
#         im = tr_data[i]
#         t_im = np.argmax(tr_target[i], axis=-1).astype(np.uint8)
#         t_im = cv.resize(t_im, (t_size[0], t_size[1])).reshape((t_size[1], t_size[0], 1))
#         t_im *= 4
#         f_im = np.vstack([im, t_im]).astype(np.uint8)
#
#         cv.imshow(str(i), f_im)
#
# =============================================================================

