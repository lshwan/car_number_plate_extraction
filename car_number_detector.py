# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 16:26:42 2020

@author: LSH
"""

import tensorflow as tf
from tensorflow.keras import regularizers, activations, optimizers
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, LeakyReLU, TimeDistributed, LSTM, SimpleRNN, Reshape, Bidirectional, GRU
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
import matplotlib.pyplot as plt
import pickle

class car_number_detector(utils.annotator.annotator):
    def __init__(self, seed=0, pre_model=False):
        super().__init__(seed)

# =============================================================================
#         self.num_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
#                          '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
#                          '한': 10, 'None': 11, 'SP': 12}
# =============================================================================
        self.num_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                         '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                         '가': 10, '거': 11, '고': 12, '구': 13,
                         '나': 14, '너': 15, '노': 16, '누': 17,
                         '다': 18, '더': 19, '도': 20, '두': 21,
                         '라': 22, '러': 23, '로': 24, '루': 25,
                         '마': 26, '머': 27, '모': 28, '무': 29,
                         '바': 30, '버': 31, '보': 32, '부': 33,
                         '사': 34, '서': 35, '소': 36, '수': 37,
                         '아': 38, '어': 39, '오': 40, '우': 41,
                         '자': 42, '저': 43, '조': 44, '주': 45,
                         '하': 46, '허': 47, '호': 48, '배': 49,
                         '울': 50, '경': 51, '기': 52, '인': 53,
                         '천': 54, '전': 55, '북': 56, 'None': 57, 'SP': 58}
        self.char_dict = self.__num_to_char__()
        self.div = 4

        if pre_model:
            self.model = self.__load_preset_model__()

        self.C = np.array([8, 4], ndmin=2)
        return

    def __num_to_char__(self):
        dic = dict()
        for key, value in self.num_dict.items():
            dic.update({value: key})

        return dic

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

    def __data_augment_flipud__(self, data, target):
        aug_data = cp.deepcopy(np.flip(data, axis=2))
        aug_target = cp.deepcopy(target)
        temp = cp.deepcopy(aug_target[:, :, :len(self.num_dict)])
        aug_target[:, :, :len(self.num_dict)] = aug_target[:, :, len(self.num_dict):]
        aug_target[:, :, len(self.num_dict):] = temp

        return aug_data, aug_target

    def __target_preprocessing__(self, target):
        car_num_list = []
        char_cord_list = []

        for plate_cord, car_num, char_cord in zip(target[0], target[1], target[2]):
            temp_car_num = ["", ""]
            temp_char_cord = [[], []]

            x1, y1 = char_cord[0][0], char_cord[0][1]
            x2, y2 = char_cord[-1][0], char_cord[-1][1]

            slope = (y2 - y1) / (x2 - x1)
            b = y1 - x1 * slope

            line = lambda x: slope * x + b

            if abs(char_cord[0][0] - char_cord[1][0]) < char_cord[0][2] / 4:
                temp_car_num[0] += car_num[0]
                temp_char_cord[0].extend(char_cord[0:1])

                temp_car_num[1] += car_num[1:]
                temp_char_cord[1].extend(char_cord[1:])
            else:
                matched = 0
                for char_box, char in zip(char_cord, car_num):
                    if char_box[1] - char_box[3] // 2 > line(char_box[0]) or line(char_box[0]) > char_box[1] + char_box[3] // 2:
                        matched = 1

                        break

                if matched == 1:
                    line_num = 0
                    x_p = 0

                    for char_box, char in zip(char_cord, car_num):
                        if x_p > char_box[0] and line_num == 0:
                            line_num = 1
                        x_p = char_box[0]
                        temp_car_num[line_num] += char
                        temp_char_cord[line_num].append(char_box)
                else:
                    temp_car_num[1] += car_num
                    temp_char_cord[1].extend(char_cord)

            car_num_list.append(temp_car_num)
            char_cord_list.append(temp_char_cord)

        return car_num_list, char_cord_list

    def __data_processing__(self, data, target, augment=True):
        ret_data = []
        ret_target = []
        ret_car_num = []

        t_size = [512, 128]

        r_number_y1 = 0.5 * np.random.rand(len(data))
        r_number_y2 = np.hstack([r_number_y1[1:], r_number_y1[0]])

        for n, (d, t, r1, r2) in enumerate(zip(data, target, r_number_y1, r_number_y2)):
            num_proc, cord_proc = self.__target_preprocessing__(t)

            if len(num_proc) == 0:
                temp_im = d[d.shape[0] // 2 - 32: d.shape[0] // 2 + 32, d.shape[1] // 2 - 64: d.shape[1] // 2 + 64]
                temp_im = cv.resize(temp_im, (t_size[0], t_size[1]))
                temp_im = cv.rotate(temp_im, cv.ROTATE_90_CLOCKWISE)
                temp_im = temp_im.reshape((temp_im.shape[0], temp_im.shape[1], 1))

                temp_t = np.zeros((t_size[0] // self.div, 2 * len(self.num_dict)))
                for i in range(2):
                    temp_t[:, i * len(self.num_dict) + self.num_dict['None']] = 1
                temp_car_num = ['', '']

                ret_data.append(temp_im)
                ret_target.append(temp_t)
                ret_car_num.append(temp_car_num)

                continue

            for plate_cord, car_num_proc, char_cord_proc in zip(t[0], num_proc, cord_proc):
                if augment:
                    idx_s = int(plate_cord[1] - ((1 + r1) * plate_cord[3]) // 2)
                    if idx_s < 0:
                        idx_s = 0

                    idx_e = int(plate_cord[1] + ((1 + r2) * plate_cord[3]) // 2)

                    temp_im = d[idx_s: idx_e,
                                plate_cord[0] - plate_cord[2] // 2: plate_cord[0] + plate_cord[2] // 2]
                else:
                    temp_im = d[plate_cord[1] - plate_cord[3] // 2: plate_cord[1] + plate_cord[3] // 2,
                                plate_cord[0] - plate_cord[2] // 2: plate_cord[0] + plate_cord[2] // 2]

                temp_t = np.zeros((t_size[0] // self.div, 2 * len(self.num_dict)))
                matched = np.zeros((t_size[0] // self.div, 2))
                temp_car_num = ['', '']

                h, w = temp_im.shape[0], temp_im.shape[1]

                temp_im = cv.resize(temp_im, (t_size[0], t_size[1]))
                temp_im = cv.rotate(temp_im, cv.ROTATE_90_CLOCKWISE)
                temp_im = temp_im.reshape((temp_im.shape[0], temp_im.shape[1], 1))

                im_x, im_y, _ = temp_im.shape

                for i, (car_num, char_cord) in enumerate(zip(car_num_proc, char_cord_proc)):
                    last_e_x = 0

                    if n == 252:
                        n = 252

                    for j, (num, cord) in enumerate(zip(car_num, char_cord)):
# =============================================================================
#                         if not num.isdigit():
#                             num = '한'
# =============================================================================

                        temp_car_num[i] += num

                        s_x, s_y, c_w, c_h = cord
                        s_x = (s_x * im_x) // (self.div * w)
                        c_w = int((c_w * im_x) / (self.div * w))

                        if s_x - c_w // 3 >= s_x:
                            temp_s_x = s_x
                        else:
                            temp_s_x = s_x - c_w // 3

                        temp_e_x = s_x + c_w // 3 + 1

                        temp_sum = np.sum(temp_t[temp_s_x: temp_e_x, i * len(self.num_dict): (i + 1) * len(self.num_dict)], axis=-1)
                        idx_sp = np.argwhere(temp_sum > 0)

                        if idx_sp.shape[0] > 0:
                            temp_t[temp_s_x + (idx_sp.shape[0] - 1) // 2: temp_e_x, i * len(self.num_dict): (i + 1) * len(self.num_dict)] = 0
                            temp_t[temp_s_x + (idx_sp.shape[0] - 1) // 2, i * len(self.num_dict) + self.num_dict['SP']] = 1

                            last_e_x = temp_s_x + (idx_sp.shape[0] - 1) // 2

                            if temp_s_x + (idx_sp.shape[0] - 1) // 2 + 1 >= s_x:
                                temp_s_x = temp_s_x + (idx_sp.shape[0] - 1) // 2 + 1
                            else:
                                temp_s_x = s_x

# =============================================================================
#                             temp_s_x = temp_s_x + (idx_sp.shape[0] - 1) // 2 + 1
# =============================================================================

                            temp_t[temp_s_x: temp_e_x, i * len(self.num_dict) + self.num_dict[num]] = 1
                            cur_s_x = temp_s_x
                        else:
                            temp_t[s_x: temp_e_x, i * len(self.num_dict) + self.num_dict[num]] = 1

                            cur_s_x = s_x

                        if last_e_x != 0:
                            temp_sum = np.sum(temp_t[last_e_x: cur_s_x, i * len(self.num_dict): (i + 1) * len(self.num_dict)], axis=-1)
                            temp_t[last_e_x: cur_s_x, :][temp_sum == 0, i * len(self.num_dict) + self.num_dict['SP']] = 1

                        if j + 1 < len(char_cord):
                            if temp_e_x  >= t_size[0] // self.div:
                                temp_t[-1, i * len(self.num_dict) + self.num_dict['SP']] = 1
                                last_e_x = -1
                            else:
                                temp_t[temp_e_x, i * len(self.num_dict) + self.num_dict['SP']] = 1
                                last_e_x = temp_e_x

                    temp_idx = np.sum(temp_t[:, i * len(self.num_dict): (i + 1) * len(self.num_dict)], axis=-1) == 0
                    temp_t[temp_idx, i * len(self.num_dict) + self.num_dict['None']] = 1

# =============================================================================
#                     temp_idx = np.sum(temp_t[:, i * len(self.num_dict): (i + 1) * len(self.num_dict)], axis=-1) > 1
#                     temp_t[temp_idx, i * len(self.num_dict): (i + 1) * len(self.num_dict)] = 0
#                     temp_t[temp_idx, i * len(self.num_dict) + self.num_dict['SP']] = 1
# =============================================================================

                ret_data.append(temp_im)
                ret_target.append(temp_t)
                ret_car_num.append(temp_car_num)

        ret_data = np.array(ret_data, dtype=np.float32)
        ret_target = np.array(ret_target, dtype=np.float32)

        if augment:
            aug_data4, aug_target4 = self.__data_augment_flipud__(ret_data, ret_target)

            ret_data = np.vstack([ret_data, aug_data4])
            ret_target = np.vstack([ret_target, aug_target4])

            aug_data3, aug_target3 = self.__data_augment_color_reverse__(ret_data, ret_target)

            ret_data = np.vstack([ret_data, aug_data3])
            ret_target = np.vstack([ret_target, aug_target3])

            aug_data1, aug_target1 = self.__data_augment_brightness__(ret_data, ret_target, 2)
            aug_data2, aug_target2 = self.__data_augment_brightness__(ret_data, ret_target, 0.25)

            ret_data = np.vstack([ret_data, aug_data1, aug_data2])
            ret_target = np.vstack([ret_target, aug_target1, aug_target2])

        return ret_data, ret_target, ret_car_num

    def get_data(self, root_path, augment=True):
        tr_data, tr_target, val_data, val_target, te_data, te_target = super().get_data_car_number(root_path)

        tr_data, tr_target, tr_raw_target = self.__data_processing__(tr_data, tr_target, augment=augment)
        val_data, val_target, val_raw_target = self.__data_processing__(val_data, val_target, augment=False)
        te_data, te_target, te_raw_target = self.__data_processing__(te_data, te_target, augment=False)

        self.tr_data, self.tr_target, self.val_data, self.val_target, self.te_data, self.te_target = \
            tr_data, tr_target, val_data, val_target, te_data, te_target

        return tr_data, tr_target, val_data, val_target, te_data, te_target, tr_raw_target, val_raw_target, te_raw_target

    def create_model(self, input_size):
        max_pool = 0
        model = Sequential()
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_size,
                             kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu',
                             kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
        model.add(MaxPooling2D((2, 2)))
        max_pool += 1

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                             kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                             kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
        model.add(MaxPooling2D((2, 2)))
        max_pool += 1

        feature_size = [input_size[0] // 2**max_pool, input_size[1] // 2**max_pool]

        model.add(TimeDistributed(Reshape((-1, 1))))

        model.add(Conv2D(64, (5, 5), strides=(1, 2), padding='same', activation='relu',
                             kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                             kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
        model.add(MaxPooling2D((2, 2), strides=(1, 2), padding='same'))

        model.add(Conv2D(128, (5, 5), strides=(1, 2), padding='same', activation='relu',
                             kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
                             kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
        model.add(MaxPooling2D((2, 2), strides=(1, 2), padding='same'))

        model.add(Conv2D(256, (5, 5), strides=(1, 2), padding='same', activation='relu',
                             kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
                             kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
        model.add(MaxPooling2D((2, 2), strides=(1, 2), padding='same'))

        model.add(Conv2D(8, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))

        model.add(TimeDistributed(Flatten()))

        model.add(TimeDistributed(Dense(1024, activation='relu',
                        kernel_regularizer=regularizers.l2(10**(-5)), bias_regularizer=regularizers.l2(10**(-5)))))
        model.add(Dropout(0.5))

        model.add(TimeDistributed(Dense(1024, activation='relu',
                         kernel_regularizer=regularizers.l2(10**(-5)), bias_regularizer=regularizers.l2(10**(-5)))))
        model.add(Dropout(0.5))

        model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True,
                       kernel_regularizer=regularizers.l2(10**(-5)), bias_regularizer=regularizers.l2(10**(-5)))))
        model.add(Dropout(0.5))

        model.add(TimeDistributed(Dense(2 * len(self.num_dict), activation='linear',
                         kernel_regularizer=regularizers.l2(10**(-5)), bias_regularizer=regularizers.l2(10**(-5)))))

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

        batch = 16

        self.model.fit(data, target,
                       validation_data=(val_data, val_target),
                       batch_size=batch, epochs=epoch,
                       steps_per_epoch=data.shape[0] // batch, shuffle=True)

    def get_weight(self, num_dict):
        key_list = list(self.num_dict.keys())
        w_list = [0 for _ in range(len(key_list))]

        with open('./utils/weight.txt', 'r') as f:
            for l in f.readlines():
                key, w = l.split()
                w_list[key_list.index(key)] = 100 * float(w)
# =============================================================================
#                 w_list[0][key_list.index(key)] = 120 * float(w)
# =============================================================================

        w_list[-2] = w_list[0]
# =============================================================================
#         w_list[0][-2] = 0.5 * w_list[0][0]
# =============================================================================
        w_list[-1] = 0.7 * w_list[0]
# =============================================================================
#         w_list[0][-1] = 0.7 * w_list[0][0]
# =============================================================================

        return w_list

    def loss(self, y_true, y_pred):
        c = self.get_weight(self.num_dict)
        out = 0

        eps = 0

        for i in range(2):
            base = i * len(self.num_dict)
            y_output = K.softmax(y_pred[:, :, i * len(self.num_dict): (i + 1) * len(self.num_dict)], axis=-1)

            for idx in range(len(self.num_dict)):
                out -= c[idx] * K.sum(y_true[:, :, base + idx: base + idx + 1] * K.log(y_output[:, :, idx: idx + 1] + eps))

        return out

    def predict_car_plate(self, d_type="train"):
        start = time.time()

        if d_type == "train":
            y_pred = [self.model.predict(d.reshape(1, d.shape[0], d.shape[1], d.shape[2]))[0] for d in self.tr_data]
            #y_pred = self.model.predict(self.tr_data)
            print("%d ms for 1 sample" %(1000 * (time.time() - start) / self.tr_data.shape[0]))
        elif d_type == "val":
            y_pred = [self.model.predict(d.reshape(1, d.shape[0], d.shape[1], d.shape[2]))[0] for d in self.val_data]
            #y_pred = self.model.predict(self.val_data)
            print("%d ms for 1 sample" %(1000 * (time.time() - start) / self.val_data.shape[0]))
        elif d_type == "test":
            y_pred = [self.model.predict(d.reshape(1, d.shape[0], d.shape[1], d.shape[2]))[0] for d in self.te_data]
            #y_pred = self.model.predict(self.te_data)
            print("%d ms for 1 sample" %(1000 * (time.time() - start) / self.te_data.shape[0]))

        return np.array(y_pred)

    def evaluate(self, y_pred, y_true):
        car_num_pred, _, cnt = self.car_number_extraction(y_pred)
# =============================================================================
#         car_num_pred, _ = self.post_processing(car_num_pred, cnt)
# =============================================================================
        car_num_true = y_true# self.car_number_extraction(y_true)

        assert len(car_num_pred) == len(car_num_true), 'y_pred size shall equal to y_true'

        correct = 0
        in_correct = []

        for i, (p, t) in enumerate(zip(car_num_pred, car_num_true)):
            if p[1][-4:] == t[1][-4:]:
            #if p[0] == t[0] and p[1] == t[1]:
                correct += 1
            else:
                in_correct.append(i)

        return correct / y_pred.shape[0], np.array(in_correct)

    def draw_roc(self, y_pred, y_true):
        import matplotlib.pyplot as plt

        precision = [[] for _ in range(len(self.num_dict))]
        recall = [[] for _ in range(len(self.num_dict))]
        for i in range(len(self.num_dict)):
            for thr in np.arange(100) / 100:
                p, r, _ = self.evaluate(y_pred, y_true, i, p_thr=thr)

                precision[i].append(p)
                recall[i].append(r)

            plt.figure()
            plt.plot(recall[i], precision[i])
            plt.ylabel("precision")
            plt.xlabel("recall")

        return precision, recall, np.arange(100) / 100

    def num_extract(self, line, line_p):
        temp_char_line = ""
        temp_cnt_line = []

        last_char = ""
        temp_cnt = 0
        max_p = 0
        for i, (l, p) in enumerate(zip(line, line_p)):
            if self.char_dict[l] != 'None':
                if self.char_dict[l] == 'SP':
                    last_char = ""

                    if temp_cnt > 0:
                        temp_cnt_line.append(temp_cnt)

                    temp_cnt = 0
                    max_p = 0
                elif self.char_dict[l] != last_char:
                    if last_char == "":
                        temp_char_line += self.char_dict[l]
                        last_char = self.char_dict[l]

                        if temp_cnt > 0:
                            temp_cnt_line.append(temp_cnt)

                        temp_cnt = 1
                        max_p = p
                    elif max_p < p:
                        temp_char_line = temp_char_line[:-1] + self.char_dict[l]
                        last_char = self.char_dict[l]

                        temp_cnt += 1
                        max_p = p
                else:
                    if p > max_p:
                        max_p = p
                    temp_cnt += 1
            else:
                if temp_cnt > 0:
                    temp_cnt_line.append(temp_cnt)
                last_char = ""
                temp_cnt = 0
                max_p = 0

        return temp_char_line, temp_cnt_line

    def car_number_extraction(self, y_pred):
        logit = np.exp(y_pred)
        line1 = logit[:, :, :logit.shape[2] // 2] / np.sum(logit[:, :, :logit.shape[2] // 2], axis=-1, keepdims=True)
        line2 = logit[:, :, logit.shape[2] // 2:] / np.sum(logit[:, :, logit.shape[2] // 2:], axis=-1, keepdims=True)

        line1_p = np.max(line1, axis=-1)
        line1 = np.argmax(line1, axis=-1)

        ret_line1 = np.zeros((line1.shape[0], self.div * line1.shape[1]))

        line2_p = np.max(line2, axis=-1)
        line2 = np.argmax(line2, axis=-1)
        ret_line2 = np.zeros((line1.shape[0], self.div * line1.shape[1]))

        for i in range(self.div):
            ret_line1[:, i::self.div] = line1
            ret_line2[:, i::self.div] = line2

        char_line = []
        char_cnt = []
        for j, (l1, p1, l2, p2) in enumerate(zip(line1, line1_p, line2, line2_p)):
            if j == 97:
                j = j
            temp_char_line1, temp_cnt_line1 = self.num_extract(l1, p1)
            temp_char_line2, temp_cnt_line2 = self.num_extract(l2, p2)

            char_line.append([temp_char_line1, temp_char_line2])
            char_cnt.append([temp_cnt_line1, temp_cnt_line2])

        return char_line, np.stack([ret_line1, ret_line2], axis=1), char_cnt

    def post_processing(self, car_char, car_cnt):
        ret_char = []
        ret_cnt = []

        for idx, (char, cnt) in enumerate(zip(car_char, car_cnt)):
            if idx == 97:
                idx = idx

            temp_char = ["", ""]
            temp_cnt = [[], []]

            temp_char[0] = char[0]
            temp_cnt[0] = cnt[0]

            p = -1
            line_char2 = char[1][::-1]
            line_cnt2 = np.array(cnt[1][::-1])

            sorted_line2 = np.sort(line_cnt2)
            median_cnt = np.median(sorted_line2)

# =============================================================================
#             line_cnt2 = np.delete(line_cnt2, save_i)
# =============================================================================

# =============================================================================
#             upper_thr = (3 * median_cnt) // 2
#             lower_thr = (2 * median_cnt) // 3
# =============================================================================

            for i, c in enumerate(line_char2):
                if self.num_dict['9'] < self.num_dict[c] < self.num_dict['None']:
                    p = i

                    length = i

                    break

            if p == -1:
                length = i + 1

# =============================================================================
#             else:
#                 flag = 1
#                 p_len = -1
#                 for i in range(p, len(line_char2)):
#                     if line_char2[i] == "한":
#                         if flag == 1:
#                             line_char2[i] = " "
#                             line_cnt2[i] = 0
#                     else:
#                         p_len += 1
#                         flag = 0
# =============================================================================

            for i in range(len(line_cnt2) - 1, -1, -1):
                c = line_cnt2[i]

                if c >= 3 * median_cnt:
                    if length != 4:
                        c_len = line_cnt2[i]

                        while c_len > 1.5 * median_cnt:
                            c_len -= median_cnt

                            line_cnt2[i] = c_len

                            line_char2 = line_char2[:i + 1] + line_char2[i] + line_char2[i + 1:]
                            line_cnt2 = np.hstack([line_cnt2[:i + 1], [median_cnt], line_cnt2[i + 1:]])

                elif c <= median_cnt // 3:
                    if i - 1 >= 0 and line_char2[i - 1] == line_char2[i]:
                        line_cnt2[i - 1] += (c + 1)
                        line_char2 = line_char2[:i] + line_char2[i + 1:]
                        line_cnt2 = np.hstack([line_cnt2[:i], line_cnt2[i + 1:]])
                    elif i + 1 < len(line_char2) and line_char2[i + 1] == line_char2[i]:
                        line_cnt2[i + 1] += (c + 1)
                        line_char2 = line_char2[:i] + line_char2[i + 1:]
                        line_cnt2 = np.hstack([line_cnt2[:i], line_cnt2[i + 1:]])


            for i, c in enumerate(line_char2):
                if self.num_dict['9'] < self.num_dict[c] < self.num_dict['None']:
                    p = i

                    length = i

                    break

            if p == -1:
                length = i + 1

            if length == 4:
                temp_char[1] = line_char2[::-1]
                temp_cnt[1] = line_cnt2[::-1]
            elif length < 4:
                median_cnt = np.median(sorted_line2[:length - 4])
                upper_thr = 15 * median_cnt / 10

                temp_len = length
                for _ in range(4 - length):
                    max_cnt = np.max(line_cnt2[:temp_len])
                    max_i = np.argmax(line_cnt2[:temp_len])

# =============================================================================
#                     if max_cnt <= upper_thr:
#                         break
# =============================================================================

                    line_cnt2[max_i] = median_cnt

                    line_char2 = line_char2[:max_i] + line_char2[max_i] + line_char2[max_i:]
                    line_cnt2 = np.hstack([line_cnt2[:max_i], [max_cnt - median_cnt], line_cnt2[max_i:]])

                    temp_len += 1

                temp_char[1] = line_char2[::-1]
                temp_cnt[1] = list(line_cnt2[::-1])
            elif length > 4:
                median_cnt = np.median(sorted_line2[length - 4:])
                lower_thr = 10 * np.median(median_cnt) / 15

                temp_len = length
                for _ in range(length - 4):
                    min_cnt = np.min(line_cnt2[:temp_len])
                    min_i = np.argmin(line_cnt2[:temp_len])

# =============================================================================
#                     if min_cnt >= lower_thr:
#                         break
# =============================================================================

                    if min_i == temp_len - 1:
                        line_char2 = line_char2[:min_i]
                        line_cnt2 = line_cnt2[:min_i]
                    else:
                        line_char2 = line_char2[:min_i] + line_char2[min_i + 1:]
                        line_cnt2 = np.hstack([line_cnt2[:min_i], line_cnt2[min_i + 1:]])

                    temp_len -= 1

                temp_char[1] = line_char2[::-1]
                temp_cnt[1] = list(line_cnt2[::-1])

            ret_char.append(temp_char)
            ret_cnt.append(temp_cnt)

        return ret_char, ret_cnt

if __name__ == "__main__":
    main_class = car_number_detector(pre_model=True)
    tr_data, tr_target, val_data, val_target, te_data, te_target, tr_car_num, val_car_num, te_car_num = main_class.get_data('./data', augment=True)
# =============================================================================
#     main_class.draw_plate_box(val_data, val_target, p_thr=0.5)
# =============================================================================
    main_class.create_model(tr_data[0].shape)
    main_class.train_step(tr_data, tr_target, lr=0.0001, epoch=10)
    main_class.train_step(tr_data, tr_target, lr=0.00001, epoch=2)

# =============================================================================
#     main_class.save_model()
# =============================================================================

    thr = 0.3

    t = main_class.predict_car_plate(d_type="val")

    car_num, line, line_cnt = main_class.car_number_extraction(t)
# =============================================================================
#     car_num, line_cnt = main_class.post_processing(car_num, line_cnt)
# =============================================================================
    acc, inc_idx = main_class.evaluate(t, val_car_num)


    for i in inc_idx:
# =============================================================================
#         cv.imshow(str(i), cv.rotate(val_data[i].astype(np.uint8), cv.ROTATE_90_COUNTERCLOCKWISE))
# =============================================================================
        plt.figure(num=i, figsize=(9, 15))
        ax = plt.subplot(311)
        plt.imshow(cv.rotate(val_data[i].astype(np.uint8), cv.ROTATE_90_COUNTERCLOCKWISE))
        plt.subplot(312, sharex=ax)
        plt.plot(line[i][0])
        plt.subplot(313, sharex=ax)
        plt.plot(line[i][1])
# =============================================================================
#     p, r, x = main_class.draw_roc(t, val_target)
# =============================================================================
# =============================================================================
#     t = main_class.nms(t, p_thr=thr, iou_thr=0.4, class_num_thr=0, class_char_thr=0)
#
#     car_pred = main_class.car_number_extraction(t)
#
#     main_class.draw_plate_box(val_data, t, p_thr=thr)
#
# =============================================================================
# =============================================================================
#     y_pred = cp.deepcopy(t)
#
#     contour = []
#     xx, yy = np.meshgrid(np.arange(main_class.C[0, 0]), np.arange(main_class.C[0, 1]))
#     for i, j in zip(xx.flatten(), yy.flatten()):
#         x, y = np.meshgrid(np.arange(i, i + 2), np.arange(j, j + 2))
#         x[0, 0], x[0, 1] = x[0, 1], x[0, 0]
#         mesh = (np.array([tr_data.shape[2], tr_data.shape[1]], ndmin=2) * np.array([x.flatten(),  y.flatten()]).T) // main_class.C[0]
#
#         contour.append(mesh.reshape((-1, 1, 2)))
#
#     for i in range(val_data.shape[0]):
#         im = val_data[i]
#         t = y_pred[i]
#
#         for j, k in zip(xx.flatten(), yy.flatten()):
#             t_a = t[k][j]
#
#             if t_a[0] > 0.5:
# # =============================================================================
# #                 cv.circle(im, (
# #                     int((t_a[1] + k) * (im.shape[1] // main_class.C)),
# #                     int((t_a[2] + j) * (im.shape[0] // main_class.C))),
# #                     3,
# #                     (0, 0, 0),
# #                     1)
# # =============================================================================
#
#                 cv.rectangle(im, (
#                     int((t_a[1] + j) * (im.shape[1] // main_class.C[0, 0]) - (t_a[3] * im.shape[1]) // 2),
#                     int((t_a[2] + k) * (im.shape[0] // main_class.C[0, 1]) - (t_a[4] * im.shape[0]) // 2)),
#                     (
#                     int((t_a[1] + j) * (im.shape[1] // main_class.C[0, 0]) + (t_a[3] * im.shape[1]) // 2),
#                     int((t_a[2] + k) * (im.shape[0] // main_class.C[0, 1]) + (t_a[4] * im.shape[0]) // 2)),
#                     (0, 0, 0),
#                     1)
#
# # =============================================================================
# #         for c in contour:
# #             cv.drawContours(im, [c], 0, [255, 0, 0], 1)
# #
# # =============================================================================
#         cv.imshow(str(i), im)
# =============================================================================

