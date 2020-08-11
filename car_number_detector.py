# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 16:26:42 2020

@author: LSH
"""

import tensorflow as tf
from tensorflow.keras import regularizers, activations, optimizers
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, Dropout, LeakyReLU, TimeDistributed, LSTM, SimpleRNN, Reshape, Bidirectional
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

        self.num_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
                         '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
                         '한': 10, 'None': 11, 'SP': 12}
# =============================================================================
#         self.num_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
#                          '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
#                          '가': 10, '거': 11, '고': 12, '구': 13,
#                          '나': 14, '너': 15, '노': 16, '누': 17,
#                          '다': 18, '더': 19, '도': 20, '두': 21,
#                          '라': 22, '러': 23, '로': 24, '루': 25,
#                          '마': 26, '머': 27, '모': 28, '무': 29,
#                          '바': 30, '버': 31, '보': 32, '부': 33,
#                          '사': 34, '서': 35, '소': 36, '수': 37,
#                          '아': 38, '어': 39, '오': 40, '우': 41,
#                          '자': 42, '저': 43, '조': 44, '주': 45,
#                          '하': 46, '허': 47, '호': 48, '배': 49,
#                          '울': 50, '경': 51, '기': 52, '인': 53,
#                          '천': 54, '전': 55, '북': 56, 'None': 57, 'SP': 58}
# =============================================================================
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

        t_size = [256, 128]

        for n, (d, t) in enumerate(zip(data, target)):
            num_proc, cord_proc = self.__target_preprocessing__(t)

            for plate_cord, car_num_proc, char_cord_proc in zip(t[0], num_proc, cord_proc):
                temp_im = d[plate_cord[1] - plate_cord[3] // 2: plate_cord[1] + plate_cord[3] // 2,
                            plate_cord[0] - plate_cord[2] // 2: plate_cord[0] + plate_cord[2] // 2]

                temp_t = np.zeros((t_size[0] // self.div, 2 * len(self.num_dict)))
                matched = np.zeros((t_size[0] // self.div, 2))

                h, w = temp_im.shape[0], temp_im.shape[1]

                temp_im = cv.resize(temp_im, (t_size[0], t_size[1]))
                temp_im = cv.rotate(temp_im, cv.ROTATE_90_CLOCKWISE)
                temp_im = temp_im.reshape((temp_im.shape[0], temp_im.shape[1], 1))

                im_x, im_y, _ = temp_im.shape

                for i, (car_num, char_cord) in enumerate(zip(car_num_proc, char_cord_proc)):
                    last_e_x = 0

                    if n == 252:
                        n = 252

                    for num, cord in zip(car_num, char_cord):
                        if not num.isdigit():
                            num = '한'
                        s_x, s_y, c_w, c_h = cord
                        s_x = (s_x * im_x) // (self.div * w)
                        c_w = int((c_w * im_x) / (self.div * w))

                        if s_x - c_w // 4 >= s_x:
                            temp_s_x = s_x
                        else:
                            temp_s_x = s_x - c_w // 4

                        temp_e_x = s_x + c_w // 4 + 1

                        temp_sum = np.sum(temp_t[temp_s_x: temp_e_x, i * len(self.num_dict): (i + 1) * len(self.num_dict)], axis=-1)
                        idx_sp = np.argwhere(temp_sum > 0)

                        if idx_sp.shape[0] > 0:
                            temp_t[temp_s_x + (idx_sp.shape[0] - 1) // 2: temp_e_x, i * len(self.num_dict): (i + 1) * len(self.num_dict)] = 0
                            temp_t[temp_s_x + (idx_sp.shape[0] - 1) // 2, i * len(self.num_dict) + self.num_dict['SP']] = 1

                            if s_x - c_w // 4 + (idx_sp.shape[0] - 1) // 2 + 1 >= s_x:
                                temp_s_x = s_x - c_w // 4 + (idx_sp.shape[0] - 1) // 2 + 1
                            else:
                                temp_s_x = s_x

                            temp_t[temp_s_x: temp_e_x, i * len(self.num_dict) + self.num_dict[num]] = 1
                            cur_s_x = temp_s_x
                        else:
                            temp_t[s_x: s_x + c_w // 4 + 1, i * len(self.num_dict) + self.num_dict[num]] = 1

                            cur_s_x = s_x

                        if last_e_x != 0:
                            temp_sum = np.sum(temp_t[last_e_x: cur_s_x, i * len(self.num_dict): (i + 1) * len(self.num_dict)], axis=-1)
                            temp_t[last_e_x: cur_s_x, :][temp_sum == 0, i * len(self.num_dict) + self.num_dict['SP']] = 1

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

        ret_data = np.array(ret_data, dtype=np.float32)
        ret_target = np.array(ret_target, dtype=np.float32)

        if augment:
            aug_data3, aug_target3 = self.__data_augment_color_reverse__(ret_data, ret_target)

            ret_data = np.vstack([ret_data, aug_data3])
            ret_target = np.vstack([ret_target, aug_target3])

            aug_data1, aug_target1 = self.__data_augment_brightness__(ret_data, ret_target, 1.5)
            aug_data2, aug_target2 = self.__data_augment_brightness__(ret_data, ret_target, 0.5)

            ret_data = np.vstack([ret_data, aug_data1, aug_data2])
            ret_target = np.vstack([ret_target, aug_target1, aug_target2])

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
                         kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
# =============================================================================
#         model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
#                          kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
# =============================================================================

        model.add(MaxPooling2D((2, 2)))
        max_pool += 1
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
# =============================================================================
#         model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
#                          kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
# =============================================================================

        model.add(MaxPooling2D((2, 2)))
        max_pool += 1

# =============================================================================
#         model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
#                          kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
# =============================================================================
# =============================================================================
#         model.add(Conv2D(256, (3, 3), padding='same', activation='relu',
#                          kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
# =============================================================================

# =============================================================================
#         model.add(MaxPooling2D((2, 2)))
#         max_pool += 1
#
# =============================================================================
        feature_size = [input_size[0] // 2**max_pool, input_size[1] // 2**max_pool]

        model.add(TimeDistributed(Flatten()))
        model.add(TimeDistributed(Reshape((-1, 1))))

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))
        model.add(Conv2D(1, (3, 3), padding='same', activation='relu',
                         kernel_regularizer=regularizers.l2(10**(-4)), bias_regularizer=regularizers.l2(10**(-4))))

        model.add(TimeDistributed(Flatten()))

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

    def loss(self, y_true, y_pred):
        c1 = 1
        c2 = 0.9
        c3 = 1

        term1 = 0
        term2 = 0
        term3 = 0

        for i in range(2):
            base = i * len(self.num_dict)

            y_output = K.softmax(y_pred[:, :, i * len(self.num_dict): (i + 1) * len(self.num_dict)], axis=-1)

            term1 -= K.sum(y_true[:, :, self.num_dict['None'] + base: self.num_dict['None'] + 1 + base] * K.log(y_output[:, :, self.num_dict['None']: self.num_dict['None'] + 1]))

            term2 -= K.sum(y_true[:, :, self.num_dict['SP'] + base: self.num_dict['SP'] + 1 + base] * K.log(y_output[:, :, self.num_dict['SP']: self.num_dict['SP'] + 1]))

            term3 -= K.sum(y_true[:, :, base: self.num_dict['None'] + base] * K.log(y_output[:, :, :self.num_dict['None']]))
# =============================================================================
#             term2 -= term1
# =============================================================================

        return c1 * term1 + c2 * term2 + c3 * term3

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

    def evaluate(self, y_pred, y_true):
        assert y_pred.shape[0] == y_true.shape[0], 'y_pred size shall equal to y_true'

        car_num_pred = self.car_number_extraction(y_pred)
        car_num_true = self.car_number_extraction(y_true)

        correct = 0
        in_correct = []

        for i, (p, t) in enumerate(zip(car_num_pred, car_num_true)):
            #if p[1][-4:] == t[1][-4:]:
            if p[0] == t[0] and p[1] == t[1]:
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

    def car_number_extraction(self, y_pred):
        logit = np.exp(y_pred)
        line1 = logit[:, :, :logit.shape[2] // 2] / np.sum(logit[:, :, :logit.shape[2] // 2], axis=-1, keepdims=True)
        line2 = logit[:, :, logit.shape[2] // 2:] / np.sum(logit[:, :, logit.shape[2] // 2:], axis=-1, keepdims=True)

        line1 = np.argmax(line1, axis=-1)
        line2 = np.argmax(line2, axis=-1)

        char_line = []
        for l1, l2 in zip(line1, line2):
            temp_char_line1 = ""
            temp_char_line2 = ""
            last_char = ""
            for l in l1:
                if self.char_dict[l] != 'None':
                    if self.char_dict[l] == 'SP':
                        last_char = ""
                    elif self.char_dict[l] != last_char:
                        temp_char_line1 += self.char_dict[l]
                        last_char = self.char_dict[l]

            last_char = ""
            for l in l2:
                if self.char_dict[l] != 'None':
                    if self.char_dict[l] == 'SP':
                        last_char = ""
                    elif self.char_dict[l] != last_char:
                        temp_char_line2 += self.char_dict[l]
                        last_char = self.char_dict[l]

            char_line.append([temp_char_line1, temp_char_line2])

        return char_line

if __name__ == "__main__":
    main_class = car_number_detector(pre_model=True)
    tr_data, tr_target, val_data, val_target, te_data, te_target = main_class.get_data('./data', augment=True)
# =============================================================================
#     main_class.draw_plate_box(val_data, val_target, p_thr=0.5)
# =============================================================================
    main_class.create_model(tr_data[0].shape)
    main_class.train_step(tr_data, tr_target, lr=0.0001, epoch=7)
    main_class.train_step(tr_data, tr_target, lr=0.00001, epoch=5)

# =============================================================================
#     main_class.save_model()
# =============================================================================

    thr = 0.3
    t = main_class.predict_car_plate(d_type="val")

    car_num = main_class.car_number_extraction(t)
    acc, inc_idx = main_class.evaluate(t, val_target)
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

