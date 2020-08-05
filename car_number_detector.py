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
                         '천': 54, '전': 55, '북': 56}
        self.char_dict = self.__num_to_char__()

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

    def __data_processing__(self, data, target, augment=True):
        ret_data = []
        ret_target = []

        t_size = [256, 128]
        div = 8
        for n, (d, t) in enumerate(zip(data, target)):
            for plate_cord, car_num, char_cord in zip(t[0], t[1], t[2]):
                temp_im = d[plate_cord[1] - plate_cord[3] // 2: plate_cord[1] + plate_cord[3] // 2,
                            plate_cord[0] - plate_cord[2] // 2: plate_cord[0] + plate_cord[2] // 2]

                temp_t = np.zeros((self.C[0, 1], self.C[0, 0], 2 * (5 + len(self.num_dict))))

                h, w, _ = temp_im.shape

                temp_im = cv.resize(temp_im, (t_size[0], t_size[1]))
                temp_im = temp_im.reshape((t_size[1], t_size[0], 1))

                im_size = np.array([temp_im.shape[1], temp_im.shape[0]], ndmin=2)

                char_cord = np.array(char_cord, dtype=int)
                if len(char_cord) > 0:
                    char_cord[:, 0] = (t_size[0] * char_cord[:, 0]) // w
                    char_cord[:, 1] = (t_size[1] * char_cord[:, 1]) // h
                    char_cord[:, 2] = (t_size[0] * char_cord[:, 2]) // w
                    char_cord[:, 3] = (t_size[1] * char_cord[:, 3]) // h

                    idx = (self.C * char_cord[:,:2]) // im_size
                    char_cord = char_cord.astype(float)
                    char_cord[:,:2] = (char_cord[:,:2] % (im_size // self.C)) / (im_size // self.C)
                    char_cord[:,2:] /= im_size


                    for i, (j, k) in enumerate(idx):
                        class_char = np.zeros(len(self.num_dict))
                        class_char[self.num_dict[car_num[i]]] = 1

                        if temp_t[k, j, 0] == 0:
                            temp_t[k, j, :5 + len(self.num_dict)] = np.hstack([[1], char_cord[i], class_char])
                        elif temp_t[k, j, 5 + len(self.num_dict)] == 0:
                            temp_t[k, j, 5 + len(self.num_dict):] = np.hstack([[1], char_cord[i], class_char])

# =============================================================================
#                     temp_t[temp_t[:, :, 0] == 0, 5 + len(self.num_dict) - 1] = 1
# =============================================================================
                ret_data.append(temp_im)
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
        model.add(Conv2D(64, (3, 3), padding='same', activation='linear', input_shape=input_size,
                         kernel_regularizer=regularizers.l2(10**(-3)), bias_regularizer=regularizers.l2(10**(-3))))
        model.add(LeakyReLU(0.01))
        model.add(Conv2D(64, (3, 3), padding='same', activation='linear', input_shape=input_size,
                         kernel_regularizer=regularizers.l2(10**(-3)), bias_regularizer=regularizers.l2(10**(-3))))
        model.add(LeakyReLU(0.01))
        model.add(Conv2D(96, (3, 3), padding='same', activation='linear', input_shape=input_size,
                         kernel_regularizer=regularizers.l2(10**(-3)), bias_regularizer=regularizers.l2(10**(-3))))
        model.add(LeakyReLU(0.01))
        model.add(MaxPooling2D((2, 2)))
        max_pool += 1
        model.add(Conv2D(128, (3, 3), padding='same', activation='linear',
                         kernel_regularizer=regularizers.l2(10**(-3)), bias_regularizer=regularizers.l2(10**(-3))))
        model.add(LeakyReLU(0.01))
        model.add(Conv2D(128, (3, 3), padding='same', activation='linear',
                         kernel_regularizer=regularizers.l2(10**(-3)), bias_regularizer=regularizers.l2(10**(-3))))
        model.add(LeakyReLU(0.01))
        model.add(Conv2D(192, (3, 3), padding='same', activation='linear',
                         kernel_regularizer=regularizers.l2(10**(-3)), bias_regularizer=regularizers.l2(10**(-3))))
        model.add(LeakyReLU(0.01))
        model.add(MaxPooling2D((2, 2)))
        max_pool += 1
        model.add(Conv2D(256, (3, 3), padding='same', activation='linear',
                         kernel_regularizer=regularizers.l2(10**(-3)), bias_regularizer=regularizers.l2(10**(-3))))
        model.add(LeakyReLU(0.01))
        model.add(Conv2D(256, (3, 3), padding='same', activation='linear',
                         kernel_regularizer=regularizers.l2(10**(-3)), bias_regularizer=regularizers.l2(10**(-3))))
        model.add(LeakyReLU(0.01))
        model.add(Conv2D(384, (3, 3), padding='same', activation='linear',
                         kernel_regularizer=regularizers.l2(10**(-3)), bias_regularizer=regularizers.l2(10**(-3))))
        model.add(LeakyReLU(0.01))
        model.add(MaxPooling2D((2, 2)))
        max_pool += 1

        feature_size = [input_size[0] // 2**max_pool, input_size[1] // 2**max_pool]
        model.add(Conv2D(4096, (feature_size[0] // self.C[0, 1], feature_size[1] // self.C[0, 0]), padding='valid', strides=(feature_size[0] // self.C[0, 1], feature_size[1] // self.C[0, 0]), activation='linear',
                         kernel_regularizer=regularizers.l2(5*10**(-4)), bias_regularizer=regularizers.l2(5*10**(-4))))
        model.add(LeakyReLU(0.01))
        model.add(Dropout(0.5))
# =============================================================================
#         model.add(Conv2D(2048, (1, 1), padding='valid', activation='linear',
#                          kernel_regularizer=regularizers.l2(5*10**(-4)), bias_regularizer=regularizers.l2(5*10**(-4))))
#         model.add(LeakyReLU(0.01))
#         model.add(Dropout(0.5))
# =============================================================================
        model.add(Conv2D(2 * (5 + len(self.num_dict)), (1, 1), padding='valid', activation='linear',
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
        c3 = 1
        c4 = 1
        c5 = 1

        term1 = 0
        term2 = 0
        term3 = 0
        term4 = 0
        term5 = 0

        for i in range(2):
            base = i * (5 + len(self.num_dict))

            term1 += K.sum(
                K.square(
                    y_true[:, :, :, 0 + base: 1 + base] * (y_true[:, :, :, 0 + base: 3 + base] - y_pred[:, :, :, 0 + base: 3 + base])
                    )) # exist: p, x, y

# =============================================================================
#             term2 += K.sum(
#                 K.square(
#                     y_true[:, :, :, 0 + base: 1 + base] * (y_true[:, :, :, 3 + base: 5 + base] - y_pred[:, :, :, 3 + base: 5 + base])
#                     )) # exist: w, h
# =============================================================================
            term2 += K.sum(
                K.square(
                    y_true[:, :, :, 0 + base: 1 + base] * (K.sqrt(y_true[:, :, :, 3 + base: 5 + base]) - K.sqrt(y_pred[:, :, :, 3 + base: 5 + base]))
                    )) # exist: w, h

            term3 += K.sum(
                K.square(
                    y_true[:, :, :, 0 + base: 1 + base] * (y_true[:, :, :, 5 + base: 15 + base] - y_pred[:, :, :, 5 + base: 15 + base])
                    )) # exist: num class

            term4 += K.sum(
                K.square(
                    y_true[:, :, :, 0 + base: 1 + base] * (y_true[:, :, :, 15 + base:5 + len(self.num_dict) + base] - y_pred[:, :, :, 15 + base: 5 + len(self.num_dict) + base])
                    )) # exist: char class

            term5 += K.sum(
                K.square(
                    (1 - y_true[:, :, :, 0 + base: 1 + base]) * (y_true[:, :, :, 0 + base: 1 + base] - y_pred[:, :, :, 0 + base: 1 + base]))) # non_exist: p

        return c1*term1 + c2*term2 + c3*term3 + c4*term4 + c5*term5

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

    def draw_plate_box(self, im, target, p_thr):
        for n, x, y in zip(np.arange(im.shape[0]), im, target):
            xx, yy = np.meshgrid(np.arange(self.C[0, 1]), np.arange(self.C[0, 0]))

            for i, j in zip(xx.flatten(), yy.flatten()):
                t_a = y[i, j]

                for idx in range(2):
                    if t_a[idx * (5 + len(self.num_dict))] > p_thr:
                        p1 = (
                            int((x.shape[1] // self.C[0, 0]) * (j + t_a[1 + idx * (5 + len(self.num_dict))]) - (x.shape[1] // 1) * t_a[3 + idx * (5 + len(self.num_dict))] // 2),
                            int((x.shape[0] // self.C[0, 1]) * (i + t_a[2 + idx * (5 + len(self.num_dict))]) - (x.shape[0] // 1) * t_a[4 + idx * (5 + len(self.num_dict))] // 2))

                        p2 = (
                            int((x.shape[1] // self.C[0, 0]) * (j + t_a[1 + idx * (5 + len(self.num_dict))]) + (x.shape[1] // 1) * t_a[3 + idx * (5 + len(self.num_dict))] // 2),
                            int((x.shape[0] // self.C[0, 1]) * (i + t_a[2 + idx * (5 + len(self.num_dict))]) + (x.shape[0] // 1) * t_a[4 + idx * (5 + len(self.num_dict))] // 2))

                        cv.rectangle(x, p1, p2, (255, 0, 0), int(3*t_a[idx * (5 + len(self.num_dict))]) + 1)

            cv.imshow(str(n), x)

    def __union__(self, y_pred, y_true):
        _, x_p, y_p, w_p, h_p = y_pred
        _, x_t, y_t, w_t, h_t = y_true

        inter_x = max(x_p + w_p / 2, x_t + w_t / 2) - min(x_p - w_p / 2, x_t - w_t / 2)

        return inter_x

    def __intersection__(self, y_pred, y_true):
        _, x_p, y_p, w_p, h_p = y_pred
        _, x_t, y_t, w_t, h_t = y_true

        inter_x = min(x_p + w_p / 2, x_t + w_t / 2) - max(x_p - w_p / 2, x_t - w_t / 2)

        return inter_x

    def __iou__(self, y_pred, y_true):
        return self.__intersection__(y_pred, y_true) / self.__union__(y_pred, y_true)

    def nms(self, y_pred, p_thr, iou_thr=0.5, class_num_thr=0.5, class_char_thr=0.5):
        y_pred = cp.deepcopy(y_pred)

        for i in range(y_pred.shape[-1] // (5 + len(self.num_dict))):
            y_pred[y_pred[:, :, :, i * (5 + len(self.num_dict))] < p_thr, i * (5 + len(self.num_dict))] = 0

        for i, y in enumerate(y_pred):
            y_list = [[] for _ in range(len(self.num_dict))]

            p_total = y[:, :, ::5 + len(self.num_dict)]
            p_total[p_total < p_thr]= 0

            p_sorted = np.sort(p_total.flatten())[::-1]

            for p in p_sorted:
                if p == 0:
                    break

                idx_x, idx_y, idx_i = np.argwhere(p_total == p)[0]
                y_cls = y[idx_x, idx_y, idx_i * (5 + len(self.num_dict)) + 5: (idx_i + 1) * (5 + len(self.num_dict))]
                y_cls[:10][y_cls[:10] < class_num_thr] = 0
                y_cls[10:][y_cls[10:] < class_char_thr] = 0

                if np.max(y_cls) > 0:
                    clss = np.argmax(y_cls)
                else:
                    y_pred[i, idx_x, idx_y, idx_i * (5 + len(self.num_dict))] = 0

                    continue


                if clss == len(self.num_dict) - 1:
                    y_pred[i, idx_x, idx_y, idx_i * (5 + len(self.num_dict))] = 0

                    continue

                temp_p = cp.deepcopy(y[idx_x, idx_y, idx_i * (5 + len(self.num_dict)): idx_i * (5 + len(self.num_dict)) + 5])
                temp_p[1] += idx_y
                temp_p[1] /= self.C[0, 0]
                temp_p[2] += idx_x
                temp_p[2] /= self.C[0, 1]

                if len(y_list[clss]) == 0:
                    y_list[clss].append(temp_p)
                else:
                    for temp_y in y_list[clss]:
                        if self.__iou__(temp_p, temp_y) > iou_thr:
                            y_pred[i, idx_x, idx_y, idx_i * (5 + len(self.num_dict))] = 0

                            break
                    else:
                        y_list[clss].append(temp_p)

        return y_pred

    def evaluate(self, y_pred, y_true, clss, p_thr=0.5, iou_thr=0.5):
        assert y_pred.shape[0] == y_true.shape[0], 'y_pred size shall equal to y_true'

        y_pred = cp.deepcopy(y_pred)
        y_true = cp.deepcopy(y_true)

        ret_iou = []
        for p, t in zip(y_pred, y_true):
            temp_eval = [0, 0, 0] #[TP, FP, FN]
            for k in range(p.shape[-1] // (5 + len(self.num_dict))):
                temp_p_cls = p[:, :, k * (5 + len(self.num_dict)) + 5: (k + 1) * (5 + len(self.num_dict))]
                temp_p_cls = np.argmax(temp_p_cls, axis=-1)

                for i, j in np.argwhere(p[:, :, k * (5 + len(self.num_dict))] > p_thr):
                    matched = 0

                    if temp_p_cls[i, j] != clss:
                        continue

                    temp_p = p[i, j, k * (5 + len(self.num_dict)): k * (5 + len(self.num_dict)) + 5]
                    temp_p[1] += j
                    temp_p[1] /= self.C[0, 0]
                    temp_p[2] += i
                    temp_p[2] /= self.C[0, 1]

                    for l in range(t.shape[-1] // (5 + len(self.num_dict))):
                        if matched == 1:
                            break

                        temp_t_cls = t[:, :, l * (5 + len(self.num_dict)) + 5: (l + 1) * (5 + len(self.num_dict))]
                        temp_t_cls = np.argmax(temp_t_cls, axis=-1)

                        for x, y in np.argwhere(t[:, :, l * (5 + len(self.num_dict))] > 0):
                            if temp_t_cls[i, j] != clss:
                                continue

                            temp_t = t[x, y, l * (5 + len(self.num_dict)): l * (5 + len(self.num_dict)) + 5]
                            temp_t[1] += y
                            temp_t[1] /= self.C[0, 0]
                            temp_t[2] += x
                            temp_t[2] /= self.C[0, 1]

                            if self.__iou__(temp_p, temp_t) > iou_thr:
                                t[x, y, l * (5 + len(self.num_dict))] = 0.5
                                temp_eval[0] += 1

                                matched = 1

                                break
                    else:
                        if matched == 0:
                            temp_eval[1] += 1

            t[t == 0.5] = 0
            temp_eval[2] = 0
            for k in range(t.shape[-1] // (5 + len(self.num_dict))):
                temp_cls = t[:, :, k * (5 + len(self.num_dict)) + 5: (k + 1) * (5 + len(self.num_dict))]
                temp_eval[2] += np.count_nonzero(t[:, :, k * (5 + len(self.num_dict))][np.argmax(temp_cls, axis=-1) == clss])
            ret_iou.append(temp_eval)

        ret_iou = np.array(ret_iou)
        TP, FP, FN = np.sum(ret_iou, axis=0)
        precision = TP / (TP + FP) if TP + FP != 0 else 0
        recall = TP / (TP + FN) if TP + FN != 0 else 0

        return precision, recall, ret_iou #Precision, Recall

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
        car_num = []

        for y in y_pred:
            car_cord = []
            for k in range(y.shape[-1] // (5 + len(self.num_dict))):
                for ind_x, ind_y in np.argwhere(y[:, :, k * (5 + len(self.num_dict))] > 0):
                    temp_y = cp.deepcopy(y[ind_x, ind_y, k * (5 + len(self.num_dict)): (k + 1) * (5 + len(self.num_dict))])
                    temp_y[5] = np.argmax(temp_y[5: len(self.num_dict)], axis=-1)
                    temp_y = temp_y[:6]
                    temp_y[1] += ind_y
                    temp_y[1] /= self.C[0, 0]
                    temp_y[2] += ind_x
                    temp_y[2] /= self.C[0, 1]

                    car_cord.append(temp_y[1:])

            car_cord = sorted(car_cord, key=lambda x: x[0])
            car_cord = np.array(car_cord)
            num = ''
            for n in car_cord[:,-1]:
                num += self.char_dict[int(n)]
            car_num.append(num)

        return car_num

if __name__ == "__main__":
    main_class = car_number_detector(pre_model=True)
    tr_data, tr_target, val_data, val_target, te_data, te_target = main_class.get_data('./data', augment=True)
# =============================================================================
#     main_class.draw_plate_box(val_data, val_target, p_thr=0.5)
# =============================================================================
# =============================================================================
#     main_class.create_model(tr_data[0].shape)
#     main_class.train_step(tr_data, tr_target, lr=0.0001, epoch=20)
#     main_class.train_step(tr_data, tr_target, lr=0.00001, epoch=15)
#
#     main_class.save_model()
# =============================================================================

    thr = 0.3
    with tf.device('/cpu:0'):
        t = main_class.predict_car_plate(d_type="val")
# =============================================================================
#     p, r, x = main_class.draw_roc(t, val_target)
# =============================================================================
    t = main_class.nms(t, p_thr=thr, iou_thr=0.4, class_num_thr=0, class_char_thr=0)

    car_pred = main_class.car_number_extraction(t)

    main_class.draw_plate_box(val_data, t, p_thr=thr)

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

