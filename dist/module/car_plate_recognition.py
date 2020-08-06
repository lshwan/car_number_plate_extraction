# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:14:26 2020

@author: LSH
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, MaxPooling2D, LeakyReLU
from tensorflow.keras.models import Sequential
import numpy as np
import cv2 as cv
import time
import os.path

class car_plate_detector():
    def __init__(self, gpu_only=False):
        self.__C__ = 16
        self.__im_size__ = [384, 512, 1]
        self.__p_thr__ = 0.26
        self.__iou_thr__ = 0.5
        self.gpu_only = gpu_only

        self.__model__ = self.__load_model__()

        return

    def __load_model__(self):
        assert os.path.isfile('../model/cpd_model_weight.h5'), 'No model available'

        model = self.__create_model__(self.__im_size__)

        if len(tf.config.list_physical_devices('GPU')) > 0 and self.gpu_only:
            device = '/gpu:0'
        else:
            device = '/cpu:0'

        with tf.device(device):
            model.load_weights('../model/cpd_model_weight.h5')

        return model

    def __create_model__(self, input_size):
        if len(tf.config.list_physical_devices('GPU')) > 0 and self.gpu_only:
            device = '/gpu:0'
        else:
            device = '/cpu:0'

        with tf.device(device):
            max_pool = 0
            model = Sequential()
            model.add(Conv2D(16, (3, 3), padding='same', activation='linear', input_shape=input_size, trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(Conv2D(16, (3, 3), padding='same', activation='linear', trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(MaxPooling2D((2, 2)))
            max_pool += 1

            model.add(Conv2D(32, (3, 3), padding='same', activation='linear', trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(Conv2D(32, (3, 3), padding='same', activation='linear', trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(MaxPooling2D((2, 2)))
            max_pool += 1

            model.add(Conv2D(64, (3, 3), padding='same', activation='linear', trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(MaxPooling2D((2, 2)))
            max_pool += 1

            feature_size = [input_size[0] // 2**max_pool, input_size[1] // 2**max_pool]
            model.add(Conv2D(1024, (feature_size[0] // self.__C__, feature_size[1] // self.__C__),
                             padding='valid', strides=(feature_size[0] // self.__C__, feature_size[1] // self.__C__), activation='linear', trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(Conv2D(5, (1, 1), padding='valid', activation='linear', trainable=False))

        return model

    def __union__(self, y_pred, y_true):
        _, x_p, y_p, w_p, h_p = y_pred
        _, x_t, y_t, w_t, h_t = y_true

        ret = self.__im_size__[1] * w_p * self.__im_size__[0] * h_p + \
            self.__im_size__[1] * w_t * self.__im_size__[0] * h_t

        ret -= self.__intersection__(y_pred, y_true)

        return ret

    def __intersection__(self, y_pred, y_true):
        _, x_p, y_p, w_p, h_p = y_pred
        _, x_t, y_t, w_t, h_t = y_true

        inter_x = min(x_p + w_p / 2, x_t + w_t / 2) - max(x_p - w_p / 2, x_t - w_t / 2)
        inter_x = 0 if inter_x < 0 else self.__im_size__[1] * inter_x

        inter_y = min(y_p + h_p / 2, y_t + h_t / 2) - max(y_p - h_p / 2, y_t - h_t / 2)
        inter_y = 0 if inter_y < 0 else self.__im_size__[0] * inter_y

        return inter_x * inter_y

    def __iou__(self, y_pred, y_true):
        return self.__intersection__(y_pred, y_true) / self.__union__(y_pred, y_true)

    def __preprocessing__(self, im):
        assert len(im.shape) >= 3, 'Please input image'

        if im.shape[2] > 1:
            im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)

        im = cv.resize(im, (self.__im_size__[1], self.__im_size__[0]))
        im = cv.medianBlur(im, 3)
        im = im.reshape((1, im.shape[0], im.shape[1], 1))

        return im

    def __nmu__(self, y_pred, p_thr, iou_thr=0.5):
        y_pred[y_pred[:, :, :, 0] < p_thr, 0] = 0

        for i, y in enumerate(y_pred):
            y_list = []

            p_list = np.sort(y[:, :, 0].flatten())[::-1]
            p_list[p_list < p_thr]= 0

            for p in p_list:
                if p == 0:
                    break

                idx_x, idx_y = np.argwhere(y[:, :, 0] == p)[0]
                temp_p = y[idx_x, idx_y]
                temp_p[1] += idx_y
                temp_p[1] /= self.__C__
                temp_p[2] += idx_x
                temp_p[2] /= self.__C__

                if len(y_list) == 0:
                    y_list.append(temp_p)
                else:
                    for temp_y in y_list:
                        if self.__iou__(temp_p, temp_y) > iou_thr:
                            y_x, y_y = np.argwhere(y[:, :, 0] == temp_y[0])[0]
                            y_pred[i, idx_x, idx_y, 0] = 0
                            y_pred[i, y_x, y_y, 3] = np.max([temp_p[1] + temp_p[3] / 2, temp_y[1] + temp_y[3] / 2]) - \
                                                        np.min([temp_p[1] - temp_p[3] / 2, temp_y[1] - temp_y[3] / 2])

                            y_pred[i, y_x, y_y, 4] = np.max([temp_p[2] + temp_p[4] / 2, temp_y[2] + temp_y[4] / 2]) - \
                                                    np.min([temp_p[2] - temp_p[4] / 2, temp_y[2] - temp_y[4] / 2])

                            break
                    else:
                        y_list.append(temp_p)

        return y_pred

    def __out2box__(self, out, orginal_shape):
        bbox = []
        for i, j in np.argwhere(out[:, :, 0] > self.__p_thr__):
            temp_out = out[i][j]
            temp_out[1] += j
            temp_out[1] /= self.__C__
            temp_out[2] += i
            temp_out[2] /= self.__C__

            temp_out[1] *= orginal_shape[1]
            temp_out[3] *= orginal_shape[1]
            temp_out[2] *= orginal_shape[0]
            temp_out[4] *= orginal_shape[0]

            bbox.append(temp_out[1:5].astype(int))

        return bbox

    def detect_car_plate(self, im, out_time=False):
        if out_time:
            start = time.time()

        if len(tf.config.list_physical_devices('GPU')) > 0 and self.gpu_only:
            device = '/gpu:0'
        else:
            device = '/cpu:0'

        with tf.device(device):
            y = self.__model__.predict(self.__preprocessing__(im))

        y = self.__nmu__(y, p_thr=self.__p_thr__, iou_thr=self.__iou_thr__)

        bbox = self.__out2box__(y[0], im.shape)

        if out_time:
            out_time = 1000 * (time.time() - start)

        if out_time:
            return bbox, out_time
        else:
            return bbox

    def draw_plate_box(self, im):
        bbox, out_time = self.detect_car_plate(im, out_time=True)

        for box in bbox:
            p1 = (box[0] - box[2] // 2, box[1] - box[3] // 2)
            p2 = (box[0] + box[2] // 2, box[1] + box[3] // 2)

            cv.rectangle(im, p1, p2, (255, 0, 0), 2)

        cv.imshow('image', im)

        return out_time

class car_plate_recognition():
    def __init__(self, gpu_only=False):
        self.__cpd__ = car_plate_detector(gpu_only=gpu_only)

        self.__num_dict__ = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
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

        self.__char_dict__ = self.__num_to_char__()
        self.__C__ = np.array([8, 4], ndmin=2)
        self.__im_size__ = [128, 256, 1]
        self.__p_thr__ = 0.3
        self.__iou_thr__ = 0.5
        self.__class_num_thr__ = 0.3
        self.__class_char_thr__ = 0.1

        self.gpu_only = gpu_only

        self.__model__ = self.__load_model__()

        return

    def __load_model__(self):
        assert os.path.isfile('../model/cpr_model_weight.h5'), 'No model available'

        model = self.__create_model__(self.__im_size__)

        if len(tf.config.list_physical_devices('GPU')) > 0 and self.gpu_only:
            device = '/gpu:0'
        else:
            device = '/cpu:0'

        with tf.device(device):
            model.load_weights('../model/cpr_model_weight.h5')

        return model

    def __num_to_char__(self):
        dic = dict()
        for key, value in self.__num_dict__.items():
            dic.update({value: key})

        return dic

    def __preprocessing__(self, im):
        assert len(im.shape) >= 3, 'Please input image'

        if im.shape[2] > 1:
            im = cv.cvtColor(im, cv.COLOR_RGB2GRAY)

        im = cv.resize(im, (self.__im_size__[1], self.__im_size__[0]))
        im = im.reshape((1, im.shape[0], im.shape[1], 1))

        return im

    def __create_model__(self, input_size):
        if len(tf.config.list_physical_devices('GPU')) > 0 and self.gpu_only:
            device = '/gpu:0'
        else:
            device = '/cpu:0'

        with tf.device(device):
            max_pool = 0
            model = Sequential()
            model.add(Conv2D(64, (3, 3), padding='same', activation='linear', input_shape=input_size, trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(Conv2D(64, (3, 3), padding='same', activation='linear', input_shape=input_size, trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(Conv2D(96, (3, 3), padding='same', activation='linear', input_shape=input_size, trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(MaxPooling2D((2, 2)))
            max_pool += 1

            model.add(Conv2D(128, (3, 3), padding='same', activation='linear', trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(Conv2D(128, (3, 3), padding='same', activation='linear', trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(Conv2D(192, (3, 3), padding='same', activation='linear', trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(MaxPooling2D((2, 2)))
            max_pool += 1

            model.add(Conv2D(256, (3, 3), padding='same', activation='linear', trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(Conv2D(256, (3, 3), padding='same', activation='linear', trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(Conv2D(384, (3, 3), padding='same', activation='linear', trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(MaxPooling2D((2, 2)))
            max_pool += 1

            feature_size = [input_size[0] // 2**max_pool, input_size[1] // 2**max_pool]
            model.add(Conv2D(4096, (feature_size[0] // self.__C__[0, 1], feature_size[1] // self.__C__[0, 0]),
                             padding='valid', strides=(feature_size[0] // self.__C__[0, 1], feature_size[1] // self.__C__[0, 0]), activation='linear', trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(Conv2D(2 * (5 + len(self.__num_dict__)), (1, 1), padding='valid', activation='linear', trainable=False))

        return model

    def __union__(self, y_pred, y_true):
        _, x_p, y_p, w_p, h_p = y_pred
        _, x_t, y_t, w_t, h_t = y_true

        ret = self.__im_size__[1] * w_p * self.__im_size__[0] * h_p + \
            self.__im_size__[1] * w_t * self.__im_size__[0] * h_t

        ret -= self.__intersection__(y_pred, y_true)

        return ret

    def __intersection__(self, y_pred, y_true):
        _, x_p, y_p, w_p, h_p = y_pred
        _, x_t, y_t, w_t, h_t = y_true

        inter_x = min(x_p + w_p / 2, x_t + w_t / 2) - max(x_p - w_p / 2, x_t - w_t / 2)
        inter_x = 0 if inter_x < 0 else self.__im_size__[1] * inter_x

        inter_y = min(y_p + h_p / 2, y_t + h_t / 2) - max(y_p - h_p / 2, y_t - h_t / 2)
        inter_y = 0 if inter_y < 0 else self.__im_size__[0] * inter_y

        return inter_x * inter_y

    def __iou__(self, y_pred, y_true):
        return self.__intersection__(y_pred, y_true) / self.__union__(y_pred, y_true)

    def __nms__(self, y_pred, p_thr, iou_thr=0.5, class_num_thr=0.5, class_char_thr=0.5):
        for i in range(y_pred.shape[-1] // (5 + len(self.__num_dict__))):
            y_pred[y_pred[:, :, :, i * (5 + len(self.__num_dict__))] < p_thr, i * (5 + len(self.__num_dict__))] = 0

        for i, y in enumerate(y_pred):
            y_list = [[] for _ in range(len(self.__num_dict__))]

            p_total = y[:, :, ::5 + len(self.__num_dict__)]
            p_total[p_total < p_thr]= 0

            p_sorted = np.sort(p_total.flatten())[::-1]

            for p in p_sorted:
                if p == 0:
                    break

                idx_x, idx_y, idx_i = np.argwhere(p_total == p)[0]
                y_cls = y[idx_x, idx_y, idx_i * (5 + len(self.__num_dict__)) + 5: (idx_i + 1) * (5 + len(self.__num_dict__))]
                y_cls[:10][y_cls[:10] < class_num_thr] = 0
                y_cls[10:][y_cls[10:] < class_char_thr] = 0

                if np.max(y_cls) > 0:
                    clss = np.argmax(y_cls)
                else:
                    y_pred[i, idx_x, idx_y, idx_i * (5 + len(self.__num_dict__))] = 0

                    continue


                if clss == len(self.__num_dict__) - 1:
                    y_pred[i, idx_x, idx_y, idx_i * (5 + len(self.__num_dict__))] = 0

                    continue

                temp_p = y[idx_x, idx_y, idx_i * (5 + len(self.__num_dict__)): idx_i * (5 + len(self.__num_dict__)) + 5]
                temp_p[1] += idx_y
                temp_p[1] /= self.__C__[0, 0]
                temp_p[2] += idx_x
                temp_p[2] /= self.__C__[0, 1]

                if len(y_list[clss]) == 0:
                    y_list[clss].append(temp_p)
                else:
                    for temp_y in y_list[clss]:
                        if self.__iou__(temp_p, temp_y) > iou_thr:
                            y_pred[i, idx_x, idx_y, idx_i * (5 + len(self.__num_dict__))] = 0

                            break
                    else:
                        y_list[clss].append(temp_p)

        return y_pred

    def __out2box__(self, out):
        bbox = []
        for i, j in np.argwhere(out[:, :, 0] > self.__p_thr__):
            temp_out = out[i][j]
            temp_out[1] += j
            temp_out[1] /= self.__C__[0, 0]
            temp_out[2] += i
            temp_out[2] /= self.__C__[0, 1]

            temp_out[1] *= self.__im_size__[1]
            temp_out[3] *= self.__im_size__[1]
            temp_out[2] *= self.__im_size__[0]
            temp_out[4] *= self.__im_size__[0]

            temp_out[5] = np.argmax(temp_out[5:], axis=-1)

            bbox.append(temp_out[1:6].astype(int))

        return bbox

    def __car_number_extraction__(self, y_pred):
        y_len = 5 + len(self.__num_dict__)

        bbox = []
        for k in range(y_pred.shape[-1] // y_len):
            bbox.extend(self.__out2box__(y_pred[0, :, :, k * y_len: (k + 1) * y_len]))

        car_cord = sorted(bbox, key=lambda x: x[0])

        car_num = ''
        if len(car_cord) > 0:
            car_cord = np.array(car_cord)

            for n in car_cord[:,-1]:
                car_num += self.__char_dict__[int(n)]

        return car_num

    def car_plate_recognition(self, im, out_time=False):
        if out_time:
            start = time.time()

        pbox = self.__cpd__.detect_car_plate(im)

        if len(tf.config.list_physical_devices('GPU')) > 0 and self.gpu_only:
            device = '/gpu:0'
        else:
            device = '/cpu:0'

        car_num = []
        car_box = []
        for i, box in enumerate(pbox):
            y_s = box[1] - box[3] // 2
            if y_s < 0:
                y_s = 0

            y_e = box[1] + box[3] // 2
            if y_e > im.shape[0]:
                y_e = im.shape[0]

            x_s = box[0] - box[2] // 2
            if x_s < 0:
                x_s = 0

            x_e = box[0] + box[2] // 2
            if x_e > im.shape[1]:
                x_e = im.shape[1]

            temp_im = im[y_s: y_e, x_s: x_e, :]

# =============================================================================
#             cv.imshow(str(i), temp_im)
# =============================================================================

            with tf.device(device):
                y = self.__model__.predict(self.__preprocessing__(temp_im))

            y = self.__nms__(y, p_thr=self.__p_thr__, iou_thr=self.__iou_thr__, class_num_thr=self.__class_num_thr__, class_char_thr=self.__class_char_thr__)

            num = self.__car_number_extraction__(y)

            if len(num) >= 5:
                car_box.append(box)
                car_num.append(num)

        if out_time:
            out_time = 1000 * (time.time() - start)

        if out_time:
            return car_box, car_num, out_time
        else:
            return car_box, car_num

if __name__ == "__main__":
    cpr = car_plate_recognition()

    im = cv.imread('D:\\Programming\\Python\\My Project\\Car_Plate_Number_Extract\\data\\0\\20200526\\20200526181342034.jpg')

    car_box, car_num, t = cpr.car_plate_recognition(im, out_time=True)

    for box in car_box:
        cv.rectangle(im,
                     (box[0] - box[2] // 2, box[1] - box[3] // 2),
                     (box[0] + box[2] // 2, box[1] + box[3] // 2),
                     (255, 0, 0),
                     2)

    cv.imshow('test', im)
