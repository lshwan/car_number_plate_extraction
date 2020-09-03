# -*- coding: utf-8 -*-
"""
Created on Thu Aug  6 14:14:26 2020

@author: LSH
"""

import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, LeakyReLU, TimeDistributed, LSTM, Reshape, Bidirectional
from tensorflow.keras.models import Sequential
import numpy as np
import cv2 as cv
import time
import os.path

class car_plate_detector():
    def __init__(self, gpu_only=False, model_path='../model/'):
        self.__C__ = 16
        self.__im_size__ = [384, 512, 1]
        self.__p_thr__ = 0.26
        self.__iou_thr__ = 0.3
        self.gpu_only = gpu_only

        self.__model__ = self.__load_model__(model_path)

        return

    def __load_model__(self, path):
        assert os.path.isfile(path + 'cpd_model_weight.h5'), 'No model available'

        model = self.__create_model__(self.__im_size__)

        if len(tf.config.list_physical_devices('GPU')) > 0 and self.gpu_only:
            device = '/gpu:0'
        else:
            device = '/cpu:0'

        with tf.device(device):
            model.load_weights(path + 'cpd_model_weight.h5')

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
            im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

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

    def __nms__(self, y_pred):
        for i, y in enumerate(y_pred):
            y_list = []

            p_list = np.sort(y[:, :, 0].flatten())[::-1]

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
                        if self.__iou__(temp_p, temp_y) > 0:
                            y_x, y_y = np.argwhere(y[:, :, 0] == temp_y[0])[0]
                            y_pred[i, idx_x, idx_y, 0] = 0

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
        y = self.__nms__(y)

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
    def __init__(self, gpu_only=False, model_path='../model/'):
        self.__cpd__ = car_plate_detector(gpu_only=gpu_only, model_path=model_path)

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
                         '천': 54, '전': 55, '북': 56, 'None': 57, 'SP': 58}

        self.__region_char__ = ['서', '울', '경', '기', '인', '천', '전', '북']

        self.__im_size__ = [512, 128, 1]

        self.gpu_only = gpu_only

        self.__model__ = self.__load_model__(model_path)

        self.__char_dict__ = self.__num_to_char__()
        self.__div__ = 4

        return

    def __num_to_char__(self):
        dic = dict()
        for key, value in self.__num_dict__.items():
            dic.update({value: key})

        return dic

    def __load_model__(self, path):
        assert os.path.isfile(path + 'cpr_model_weight.h5'), 'No model available'

        model = self.__create_model__(self.__im_size__)

        if len(tf.config.list_physical_devices('GPU')) > 0 and self.gpu_only:
            device = '/gpu:0'
        else:
            device = '/cpu:0'

        with tf.device(device):
            model.load_weights(path + 'cpr_model_weight.h5')

        return model

    def __preprocessing__(self, im):
        assert len(im.shape) >= 3, 'Please input image'

        if im.shape[2] > 1:
            im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

        im = cv.resize(im, (self.__im_size__[0], self.__im_size__[1]))
        im = cv.rotate(im, cv.ROTATE_90_CLOCKWISE)
        im = im.reshape((1, im.shape[0], im.shape[1], 1))

        return im

    def __create_model__(self, input_size):
        if len(tf.config.list_physical_devices('GPU')) > 0 and self.gpu_only:
            device = '/gpu:0'
        else:
            device = '/cpu:0'

        with tf.device(device):
            model = Sequential()
            model.add(Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_size, trainable=False))
            model.add(Conv2D(32, (3, 3), padding='same', activation='relu', trainable=False))
            model.add(MaxPooling2D((2, 2)))

            model.add(Conv2D(64, (3, 3), padding='same', activation='relu', trainable=False))
            model.add(Conv2D(64, (3, 3), padding='same', activation='relu', trainable=False))
            model.add(MaxPooling2D((2, 2)))

            model.add(TimeDistributed(Reshape((-1, 1))))

            model.add(Conv2D(64, (5, 5), strides=(1, 2), padding='same', activation='relu', trainable=False))
            model.add(Conv2D(64, (3, 3), padding='same', activation='relu', trainable=False))
            model.add(MaxPooling2D((2, 2), strides=(1, 2), padding='same'))

            model.add(Conv2D(128, (5, 5), strides=(1, 2), padding='same', activation='relu', trainable=False))
            model.add(Conv2D(128, (3, 3), padding='same', activation='relu', trainable=False))
            model.add(MaxPooling2D((2, 2), strides=(1, 2), padding='same'))

            model.add(Conv2D(256, (5, 5), strides=(1, 2), padding='same', activation='relu', trainable=False))
            model.add(Conv2D(256, (3, 3), padding='same', activation='relu', trainable=False))
            model.add(MaxPooling2D((2, 2), strides=(1, 2), padding='same'))

            model.add(Conv2D(8, (3, 3), padding='same', activation='relu', trainable=False))

            model.add(TimeDistributed(Flatten()))

            model.add(TimeDistributed(Dense(1024, activation='relu', trainable=False)))
            model.add(TimeDistributed(Dense(1024, activation='relu', trainable=False)))

            model.add(Bidirectional(LSTM(128, activation='tanh', return_sequences=True, trainable=False)))

            model.add(TimeDistributed(Dense(2 * len(self.__num_dict__), activation='linear', trainable=False)))

            return model

    def __line_number_extraction__(self, line):
        logit = np.exp(line)
        softmax = logit / np.sum(logit, axis=-1, keepdims=True)

        line_max = np.argmax(softmax, axis=-1)
        line_p = np.max(softmax, axis=-1)

        line_car_num = ""
        last_char = ""

        max_p = 0

        for i, (l, p) in enumerate(zip(line_max, line_p)):
            if self.__char_dict__[l] != 'None':
                if self.__char_dict__[l] == 'SP':
                    last_char = ""
                    max_p = 0
                elif self.__char_dict__[l] != last_char:
                    if last_char == "":
                        line_car_num += self.__char_dict__[l]
                        last_char = self.__char_dict__[l]
                        max_p = p
                    elif max_p < p:
                            line_car_num = line_car_num[:-1] + self.__char_dict__[l]
                            last_char = self.__char_dict__[l]
                            max_p = p
            else:
                last_char = ""
                max_p = 0

        return line_car_num

    def __sanity_check__(self, num):
        if len(num) < 7:
            return ""

        last = num[-5:]
        if last[0].isdigit():
            return ""

        for n in last[1:]:
            if not n.isdigit():
                return ""

        region_flag = num[0] in self.__region_char__

        if region_flag and num[1] not in self.__region_char__:
            return ""

        st = 2 if region_flag else 0

        cnt = 0
        for n in num[st: -5]:
            if not n.isdigit():
                return ""

            cnt += 1
        else:
            if 0 < cnt < 4:
                if not region_flag and cnt == 1:
                    return ""

                if region_flag and cnt == 3:
                    return ""
            else:
                return ""

        return num

    def __car_number_extraction__(self, y_pred):
        line1_car_num = self.__line_number_extraction__(
            y_pred[0, :, :y_pred.shape[2] // 2])

        line2_car_num = self.__line_number_extraction__(
            y_pred[0, :, y_pred.shape[2] // 2:])

        ret_car_num = self.__sanity_check__(line1_car_num + line2_car_num)

        return ret_car_num

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

            num = self.__car_number_extraction__(y)

            if num != "":
                car_box.append(box)
                car_num.append(num)

        if out_time:
            out_time = 1000 * (time.time() - start)

        if out_time:
            return car_box, car_num, out_time
        else:
            return car_box, car_num

if __name__ == "__main__":
    # gpu_only=true for gpu operation
    cpr = car_plate_recognition(gpu_only=True, model_path='../model/')

    vc = cv.VideoCapture("C:\\Users\\LSH\\Desktop\\output3.avi")

    tt = []
    while vc.isOpened():
        ret, im = vc.read()

# =============================================================================
#     import utils.annotator as ann
#
#     anno = ann.annotator()
#     loc = anno.__file_capture__('../../data', ext='jpg', except_ext='txt')
#
#     for l in loc:
#         im = cv.imread(l)
# =============================================================================

# =============================================================================
#         for _ in range(3):
# =============================================================================
        car_box, car_num, t = cpr.car_plate_recognition(im, out_time=True)
        tt.append([car_num, t])


        from PIL import ImageFont, ImageDraw, Image

        fontpath = "fonts/gulim.ttc"
        font = ImageFont.truetype(fontpath, 25)
        img_pil = Image.fromarray(im)
        draw = ImageDraw.Draw(img_pil)
        for b, n in zip(car_box, car_num):
            draw.text((b[0] - b[2] // 2, b[1] - b[3] // 2 - 25), n, font=font, fill=(0,0,255,0), stroke_width=1)

        im = np.array(img_pil)

        for box in car_box:
            cv.rectangle(im,
                         (box[0] - box[2] // 2, box[1] - box[3] // 2),
                         (box[0] + box[2] // 2, box[1] + box[3] // 2),
                         (255, 0, 0),
                         2)

        cv.imshow('test', im)
        cv.waitKey(100)

# =============================================================================
#         cv.destroyAllWindows()
#         cv.imshow('%.2f' %(t), im)
#         cv.waitKey()
# =============================================================================
