# -*- coding: utf-8 -*-
"""
Created on Wed Oct 21 19:34:25 2020

@author: LSH
"""

import cv2 as cv
import time
from multiprocessing import Process, Pipe
import csv
import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Flatten, Dense, MaxPooling2D, LeakyReLU, TimeDistributed, LSTM, Reshape, Bidirectional, BatchNormalization
from tensorflow.keras.models import Sequential
import numpy as np
import os


class car_plate_detector():
    def __init__(self, gpu_only=False, model_path='../model/'):
        self.__C__ = 16
        self.__im_size__ = [384, 512, 1]
        self.__p_thr__ = 0.64
        self.__iou_thr__ = 0.5
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
            model.add(Conv2D(32, (3, 3), padding='same', activation='linear', input_shape=input_size, trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(Conv2D(32, (3, 3), padding='same', activation='linear', trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(MaxPooling2D((2, 2)))
            max_pool += 1

            model.add(Conv2D(64, (3, 3), padding='same', activation='linear', trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(Conv2D(64, (3, 3), padding='same', activation='linear', trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(MaxPooling2D((2, 2)))
            max_pool += 1

            model.add(Conv2D(128, (3, 3), padding='same', activation='linear', trainable=False))
            model.add(LeakyReLU(0.01))

            model.add(Conv2D(128, (3, 3), padding='same', activation='linear', trainable=False))
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

    def __out2sorted_serial_box__(self, y_pred):
        ret_y = []
        for y in y_pred:
            temp_y = y.reshape(-1, y_pred.shape[-1])
            sorted_idx = np.argsort(temp_y[:, 0])
            sorted_idx = sorted_idx[::-1]

            temp_y = temp_y[sorted_idx, :]
            temp_y = np.hstack([temp_y, sorted_idx.reshape(-1, 1)])

            ret_y.append(temp_y)

        return np.array(ret_y)

    def __nmu__(self, y_pred, p_thr, iou_thr=0.5):
        for i, y in enumerate(y_pred):
            y_list = []
            y_idx = []

            cand_idx = np.argwhere(y[:, 0] > p_thr)
            for j in cand_idx:
                j = j[0]
                p = y[j]

                idx_x, idx_y = p[-1] // self.__C__, p[-1] % self.__C__
                temp_p = p[:-1]
                temp_p[1] += idx_y
                temp_p[1] /= self.__C__
                temp_p[2] += idx_x
                temp_p[2] /= self.__C__

                if len(y_list) == 0:
                    y_list.append(temp_p)
                    y_idx.append(j)
                else:
                    for temp_y, temp_idx in zip(y_list, y_idx):
                        if self.__iou__(temp_p, temp_y) > iou_thr:
                            y_pred[i, j, 0] = 0
                            y_pred[i, temp_idx, 3] = np.max([temp_p[1] + temp_p[3] / 2, temp_y[1] + temp_y[3] / 2]) - \
                                                        np.min([temp_p[1] - temp_p[3] / 2, temp_y[1] - temp_y[3] / 2])

# =============================================================================
#                             y_pred[i, temp_idx, 4] = np.max([temp_p[2] + temp_p[4] / 2, temp_y[2] + temp_y[4] / 2]) - \
#                                                     np.min([temp_p[2] - temp_p[4] / 2, temp_y[2] - temp_y[4] / 2])
# =============================================================================

                            break
                    else:
                        y_list.append(temp_p)
                        y_idx.append(j)

        return y_pred

    def __nms__(self, y_pred, p_thr):
        for i, y in enumerate(y_pred):
            y_list = []
            y_idx = []

            cand_idx = np.argwhere(y[:, 0] > p_thr)
            for j in cand_idx:
                j = j[0]
                p = y[j]

                idx_x, idx_y = p[-1] // self.__C__, p[-1] % self.__C__
                temp_p = p[:-1]
                temp_p[1] += idx_y
                temp_p[1] /= self.__C__
                temp_p[2] += idx_x
                temp_p[2] /= self.__C__

                if len(y_list) == 0:
                    y_list.append(temp_p)
                    y_idx.append(j)
                else:
                    for temp_y, temp_idx in zip(y_list, y_idx):
                        if self.__iou__(temp_p, temp_y) > 0:
                            y_pred[i, j, 0] = 0

                            break
                    else:
                        y_list.append(temp_p)
                        y_idx.append(j)

        return y_pred

    def __out2box__(self, out, orginal_shape, p_thr):
        bbox = []

        cand_idx = np.argwhere(out[:, 0] > p_thr)
        for j in cand_idx:
            j = j[0]
            p = out[j]

            temp_out = p
            temp_out[1] += p[-1] % self.__C__
            temp_out[1] /= self.__C__
            temp_out[2] += p[-1] // self.__C__
            temp_out[2] /= self.__C__

            temp_out[1] *= orginal_shape[1]
            temp_out[3] *= orginal_shape[1]
            temp_out[2] *= orginal_shape[0]
            temp_out[4] *= orginal_shape[0]

            bbox.append(temp_out[1:5].astype(int))

        return bbox

    def __sanity_check__(self, boxs):
        ret_box = []
        for b in boxs:
            if b[-1] > 0 and b[-2] > 0:
                ret_box.append(b)

        return ret_box

    def detect_car_plate(self, im, out_time=False):
        if out_time:
            start = time.time()

        if len(tf.config.list_physical_devices('GPU')) > 0 and self.gpu_only:
            device = '/gpu:0'
        else:
            device = '/cpu:0'

        with tf.device(device):
            y = self.__model__.predict(self.__preprocessing__(im))

        y = self.__out2sorted_serial_box__(y)
# =============================================================================
#         y = self.__nmu__(y, p_thr=self.__p_thr__, iou_thr=self.__iou_thr__)
# =============================================================================
        y = self.__nms__(y, p_thr=self.__p_thr__)

        bbox = self.__out2box__(y[0], im.shape, p_thr=self.__p_thr__)

        bbox = self.__sanity_check__(bbox)

        if out_time:
            elapsed_time = 1000 * (time.time() - start)

        if out_time:
            return bbox, elapsed_time
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
                             '천': 54, '전': 55, '북': 56, '대': 57,
                             '산': 58, '광': 59, '강': 60, '원': 61,
                             '충': 62, '남': 63, '제': 64, '세': 65,
                             '종': 66, 'None': 67, 'SP': 68}

        self.__region_char__ = ['서', '울', '대', '전', '구', '부', '산', '인', '천', '광', '주', '경', '기', '강', '원', '충', '북', '남', '제', '세', '종']
        self.__region_word__ = ['서울', '대전', '대구', '부산', '인천', '광주', '울산', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주', '세종']

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
            model.add(BatchNormalization(trainable=False))
            model.add(Conv2D(32, (3, 3), padding='same', activation='relu', trainable=False))
            model.add(BatchNormalization(trainable=False))
            model.add(MaxPooling2D((2, 2)))

            model.add(Conv2D(64, (3, 3), padding='same', activation='relu', trainable=False))
            model.add(BatchNormalization(trainable=False))
            model.add(Conv2D(64, (3, 3), padding='same', activation='relu', trainable=False))
            model.add(BatchNormalization(trainable=False))
            model.add(MaxPooling2D((2, 2)))

            model.add(TimeDistributed(Reshape((-1, 1))))

            model.add(Conv2D(64, (5, 5), strides=(1, 2), padding='same', activation='relu', trainable=False))
            model.add(BatchNormalization(trainable=False))
            model.add(Conv2D(64, (3, 3), padding='same', activation='relu', trainable=False))
            model.add(BatchNormalization(trainable=False))
            model.add(MaxPooling2D((2, 2), strides=(1, 2), padding='same'))

            model.add(Conv2D(128, (5, 5), strides=(1, 2), padding='same', activation='relu', trainable=False))
            model.add(BatchNormalization(trainable=False))
            model.add(Conv2D(128, (3, 3), padding='same', activation='relu', trainable=False))
            model.add(BatchNormalization(trainable=False))
            model.add(MaxPooling2D((2, 2), strides=(1, 2), padding='same'))

            model.add(Conv2D(256, (5, 5), strides=(1, 2), padding='same', activation='relu', trainable=False))
            model.add(BatchNormalization(trainable=False))
            model.add(Conv2D(256, (3, 3), padding='same', activation='relu', trainable=False))
            model.add(BatchNormalization(trainable=False))
            model.add(MaxPooling2D((2, 2), strides=(1, 2), padding='same'))

            model.add(Conv2D(8, (3, 3), padding='same', activation='relu', trainable=False))
            model.add(BatchNormalization(trainable=False))

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

    def __sanity_check2__(self, num2):
        return ""

        if len(num2) < 4:
            return ""

        for n in num2[-4:]:
            if not n.isdigit():
                return ""

        if len(num2) == 4:
            return num2
        elif not num2[-5].isdigit():
            return num2[-4:]
        else:
            return ""

    def __sanity_check__(self, num1, num2):
        num = num1 + num2

        if len(num) < 7:
            return self.__sanity_check2__(num2)

        last = num[-5:]
        if last[0].isdigit():
            return self.__sanity_check2__(num2)

        for n in last[1:]:
            if not n.isdigit():
                return self.__sanity_check2__(num2)

        region_flag = num[0] in self.__region_char__

        if region_flag:
            if num[1] not in self.__region_char__:
                return self.__sanity_check2__(num2)
            elif num[:2] not in self.__region_word__:
                return self.__sanity_check2__(num2)

        st = 2 if region_flag else 0

        cnt = 0
        for n in num[st: -5]:
            if not n.isdigit():
                return self.__sanity_check2__(num2)

            cnt += 1
        else:
            if 0 < cnt < 4:
                if not region_flag and cnt == 1:
                    return self.__sanity_check2__(num2)

                if region_flag and cnt == 3:
                    return self.__sanity_check2__(num2)
            else:
                return self.__sanity_check2__(num2)

        return num

    def __car_number_extraction__(self, y_pred):
        line1_car_num = self.__line_number_extraction__(
            y_pred[0, :, :y_pred.shape[-1] // 2])

        line2_car_num = self.__line_number_extraction__(
            y_pred[0, :, y_pred.shape[-1] // 2:])

        ret_car_num = self.__sanity_check__(line1_car_num, line2_car_num)

        return ret_car_num

    def car_plate_recognition(self, im, out_time=False):
        elapsed_time = 0
        if out_time:
            start = time.time()

        pbox = self.__cpd__.detect_car_plate(im)

        if len(pbox) > 2:
            pbox = pbox[:2]

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

            with tf.device(device):
                y = self.__model__.predict(self.__preprocessing__(temp_im))

            num = self.__car_number_extraction__(y)

            if num != "":
                car_box.append(box)
                car_num.append(num)
# =============================================================================
#             cv.imshow(str(i), temp_im)
# =============================================================================

        if out_time:
            elapsed_time = 1000 * (time.time() - start)

        if out_time:
            return car_box, car_num, elapsed_time
        else:
            return car_box, car_num

class park_entry_processor():
    def __init__(self):
        self.__car__ = []

        return

    def __car_check__(self, num1, num2):
        cnt = 0
        for i in range(min(len(num1), len(num2))):
            if num1[-1 - i] == num2[-1 - i]:
                cnt += 2

        return cnt / (len(num1) + len(num2))

    def __car_check_and_insert_list__(self, car_num_list, cur_car_num, car_im):
        match = False
        max_similarity = 0

        idx = 0
        similarity = 0

        for i, car in enumerate(car_num_list):
            if similarity == 1:
                break

            for k, d in enumerate(car['car_num']):
                similarity = self.__car_check__(d[0], cur_car_num)

                if max_similarity < similarity:
                    idx = i
                    max_similarity = similarity

                if similarity >= 0.5:
                    match = True

                    if similarity == 1:
                        d[1] += 1
                        d[2].append(car_im)

                        if k > 0:
                            for j in range(k):
                                if car['car_num'][k][1] > car['car_num'][j][1]:
                                    car['car_num'][k], car['car_num'][j] = car['car_num'][j], car['car_num'][k]

                        break

        if match and max_similarity < 1:
            car_num_list[idx]['car_num'].append([cur_car_num, 1, [car_im]])

        return match, idx

    def __write_car__(self, msg):
        with open('./car.csv', 'a', encoding='cp949', newline='') as f:
            wr = csv.writer(f)
            wr.writerow(msg)

    def process_main(self, im, car_info):
        car_nums = car_info[1]

        cur_time = time.time()

        for num in car_nums:
            match = False

            match, idx = self.__car_check_and_insert_list__(self.__car__, num, im)

            if match:
                self.__car__[idx]["time"] = cur_time
            else:
                if not match:
                    self.__car__.append({"time": cur_time, "car_num": [[num, 1, [im]]]})

        for i in range(len(self.__car__) - 1, -1, -1):
            if cur_time - self.__car__[i]['time'] > 1:
                if self.__car__[i]['car_num'][0][1] > 5:
                    yymmss = time.localtime()
                    self.__write_car__(["%4d-%2d-%2d" %(yymmss[0], yymmss[1], yymmss[2]),
                                       "%2d: %2d: %2d" %(yymmss[3], yymmss[4], yymmss[5]),
                                       "%s" %(self.__car__[i]['car_num'][0][0]),
                                       "%d" %(self.__car__[i]['car_num'][0][1])])

                    if not os.path.isdir('./image_save/'):
                        os.mkdir('./image_save/')

                    cv.imwrite(u"./image_save/%s%s%s%s%s%s-%s.jpg"
                               %(str(yymmss[0]).zfill(4),
                                 str(yymmss[1]).zfill(2),
                                 str(yymmss[2]).zfill(2),
                                 str(yymmss[3]).zfill(2),
                                 str(yymmss[4]).zfill(2),
                                 str(yymmss[5]).zfill(2),
                                 self.__car__[i]['car_num'][0][0][-4:]),
                               self.__car__[i]['car_num'][0][2][len(self.__car__[i]['car_num'][0][2]) // 2])

                self.__car__.pop(i)

def write_log(msg, new=False):
    yymmss = time.time()

    mode = 'w' if new else 'a'

    with open('./system.log', mode) as f:
        f.write("%.6f\t%s\n" %(yymmss, msg))

def open_image(conn):
    vc = cv.VideoCapture("C:\\Users\\LSH\\Desktop\\plate1_test.avi")

    write_log('video capture process started')

    start_time = time.time()
    ms = 0

    video_len = 1000 * vc.get(cv.CAP_PROP_FRAME_COUNT) // vc.get(cv.CAP_PROP_FPS)

    while video_len - 10 > ms:
        vc.set(cv.CAP_PROP_POS_MSEC, ms)

        ret, im = vc.read()

        if not ret:
            break

        conn.send(im)

        try:
            check = conn.recv()
        except Exception as e:
            write_log(e)
            break

        if check != "done":
            write_log("unexpected msg")
            break

        ms = int(1000 * (time.time() - start_time))

    write_log('video capture process ended')

    conn.send("final")
    conn.close()

if __name__ == '__main__':
    parent_conn, child_conn = Pipe()

    write_log('system start', True)

    p = Process(target=open_image, args=(child_conn,))
    p.start()

    cpr_ins = car_plate_recognition(gpu_only=True, model_path='../model/')
    main_process = park_entry_processor()

    while True:
        save_im = parent_conn.recv()

        if type(save_im) == str:
            write_log('last frame')
            break

        im = save_im[300:300 + 360, 460:460 + 650, :]

        car_info = cpr_ins.car_plate_recognition(im, out_time=True)
        main_process.process_main(save_im, car_info)

        parent_conn.send("done")

        cv.imshow("test", im)
        cv.waitKey(5)

    write_log('system finished')
    p.join()


