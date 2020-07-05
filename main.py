# -*- coding: utf-8 -*-
"""
Created on Wed Jun 24 22:54:51 2020

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
import cv2 as cv
import copy as cp
import time

class car_plate_detector(utils.annotator.annotator):
    C = 0
    model = None

    def __init__(self, C=7, seed=0, pre_model=False):
        super().__init__(seed)
        self.C = C
        if pre_model:
            self.model = self.__load_preset_model__()

        return

    def __load_preset_model__(self):
        return load_model('./model/model.h5', custom_objects={'loss': self.loss})

    def save_model(self):
        save_model(self.model, './model/model.h5')

    def __data_augment_brightness__(self, data, target, c=1.5):
        aug_data = cp.deepcopy(c * data.astype(np.uint32))
        aug_data[aug_data > 255] = 255
        aug_data = aug_data.astype(np.uint8)

        aug_target = cp.deepcopy(target)

        return aug_data, aug_target

    def __data_augment_color_reverse__(self, data, target):
        aug_data = cp.deepcopy(255 - data)
        aug_target = cp.deepcopy(target)

        return aug_data, aug_target

    def __data_augment_fliplr__(self, data, target):
        aug_data = cp.deepcopy(np.flip(data, axis=2))
        aug_target = cp.deepcopy(np.flip(target, axis=2))
        aug_target[:, :, :, 1] = 1 - aug_target[:, :, :, 1]

        return aug_data, aug_target

    def __data_augment_flipud__(self, data, target):
        aug_data = cp.deepcopy(np.flip(data, axis=1))
        aug_target = cp.deepcopy(np.flip(target, axis=1))
        aug_target[:, :, :, 2] = 1 - aug_target[:, :, :, 2]

        return aug_data, aug_target

    def __data_processing__(self, data, target, augment=False):
        #TODO: consider IOU to ret_target[:,:,:,0]
        t_size = [384, 512]
        ret_target = np.zeros((len(data), self.C, self.C, 5))
        ret_data = np.zeros((len(data), t_size[0], t_size[1], 1), dtype=np.uint8)

# =============================================================================
#         data = data[:,:self.C * (data.shape[1] // self.C),:self.C * (data.shape[2] // self.C), :]
# =============================================================================

        for n, (d, t) in enumerate(zip(data, target)):
            h, w = d.shape[0], d.shape[1]

            temp_data = cv.resize(d, (t_size[1], t_size[0]))
            temp_data = cv.medianBlur(temp_data, 3)
# =============================================================================
#             temp_data = cv.adaptiveThreshold(temp_data, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 17, 1)
#             temp_data = cv.dilate(temp_data, kernel=cv.getStructuringElement(cv.MORPH_RECT, (2, 2)))
#             temp_data = cv.erode(temp_data, kernel=cv.getStructuringElement(cv.MORPH_RECT, (2, 2)))
# =============================================================================
            t[:, 0] = (t_size[1] * t[:, 0]) // w
            t[:, 1] = (t_size[0] * t[:, 1]) // h
            t[:, 2] = (t_size[1] * t[:, 2]) // w
            t[:, 3] = (t_size[0] * t[:, 3]) // h

            im_size = np.array([temp_data.shape[1], temp_data.shape[0]], ndmin=2)

            temp_data = temp_data.reshape((temp_data.shape[0], temp_data.shape[1], 1))
            ret_data[n] = temp_data

            idx = (self.C * t[:,:2]) // im_size
            t = t.astype(float)
            t[:,:2] = (t[:,:2] % (im_size // self.C)) / (im_size // self.C)
            t[:,2:] /= im_size

            for i, (j, k) in enumerate(idx):
                ret_target[n, k, j] = np.hstack([[1], t[i]])

        if augment:

            aug_data1, aug_target1 = self.__data_augment_brightness__(ret_data, ret_target, 1.5)
            aug_data2, aug_target2 = self.__data_augment_brightness__(ret_data, ret_target, 0.5)

            ret_data = np.vstack([ret_data, aug_data1, aug_data2])
            ret_target = np.vstack([ret_target, aug_target1, aug_target2])

            aug_data3, aug_target3 = self.__data_augment_flipud__(ret_data, ret_target)

            ret_data = np.vstack([ret_data, aug_data3])
            ret_target = np.vstack([ret_target, aug_target3])

            aug_data4, aug_target4 = self.__data_augment_fliplr__(ret_data, ret_target)

            ret_data = np.vstack([ret_data, aug_data4])
            ret_target = np.vstack([ret_target, aug_target4])

            aug_data5, aug_target5 = self.__data_augment_color_reverse__(ret_data, ret_target)

            ret_data = np.vstack([ret_data, aug_data5])
            ret_target = np.vstack([ret_target, aug_target5])

        return ret_data, ret_target

    def get_data(self, root_path):
        tr_data, tr_target, val_data, val_target, te_data, te_target = super().get_data(root_path)

        tr_data, tr_target = self.__data_processing__(tr_data, tr_target, augment=True)
        val_data, val_target = self.__data_processing__(val_data, val_target, augment=False)
        te_data, te_target = self.__data_processing__(te_data, te_target, augment=False)

        self.tr_data, self.tr_target, self.val_data, self.val_target, self.te_data, self.te_target = \
            tr_data, tr_target, val_data, val_target, te_data, te_target

        return tr_data, tr_target, val_data, val_target, te_data, te_target

    def create_model(self, input_size):
        max_pool = 0
        model = Sequential()
        model.add(Conv2D(16, (3, 3), padding='same', activation='linear', input_shape=input_size))
        model.add(LeakyReLU(0.01))
        model.add(Conv2D(16, (3, 3), padding='same', activation='linear', input_shape=input_size))
        model.add(LeakyReLU(0.01))
        model.add(MaxPooling2D((2, 2)))
        max_pool += 1
        model.add(Conv2D(32, (3, 3), padding='same', activation='linear'))
        model.add(LeakyReLU(0.01))
        model.add(Conv2D(32, (3, 3), padding='same', activation='linear'))
        model.add(LeakyReLU(0.01))
        model.add(MaxPooling2D((2, 2)))
        max_pool += 1
        model.add(Conv2D(64, (3, 3), padding='same', activation='linear'))
        model.add(LeakyReLU(0.01))
        model.add(Conv2D(64, (3, 3), padding='same', activation='linear'))
        model.add(LeakyReLU(0.01))
        model.add(MaxPooling2D((2, 2)))
        max_pool += 1
        feature_size = [input_size[0] // 2**max_pool, input_size[1] // 2**max_pool]
        model.add(Conv2D(1024, (feature_size[0] // self.C, feature_size[1] // self.C), padding='valid', strides=(feature_size[0] // self.C, feature_size[1] // self.C), activation='linear'))
        model.add(Dropout(0.5))
        model.add(LeakyReLU(0.01))
        model.add(Conv2D(1024, (1, 1), padding='valid', activation='linear'))
        model.add(Dropout(0.5))
        model.add(LeakyReLU(0.01))
        model.add(Conv2D(5, (1, 1), padding='valid', activation='linear'))

        self.model = model

        return model

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
        c2 = 0.1

        term1 = K.sum(K.square(y_true[:, :, :, 0:1] * (y_true[:, :, :, 0:3] - y_pred[:, :, :, 0:3])))
        term2 = K.sum(K.square((1 - y_true[:, :, :, 0:1]) * (y_true[:, :, :, 0:1] - y_pred[:, :, :, 0:1])))
        term3 = K.sum(K.square(y_true[:, :, :, 0:1] * (K.sqrt(y_true[:, :, :, 3:]) - K.sqrt(y_pred[:, :, :, 3:]))))

        return c1*term1 + c2*term2 + c1*term3

if __name__ == "__main__":
    main_class = car_plate_detector(16, pre_model=False)
    tr_data, tr_target, val_data, val_target, te_data, te_target = main_class.get_data('./data')
    main_class.create_model(tr_data[0].shape)
    main_class.train_step(tr_data, tr_target, lr=0.0001, epoch=30)
    #main_class.train_step(tr_data, tr_target, lr=0.00001, epoch=30)

    main_class.save_model()
    model = main_class.model

    start = time.time()
    t = model.predict(val_data)
    print("%d ms for 1 sample" %(1000 * (time.time() - start) / val_data.shape[0]))

    for n, x, y, y_t in zip(np.arange(val_data.shape[0]), val_data, t, val_target):
        xx, yy = np.meshgrid(np.arange(main_class.C), np.arange(main_class.C))

        for i, j in zip(xx.flatten(), yy.flatten()):
            t_y = y_t[i, j]
            if t_y[0]:
                p1 = (
                    int((x.shape[1] // main_class.C) * (j + t_y[1]) - (x.shape[1] // 1) * t_y[3] // 2),
                    int((x.shape[0] // main_class.C) * (i + t_y[2]) - (x.shape[0] // 1) * t_y[4] // 2))

                p2 = (
                    int((x.shape[1] // main_class.C) * (j + t_y[1]) + (x.shape[1] // 1) * t_y[3] // 2),
                    int((x.shape[0] // main_class.C) * (i + t_y[2]) + (x.shape[0] // 1) * t_y[4] // 2))

                cv.rectangle(x, p1, p2, (0, 0, 0), 1)

            t_a = y[i, j]
            if t_a[0] > 0.05:
                p1 = (
                    int((x.shape[1] // main_class.C) * (j + t_a[1]) - (x.shape[1] // 1) * t_a[3] // 2),
                    int((x.shape[0] // main_class.C) * (i + t_a[2]) - (x.shape[0] // 1) * t_a[4] // 2))

                p2 = (
                    int((x.shape[1] // main_class.C) * (j + t_a[1]) + (x.shape[1] // 1) * t_a[3] // 2),
                    int((x.shape[0] // main_class.C) * (i + t_a[2]) + (x.shape[0] // 1) * t_a[4] // 2))

                cv.rectangle(x, p1, p2, (255, 0, 0), int(3*t_a[0]) + 1)

        cv.imshow(str(n), x)
# =============================================================================
#     contour = []
#     xx, yy = np.meshgrid(np.arange(main_class.C), np.arange(main_class.C))
#     for i, j in zip(xx.flatten(), yy.flatten()):
#         x, y = np.meshgrid(np.arange(i, i + 2), np.arange(j, j + 2))
#         x[0, 0], x[0, 1] = x[0, 1], x[0, 0]
#         mesh = (np.array([tr_data.shape[2], tr_data.shape[1]], ndmin=2) * np.array([x.flatten(),  y.flatten()]).T) // main_class.C
#
#         contour.append(mesh.reshape((-1, 1, 2)))
#     for i in range(6):
#         im = tr_data[454*i]
#         t = tr_target[454*i]
#
#         for j, k in zip(xx.flatten(), yy.flatten()):
#             t_a = t[j][k]
#
#             if t_a[0]:
#                 cv.circle(im, (
#                     int((t_a[1] + k) * (im.shape[1] // main_class.C)),
#                     int((t_a[2] + j) * (im.shape[0] // main_class.C))),
#                     3,
#                     (0, 0, 0),
#                     1)
#
#                 cv.rectangle(im, (
#                     int((t_a[1] + k) * (im.shape[1] // main_class.C) - (t_a[3] * im.shape[1]) // 2),
#                     int((t_a[2] + j) * (im.shape[0] // main_class.C) - (t_a[4] * im.shape[0]) // 2)),
#                     (
#                     int((t_a[1] + k) * (im.shape[1] // main_class.C) + (t_a[3] * im.shape[1]) // 2),
#                     int((t_a[2] + j) * (im.shape[0] // main_class.C) + (t_a[4] * im.shape[0]) // 2)),
#                     (0, 0, 0),
#                     1)
#
#         for c in contour:
#             cv.drawContours(im, [c], 0, [255, 0, 0], 1)
#
#         cv.imshow(str(i), im)
#         cv.waitKey(500)
# =============================================================================


# =============================================================================
# ld = ld.load_data()
# data = ld.get_data()
#
# data['train'][1] = tf.keras.utils.to_categorical(data['train'][1], 10)
#
# model = Sequential()
# model.add(Conv2D(4, (3, 3), padding='same', activation='relu', input_shape=(28, 28, 1)))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(8, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
# model.add(MaxPooling2D((2, 2)))
# model.add(Flatten())
# model.add(Dense(512, activation='relu'))
# model.add(Dropout(0.3))
# model.add(Dense(10, activation='softmax'))
#
# model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['categorical_accuracy'])
#
# model.fit(data['train'][0], data['train'][1], epochs=100, batch_size=256)
#
# y_pred = model.predict(data['valid'][0])
# y_pred = np.argmax(y_pred, axis=1)
# y_true = data['valid'][1]
#
# tt = y_pred - y_true
#
# print(float(tt[tt == 0].shape[0]) / tt.shape[0])
# =============================================================================
