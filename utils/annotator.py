# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 11:21:03 2020

@author: LSH
"""

import numpy as np
import cv2 as cv
import os
import sys
from itertools import islice


class annotator():
    __seed__ = 0

    def __init__(self, seed=0):
        self.__seed__ = seed

        return

    def __file_capture__(self, path, ext='jpg', except_ext=None):
        ret_list = []
        try:
            ls = os.listdir(path)
        except NotADirectoryError:
            return []


        for p in ls:
            if p.endswith(ext):
                if not except_ext or '%s.%s' %(p[:-4], except_ext) not in ls:
                    ret_list.append(path + '/' + p)
            else:
                ret_list.extend(self.__file_capture__(path + '/' + p, ext=ext, except_ext=except_ext))

        return ret_list

    def __xywh2p__(self, xywh, offset=[0, 0]):
        return [[xywh[0] - xywh[2] // 2 + offset[0], xywh[1] - xywh[3] // 2 + offset[1]],
                [xywh[0] + xywh[2] // 2 + offset[0], xywh[1] - xywh[3] // 2 + offset[1]],
                [xywh[0] + xywh[2] // 2 + offset[0], xywh[1] + xywh[3] // 2 + offset[1]],
                [xywh[0] - xywh[2] // 2 + offset[0], xywh[1] + xywh[3] // 2 + offset[1]]]

    def __ppwh2xywh__(self, ppwh):
        return [ppwh[0] + ppwh[2] // 2,
                ppwh[1] + ppwh[3] // 2,
                ppwh[2],
                ppwh[3]]

    def __parse_annotation__(self, anno_path):
        with open(anno_path, 'r') as f:
            _ = f.readline().rstrip()
            plate_num = int(f.readline().rstrip())

            car_num = []
            plate_cord = []
            char_cord = []
            for i in range(plate_num):
                plate_loc = list(map(int, f.readline().rstrip().split()))
                plate_cord.append(self.__ppwh2xywh__(plate_loc))

                car_num.append(f.readline().rstrip())

                char_num = int(f.readline().rstrip())

                temp = []
                for j in range(char_num):
                    char_loc = list(map(int, f.readline().rstrip().split()))
                    temp.append(self.__ppwh2xywh__(char_loc))

                char_cord.append(temp)

        return car_num, plate_cord, char_cord, plate_num

    def annotation(self, root_path):
        im_list = self.__file_capture__(root_path, except_ext='txt')

        with open('./ann.conf', 'r') as f:
            for l in f.readlines():
                if "image_x=" in l:
                    idx = l.find("image_x=")
                    x = int(l[idx+len("image_x="):].rstrip('\n'))

                if "car_image_y=" in l:
                    idx = l.find("car_image_y=")
                    y = int(l[idx+len("car_image_y="):].rstrip('\n'))

                if "plate_image_y=" in l:
                    idx = l.find("plate_image_y=")
                    y1 = int(l[idx+len("plate_image_y="):].rstrip('\n'))

        for im_name in im_list:
            im = cv.imread(im_name)

            ref_char = '%s\n' %(im_name)

            cv.namedWindow('Select Car Plate')
            cv.moveWindow('Select Car Plate', x, y)
            rects = cv.selectROIs('Select Car Plate', im, showCrosshair=False)

            try:
                ref_char += '%d\n' %(rects.shape[0])

                re = True
                while re:
                    re = False
                    for r in rects:
                        crop = im[r[1]:r[1]+r[3], r[0]:r[0]+r[2], :]

                        cv.destroyWindow('Select Car Plate')

                        cv.namedWindow('Type Car Number')
                        cv.moveWindow('Type Car Number', x, y1)
                        cv.imshow('Type Car Number', crop)
                        cv.waitKey(500)
                        tt = input(u'차량번호: ')

                        cv.destroyWindow('Type Car Number')

                        cv.namedWindow('Select Car Numbers')
                        cv.moveWindow('Select Car Numbers', x, y1)
                        crops = cv.selectROIs('Select Car Numbers', crop, showCrosshair=False)
                        cv.waitKey(500)

                        try:
                            if crops.shape[0] != len(tt):
                                re = True

                                break
                        except:
                            print(sys.exc_info()[0])

                            re = True

                            break

                        ref_char += '%d\t%d\t%d\t%d\n' %(r[0], r[1], r[2], r[3])
                        ref_char += '%s\n' %(tt)
                        ref_char += '%d\n' %(crops.shape[0])

                        for c in crops:
                            ref_char += '%d\t%d\t%d\t%d\n' %(c[0], c[1], c[2], c[3])
            except:
                print(sys.exc_info()[0])
                ref_char += '0\n'

            cv.destroyAllWindows()

            with open('%s.txt' %(im_name[:-4]), 'w') as f:
                f.write(ref_char)

    def draw_annotation(self, im_path):
        im = cv.imread(im_path)
        car_num, plate_cord, char_cord, _ = self.__parse_annotation__('%s.txt' %(im_path[:-4]))

        contour = []
        for num, p_cord, c_cord in zip(car_num, plate_cord, char_cord):
            cord = self.__xywh2p__(p_cord)
            contour.append(np.array(cord).reshape((4, -1, 2)))

            for c in c_cord:
                contour.append(np.array(self.__xywh2p__(c, offset=cord[0])).reshape((4, -1, 2)))

            for c in contour:
                c = c.reshape((4, -1, 2))
                cv.drawContours(im, [c], 0, [255, 0, 0], 1)

        cv.imshow('image', im)
        cv.waitKey(500)

    def __preprocessing__(self, im, cord):
        im = im[70:, :]
        if cord[1] - 70 < 0:
            cord[1] = 0
        else:
            cord[1] -= 70

        return im, cord

    def get_data(self, root_path):
        ann_list = self.__file_capture__(root_path, ext='txt')

        im_set = []
        ann_set = []

        for ann in ann_list:
            im = cv.imread('%s.jpg' %(ann[:-4]), flags=cv.IMREAD_GRAYSCALE)
            _, cord, _, _ = self.__parse_annotation__(ann)

            temp = im[70:, :]

            temp_set = []
            for c in cord:
                if c[1] - 70 < 0:
                    c[1] = 0
                else:
                    c[1] -= 70

                temp_set.append(c)

            im_set.append(temp.reshape((temp.shape[0], temp.shape[1], 1)))
            ann_set.append(np.array(temp_set))

        np.random.seed(self.__seed__)
        np.random.shuffle(im_set)
        np.random.seed(self.__seed__)
        np.random.shuffle(ann_set)

        tr_set = im_set[:int(0.8*len(im_set))]
        tr_ann = ann_set[:int(0.8*len(im_set))]
        val_set = im_set[int(0.8*len(im_set)):int(0.9*len(im_set))]
        val_ann = ann_set[int(0.8*len(im_set)):int(0.9*len(im_set))]
        te_set = im_set[int(0.9*len(im_set)):]
        te_ann = ann_set[int(0.9*len(im_set)):]

        return tr_set, tr_ann, val_set, val_ann, te_set, te_ann

if __name__ == "__main__":
    with open('./ann.conf', 'r') as f:
        for l in f.readlines():
            idx = l.find("root_dir=")

            if idx >= 0:
                rt_dir = l[idx+len("root_dir="):].rstrip('\n')

                break

    ann = annotator(1411)
    #ann.draw_annotation('../data/0/20200528/20200528053754962.jpg')
    ann.annotation(rt_dir)
# =============================================================================
#     ann.__parse_annotation__('../data/0/20200526/20200526065325711.txt')
# =============================================================================
# =============================================================================
#     train_data, train_truth, val_data, val_truth, test_data, test_truth = ann.get_data('../data')
#
#     im = train_data[0]
#     test_y = train_truth[0]
#
#     for y in test_y:
#         y[0] = y[0] * im.shape[1]
#         y[1] = y[1] * im.shape[0]
#         y[2] = y[2] * im.shape[1]
#         y[3] = y[3] * im.shape[0]
#
#         y = ann.__xywh2p__(y)
#         y = np.array(y, dtype=int)
#
#         cv.drawContours(im, [y.reshape((4, -1, 2))], 0, [255, 0, 0], 1)
#
#     cv.imshow('test', im)
# =============================================================================

# =============================================================================
#     data = cv.imread('../data/0/20200526/20200526063653058.jpg', flags=cv.IMREAD_GRAYSCALE)
#
#     data = data.reshape((data.shape[0], data.shape[1], 1))
#     data = cv.resize(data, (448, 336))
#
#     data1 = cv.medianBlur(data, 3)
#     data1 = cv.adaptiveThreshold(data1, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 17, 1)
#
#     data2 = (15 * data.astype(np.uint32)) // 10
#     data2[data2 > 255] = 255
#     data2 = data2.astype(np.uint8)
#
#     data3 = cv.medianBlur(data2, 3)
#     data3 = cv.adaptiveThreshold(data3, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 7, 2)
# # =============================================================================
# #     data1 = cv.erode(data, kernel=cv.getStructuringElement(cv.MORPH_RECT, (2, 2)))
# #     data1 = cv.dilate(data1, kernel=cv.getStructuringElement(cv.MORPH_RECT, (2, 2)))
# #     data1 = cv.dilate(data1, kernel=cv.getStructuringElement(cv.MORPH_RECT, (2, 2)))
# #     data1 = cv.erode(data1, kernel=cv.getStructuringElement(cv.MORPH_RECT, (2, 2)))
# #
# # =============================================================================
#
#     cv.imshow('test', data)
#     cv.imshow('test1', data1)
#     cv.imshow('test2', data2)
#     cv.imshow('test3', data3)
#
# =============================================================================






