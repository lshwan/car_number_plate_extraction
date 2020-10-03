# -*- coding: utf-8 -*-
"""
Created on Wed Sep 23 20:11:46 2020

@author: LSH
"""

import cv2 as cv
import numpy as np
import copy as cp
import matplotlib.pyplot as plt
from PIL import ImageFont, ImageDraw, Image

def num_to_char(num_dict):
    dic = dict()
    for key, value in num_dict.items():
        dic.update({value: key})

    return dic

num_dict = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4,
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
            '하': 46, '허': 47, '호': 48, '배': 49}

region_char = ['서울', '대전', '대구', '부산', '인천', '광주', '울산', '경기', '강원', '충북', '충남', '전북', '전남', '경북', '경남', '제주', '세종']

char_dict = num_to_char(num_dict)

#D:\Programming\Python\My Project\Car_Plate_Number_Extract\data\0\20200529\20200529084834855.jpg
#'D:/Programming/Python/My Project/Car_Plate_Number_Extract/data/0/20200526/20200526082351875.jpg'
#'D:/Programming/Python/My Project/Car_Plate_Number_Extract/data/0/20200526/20200526083710193.jpg'
#'D:/Programming/Python/My Project/Car_Plate_Number_Extract/data/0/20200526/20200526095039569.jpg'
#'D:/Programming/Python/My Project/Car_Plate_Number_Extract/data/0/20200526/20200526161632316.jpg'


for k in range(1):
    rand_num = np.random.randint(0, 10, 6)
    rand_num1 = np.random.randint(10, len(num_dict), 1)

    car_line1 = "%s%s" %(char_dict[rand_num[0]], char_dict[rand_num[1]])
    car_line2 = "%s" %(char_dict[rand_num1[0]])
    car_line3 = "%s%s%s%s" %(char_dict[rand_num[2]], char_dict[rand_num[3]], char_dict[rand_num[4]], char_dict[rand_num[5]])

    car_num = [car_line1, car_line2, car_line3]
# =============================================================================
#     draw_prop = [{'font_size': 125, 's_x': 179, 'e_x': 196, 'y': 189, 'font': 'malgunbd.ttf'}, # green
#                  {'font_size': 110, 's_x': 196, 'e_x': 201, 'y': 190, 'font': 'malgunbd.ttf'},
#                  {'font_size': 215, 's_x': 170, 'e_x': 217, 'y': 203, 'font': 'HMFMMUEX.TTC'}]
# =============================================================================

    draw_prop = [{'font_size': 100, 's_x': 290, 'e_x': 308, 'y': 185, 'font': 'HANDotum.ttf'}, # white
                 {'font_size': 120, 's_x': 310, 'e_x': 326, 'y': 184, 'font': 'HANDotum.ttf'},
                 {'font_size': 155, 's_x': 273, 'e_x': 290, 'y': 197, 'font': 'HANDotum.ttf'}]

    mul = 10

    bg_color = 116
    num_color = (51, 51, 51, 0)
    shadow_color = (51, 51, 51)
    shadow_thick = 3

    im = cv.imread('D:/Programming/Python/My Project/Car_Plate_Number_Extract/data/0/20200609/20200609165430665.jpg')

    pts1 = np.float32([[269, 185],[270, 223],[344, 192],[346, 231]])

    y = int(min(pts1[0][1], pts1[2][1]))
    dy = int(np.sqrt((pts1[0][1] - pts1[1][1]) ** 2 + (pts1[0][0] - pts1[1][0]) ** 2))

    x = int(min(pts1[0][0], pts1[1][0]))
    dx = int(np.sqrt((pts1[0][0] - pts1[2][0]) ** 2 + (pts1[0][1] - pts1[2][1]) ** 2))

    pts2 = np.float32([[x, y],[x, y + dy],[x + dx, y],[x + dx, y + dy]])

    M = cv.getPerspectiveTransform(pts1, pts2)

    im = cv.warpPerspective(im, M, (512,384))
    im = im[:339, :482]

    region = [[[288, 187], [327, 198]], [[272, 197], [341, 222]]]

    for r in region:
        target = np.argwhere(im[r[0][1]: r[1][1], r[0][0]:r[1][0], 0] > 255)
        n_s = bg_color * np.ones(target.shape[0])
        im[r[0][1]: r[1][1], r[0][0]:r[1][0]][target[:, 0], target[:, 1], 0] = n_s
        im[r[0][1]: r[1][1], r[0][0]:r[1][0]][target[:, 0], target[:, 1], 1] = n_s
        im[r[0][1]: r[1][1], r[0][0]:r[1][0]][target[:, 0], target[:, 1], 2] = n_s

    for r in region:
        target = np.argwhere(im[r[0][1]: r[1][1], r[0][0]:r[1][0], 0] < 110)
        n_s = bg_color * np.ones(target.shape[0])
        im[r[0][1]: r[1][1], r[0][0]:r[1][0]][target[:, 0], target[:, 1], 0] = n_s
        im[r[0][1]: r[1][1], r[0][0]:r[1][0]][target[:, 0], target[:, 1], 1] = n_s
        im[r[0][1]: r[1][1], r[0][0]:r[1][0]][target[:, 0], target[:, 1], 2] = n_s

    im_size_x, im_size_y = im.shape[1], im.shape[0]
    im = Image.fromarray(cv.resize(im, (mul * im_size_x, mul * im_size_y)))
    draw = ImageDraw.Draw(im)

    for num, prop in zip(car_num, draw_prop):
        font_size, s_x, e_x, y, font_type = prop['font_size'], prop['s_x'], prop['e_x'], prop['y'], prop['font']
        fontpath = "fonts/%s" %(font_type)

        font = ImageFont.truetype(fontpath, font_size)

        for i, n in enumerate(num):
            draw.text((mul * s_x + i * mul * (e_x - s_x) // len(num), mul * y),  n, font=font, fill=num_color, stroke_width=shadow_thick, stroke_fill=shadow_color)

    im = np.array(im, dtype=np.uint8)

    for r in region:
        im[mul * r[0][1]: mul * r[1][1], mul * r[0][0]: mul * r[1][0]] = cv.blur(im[mul * r[0][1]: mul * r[1][1], mul * r[0][0]: mul * r[1][0]], (5, 5))

    im = im.astype(float)
    for r in region:
        n_s = (15 * (np.random.rand(mul * (r[1][1] - r[0][1]), mul * (r[1][0] - r[0][0])) - 0.5))

        im[mul * r[0][1]: mul * r[1][1], mul * r[0][0]: mul * r[1][0], 0] += n_s
        im[mul * r[0][1]: mul * r[1][1], mul * r[0][0]: mul * r[1][0], 1] += n_s
        im[mul * r[0][1]: mul * r[1][1], mul * r[0][0]: mul * r[1][0], 2] += n_s

    im *= (0.5 + np.random.rand(1))

    im[im > 255] = 255
    im[im < 0] = 0
    im = im.astype(np.uint8)

    im = cv.resize(im, (im_size_x, im_size_y))

    r_y = np.random.randint(-20, 20, 1)

    pts3 = cp.deepcopy(pts2)
    pts3[2][1] += r_y
    pts3[3][1] += r_y

    M = cv.getPerspectiveTransform(pts2, pts3)

    im = cv.warpPerspective(im, M, (im_size_x,im_size_y))

# =============================================================================
#     cv.imwrite('../data/augment2/augment%d.jpg' %(k), im)
#
#     m_x, m_x1 = np.random.randint(13, 25, 2)
#     m_y, m_y1 = np.random.randint(6, 13, 2)
#
#     if r_y > 0:
#         m_y1 += r_y
#     else:
#         m_y -= r_y
#
#     w = []
#     with open('../data/augment2/test.txt', 'r') as f:
#         for l in f.readlines():
#             w.append(l)
#
#     w[0] = '../data/augment2/augment%d.jpg\n' %(k)
#
#     x, y, width, height = map(int, w[2].split())
#     x -= m_x
#     y -= m_y
#     width += (m_x + m_x1)
#     height += (m_y + m_y1)
#
#     w[2] = "%d\t%d\t%d\t%d\n" %(x, y, width, height)
#
#     num = ""
#     for n in car_num:
#         num += n
#
#     w[3] = "%s\n" %(num)
#
#     for i in range(5, 5 + int(w[4].strip())):
#         x, y, width, height = map(int, w[i].split())
#
#         x += m_x
#         y += m_y
#
#         w[i] = "%d\t%d\t%d\t%d\n" %(x, y, width, height)
#
#
#     with open('../data/augment2/augment%d.txt' %(k), 'w') as f:
#         for l in w:
#             f.writelines(l)
# =============================================================================
