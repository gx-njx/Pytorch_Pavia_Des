import cv2
import math
import numpy as np


def creatimg(label, img_fname):
    answerlabel = np.zeros([label.shape[0], 1])
    for i in range(label.shape[0]):
        '''if label[i] != 0:
            answerlabel[i] = label[i] - 1
        else:'''
        answerlabel[i] = np.argmax(label[i, :])
    picture = np.zeros([340, 610, 3])
    color = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0], [1, 0.08, 0.52], [0.33, 0.46, 0.75], [0.58, 0.51, 0.38],
             [0.5, 0.96, 0.5], [0.96, 0.38, 0.2]]
    color = np.array(color)
    for i in range(label.shape[0]):
        line = int(math.floor(i / 610))
        column = int(i - line * 610)
        picture[line, column, :] = color[int(answerlabel[i]), :] * 255
    # cv2.imshow('ClassifyAnswer', picture/255)
    cv2.imwrite(img_fname, picture)
    # cv2.waitKey()
