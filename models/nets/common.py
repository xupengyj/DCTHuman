from enum import Enum

import cv2



class CocoPart(Enum):
    LeftAnkle = 0
    LeftKnee = 1
    LeftStraddle = 2
    RightStraddle = 3
    RightKnee = 4
    RightAnkle = 5
    LeftHand = 6
    LeftElbow = 7
    LeftShoulder = 8
    RightShoulder = 9
    RightElbow = 10
    RightHand = 11
    Nick = 12
    Head = 13
    Background = 14


CocoPairs = [
    (0, 1), (1, 2), (2, 8), (7, 8), (6, 7), (8, 12), (12, 13), (9, 12), 
    (4, 5), (3, 4), (3, 9), (9, 10), (10, 11)
]   # = 13链接路径
CocoPairsRender = CocoPairs

#颜色
CocoColors = [[255, 0, 0],[255, 32, 0], [255, 85, 0], [255, 170, 0], [255, 200, 0],
[255, 255, 0], [200, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
[0, 255, 35],[0, 255, 85], [0, 255, 170], [0, 255, 200],[0, 255, 255], 
[0, 170, 255], [0, 100, 255],[0, 85, 255], [0, 0, 255], [85, 0, 255],
[100, 0, 255],[170, 0, 255], [255, 0, 255],[255, 0, 200], [255, 0, 170], [255, 0, 85]]


def read_imgfile(path, width=None, height=None):
    val_image = cv2.imread(path, cv2.IMREAD_COLOR)
    if width is not None and height is not None:
        val_image = cv2.resize(val_image, (width, height))
    return val_image



def to_str(s):
    if not isinstance(s, str):
        return s.decode('utf-8')
    return s
