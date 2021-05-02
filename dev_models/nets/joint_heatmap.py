import random
import numpy as np
import math
import cv2
import os.path as osp
import subprocess as sp
import torch
import random
# import common
from opendr.camera import ProjectPoints
from opendr.renderer import ColoredRenderer
from opendr.lighting import LambertianPointLight


def recover_img(image):
    # normalize format
    if isinstance(image, torch.Tensor):
        if image.device.type == 'cuda':
            image = image.cpu().detach().numpy()
        else:
            image = image.detach().numpy()
    else:
        assert isinstance(image, np.ndarray)

    # recover image
    if np.max(image) < 1.1:
        image = (image+1)* 0.5 * 255
    if image.shape[2] != 3:
        image = np.transpose(image, (1, 2, 0))

    image = image.copy() # necessary !!!
    return image.astype(np.uint8)


def recover_keypoint(kps, kps_weight, img_size, resize=True):
    # normalize format
    if isinstance(kps, torch.Tensor):
        if kps.device.type == 'cuda':
            kps = kps.cpu().detach().numpy()
            kps_weight = kps_weight.cpu().detach().numpy()
        else:
            kps = kps.detach().numpy()
            kps_weight = kps_weight.detach().numpy()
    else:
        assert isinstance(kps, np.ndarray)
    # recover keypoint
    if resize:
        kps = (kps+1)*0.5 * img_size
    # return 
    return kps, kps_weight

def heat_map(data, kps_weight,pred_cam_params,pred_kp3D,depth_color,map_size=224):
    map = np.array([0] * pow(map_size, 2), dtype=np.uint8).reshape((map_size, map_size))
    depth_color = depth_color[:, :, :3]
    centers = {}
    idx = 0

    for i,d in enumerate(data):
        u = d[0]
        v = d[1]
        weight = kps_weight[i][0].numpy() 
        if weight > 0:
            if int(u) >=224:
                r1 = np.linalg.norm(pred_kp3D[i]-depth_color[int(223)][int(v)])
            elif int(v) >=224:
                r1 = np.linalg.norm(pred_kp3D[i]-depth_color[int(u)][int(223)])
            else:
                r1 = np.linalg.norm(pred_kp3D[i]-depth_color[int(u)][int(v)])
            r = 10+0.011*r1
            e = 0.25*r+2.5
            center = (u,v,e)
            centers[idx] = center
            idx += 1
            weight = min(255, int(random.uniform(1.0,weight) * 200))
            attention(int(u), int(v), weight, map,int(r))
    if len(centers) <= 13:
        for i in range(len(centers),14):
            centers[i] = (-1,-1,-1)
    # print(centers)

    heat_img = cv2.applyColorMap(map, cv2.COLORMAP_JET)  # 注意此处的三通道热力图是cv2专有的GBR排列
    heat_img[(heat_img[:, :, 0] == 128) & (heat_img[:, :, 1] == 0) & (heat_img[:, :, 2] == 0)] = [0, 0, 0]
    heat_img = cv2.transpose(heat_img)
    return heat_img,centers

def draw_keypoints(image, kps, kps_weight,pred_cam_params, pred_kp3D,depth_color,color=(0,0,255), img_size=224):
    image = recover_img(image)
    kps, kps_weight = recover_keypoint(kps, kps_weight, img_size)
    # recover color 
    if color == 'red':
        color = (0, 0, 255)
    elif color == 'green':
        color = (0, 255, 0)
    elif color == 'blue':
        color = (255, 0, 0)
    else:
        assert isinstance(color, tuple) and len(color) == 3
    # draw heatmaps
    heat_maps,centers = heat_map(kps,kps_weight,pred_cam_params,pred_kp3D,depth_color)
    return image[:,:,::-1].astype(np.uint8),heat_maps,image,centers

def attention(u, v, val, map, r):
    shape = map.shape
    w, h = shape[0], shape[1]
 
    intensity = np.linspace(val, 50, r, dtype=np.uint8)
 
    for x in range(max(0, u-r), min(w, u+r)):
        for y in range(max(0, v-r), min(h, v+r)):
            distance = math.ceil(math.sqrt(pow(x-u, 2) + pow(y-v, 2)))
 
            if distance < r:
                if map[x][y] == 0:
                    map[x][y] = intensity[distance]
                else:
                    map[x][y] = max(map[x][y], intensity[distance])
