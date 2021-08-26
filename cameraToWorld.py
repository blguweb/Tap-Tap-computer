#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
from typing import NoReturn
import cv2 as cv
import numpy as np
from numpy import mat
import xml.etree.ElementTree as ET
import math

camera_angle = 315

camera_intrinsic = {
    # # 相机内参矩阵
    # 相机内参矩阵 matlab 求得
    "camera_matrix":  [871.086328150675740,0.0, 314.319098669115306,
                        0.0, 868.410697770935144, 254.110678266434348,
                                                    0.0, 0.0, 1.0],
    # 畸变系数
    "camera_distortion": [0.182040359674805,-0.564946010535902,0.001566542339394, 0.003396709692351,0.000000000000000 ],
    # #  # 旋转矢量
    "camera_rvec": [-1.57079633, 0.0, 0.0],
    #  平移矢量 
    # "camera_tvec": ['-29.046143504451425', '1126.526303382564', '736.155158603123']
    "camera_tvec": [0.0, 0.0, 0.0],
    # # 旋转矩阵
    # "rvec_matrix": [[1.0,0.0,0.0],
    #                  [0.0,0.0,-1.0],
    #                  [0.0,1.0,0.0]]

}

class CtoWorld(object):
    def __init__(self):
        self.image_size = (640 , 480)
        self.rvec = np.asarray(camera_intrinsic['camera_rvec'])
        self.cam_mat = np.asarray(camera_intrinsic['camera_matrix'])
        self.tvec = np.asarray(camera_intrinsic['camera_tvec'])
        self.cam_dist = np.asarray(camera_intrinsic['camera_distortion'])
        self.rot_mat = mat(cv.Rodrigues(self.rvec)[0])
        # self.cam_mat_new, roi = cv.getOptimalNewCameraMatrix(self.cam_mat, self.cam_dist, self.image_size, 1, self.image_size)
        # self.roi = np.array(roi)

    def pixel_c(self,points,depth):
        # 像素 -> 相机
        p= (depth*np.asarray(points)).T
        p = mat(p, np.float).reshape((3,1))
        self.cam_mat = mat(self.cam_mat, np.float).reshape((3, 3))
        ca_points =np.dot( np.linalg.inv(self.cam_mat),p)
        print("c",ca_points)
        return ca_points

    def c_w(self,points):
        revc =  mat(self.rot_mat, np.float).reshape((3, 3))
        T = mat(self.tvec, np.float).reshape((3, 1))
        w_points = np.dot(revc,points)+T
        print("w",w_points)
        return w_points

    def imu_get(self,message):
        mess = message.split()
        z = float(mess[0])
        x = float(mess[1])
        y = float(mess[2])
        print("3",x,y,z)
        return x,y,z
    
    def unit_vector_get(self,vx,vy,vz):
        # 摄像头与北的夹角
        c_to_n = camera_angle
        # 计算角度
        # 因为是西 所以是负数
        # xita 对于 -y 顺时针为正 逆时针为负c_to_n - (-vz)
        xita = c_to_n + vz
        fai = vx + 90
        print("fai",fai,xita)
        # 方向单位向量
        uz = math.cos(math.radians(fai))
        print("uz",uz)
        ux = - math.sin(math.radians(xita)) * math.sin(math.radians(fai))
        uy = - math.cos(math.radians(xita)) * math.sin(math.radians(fai))
        vec = [ux,uy,uz]
        print("vtype",vec)
        return vec
        
    def target_not(self,unot,uvector):
        # 需要知道在哪一个面碰壁
        # 比如y
        tx = uvector[0] * (-unot[1]) / uvector[1] + unot[0]
        tz = uvector[2] * (-unot[1]) / uvector[1] + unot[2]
        return tx,tz
        

if __name__ == '__main__':
    mctoworld = CtoWorld()  # 生产矫正对象
    # 像素坐标 x,y,depth
    points = [355,218,1]
    depth = 1540
    # 相机坐标
    camera_points = mctoworld.pixel_c(points,depth)
    w_points = mctoworld.c_w(camera_points)
    # IMU
    mes = "-42.60 6.91 0.67"
    x,y,z = mctoworld.imu_get(mes)
    mvector = mctoworld.unit_vector_get(x,y,z)
    tx,tz = mctoworld.target_not(w_points,mvector)
    print("tx: ",tx)
    print("tz: ",tz)
    if -2000 < tx < -1380 and 840 < tz < 1300:
        print("true")
    else:
        print("false")