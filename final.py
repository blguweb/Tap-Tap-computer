"""Compute depth maps for images in the input folder.
"""
from operator import eq
import os
import glob
from numpy.core.defchararray import mod
from numpy.lib.shape_base import tile
from serial.win32 import EV_BREAK
from timm.models import senet
from timm.models.byobnet import num_groups
import torch
from torch.functional import lu_unpack
import utils
import cv2
import sys
import threading
import serial  
from sys import platform
import argparse
import matplotlib.pyplot as plt
import  numpy as np
from torchvision.transforms import Compose
from midas.dpt_depth import DPTDepthModel
from midas.midas_net import MidasNet
from midas.midas_net_custom import MidasNet_small
from midas.transforms import Resize, NormalizeImage, PrepareForNet
import math
from cameraToWorld import CtoWorld
from numpy.linalg import solve
import time
ser = serial.Serial('COM9', 9600)
# 摄像头与北方向的夹角
angle = 345
ar_min = 135
ar_max = 180
radius = 500.0
mequipment = {"TV":[200,0.0,0.0,0.0]}
radius = {"TV": 500, "AIR": 200, "AUDIO": 200, "FAN": 200, "SWEEPER": 200 }
firstnode = [0,0]
def run(img_name, output_path, model_type, model, optimize=True):
    """Run MonoDepthNN to compute depth maps.

    Args:
        img_name: catch picture
        output_path (str): path to output folder
        model_path (str): path to saved model
    """
    print("initialize")
    # select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device: %s" % device)

    # load network
    if model_type == "dpt_large": # DPT-Large
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid": #DPT-Hybrid
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":
        net_w, net_h = 384, 384
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small":
        net_w, net_h = 256, 256
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False
    
    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method=resize_mode,
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )

    model.eval()
    
    if optimize==True:
        # rand_example = torch.rand(1, 3, net_h, net_w)
        # model(rand_example)
        # traced_script_module = torch.jit.trace(model, rand_example)
        # model = traced_script_module
    
        if device == torch.device("cuda"):
            model = model.to(memory_format=torch.channels_last)  
            model = model.half()

    model.to(device)


    # create output folder
    os.makedirs(output_path, exist_ok=True)

    print("start processing")

        # input

    img = utils.read_image(img_name)
    img_input = transform({"image": img})["image"]

    # compute
    with torch.no_grad():
        sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
        if optimize==True and device == torch.device("cuda"):
            sample = sample.to(memory_format=torch.channels_last)  
            sample = sample.half()
        prediction = model.forward(sample)
        prediction = (
            torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=img.shape[:2],
                mode="bicubic",
                align_corners=False,
            )
            .squeeze()
            .cpu()
            .numpy()
        )

    # output
    filename = os.path.join(
        output_path, "result"
    )
    # cv2.namedWindow('imagedepth', cv2.WINDOW_NORMAL)
    # cv2.imshow('image',prediction)
    # cv2.waitKey(0)
    mdepth = utils.write_depth(filename, prediction, bits=2)

    print("finished")

    return mdepth



def processOpenpose(image,op):
    # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    params = dict()
    params["model_folder"] = "../../../models/"

    # Construct it from system arguments
    # op.init_argv(args[1])
    # oppython = op.OpenposePython()
    # Add others in path?

    params["net_resolution"] = "320x176"

    # Starting OpenPose
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()




    imageToProcess = image
    # Process Image
    datum = op.Datum()
    # imageToProcess = cv2.imread(img)
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Display Image
    # print("Body keypoints: \n" + str(datum.poseKeypoints))
    # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    # cv2.waitKey(0)
    print("keypointtype",type(datum.cvOutputData))
    cv2.imwrite("keypoint.jpg",datum.cvOutputData) # 保存路径
    return datum.poseKeypoints
    

    # # 图片大小为320x426
    # tink = np.ones((426,320),dtype='float64')
    # tink = tink
    # print(tink.shape)
    # for i in range(datum.poseKeypoints.shape[0]):
    #     for j in range(datum.poseKeypoints.shape[1]):
    #         x = datum.poseKeypoints[i][j][0]
    #         y = datum.poseKeypoints[i][j][1]
    #         if y>426 or x>320:
    #             continue
    #         score =  datum.poseKeypoints[i][j][2]
    #         #color =  score
    #         color = 1
    #         print("x,y",int(y),int(x))
    #         tink[int(y)][int(x)] = 240 * color / 25
    #         tink[int(y)+1][int(x)] = 240 * color / 25
    #         tink[int(y)][int(x)+1] = 240 * color / 25
    #         tink[int(y)-1][int(x)] = 240 * color / 25
    #         tink[int(y)][int(x)-1] = 240 * color / 25
    #         tink[int(y) + 1][int(x)+1] = 240 * color / 25
    #         tink[int(y)-1][int(x) + 1] = 240 * color / 25
    #         tink[int(y) - 1][int(x)-1] = 240 * color / 25
    #         tink[int(y) + 1][int(x) - 1] = 240 * color / 25

    # plt.imshow(tink,cmap="gray")
    # plt.axis('off')
    # plt.show()
def isSend(l_head,l_mid,l_tail,r_head,r_mid,r_tail,label):
    """Three points on one line
    label = 0:失效 1:只有右手 2：只有左手 3：两只手
    Args:
        Keypoints :Three points
    """
    invalid = np.array([])
    if label == 0:
        return invalid
    elif label == 1:
        # right hand
        print("head tail",r_head[1],r_tail[1])
        a_2 =(r_head[0] - r_mid[0])**2 + (r_head[1] - r_mid[1])**2 + (r_head[2] - r_mid[2])**2
        b_2 =(r_tail[0] - r_mid[0])**2 + (r_tail[1] - r_mid[1])**2 + (r_tail[2] - r_mid[2])**2
        c_2 =(r_head[0] - r_tail[0])**2 + (r_head[1] - r_tail[1])**2 + (r_head[2] - r_tail[2])**2
        r_angle = math.degrees(math.acos((a_2 + b_2 - c_2)/(2 * math.sqrt(a_2) * math.sqrt(b_2))))
        print("rangle",r_angle)
        if ar_min < r_angle < ar_max:
            return r_head
        else:
            return invalid

    elif label == 2:
        #left hand
        print("head tail",l_head[1],l_tail[1])
        a_2 =(l_head[0] - l_mid[0])**2 + (l_head[1] - l_mid[1])**2 + (l_head[2] - l_mid[2])**2
        b_2 =(l_tail[0] - l_mid[0])**2 + (l_tail[1] - l_mid[1])**2 + (l_tail[2] - l_mid[2])**2
        c_2 =(l_head[0] - l_tail[0])**2 + (l_head[1] - l_tail[1])**2 + (l_head[2] - l_tail[2])**2
        l_angle = math.degrees(math.acos((a_2 + b_2 - c_2)/(2 * math.sqrt(a_2) * math.sqrt(b_2))))
        print("langle",l_angle)
        if ar_min < l_angle < ar_max:
            return l_head
        else:
            return invalid

    elif label == 3:
        #left hand
        print("head tail",l_head[1],l_tail[1])
        a_2 =(l_head[0] - l_mid[0])**2 + (l_head[1] - l_mid[1])**2 + (l_head[2] - l_mid[2])**2
        b_2 =(l_tail[0] - l_mid[0])**2 + (l_tail[1] - l_mid[1])**2 + (l_tail[2] - l_mid[2])**2
        c_2 =(l_head[0] - l_tail[0])**2 + (l_head[1] - l_tail[1])**2 + (l_head[2] - l_tail[2])**2
        l_angle = math.degrees(math.acos((a_2 + b_2 - c_2)/(2 * math.sqrt(a_2) * math.sqrt(b_2))))
        print("langle",l_angle)
        # right hand
        print("head tail",r_head[1],r_tail[1])
        a_2 =(r_head[0] - r_mid[0])**2 + (r_head[1] - r_mid[1])**2 + (r_head[2] - r_mid[2])**2
        b_2 =(r_tail[0] - r_mid[0])**2 + (r_tail[1] - r_mid[1])**2 + (r_tail[2] - r_mid[2])**2
        c_2 =(r_head[0] - r_tail[0])**2 + (r_head[1] - r_tail[1])**2 + (r_head[2] - r_tail[2])**2
        r_angle = math.degrees(math.acos((a_2 + b_2 - c_2)/(2 * math.sqrt(a_2) * math.sqrt(b_2))))
        print("rangle",r_angle)
        if ar_min < l_angle < ar_max and ar_min < r_angle < ar_max:
            if l_head[2] > r_head[2]:
                return l_head
            else:
                return r_head
        elif ar_min < l_angle < ar_max and r_angle <= ar_min:
            return l_head
        elif l_angle <= ar_min and ar_min < r_angle < ar_max:
            return r_head
        else:
            return invalid


    
def gtDepth(depth):
    # a = -2.13798
    # b = 3622.8536
    a =  -0.43476123
    b = 1647.1930877842856
    return a * depth + b

def target_not(unot,uvector):
    # 需要知道在哪一个面碰壁
    # 比如y
    tx = uvector[0] * (-unot[1]) / uvector[1] + unot[0]
    tz = uvector[2] * (-unot[1]) / uvector[1] + unot[2]
    return tx,tz


def distance(value,points,vector):
    
    P1 = np.array([value[1],value[2],value[3]])
    P2 = np.array(points).reshape(1,-1)

    # A和B两个向量尾部相连
    A = P1 - P2
    B = np.array(vector)
    # 计算叉乘
    A_B = np.cross(A, B)
    # 计算叉乘的膜
    AB_mo = np.linalg.norm(A_B)
    B_mo = np.linalg.norm(B)
    dist = AB_mo / B_mo
    return dist

def get_eq(name):
    # 需要知道在哪一个面碰壁
    # 比如y

    if name == "TV":
        return 0
    elif name == "AIR":
        return 1
    elif name == "AUDIO":
        return 2
    elif name == "FAN":
        return 3
    elif name == "SWEEPER":
        return 4
    

    

    
def destination_calibration(points,vector,model,equipment):
    # 两条直线的交点？
    if model == "calibration":
        #存点
        firstnode[0] = points
        firstnode[1] = vector
        return "a"
    elif model == "calibration2":
        A = np.array(firstnode[1]).reshape(1,-1)
        B = np.array(vector).reshape(1,-1)
        P1 = np.array(firstnode[0]).reshape(1,-1)
        P2 = np.array(points).reshape(1,-1)
        N = np.cross(A, B).reshape(1,-1)
        # dest = np.linalg.norm(np.dot(N,P2 - P1)) / np.linalg.norm(N)
        a=np.mat([[B[0][0] * N[0][1] - B[0][1] * N[0][0],N[0][0] * A[0][1] - N[0][1] * A[0][0]],[B[0][0] * N[0][2] - B[0][2] * N[0][0], N[0][0] * A[0][2] - N[0][2] * A[0][0]]])#系数矩阵
        b=np.mat([N[0][0] * P2[0][1] - P1[0][1] * N[0][0] - P2[0][0] * N[0][1] + P1[0][0] * N[0][1],N[0][0] * P2[0][2] - P1[0][2] * N[0][0] - P2[0][0] * N[0][2] + P1[0][0] * N[0][2]]).T 
        #  m=B,P2 ,t=A,P1
        x = np.array(solve(a,b)).reshape(1,-1)#方程组的解
        mequipment[equipment] = [radius[equipment],(x[0][0] * B[0][0] + P2[0][0] + x[0][1] * A[0][0] + P1[0][0]) / 2,(x[0][0] * B[0][1] + P2[0][1] + x[0][1] * A[0][1] + P1[0][1]) / 2,(x[0][0] * B[0][2] + P2[0][2] + x[0][1] * A[0][2] + P1[0][2]) / 2]
        print("input",equipment,mequipment[equipment])
        return "b"
    else:
        return "c"


def calculate(poseKeypoints,imageDepth,c_to_w,vector,model,equipment):
    global out
    for i in range(poseKeypoints.shape[0]): # people
        left = [7,6,5]
        leftKeypoints = []
        right = [4,3,2]
        left_complete = True
        right_complete = True
        for j in left:
            x = poseKeypoints[i][j][0]
            y = poseKeypoints[i][j][1]
            if x == 0.0 or y == 0.0:
                left_complete = False
            leftKeypoints.append([x,y])
        # print("left",leftKeypoints)
        # print(leftKeypoints[1][0])

        rightKeypoints = []
        for j in right:
            x = poseKeypoints[i][j][0]
            y = poseKeypoints[i][j][1]
            if x == 0.0 or y == 0.0:
                right_complete = False
            rightKeypoints.append([x,y])
        # print(rightKeypoints)
        # print(rightKeypoints[1][0])
        # print("pose_and_depth_type",poseKeypoints.shape, imageDepth.shape)
        # x, y 是针对于图像坐标系，但是depth是array 先行后列
        # right hand 
        hand_not_x, hand_not_y = int(poseKeypoints[i][4][0]),int(poseKeypoints[i][4][1])

        print("2",hand_not_x,hand_not_y)
        print("depth",gtDepth(imageDepth[hand_not_y][hand_not_x]),imageDepth[hand_not_y][hand_not_x])
        r_head = c_to_w.c_w(c_to_w.pixel_c([hand_not_x,hand_not_y,1],gtDepth(imageDepth[hand_not_y][hand_not_x])))
        hand_not_x, hand_not_y = int(poseKeypoints[i][3][0]),int(poseKeypoints[i][3][1])
        print("3",hand_not_x,hand_not_y)
        r_mid = c_to_w.c_w(c_to_w.pixel_c([hand_not_x,hand_not_y,1],gtDepth(imageDepth[hand_not_y][hand_not_x])))
        hand_not_x, hand_not_y = int(poseKeypoints[i][2][0]),int(poseKeypoints[i][2][1])
        print("4",hand_not_x,hand_not_y)
        r_tail = c_to_w.c_w(c_to_w.pixel_c([hand_not_x,hand_not_y,1],gtDepth(imageDepth[hand_not_y][hand_not_x])))

        # left hand
        hand_not_x, hand_not_y = int(poseKeypoints[i][7][0]),int(poseKeypoints[i][7][1])
        print("2",hand_not_x,hand_not_y)
        l_head = c_to_w.c_w(c_to_w.pixel_c([hand_not_x,hand_not_y,1],gtDepth(imageDepth[hand_not_y][hand_not_x])))
        hand_not_x, hand_not_y = int(poseKeypoints[i][6][0]),int(poseKeypoints[i][6][1])
        print("3",hand_not_x,hand_not_y)
        l_mid = c_to_w.c_w(c_to_w.pixel_c([hand_not_x,hand_not_y,1],gtDepth(imageDepth[hand_not_y][hand_not_x])))
        hand_not_x, hand_not_y = int(poseKeypoints[i][5][0]),int(poseKeypoints[i][5][1])
        print("4",hand_not_x,hand_not_y)
        l_tail = c_to_w.c_w(c_to_w.pixel_c([hand_not_x,hand_not_y,1],gtDepth(imageDepth[hand_not_y][hand_not_x])))


        # 必须手指都是可以看到的
        label = 0
        if right_complete == True:
            label = label + 1
        if left_complete == True:
            label = label + 2

        print("label",label)
        # label = 0:失效 1:只有右手 2：只有左手 3：两只手
        w_points = isSend(l_head,l_mid,l_tail,r_head,r_mid,r_tail,label)
        # IMU
        if w_points.size != 0:
            if model == "calibration" or model == "calibration2":
                ca_message = destination_calibration(w_points,vector,model,equipment)
                for j in range(2):
                    ser.write(str(ca_message).encode("gbk"))
            elif model == "order":
                for key,value in mequipment.items():
                    dis = distance(value,w_points,vector)
                    print("dis",dis)
                    if dis < value[0]:
                        eq = get_eq(key)
                        print("equipment",eq)
                        for j in range(2):
                            ser.write(str(eq).encode("gbk"))
                        break
        else:
            # 失效
            pass

        out = ''
            
        

def reads():
    """ 读取数据 """
    global out
    while True:
        if out == '':
            while ser.inWaiting() > 0:
                out += ser.read(1).decode()  # 一个一个的读取
        if 0xFF == ord('q'): 
            break


def imu_get(str):
    mess = str.split()
    print("model",len(mess))
    if len(mess) == 3:
        model = "calibration"
        z = float(mess[1])
        x = float(mess[2])
        equip = mess[0]
        print("imu",z,x)
    elif len(mess) == 2:
        model = "order"
        z = float(mess[0])
        x = float(mess[1])
        print("imu",z,x)
        equip = ""
    elif len(mess) == 4:
        model = "calibration2"
        z = float(mess[1])
        x = float(mess[2])
        equip = mess[0]
        print("imu",z,x)

    # 单位向量
    # 摄像头与北的夹角
    c_to_n =angle
    # 计算角度
    # 因为是西 所以是负数
    # xita 对于 -y 顺时针为正 逆时针为负c_to_n - (-z)
    xita = (c_to_n - z + 270) % 360
    fai = x + 90
    print("fai",fai,xita)
    # 方向单位向量
    uz = math.cos(math.radians(fai))
    print("uz",uz)
    uy = math.sin(math.radians(xita)) * math.sin(math.radians(fai))
    ux = math.cos(math.radians(xita)) * math.sin(math.radians(fai))
    vec = [ux,uy,uz]
    print("vtype",vec)
    return vec,model,equip

def load_model(model_type,model_path):


    # load network
    if model_type == "dpt_large": # DPT-Large
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
    elif model_type == "dpt_hybrid": #DPT-Hybrid
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
    elif model_type == "midas_v21":
        model = MidasNet(model_path, non_negative=True)
    elif model_type == "midas_v21_small":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
    else:
        print(f"model_type '{model_type}' not implemented, use: --model_type large")
        assert False
    return model


def camera():
    global out
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # 打开摄像头
    mctoworld = CtoWorld()  # 生产矫正对象
    model_ = load_model(args.model_type,args.model_weights)
    while (1):
        # get a frame
        ret, frame = cap.read()
        # frame = cv2.flip(frame, 1) # 摄像头是和人对立的，将图像左右调换回来正常显示
        # frame = cv2.flip(frame, 1) # 摄像头是和人对立的，将图像左右调换回来正常显示
        # show a frame
        cv2.imshow("capture", frame) # 生成摄像头窗口
        if cv2.waitKey(1) and out != '': # 如果按下q 就截图保存并退出
            print("okkkk")
            print(frame.shape)
            x, y = frame.shape[0:2]
            # print("x.y",x,y)
            imgecroped = cv2.resize(frame, (int(y/4), int(x/4)))
            print(imgecroped.shape)
            cv2.imwrite("test.jpg", imgecroped) # 保存路径
            
            cv2.destroyAllWindows()
            mve,order_model,equipment_c  = imu_get(out)
            # process openpose
            start = time.time()
            poseKeypoints = processOpenpose(imgecroped,op)
            end = time.time()
            print("openpose",end - start)
            start = time.time()
            # # compute depth maps
            imageDepth = run(imgecroped, args.output_path, args.model_type, model_, args.optimize)
            end = time.time()
            print("depth",end - start)
            
            calculate(poseKeypoints,imageDepth,mctoworld,mve,order_model,equipment_c)
            # out = ''
            # break
    cap.release()

if __name__ == "__main__":


    try:
        # Import Openpose (Windows/Ubuntu/OSX)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Windows Import
            if platform == "win32":
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append(dir_path + '/../../python/openpose/Release')
                os.environ['PATH']  = os.environ['PATH'] + ';' + dir_path + '/../../x64/Release;' +  dir_path + '/../../bin;'
                import pyopenpose as op
            else:
                # Change these variables to point to the correct folder (Release/x64 etc.)
                sys.path.append('../../python')
                # If you run `make install` (default path is `/usr/local/python` for Ubuntu), you can also access the OpenPose/python module from there. This will install OpenPose and the python library at your desired installation path. Ensure that this is in your python path in order to use it.
                # sys.path.append('/usr/local/python')
                from openpose import pyopenpose as op
        except ImportError as e:
            print('Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e
    except Exception as e:
        print(e)
        sys.exit(-1)

    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input_path', 
        default='input',
        help='folder with input images'
    )

    parser.add_argument('-o', '--output_path', 
        default='output',
        help='folder for output images'
    )

    parser.add_argument('-m', '--model_weights', 
        default=None,
        help='path to the trained weights of model'
    )

    parser.add_argument('-t', '--model_type', 
        default='dpt_hybrid',
        help='model type: dpt_large, dpt_hybrid, midas_v21_large or midas_v21_small'
    )

    parser.add_argument('-n', '--net_resolution', 
        default='240x160',
        help='size of image'
    )

    parser.add_argument('--optimize', dest='optimize', action='store_true')
    parser.add_argument('--no-optimize', dest='optimize', action='store_false')
    parser.set_defaults(optimize=True)

    args = parser.parse_args()
    print("canshu",args)
    # args = parser.parse_known_args()

    # # Custom Params (refer to include/openpose/flags.hpp for more parameters)
    # params = dict()
    # params["model_folder"] = "../../../models/"

    # # Add others in path?
    # for i in range(0, len(args[1])):
    #     curr_item = args[1][i]
    #     if i != len(args[1])-1: next_item = args[1][i+1]
    #     else: next_item = "1"
    #     if "--" in curr_item and "--" in next_item:
    #         key = curr_item.replace('-','')
    #         if key not in params:  params[key] = "1"
    #     elif "--" in curr_item and "--" not in next_item:
    #         key = curr_item.replace('-','')
    #         if key not in params: params[key] = next_item

    default_models = {
        "midas_v21_small": "weights/midas_v21_small-70d6b9c8.pt",
        "midas_v21": "weights/midas_v21-f6b98070.pt",
        "dpt_large": "weights/dpt_large-midas-2f21e586.pt",
        "dpt_hybrid": "weights/dpt_hybrid-midas-501f0c75.pt",
    }

    if args.model_weights is None:
        args.model_weights = default_models[args.model_type]

    # set torch options
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
 
    out = ''

    t1 = threading.Thread(target=reads, name='reads')
    t2 = threading.Thread(target=camera, name='camera')

    t1.start()
    t2.start()    
