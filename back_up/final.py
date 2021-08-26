"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
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

ser = serial.Serial('COM4', 9600)


def run(img_name, output_path, model_path, model_type="midas_v21_small", optimize=True):
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
        model = DPTDepthModel(
            path=model_path,
            backbone="vitl16_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode = "minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "dpt_hybrid": #DPT-Hybrid
        model = DPTDepthModel(
            path=model_path,
            backbone="vitb_rn50_384",
            non_negative=True,
        )
        net_w, net_h = 384, 384
        resize_mode="minimal"
        normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    elif model_type == "midas_v21":
        model = MidasNet(model_path, non_negative=True)
        net_w, net_h = 384, 384
        resize_mode="upper_bound"
        normalization = NormalizeImage(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
    elif model_type == "midas_v21_small":
        model = MidasNet_small(model_path, features=64, backbone="efficientnet_lite3", exportable=True, non_negative=True, blocks={'expand': True})
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
def isSend(Keypoints):
    """Three points on one line

    Args:
        Keypoints :Three points
    """
    a_2 =(Keypoints[0][0] - Keypoints[1][0])**2 + (Keypoints[0][1] - Keypoints[1][1])**2
    b_2 =(Keypoints[1][0] - Keypoints[2][0])**2 + (Keypoints[1][1] - Keypoints[2][1])**2
    c_2 =(Keypoints[0][0] - Keypoints[2][0])**2 + (Keypoints[0][1] - Keypoints[2][1])**2
    angle = math.acos((a_2 + b_2 - c_2)/(2 * math.sqrt(a_2) * math.sqrt(b_2)))
    print("angle",angle)
    if 0 <= angle <= math.pi :
        print("ok")
        return True
    else:
        return False
    
    

def calculate(poseKeypoints,imageDepth):
    for i in range(poseKeypoints.shape[0]): # people
        left = [7,6,5]
        leftKeypoints = []
        right = [4,3,2]
        for j in left:
            x = poseKeypoints[i][j][0]
            y = poseKeypoints[i][j][1]
            leftKeypoints.append([x,y])
        # print("left",leftKeypoints)
        # print(leftKeypoints[1][0])

        rightKeypoints = []
        for j in right:
            x = poseKeypoints[i][j][0]
            y = poseKeypoints[i][j][1]
            rightKeypoints.append([x,y])
        print(rightKeypoints)
        print(rightKeypoints[1][0])
        if(isSend(leftKeypoints)):
            # send data
            getDepth = 1
            print("leftHand")
            # getDepth = imageDepth[int(poseKeypoints[i][7][1])][int(poseKeypoints[i][7][0])]
            for i in range(5):
                success_bytes = ser.write(str(getDepth).encode("gbk"))
            # print("getDepth",imageDepth[int(poseKeypoints[i][7][1])][int(poseKeypoints[i][7][0])])
        # if(isSend(rightKeypoints)):
        #     # getDepth = imageDepth[int(poseKeypoints[i][4][1])][int(poseKeypoints[i][4][0])]
        #     getDepth = 1
        #     print("leftHand")
        #     for i in range(5):
        #         success_bytes = ser.write(str(getDepth).encode("gbk"))
        #     # print("getDepth",imageDepth[int(poseKeypoints[i][4][1])][int(poseKeypoints[i][4][0])])
        

def reads():
    """ 读取数据 """
    global out
    while True:
        if out == '':
            while ser.inWaiting() > 0:
                out += ser.read(1).decode()  # 一个一个的读取
            print(out)
        if 0xFF == ord('q'): # 如果按下q 就截图保存并退出
            break

def camera():
    global out
    cap = cv2.VideoCapture(0) # 打开摄像头
    while (1):
        # get a frame
        ret, frame = cap.read()
        # frame = cv2.flip(frame, 1) # 摄像头是和人对立的，将图像左右调换回来正常显示
        # frame = cv2.flip(frame, 1) # 摄像头是和人对立的，将图像左右调换回来正常显示
        # show a frame
        cv2.imshow("capture", frame) # 生成摄像头窗口
        
        if cv2.waitKey(1) and out != '': # 如果按下q 就截图保存并退出
            print("okkkk")
            x, y = frame.shape[0:2]
            
            imgecroped = cv2.resize(frame, (int(y / 2), int(x / 2)))
            print(imgecroped.shape)
            cv2.imwrite("test.jpg", imgecroped) # 保存路径
            
            cv2.destroyAllWindows()
            # process openpose
            poseKeypoints = processOpenpose(imgecroped,op)
            # print(type(poseKeypoints))
            # compute depth maps
            imageDepth = run(imgecroped, args.output_path, args.model_weights, args.model_type, args.optimize)
            calculate(poseKeypoints,imageDepth)
            # out = ''
            break
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
        default='midas_v21_small',
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
    
