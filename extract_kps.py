import os

import onnxruntime as ort

from utils.inference import extract_keypoints_2
from utils import get_config


def main(cfg):
    YOLOV6_PATH = cfg.yolov6_path
    VITPOSE_PATH = cfg.vitpose_path
    FOLDER_PATH = cfg.folder_path

    # if cfg.cpu:
    #     EP_list = ['CPUExecutionProvider']
    # else:
    EP_list = ['CUDAExecutionProvider']

    yolov6_sess = ort.InferenceSession(YOLOV6_PATH, providers=EP_list)
    vitpose_sess = ort.InferenceSession(VITPOSE_PATH, providers=EP_list)
    # TODO : implement smooth_net feature
    # if cfg.smooth_net:
    #     smooth_net = ort.InferenceSession('smoothnet-32.onnx', providers=EP_list)

    os.system("") # make terminal to be able to use ANSI escape

    if FOLDER_PATH:
        print("Extract keypoints from images in the folder")
        
        extract_keypoints_2(FOLDER_PATH, yolov6_sess, vitpose_sess, cfg)


if __name__ == "__main__":
    cfg = get_config()

    main(cfg)