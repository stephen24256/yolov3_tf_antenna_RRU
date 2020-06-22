import cv2
import time
import numpy as np
from PIL import Image,ImageDraw
import torch
from evaluate_mAP import YoloTest
import core.utils as utils


with torch.no_grad() as grad:
    video_path = r"./docs/video/3.mp4"  # 5FPS
    i = 0
    vid = cv2.VideoCapture(video_path)
    print(vid.get(cv2.CAP_PROP_FPS))  # 9帧每秒
    while True:
        ret,frame = vid.read()   # ret是否读成功，frame 图像
        print("12",frame.shape)
        if ret:
            # frame1 = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = np.array(frame)
            print(image.shape)
            detector = YoloTest()
            boxes = detector.predict(image)
            image = utils.draw_bbox(image, boxes, show_label=True)
        else:
            raise ValueError("No image!")

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):  # 1为 1毫秒
            break
