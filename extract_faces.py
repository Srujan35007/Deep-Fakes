import os 
import cv2
import time 
import math 
import matplotlib.pyplot as plt 
import numpy as np 
import pandas as pd 
import face_recognition
from decord import VideoReader, cpu
from tqdm import tqdm
from argparse import ArgumentParser as AP 
print(f"Imports complete")

parser = AP()
parser.add_argument('video_file_path')
parser.add_argument('req_frames')
parser.add_argument('out_folder')
args = parser.parse_args()

video_path = args.video_file_path
vr = VideoReader(video_path, ctx=cpu(0))
FPS = int(vr.get_avg_fps())
print(f"Video loaded.")

count = 0
exceptions = 0
scale = 4
thresh = 128
start_frame = 0
end_frame = len(vr)-1
frame_step = max((end_frame-start_frame)//int(args.req_frames), 1)
orig_shape = vr[0].asnumpy().shape
print(f"Original shape: {orig_shape[:-1]}")
print(f"Input shape: {np.array(orig_shape[:-1])//scale}")
print(f"Threshold: {thresh}x{thresh} Px")
os.system(f'mkdir ./{args.out_folder}')
for i in tqdm(range(start_frame, end_frame, frame_step)):
    try:
        img = np.asarray(vr[i].asnumpy())
        img2 = cv2.resize(img, (img.shape[1]//scale, img.shape[0]//scale))
        face_locs = face_recognition.face_locations(img2, model='hog')[0]
        x1,y1 = face_locs[3], face_locs[0]
        x2,y2 = face_locs[1], face_locs[2]
        p1 = (x1,y1)
        p2 = (x2,y2)
        img2 = cv2.rectangle(img2, p1,p2,(255,0,0), 2)
        cropped = img[y1*scale:y2*scale,x1*scale:x2*scale]
        if cropped.shape[0] >= thresh and cropped.shape[1] >= thresh:
            cropped = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
            cropped = cv2.resize(cropped, (thresh, thresh))
            cv2.imwrite(f'./{args.out_folder}/image_{count}.jpg', cropped)
            count += 1
    except:
        exceptions += 1

print(f"{count} frames extracted.")
print(f"{exceptions} Exceptions")
