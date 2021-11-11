import os 
import time 
from math import sqrt
import numpy as np
import cv2 
import face_recognition as FR
from decord import cpu, VideoReader
from argparse import ArgumentParser as AP 
print(f"Imports complete.")

THRESH_AREA = 256*512 # The minimum area to give face recognition module as input
THRESH_FACE = 256 # Minimum face bbox dims

def get_input_scale(image, thresh_area):
    '''
    Calculates an optimized input scale for resizing images

    returns (float) scale
    '''
    gamma = 0.98
    height, width, channels = image.shape
    area = height*width
    init_area = height*width
    while area >= thresh_area:
        height = height*gamma
        width = width*gamma
        area = height*width
    return sqrt(init_area/area)


def get_face(image, scale):
    '''
    Gets face coordinates from a numpy image array. 

    image: (numpy.array) Target image.
    returns None if there is no face 
    else returns [(x1, y1), (x2, y2)]
    where (x1, y1) and (x2, y2) are face bounding box.
    '''
    _orig_width, _orig_height = image.shape[1], image.shape[0]
    _img_orig = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _img = cv2.resize(_img_orig, (int(_orig_width/scale), int(_orig_height/scale)))
    locs = FR.face_locations(_img)
    if len(locs) > 0:
        locs = locs[0]
        x1, y1 = int(locs[3]*scale), int(locs[0]*scale)
        x2, y2 = int(locs[1]*scale), int(locs[2]*scale)
        return [(x1, y1), (x2, y2)]
    else:
        return None
    

# Get arguments
parser = AP()
parser.add_argument("src_video", 
        type=str, help="source video path")
parser.add_argument("num_frames", 
        type=int, help="number for frames to extract")
parser.add_argument("out_dir",
        type=str, help="new output dir name to extract faces")
args = parser.parse_args()


# Get video metrics
vr = VideoReader(args.src_video)
total_frames = len(vr)
fps = vr.get_avg_fps()
sample_frame =  np.array(vr[0].asnumpy())
print(f"Original shape: {sample_frame.shape}")
del(vr)


# extraction params
start_frame = int(fps*10)
end_frame = int(total_frames - fps*20)
interval = max(int((end_frame-start_frame)/args.num_frames), 1)
selected_frames = [i for i in range(start_frame, end_frame, interval)]
scale = get_input_scale(sample_frame, THRESH_AREA)
new_dims = [int(sample_frame.shape[0]/scale), int(sample_frame.shape[1]/scale)]
print(f"Input frame dims: {new_dims}")
print(f"Face dims threshold: {THRESH_FACE} x {THRESH_FACE} px\n")


# Extract faces
out_dir_path = args.out_dir
os.system(f"mkdir ./{out_dir_path}")
print(f"Extracting faces to {out_dir_path}")
cap = cv2.VideoCapture(args.src_video)
frame_idx = -1
n_faces = 0
bef_clock = time.time()
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        frame_idx += 1
        if frame_idx in selected_frames:
            bbox = get_face(frame, scale)
            if bbox:
                p1, p2 = bbox[0], bbox[1]
                x1, y1 = p1
                x2, y2 = p2
                face = np.array(frame)[y1:y2, x1:x2, :]
                # Check face dims
                if face.shape[0] >= THRESH_FACE and face.shape[1] >= THRESH_FACE:
                    face = cv2.resize(face, (THRESH_FACE, THRESH_FACE))
                    cv2.imwrite(f"./{out_dir_path}/face_{n_faces}.jpg", face)
                    n_faces += 1
        pct_faces = (n_faces/args.num_frames)*100 
        pct_frames = (frame_idx/(end_frame-start_frame))*100
        elapsed_time = (time.time()-bef_clock)/60 # Minutes
        print(f"Faces: {n_faces} ({pct_faces:.1f}%) | Frames: {frame_idx}/{total_frames} ({pct_frames:.1f}%) | elapsed time: {elapsed_time:.1f} Min.", end='\r')
    else:
        break 
cap.release()
print(f"\n{n_faces} frames captured.")
