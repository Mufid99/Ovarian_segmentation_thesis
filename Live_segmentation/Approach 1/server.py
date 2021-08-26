# from imutils import build_montages
from datetime import datetime
import numpy as np
import imagezmq
import cv2
import argparse
import subprocess
import time
import signal
import zmq 
import shutil
import os
from pathlib import Path
import server_tools
import SimpleITK as sitk

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--filename", required=True,
	help="filename for video")
ap.add_argument("-fps", "--fps", required=True, type=int,
	help="The fps at which to write the new video")
args = vars(ap.parse_args())

# image hub to receive images
image_hub = imagezmq.ImageHub(REQ_REP=True)

# initialize the dictionary which will contain  information regarding
# when a device was last active, then store the last time the check
# was made was now
lastActive = {}
lastActiveCheck = datetime.now()

client, frame = 0,0

count = 0

width = 256
height = 256
fps = args["fps"]

filename = args["filename"]
vwriter = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'MJPG'), fps, (width, height))
frames = []

while True:
    # After receiving the first image, quit if no images are received within 3 seconds
    # (client disconnect), this could be increased if images are taking longer to process
    if count != 0:
        image_hub.zmq_socket.RCVTIMEO = 3000
    try:
        # receive RPi name and frame from the RPi and acknowledge
	    # the receipt
        client_name, frame = image_hub.recv_image()
        image_hub.send_reply(b'OK')
    except zmq.error.Again:
        print("Video stream ended from client")
        break

    # if a device is not in the last active dictionary then it means
	# that its a newly connected device
    if client_name not in lastActive.keys():
        print("[INFO] receiving data from {}...".format(client_name))
    
    frames.append(frame)

    # record the last active time for the device from which we just
	# received a frame
    lastActive[client_name] = datetime.now()
 
    count += 1

# set paths to both frames and segmentations for nnU-Net
frames_path = "./frames/"
Path(frames_path).mkdir(parents=False, exist_ok=True)
segmentation_path = "./segmentations/"
Path(segmentation_path).mkdir(parents=False, exist_ok=True)

for i, frame in enumerate(frames):
    # setup images in right format for nnU-Net 
    server_tools.convert_2d_image_to_nifti_new(frame, frames_path + "frame%d" % i)

# command to make predictions with trained model
cmd = "nnUNet_predict -i " + frames_path + " -o " + segmentation_path + " -tr nnUNetTrainerV2 -ctr nnUNetTrainerV2CascadeFullRes -m 2d -p nnUNetPlansv2.1 -t Task101_Ovary"
subprocess.call(cmd.split(" "))

# colour for segmentation to take
seg_colour = [0, 255, 0]

segmentations =  os.listdir(segmentation_path)

# sort with human keys so that frames match up with segmentation
for i, seg in enumerate(sorted(segmentations, key=server_tools.human_keys)):
    if seg.endswith(".gz"):
       seg_arr = sitk.ReadImage(segmentation_path + seg)
       seg_arr = np.squeeze(sitk.GetArrayFromImage(seg_arr))
       # stacks segmentation to become rgb
       seg_arr = np.dstack([seg_arr]*3)
       # set the colour of the segmentation
       seg_arr[:,:,0][seg_arr[:,:,0] == 1] = seg_colour[0]
       seg_arr[:,:,1][seg_arr[:,:,1] == 1] = seg_colour[1]
       seg_arr[:,:,2][seg_arr[:,:,2] == 1] = seg_colour[2]
       # mix image and segmentation
       new_frame = ((0.6 * frames[i]) + (0.4 * seg_arr)).astype("uint8")
       # write the mix to the video
       vwriter.write(new_frame)

# delete directories after video is complete
shutil.rmtree(frames_path)
shutil.rmtree(segmentation_path)

# send a message to client indicating segmentation is done so video can be requested
message = image_hub.zmq_socket.recv_string()
image_hub.zmq_socket.send_string("done")

vwriter.release()
cv2.destroyAllWindows()