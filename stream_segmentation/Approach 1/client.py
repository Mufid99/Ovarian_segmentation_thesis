from imutils.video import VideoStream
import os, sys, subprocess
import time
import imagezmq
import socket
import imutils
import argparse
import client_tools
import cv2
import zmq
import SimpleITK as sitk


# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-s", "--server-ip", required=True,
	help="ip address of the server to which the client will connect")
ap.add_argument("-l", "--location", required=True,
	help="location in which to save video")
ap.add_argument("-r", "--remote-path", required=True,
	help="path in which to find video in remote server")
ap.add_argument("-c", "--ssh-config", required=True,
	help="provide host in ssh config file that will use key instead of password")
ap.add_argument("-v", "--video-path", required=True,
	help="path to play video from")
ap.add_argument("-R", "--use-recorded", action='store_true',
	help="path to play video from")
args = vars(ap.parse_args())

# initialize the ImageSender object with the socket address of the
# server
sender = imagezmq.ImageSender(connect_to="tcp://{}:5555".format(
	args["server_ip"]), REQ_REP=True)

# get current host name to send to server
host_name = socket.gethostname()

if args["use_recorded"]:
	vs = cv2.VideoCapture("./demo.avi")
else:
	vs = cv2.VideoCapture(0)

# sleep to allow time for device to start capturing
if not args["use_recorded"]:
	time.sleep(0.7)

if args["use_recorded"]:
	print("Sending video to server, please wait")
else:
	print("Press Ctrl + c to end video stream")

try:
	while True:
		# read the frame from the camera and send it to the server
		ret, frame = vs.read()
		# reached final frame
		if not ret:
			print("done sending video")
			print("Please wait until segmentation is complete")
			break
		# crop images to focus on subject and remove text
		frame = client_tools.crop_ultrasound_stream(frame)
		# applies linear image interpolation
		frame = cv2.resize(frame, dsize=(256, 256))
		# send the frame to the server to apply segmentation
		message = sender.send_image(host_name, frame)
except KeyboardInterrupt:
	print("\nPlease wait until segmentation is complete")
	pass

# socket options that allow to send multiple times before a receive
sender.zmq_socket.setsockopt(zmq.REQ_RELAXED, 1)
sender.zmq_socket.setsockopt(zmq.REQ_CORRELATE, 1)

# need to sleep here more than 3 seconds so that server would not be trying to receive an image
time.sleep(4)

# sends a message to check if the server is done processing the segmentation
sender.zmq_socket.send_string("done?")
# receives a confirmation so that the segmentation can be obtained 
message = sender.zmq_socket.recv()


# scp command to fetch video segmentation from server
cmd = "scp " + args["ssh_config"] + ":" + args["remote_path"] + " " + args["location"]
subprocess.call(cmd.split(" "))

filename = args["video_path"]

# opens the video file automatically on any OS
def open_file(filename):
    if sys.platform == "win32":
        os.startfile(filename)
    else:
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, filename])
open_file(filename)


