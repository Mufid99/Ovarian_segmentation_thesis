import inference
import numpy as np
import cv2

# Crop stream from ultrasound to desired patch
def crop_ultrasound_stream(img):
    top_boundary = int(img.shape[0] * 0.1)
    bottom_boundary =int(img.shape[0] * 0.7)
    left_boundary = int(img.shape[1] * 0.2)
    right_boundary = int(img.shape[1] * 0.8)
    return img[top_boundary:bottom_boundary, left_boundary:right_boundary]

if __name__ == '__main__': 
    # Initialises predictor including model and parameters
    prdr = inference.Predictor()

    # Colours for labels
    label1_colour = [0, 255, 0]
    label2_colour = [255, 255, 0]

    vs = cv2.VideoCapture("./demo.avi")
    # vs = cv2.VideoCapture(0)
    
    ret, frame = vs.read()
    while ret:

        frame = crop_ultrasound_stream(frame)

        # resample image to size for model to predict on
        frame = cv2.resize(frame, dsize=(256, 256))
        
        # transpose image so it can be accepted by nnU-Net
        frame_t = frame.transpose((2, 0, 1))

        # add extra dimension, required by nnU-Net
        frame_t = frame_t[:, None]

        # applies segmentation on image 
        seg_arr = prdr.predict(frame_t, ensemble=False, do_tta=False)
        
        # colours produces segmentation with desired colour
        seg_arr = np.squeeze(seg_arr)
        seg_arr = np.dstack([seg_arr]*3)
        seg_arr[:,:,0][seg_arr[:,:,0] == 1] = label1_colour[0]
        seg_arr[:,:,1][seg_arr[:,:,1] == 1] = label1_colour[1]
        seg_arr[:,:,2][seg_arr[:,:,2] == 1] = label1_colour[2]
        seg_arr[:,:,0][seg_arr[:,:,0] == 2] = label2_colour[0]
        seg_arr[:,:,1][seg_arr[:,:,1] == 2] = label2_colour[1]
        seg_arr[:,:,2][seg_arr[:,:,2] == 2] = label2_colour[2]

        # mix image and segmentation
        new_frame = ((0.7 * frame) + (0.3 * seg_arr)).astype("uint8")

        # resizable window
        cv2.namedWindow('Segmentation', cv2.WINDOW_KEEPRATIO)

        # adds text as legend
        cv2.putText(new_frame, 'Label 1', (180, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.3, label1_colour, 1, cv2.LINE_AA)
        cv2.putText(new_frame, 'Label 1 & 2', (180, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.3, label2_colour, 1, cv2.LINE_AA)

        cv2.imshow('Segmentation', new_frame)

        if cv2.waitKey(1) == ord('q'):
            break
        ret, frame = vs.read()

    