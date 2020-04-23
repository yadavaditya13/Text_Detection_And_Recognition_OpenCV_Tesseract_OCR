# NOTE :: This file can be used for both Video files and Live Cam Text detection
# importing the necessary packages

from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression

import numpy as np
import pytesseract
import argparse
import imutils
import time
import cv2


# this function is used for extracting bounding boxes with confidence score from EAST predictions
def decode_predictions(scores, geometry):
    # grabbing # of rows and columns from scores volume,
    # then initialize our set of bounding box and confidence score

    (numRows, numCols) = scores.shape[2:4]
    rects = []
    confidences = []

    # looping over the number of rows
    for row in range(0, numRows):
        # extracting scores and bounding box coordinates for currently detected text
        scoresData = scores[0, 0, row]
        xData0 = geometry[0, 0, row]
        xData1 = geometry[0, 1, row]
        xData2 = geometry[0, 2, row]
        xData3 = geometry[0, 3, row]
        anglesData = geometry[0, 4, row]

        # looping over the columns in each row
        for col in range(0, numCols):
            # filtering out weak probabilities
            if scoresData[col] < args["min_confidence"]:
                continue

            # compute the offset factor as our resulting feature
            # maps will be 4x smaller than the input image
            (offsetX, offsetY) = (col * 4.0, row * 4.0)

            # extracting rotation angle for the prediction and computing sine and cosine
            angle = anglesData[col]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # lets derive height and width of bounding box using geometry volume
            h = xData0[col] + xData2[col]
            w = xData1[col] + xData3[col]

            # getting co-ordinates of bounding box
            endX = int(offsetX + (cos * xData1[col]) + (sin * xData2[col]))
            endY = int(offsetY + (sin * xData1[col]) + (cos * xData2[col]))
            startX = int(endX - w)
            startY = int(endY - h)

            # appending box co-ords and confidence to our lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[col])

    # returning the rects and confidences
    return (rects, confidences)


# parsing arguments for script
ap = argparse.ArgumentParser()

ap.add_argument("-east", "--east", type=str, required=True,  help="path to EAST text detector!")
ap.add_argument("-v", "--video", type=str, help="path to optinal input video file")
ap.add_argument("-c", "--min_confidence", type=float, default=0.5,
                help="minimum probability required to inspect a region!")
ap.add_argument("-w", "--width", type=int, default=320, help="nearest multiple of 32 for resized width!")
ap.add_argument("-e", "--height", type=int, default=320, help="nearest multiple of 32 for resized height!")
# NOTE: Try changing confidence values if you get wrong output... it all depends on the models sensitivity of accessing the text
ap.add_argument("-p", "--padding", type=float, default=0.0, help="amount of padding to add to each border of ROI!")

args = vars(ap.parse_args())

# initialize the original frame dimensions, new frame dimensions,
# and ratio between the dimensions
(W, H) = (None, None)
(newW, newH) = (args["width"], args["height"])
(rW, rH) = (None, None)

# Defining two output layer names for EAST detector model that will be useful to us
# first being the output probabilities aka scores and second will be used for calculating bounding boxes
layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

# loading the pre-trained EAST text detector from disk
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# If a video file was not supplied to the script ... we will run live cam
if not args.get("video", False):
    print("[INFO] We are going Live...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    # If we have a video file to run
    print("[INFO] Loading the required Video File...")
    vs = cv2.VideoCapture(args["video"])

# starting the fps throughput estimator
fps = FPS().start()

# loop over frames from the video stream
while True:
    # grab the current frame, then handle if we are using a
    # VideoStream or VideoCapture object
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame

    # If stream ends
    if frame is None:
        break

    # resizing the frame and maintaining it's aspect ratio
    frame = imutils.resize(frame, width=1000)
    orig = frame.copy()
    (OH, OW) = frame.shape[:2]

    # if our frame dimensions are None, we still need to compute the
    # ratio of old frame dimensions to new frame dimensions
    if W is None or H is None:
        (origH, origW) = frame.shape[:2]
        rW = origW / float(newW)
        rH = origH / float(newH)

    # resize the frame, this time ignoring aspect ratio
    frame = cv2.resize(frame, (newW, newH))
    (H, W) = frame.shape[:2]

    # the first step is to create blobs from image and then pass the blob as input to our model which
    # will return two output layers
    blob = cv2.dnn.blobFromImage(frame, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    # decoding the predictions and then applying non-maxima suppression to suppress weak and overlapping boxes
    (rects, confidences) = decode_predictions(scores, geometry)
    boxes = non_max_suppression(np.array(rects), probs=confidences)

    # This finishes the task of detecting the texts
    # Lets begin text recognition using Tesseract

    # initializing results list
    results = []

    # looping over the bounding boxes
    for (startX, startY, endX, endY) in boxes:
        # scaling bounding box coordinates based on respective ratios
        # initially we resized the image now we are just trying to get it back to original shape
        startX = int(startX * rW)
        startY = int(startY * rH)
        endX = int(endX * rW)
        endY = int(endY * rH)

        # lets add padding to text by calculating deltas in (x, y) directions
        dx = int((endX - startX) * args["padding"])
        dy = int((endY - startY) * args["padding"])

        # applying padding to each bounding box
        startX = max(0, startX - dx)
        startY = max(0, startY - dy)
        endX = min(OW, endX + (dx * 2))
        endY = min(OH, endY + (dy * 2))

        # extracting the actual padded ROI
        roi = orig[startY:endY, startX:endX]

        # now we shall be using tesseract to OCR text
        # we need to supply three things
        # - a language
        # - an oem flag of 1 indicating the use of LSTM neural network for OCR
        # - a psm flag of value 7 to show that we will be trating the ROI as a single line of text

        config = ("-l eng --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)

        # appending bounding box coords and OCR'd text to results
        results.append(((startX, startY, endX, endY), text))

    fps.update()
    # sorting the results bounding box coordinates from top to bottom aka y coordinate
    results = sorted(results, key=lambda r: r[0][1])

    # -- use it to display the results all at once in an image
    output = orig.copy()
    # lets begin looping over the results
    for ((startX, startY, endX, endY), text) in results:
        # displaying the OCR'd text
        print("!!!...OCR TEXT...!!!")
        print("********************")
        print("********************")
        print("{}\n".format(text))

        # drawing the text on frame
        # lets filter the non-ASCII text
        text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
        #output = orig.copy()
        cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(output, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

    # displaying the result
    cv2.imshow("Frame with Text Detection : ", output)
    key = cv2.waitKey(1) & 0xFF

    # to quit the process give interrupt "q"
    if key == ord("q"):
        break

# Stopping the timer and displaying FPS information
fps.stop()
print("[INFO] Elapsed Time: {:.2f}".format(fps.elapsed()))
print("[INFO] Approx. FPS: {:.2f}".format(fps.fps()))

# if we are using a webcam, release the pointer
if not args.get("video", False):
    vs.stop()

# otherwise, release the file pointer
else:
    vs.release()

# close all windows
cv2.destroyAllWindows()