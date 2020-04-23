from imutils.object_detection import non_max_suppression
import numpy as np
import pytesseract
import argparse
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
            endY = int(offsetY - (sin * xData1[col]) + (cos * xData2[col]))
            startX = int(endX - w)
            startY = int(endY - h)

            # appending box co-ords and confidence to our lists
            rects.append((startX, startY, endX, endY))
            confidences.append(scoresData[col])

    # returning the rects and confidences
    return (rects, confidences)


# parsing arguments for script
ap = argparse.ArgumentParser()

ap.add_argument("-i", "--image", type=str, help="path to input image!")
ap.add_argument("-east", "--east", type=str, help="path to EAST text detector!")
ap.add_argument("-c", "--min_confidence", type=float, default=0.5,
                help="minimum probability required to inspect a region!")
ap.add_argument("-w", "--width", type=int, default=320, help="nearest multiple of 32 for resized width!")
ap.add_argument("-e", "--height", type=int, default=320, help="nearest multiple of 32 for resized height!")
#NOTE: Try changing confidence values if you get wrong output... it all depends on the models sensitivity of accessing the text
ap.add_argument("-p", "--padding", type=float, default=0.0, help="amount of padding to add to each border of ROI!")

args = vars(ap.parse_args())

# loading input image
image = cv2.imread(args["image"])
# copy to display script results
orig = image.copy()

# extracting image dimensions
(origH, origW) = image.shape[:2]

# set the new width and height and then determine the ratio in change for both the width and height
(newH, newW) = (args["height"], args["width"])
rH = origH / float(newH)
rW = origW / float(newW)

# resizing the image and extracting new dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# Defining two output layer names for EAST detector model that will be useful to us
# first being the output probabilities aka scores and second will be used for calculating bounding boxes
layerNames = ["feature_fusion/Conv_7/Sigmoid", "feature_fusion/concat_3"]

# loading the pre-trained EAST text detector from disk
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(args["east"])

# the first step is to create blobs from image and then pass the blob as input to our model which
# will return two output layers
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
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
    endX = min(origW, endX + (dx * 2))
    endY = min(origH, endY + (dy * 2))

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

# sorting the results bounding box coordinates from top to bottom aka y coordinate
results = sorted(results, key=lambda r: r[0][1])

#output = orig.copy() -- use it to display the results all at once in an image
# lets begin looping over the results
for((startX, startY, endX, endY), text) in results:
    # displaying the OCR'd text
    print("!!!...OCR TEXT...!!!")
    print("********************")
    print("********************")
    print("{}\n".format(text))

    # drawing the text on input image
    # lets filter the non-ASCII text
    text = "".join([c if ord(c) < 128 else "" for c in text]).strip()
    output = orig.copy()
    cv2.rectangle(output, (startX, startY), (endX, endY), (0, 0, 255), 2)
    cv2.putText(output, text, (startX, startY - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    # NOTE : This will display the detected texts one by one if you press "c" till every text is displayed.
    # displaying the result
    cv2.imshow("Text Recognition : ", output)
    key = cv2.waitKey(0) & 0xFF
    # to continue display all results give keyboard interrupt "c"
    if key == ord("c"):
        continue
    # to quit the process give interrupt "q"
    if key == ord("q"):
        break
# Uncomment this and comment the above identical block to display detections all at once in the image
# # displaying the result
# cv2.imshow("Text Detection : ", output)
# key = cv2.waitKey(0) & 0xFF
#
# if key == ord("q"):
#     cv2.destroyAllWindows()

# the cleanup
cv2.destroyAllWindows()