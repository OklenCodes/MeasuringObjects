from scipy.spatial import distance as dist
from imutils import perspective
from imutils import contours
import numpy as np
import imutils
import cv2

def midpoint(ptA, ptB):
    return ((ptA[0] + ptB[0]) *0.5, (ptA[1] + ptB[1]) * 0.5)
# Helper Method to compute the midpoint between two sets of (x, y) coordinates

image = cv2.imread("example1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7,7), 0)
edged = cv2.Canny(gray, 50, 100)
edged = cv2.dilate(edged, None, iterations=1)
edged = cv2.erode(edged, None, iterations=1)
# Dilation + erosion to close any gaps in between edges in the edge map

cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

cnts = imutils.grab_contours(cnts)

(cnts, _) = contours.sort_contours(cnts) # Grabbing Contours from left to right
pixelsPerMetric = None

for c in cnts:
    if cv2.contourArea(c) < 100:  # If contour is below a certain size ignore
        continue

    orig = image.copy()
    box = cv2.minAreaRect(c)
    box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
    box = np.array(box, dtype="int")

    box = perspective.order_points(box) # Arranging bounding box coordinates in top-left, top-right clockwise order
    cv2.drawContours(orig, [box.astype("int")], -1, (0, 255, 0), 2)

    for (x,y) in box:
        cv2.circle(orig, (int(x), int(y)), 5, (0, 0, 255), -1)

    (tl, tr, bl, br) = box #unpacks bounding box
    (tltrX, tltrY) = midpoint(tl, tr) #Midpoint between topleft, topright
    (blbrX, blbrY) = midpoint(bl, br) #midpoint between bottomleft, bottomright

    (tlblX, tlblY) = midpoint(tl, bl) #Midpoint between topleft, bottom left
    (trbrX, trbrY) = midpoint(tr, br) #midpoint between topright, bottom right

    """ Drawing the midpoints"""
    cv2.circle(orig, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
    cv2.circle(orig, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)

   #Drawing Lines between the midpoints

    cv2.line(orig, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 0, 255), 2)
    cv2.line(orig, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)), (255, 0, 255), 2)

    """Computing Euclidean distance between midpoints"""
    dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
    dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

    if pixelsPerMetric is None:
        pixelsPerMetric = dB / 1.0 # Helps to compute the size of objects in image

    # compute the dimensions of the object in
    dimA = dA / pixelsPerMetric
    dimB = dB / pixelsPerMetric

    # draw the object sizes on the image
    cv2.putText(orig, "{:.1f}in".format(dimA),
                (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)
    cv2.putText(orig, "{:.1f}in".format(dimB),
                (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, (255, 255, 255), 2)

    # show the output image
    cv2.imshow("Image", orig)
    cv2.waitKey(0)


