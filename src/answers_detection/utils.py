import cv2
import numpy as np


## TO STACK ALL THE IMAGES IN ONE WINDOW
def stackImages(imgArray, scale, lables=[]):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        hor_con = np.concatenate(imgArray)
        ver = hor
    if len(lables) != 0:
        eachImgWidth = int(ver.shape[1] / cols)
        eachImgHeight = int(ver.shape[0] / rows)
        # print(eachImgHeight)
        for d in range(0, rows):
            for c in range(0, cols):
                cv2.rectangle(ver, (c * eachImgWidth, eachImgHeight * d),
                              (c * eachImgWidth + len(lables[d][c]) * 13 + 27, 30 + eachImgHeight * d), (255, 255, 255),
                              cv2.FILLED)
                cv2.putText(ver, lables[d][c], (eachImgWidth * c + 10, eachImgHeight * d + 20),
                            cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
    return ver


def rect_contour(contours):
    rec_con = []

    for cont in contours:
        area = cv2.contourArea(cont)
        if area > 50:
            peri = cv2.arcLength(cont, True)
            approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
            if len(approx) == 4:
                rec_con.append(cont)
    rec_con = sorted(rec_con, key=cv2.contourArea, reverse=True)

    return rec_con


def get_corner_points(cont):
    peri = cv2.arcLength(cont, True)
    approx = cv2.approxPolyDP(cont, 0.02 * peri, True)
    return approx


def reorder(points):
    points = points.reshape((4, 2))
    points_new = np.zeros((4, 1, 2), np.int32)
    add = points.sum(1)
    points_new[0] = points[np.argmin(add)]  # [0, 0]
    points_new[3] = points[np.argmax(add)]  # [w, h]
    diff = np.diff(points, axis=1)
    points_new[1] = points[np.argmin(diff)]  # [w, 0]
    points_new[2] = points[np.argmax(diff)]  # [0, h]
    return points_new


def split_boxes(img) -> [np.array]:
    rows = np.array_split(img, 12)
    matrix = []
    for row in rows:
        cols = np.array_split(row.T, 15)
        row_marks = []
        for elem in cols:
            col = elem.T[5:-5, 5:-5]
            row_marks.append(col)
        matrix.append(row_marks)
    return matrix
