import cv2
import numpy as np
from detection_functions import constants
import io
from PIL import Image
from fastapi import UploadFile
from collections import defaultdict
from symbols_detection.model_eval import get_handwritten_text


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

async def file_to_cv_image(file: UploadFile):
    img_bytes = await file.read()
    stream = io.BytesIO(img_bytes)
    img = Image.open(stream)
    open_cv_image = np.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
    return open_cv_image


def np2cv(np_img: np.array):
    open_cv_image = np_img[:, :, ::-1].copy()
    return open_cv_image

def get_anses_matrix(ans_matrix):
    answers = defaultdict(list)
    for i, row in enumerate(ans_matrix):
        for j, col in enumerate(row):
            sum_elem = np.sum(col)
            if sum_elem > constants.FILL_THRESH:
                answers[i].append(j)
    return answers


def get_anses(cv_image):
    width_scan = constants.SCAN_WIDTH
    height_scan = constants.SCAN_HEIGHT
    # Prepare img
    img = cv2.resize(cv_image, (width_scan, height_scan))
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 1)
    img_canny = cv2.Canny(img_blur, 10, 50)
    # Find contours
    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # Find rectangles
    rect_con = rect_contour(contours)
    biggest_con = get_corner_points(rect_con[0])
    if biggest_con.size != 0:
        biggest_con_new = reorder(biggest_con)

        width_area = constants.ANSWER_AREA_WIDTH
        height_area = constants.ANSWER_AREA_HEIGHT
        pt1 = np.float32(biggest_con_new)
        pt2 = np.float32([[0, 0], [width_area, 0], [0, height_area], [width_area, height_area]])
        matrix = cv2.getPerspectiveTransform(pt1, pt2)
        img_area = cv2.warpPerspective(img, matrix, (width_area, height_area))
        img_area_gray = cv2.cvtColor(img_area, cv2.COLOR_BGR2GRAY)
        # Apply threshold
        img_area_thresh = cv2.threshold(img_area_gray, 220, 255, cv2.THRESH_BINARY_INV)[1]
        # Get family name
        family_name_img = constants.crop_img(img_area, constants.ANCHORS_POINTS["family_name"])
        im_pil = Image.fromarray(family_name_img)
        family_name = get_handwritten_text(im_pil)
        # Get answers images
        ans_matrix = split_boxes(constants.crop_img(img_area_thresh, constants.ANCHORS_POINTS["answers"]))
        # Get answers
        return get_anses_matrix(ans_matrix), family_name


def get_answers_stats(answers_student: dict[int, list], answers_teacher: dict[int, list]) -> (int, int):
    all_answers = 0
    correct_answers = 0

    for task, anses in answers_student.items():
        if task not in answers_teacher:
            all_answers += len(anses)
            continue
        for ans in anses:
            if ans in answers_teacher[task]:
                correct_answers += 1
        all_answers += len(anses)

    return all_answers, correct_answers
