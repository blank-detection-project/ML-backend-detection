import cv2
import numpy as np

import src.detection_functions.constants
from src.detection_functions import constants, detection_utils


def get_digits(path):

    width_img = src.answers_detection.constants.SCAN_WIDTH
    height_img = src.answers_detection.constants.SCAN_HEIGHT

    # Prepocessing
    img = cv2.imread(path)
    img = cv2.resize(img, (width_img, height_img))
    img_contours = img.copy()
    img_biggest_contour = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, (5,5), 1)
    img_canny = cv2.Canny(img_blur, 10, 50)

    #Find contours
    contours, hierarchy = cv2.findContours(img_canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 5)

    #Find rectangles
    rect_con = detection_utils.rect_contour(contours)
    biggest_con = detection_utils.get_corner_points(rect_con[0])

    if biggest_con.size != 0:
        cv2.drawContours(img_biggest_contour, biggest_con, -1, (0, 255, 0), 20)
        biggest_con_new = detection_utils.reorder(biggest_con)

        width_area = src.answers_detection.constants.ANSWER_AREA_WIDTH
        height_area = src.answers_detection.constants.ANSWER_AREA_HEIGHT
        pt1 = np.float32(biggest_con_new)
        pt2 = np.float32([[0, 0], [width_area, 0], [0, height_area], [width_area, height_area]])
        matrix = cv2.getPerspectiveTransform(pt1, pt2)
        img_area = cv2.warpPerspective(img, matrix, (width_area, height_area))
        img_area_gray = cv2.cvtColor(img_area, cv2.COLOR_BGR2GRAY)

        # Apply threshold
        img_area_thresh = cv2.threshold(img_area_gray, 220, 255, cv2.THRESH_BINARY_INV)[1]

        # To Lev
        img1 = constants.crop_img(img_area_thresh, constants.ANCHORS_POINTS["group_1"])
        img2 = constants.crop_img(img_area_thresh, constants.ANCHORS_POINTS["group_2"])
        img3 = constants.crop_img(img_area_thresh, constants.ANCHORS_POINTS["list_number_1"])
        img4 = constants.crop_img(img_area_thresh, constants.ANCHORS_POINTS["list_number_2"])
        img5 = constants.crop_img(img_area_thresh, constants.ANCHORS_POINTS["variant"])
        result = [img1, img2, img3, img4, img5]

        mnist_size = 28
        for i in range(len(result)):
            result[i] = cv2.resize(result[i], (mnist_size, mnist_size))

        return result

groups_all = []
lists_all = []
variants_all = []

for i in range(1, 10):
    path = f"../dataset_blanks/stud{i}.jpg"
    images = get_digits(path)
    for j, (name, img) in enumerate(zip(["group1", "group2", "list1", "list2", "variant"], images)):
        cv2.imwrite(f"./result_lev/{(i - 1) * 5 + j}.jpg", img)
        if name.startswith("group"):
            groups_all.append(img)
        elif name.startswith("list"):
            lists_all.append(img)
        else:
            variants_all.append(img)

stacked = detection_utils.stackImages((groups_all, lists_all), 0.5)
cv2.imwrite(f"./result_lev/all.jpg", stacked)
cv2.imshow("Original", stacked)
cv2.waitKey(0)
