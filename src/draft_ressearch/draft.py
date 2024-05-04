import cv2
import numpy as np

import src.detection_functions.constants
from src.detection_functions import constants, detection_utils

path = "../../dataset_blanks/stud1.jpg"
path = "res_tmp/page-0.jpg"
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

    # Get rectangles
    img_area_contours = img_area.copy()
    img_area_contours_all = img_area.copy()
    img_area_blur = cv2.GaussianBlur(img_area, (5, 5), 1)
    img_area_canny = cv2.Canny(img_area_blur, 10, 50)
    contours, hierarchy = cv2.findContours(img_area_canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    rect_con = detection_utils.rect_contour(contours)
    cv2.drawContours(img_area_contours, rect_con, -1, (0, 255, 0), 3)
    cv2.drawContours(img_area_contours_all, contours, -1, (0, 255, 0), 2)
    # Apply threshold
    img_area_thresh = cv2.threshold(img_area_gray, 220, 255, cv2.THRESH_BINARY_INV)[1]


img_blank = np.zeros_like(img)
image_array = ([img, img_gray, img_blur, img_canny],
               [img_contours, img_biggest_contour, img_area, img_blank])
#img_stacked = utils.stackImages(image_array, 0.5)

# img_area_stacked = utils.stackImages(([img_area, img_area_canny, img_area_contours_all, img_area_contours, img_area_thresh],), 0.5)


constants.draw_points(img_area_contours)
ans_matrix = detection_utils.split_boxes(constants.crop_img(img_area_thresh, constants.ANCHORS_POINTS["answers"]))
sums = []

answers = []
for i, row in enumerate(ans_matrix):
    for j, col in enumerate(row):
        sum_elem = np.sum(col)
        sums.append(sum_elem)
        if sum_elem > constants.FILL_THRESH:
            answers.append([i+1, j+1])

print({"answers": answers})

#To Lev
img1 = constants.crop_img(img_area_thresh, constants.ANCHORS_POINTS["group_1"])
img2 = constants.crop_img(img_area_thresh, constants.ANCHORS_POINTS["group_2"])
img3 = constants.crop_img(img_area_thresh, constants.ANCHORS_POINTS["list_number_1"])
img4 = constants.crop_img(img_area_thresh, constants.ANCHORS_POINTS["list_number_2"])
variant = constants.crop_img(img_area_thresh, constants.ANCHORS_POINTS["variant"])
#cv2.imwrite("./group1.jpg", img1)
#cv2.imwrite("./group2.jpg", img2)


img_area_stacked = detection_utils.stackImages(([img_area, img_area_canny, img_area_contours, img_area_thresh],), 0.5)
cv2.imshow("Original", img_area_stacked)
cv2.waitKey(0)
