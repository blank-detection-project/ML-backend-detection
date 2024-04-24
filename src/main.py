import cv2
import numpy as np
import utils

path = "../dataset_blanks/student_var3.jpg"
width_img = 1242
height_img = 1756

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
rect_con = utils.rect_contour(contours)
biggest_con = utils.get_corner_points(rect_con[0])

if biggest_con.size != 0:
    cv2.drawContours(img_biggest_contour, biggest_con, -1, (0, 255, 0), 20)
    biggest_con_new = utils.reorder(biggest_con)

    width_area = 1500
    height_area = 1300
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
    rect_con = utils.rect_contour(contours)
    cv2.drawContours(img_area_contours, rect_con, -1, (0, 255, 0), 2)
    cv2.drawContours(img_area_contours_all, contours, -1, (0, 255, 0), 2)
    # Apply threshold
    img_area_thresh = cv2.threshold(img_area_gray, 220, 255, cv2.THRESH_BINARY_INV)[1]


img_blank = np.zeros_like(img)
image_array = ([img, img_gray, img_blur, img_canny],
               [img_contours, img_biggest_contour, img_area, img_blank])
#img_stacked = utils.stackImages(image_array, 0.5)

# img_area_stacked = utils.stackImages(([img_area, img_area_canny, img_area_contours_all, img_area_contours, img_area_thresh],), 0.5)
img_area_stacked = utils.stackImages(([img_area, img_area_canny, img_area_contours],), 0.5)
cv2.imshow("Original", img_area_stacked)
cv2.waitKey(0)
