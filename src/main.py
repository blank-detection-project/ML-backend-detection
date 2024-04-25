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
    cv2.drawContours(img_area_contours, rect_con, -1, (0, 255, 0), 1)
    cv2.drawContours(img_area_contours_all, contours, -1, (0, 255, 0), 1)
    # Apply threshold
    img_area_thresh = cv2.threshold(img_area_gray, 220, 255, cv2.THRESH_BINARY_INV)[1]


img_blank = np.zeros_like(img)
image_array = ([img, img_gray, img_blur, img_canny],
               [img_contours, img_biggest_contour, img_area, img_blank])
#img_stacked = utils.stackImages(image_array, 0.5)

# img_area_stacked = utils.stackImages(([img_area, img_area_canny, img_area_contours_all, img_area_contours, img_area_thresh],), 0.5)


# Подбор параметров
# Таблица ответов
w1_ans = 276
h1_ans = 393
img_area_contours = cv2.circle(img_area_contours, (w1_ans, h1_ans), radius=10, color=(0, 0, 255), thickness=3)
w2_ans = 1498
h2_ans = 1218
img_area_contours = cv2.circle(img_area_contours, (w2_ans, h2_ans), radius=10, color=(0, 0, 255), thickness=3)
# Фамилия
w1_family = 276
h1_family = 180
img_area_contours = cv2.circle(img_area_contours, (w1_family, h1_family), radius=10, color=(0, 0, 255), thickness=3)
w2_family = 1498
h2_family = 230
img_area_contours = cv2.circle(img_area_contours, (w2_family, h2_family), radius=10, color=(0, 0, 255), thickness=3)
# Группа
h1_group = 130
w1_group = w1_family
img_area_contours = cv2.circle(img_area_contours, (w1_group, h1_group), radius=10, color=(0, 0, 255), thickness=3)
w2_group = w2_family
h2_group = h1_family
img_area_contours = cv2.circle(img_area_contours, (w2_group, h2_group), radius=10, color=(0, 0, 255), thickness=3)
# Номер по списку
w1_gl = w1_family
h1_gl = 51
img_area_contours = cv2.circle(img_area_contours, (w1_gl, h1_gl), radius=10, color=(0, 0, 255), thickness=3)
w2_gl = w2_family
h2_gl = h1_group
img_area_contours = cv2.circle(img_area_contours, (w2_gl, h2_gl), radius=10, color=(0, 0, 255), thickness=3)
#Вариант
w1_var = w1_family
h1_var = 1
img_area_contours = cv2.circle(img_area_contours, (w1_var, h1_var), radius=10, color=(0, 0, 255), thickness=3)
w2_var = w2_family
h2_var = h1_gl
img_area_contours = cv2.circle(img_area_contours, (w2_var, h2_var), radius=10, color=(0, 0, 255), thickness=3)




img_area_stacked = utils.stackImages(([img_area, img_area_canny, img_area_contours, img_area_thresh],), 0.5)
cv2.imshow("Original", img_area_stacked)
cv2.waitKey(0)
