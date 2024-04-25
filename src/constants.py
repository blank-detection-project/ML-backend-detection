import cv2

# Подбор параметров
#Вариант
W1_VAR = 276
H1_VAR = 1
W2_VAR = 360
H2_VAR = 51
# Номер по списку
W11_GL = W1_VAR
H11_GL = H2_VAR
W12_GL = W2_VAR
H12_GL = 100

W21_GL = W2_VAR
H21_GL = H11_GL
W22_GL = 444
H22_GL = H12_GL
# Группа
W11_GROUP = W1_VAR + 15
H11_GROUP = 100 + 10
W12_GROUP = W2_VAR - 7
H12_GROUP = 150

W21_GROUP = W2_VAR + 7
H21_GROUP = H12_GL + 10
W22_GROUP = W21_GROUP + 84 - 25
H22_GROUP = H21_GROUP + 50 - 10
# Фамилия
W1_FAMILY = 276
H1_FAMILY = 150
W2_FAMILY = 1498
H2_FAMILY = 200
# Таблица ответов
W1_ANS = 276
H1_ANS = 367
W2_ANS = 1498
H2_ANS = 1218


ANCHORS_POINTS = {
    "answers": [(W1_ANS, H1_ANS), (W2_ANS, H2_ANS)],
    "family_name": [(W1_FAMILY, H1_FAMILY), (W2_FAMILY, H2_FAMILY)],
    "group_1": [(W11_GROUP, H11_GROUP), (W12_GROUP, H12_GROUP)],
    "group_2": [(W21_GROUP, H21_GROUP), (W22_GROUP, H22_GROUP)],
    "list_number_1": [(W11_GL, H11_GL), (W12_GL, H12_GL)],
    "list_number_2": [(W21_GL, H21_GL), (W22_GL, H22_GL)],
    "variant": [(W1_VAR, H1_VAR), (W2_VAR, H2_VAR)]
}

FILL_THRESH = 200000

def draw_points(img):
    for _, anchor in ANCHORS_POINTS.items():
        for point in anchor:
            w, h = point
            cv2.circle(img, (w, h), radius=10, color=(0, 0, 255), thickness=3)

def crop_img(img, points):
    w1, h1 = points[0]
    w2, h2 = points[1]
    return img[h1:h2+1, w1:w2+1]

