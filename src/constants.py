import cv2

# Подбор параметров
# Таблица ответов
W1_ANS = 276
H1_ANS = 393
W2_ANS = 1498
H2_ANS = 1218
# Фамилия
W1_FAMILY = 276
H1_FAMILY = 180
W2_FAMILY = 1498
H2_FAMILY = 230
# Группа
H1_GROUP = 130
W1_GROUP = W1_FAMILY
W2_GROUP = W2_FAMILY
H2_GROUP = H1_FAMILY
# Номер по списку
W1_GL = W1_FAMILY
H1_GL = 51
W2_GL = W2_FAMILY
H2_GL = H1_GROUP
#Вариант
W1_VAR = W1_FAMILY
H1_VAR = 1
W2_VAR = W2_FAMILY
H2_VAR = H1_GL

ANCHORS_POINTS = {
    "answers": [(W1_ANS, H1_ANS), (W2_ANS, H2_ANS)],
    "family_name": [(W1_FAMILY, H1_FAMILY), (W2_FAMILY, H2_FAMILY)],
    "group": [(W1_GROUP, H1_GROUP), (W2_GROUP, H2_GROUP)],
    "list_number": [(W1_GL, H1_GL), (W2_GL, H2_GL)],
    "variant": [(W1_VAR, H1_VAR), (W2_VAR, H2_VAR)]
}

FILL_THRESH = 250000

def draw_points(img):
    for _, anchor in ANCHORS_POINTS.items():
        for point in anchor:
            print
            w, h = point
            cv2.circle(img, (w, h), radius=10, color=(0, 0, 255), thickness=3)

def crop_img(img, points):
    w1, h1 = points[0]
    w2, h2 = points[1]
    return img[h1:h2+1, w1:w2+1]


