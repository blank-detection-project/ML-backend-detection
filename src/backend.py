import cv2
import io
import numpy as np
from answers_detection import constants, utils
from PIL import Image
from fastapi import FastAPI, UploadFile
from collections import defaultdict
from fastapi.middleware.cors import CORSMiddleware
from symbols_detection.model_eval import get_handwritten_text


async def file_to_cv_image(file: UploadFile):
    img_bytes = await file.read()
    stream = io.BytesIO(img_bytes)
    img = Image.open(stream)
    open_cv_image = np.array(img)
    open_cv_image = open_cv_image[:, :, ::-1].copy()
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
    rect_con = utils.rect_contour(contours)
    biggest_con = utils.get_corner_points(rect_con[0])
    if biggest_con.size != 0:
        biggest_con_new = utils.reorder(biggest_con)

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
        ans_matrix = utils.split_boxes(constants.crop_img(img_area_thresh, constants.ANCHORS_POINTS["answers"]))
        # Get answers
        return get_anses_matrix(ans_matrix), family_name


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/api/blank")
async def upload_file(file_student: UploadFile, file_teacher: UploadFile):

    cv_image_st = await file_to_cv_image(file_student)
    cv_image_te = await file_to_cv_image(file_teacher)
    answers_student, family = get_anses(cv_image_st)
    answers_teacher, _ = get_anses(cv_image_te)

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

    return {"correctAnswers": correct_answers, "allAnswers": all_answers, "name": family}


