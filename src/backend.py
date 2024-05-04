from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from answers_detection.detection_utils import get_anses, file_to_cv_image


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


