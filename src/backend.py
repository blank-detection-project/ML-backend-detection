from fastapi import FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from answers_detection.detection_utils import get_anses, file_to_cv_image, get_answers_stats


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

    all_answers, correct_answers = get_answers_stats(answers_student, answers_teacher)

    return {"correctAnswers": correct_answers, "allAnswers": all_answers, "name": family}
