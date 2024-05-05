import pandas as pd
import numpy as np
import io
from fastapi import FastAPI, UploadFile, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from src.detection_functions.detection_utils import (
    get_anses,
    file_to_cv_image,
    get_answers_stats,
    to_bites
)
from src.detection_functions.pdf_to_img import (
    upload_file2flitz_doc,
    get_pdf_images
)


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


@app.post("/api/blanks_pdf")
async def upload_file(pdf_students: UploadFile, file_teacher: UploadFile):
    cv_image_teacher = await file_to_cv_image(file_teacher)
    answers_teacher, _ = get_anses(cv_image_teacher)

    flitz_doc = await upload_file2flitz_doc(pdf_students)
    np_images = get_pdf_images(flitz_doc)
    result = []
    for page_number, cv_image_student in enumerate(np_images):
        answers_student, family = get_anses(cv_image_student)
        all_answers, correct_answers = get_answers_stats(answers_student, answers_teacher)
        result.append({"correctAnswers": correct_answers, "allAnswers": all_answers, "name": family})
    #return {"result": result, "answers": answers_teacher}

    df = pd.DataFrame(np.zeros((4, 5)))
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer) as writer:
        df.to_excel(writer, index=False)
    return StreamingResponse(
        io.BytesIO(buffer.getvalue()),
        media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
        headers={"Content-Disposition": f"attachment; filename=data.xlsx"}
    )
