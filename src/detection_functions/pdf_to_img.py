import fitz
import numpy as np
import io
from fastapi import UploadFile


def pix2np(pix):
    im = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
    im = np.ascontiguousarray(im[..., [2, 1, 0]])  # rgb to bgr
    return im


async def upload_file2flitz_doc(file: UploadFile) -> fitz.Document:
    img_bytes = await file.read()
    stream = io.BytesIO(img_bytes)
    doc = fitz.Document(stream=stream)
    return doc


def get_pdf_images(pdf_file: fitz.Document) -> [np.array]:
    images_np = []
    for page in pdf_file:  # iterate through the pages
        pix = page.get_pixmap(dpi=150)   # render page to an image
        #pix.save("res_tmp/page-%i.jpg" % page.number)  # store image as a PNG
        img = pix2np(pix)
        images_np.append(img)
    return images_np