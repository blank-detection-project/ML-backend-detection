import os

from fastapi import HTTPException
import torch
import cv2

from transformers import TrOCRProcessor, VisionEncoderDecoderModel

processor = TrOCRProcessor.from_pretrained('raxtemur/trocr-base-ru')
trocr_model = VisionEncoderDecoderModel.from_pretrained('raxtemur/trocr-base-ru')

def process_image(image_path: str, binarization=True) -> torch.Tensor:
    """
    Предобработка изображения и возврат его тензорного представления.

    Args:
        image_path (str): Путь к файлу изображения.

    Returns:
        torch.Tensor: Тензорное представление изображения.
    """
    if not os.path.isfile(image_path):
        raise HTTPException(status_code=404, detail="Image does not exist.")

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    img_resized = cv2.resize(image, (28, 28), interpolation=cv2.INTER_LINEAR)
    (thresh, im_bw) = cv2.threshold(img_resized, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    if binarization:
        im_bw = cv2.bitwise_not(im_bw)

    return torch.FloatTensor(im_bw)[None, None, :, :]


def get_handwritten_text(image) -> str:
    """
    Обрабатывает одно изображение с помощью trocr_ru и возвращает распознанный текст.

    Args:
        image_path (str): Путь к файлу изображения.

    Returns:
        str: Распознанный текст.
    """
    pixel_values = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values

    generated_ids = trocr_model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

    return generated_text


# Пример использования:
# - POST на /get_digits_for_array/ с массивом путей к файлам изображений
# - POST на /get_digit/ с путем к файлу изображения
# - POST на /get_handwritten_text/ с путем к файлу изображения