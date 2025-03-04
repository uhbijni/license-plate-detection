import pytesseract
from PIL import Image


def extract_text_from_box(image, box):
    """
    Extracts text from the regions of the image defined by the bounding boxes.

    Args:
        image (PIL.Image): The image from which to extract text.
        boxes (list of list of int): The bounding boxes defining the regions to extract text from.

    Returns:
        list of str: The extracted text from each bounding box.
    """

    xmin, ymin, xmax, ymax = box
    cropped_image = image.crop((xmin, ymin, xmax, ymax))
    if (xmax - xmin < 2) or (ymax - ymin < 2):
        return "bounding box too small!"

    text = pytesseract.image_to_string(cropped_image)
    if len(text) == 0:
        return "no text detected!"

    return text
