import pytesseract
from PIL import Image, ImageFilter, ImageEnhance


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
    if (xmax - xmin < 2) or (ymax - ymin < 2):
        return "bounding box too small!"
    
    cropped_image = image.crop((xmin, ymin, xmax, ymax))
    cropped_image = preprocess_image(cropped_image)


    text = pytesseract.image_to_string(cropped_image)
    if len(text) == 0:
        return "no text detected!"

    return text

def preprocess_image(image):
    """
    Preprocesses the input image.

    Args:
        image (PIL.Image): The input image.

    Returns:
        PIL.Image: The preprocessed image.
    """
    image = image.convert('L')
    og = image.copy()
    image = image.resize((image.width * 4, image.height * 4), Image.BILINEAR)      
    
    image.show()
    og.show()
    return image