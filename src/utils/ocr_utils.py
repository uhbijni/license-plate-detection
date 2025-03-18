import pytesseract
from PIL import Image, ImageFilter


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
    cropped_image = preprocess_image(cropped_image)
    if (xmax - xmin < 2) or (ymax - ymin < 2):
        return "bounding box too small!"

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
    
    # Apply thresholding
    #threshold = 50
    #image = image.point(lambda p: 255 if p > threshold else 0)
    
    #image.show()
    #print("test")


    return image