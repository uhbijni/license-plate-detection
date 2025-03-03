import numpy as np
from PIL import Image, ImageDraw

def draw_bounding_boxes(image, boxpred, boxtruth, color=(255, 0, 0), thickness=2):
    """
    Draws bounding boxes on an image.

    Args:
        image (PIL.Image or np.ndarray): The image on which to draw.
        boxpred: The predicted bounding box to draw
        boxtruth: The ground truth bounding box to draw
        color (tuple): The color of the bounding box.
        thickness (int): The thickness of the bounding box lines.

    Returns:
        PIL.Image: The image with bounding boxes drawn.
    """
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    
    draw = ImageDraw.Draw(image)
    draw.rectangle(boxpred, outline="red", width=thickness)
    draw.rectangle(boxtruth, outline="yellow", width=thickness)
    
    return image