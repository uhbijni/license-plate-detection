def load_image(image_path):
    import cv2
    image = cv2.imread(image_path)
    return image

def save_image(image, output_path):
    import cv2
    cv2.imwrite(output_path, image)

def preprocess_image(image):
    import cv2
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized_image = cv2.resize(gray_image, (640, 480))
    return resized_image