# predict.py
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.preprocessing import image

# ----------------------------
# Skeletonization function
# ----------------------------
def skeletonize(img):
    if img.max() > 1:
        img = img // 255

    skel = np.zeros(img.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    while True:
        opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, element)
        temp = cv2.subtract(img, opened)
        eroded = cv2.erode(img, element)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        if cv2.countNonZero(img) == 0:
            break

    return skel

# ----------------------------
# Preprocessing
# ----------------------------
def preprocess_image(image_path, img_size=224):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    bilateral = cv2.bilateralFilter(gray, 11, 75, 75)

    thresh = cv2.adaptiveThreshold(
        bilateral, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        9, 1
    )

    skeleton = skeletonize(thresh)

    skeleton_rgb = cv2.cvtColor(skeleton, cv2.COLOR_GRAY2RGB)
    resized = cv2.resize(skeleton_rgb, (img_size, img_size))

    x = image.img_to_array(resized)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    return x

# ----------------------------
# Prediction
# ----------------------------
def predict(image_path, model_path, class_names):
    model = tf.keras.models.load_model(model_path)

    x = preprocess_image(image_path)
    preds = model.predict(x)

    class_index = np.argmax(preds[0])
    confidence = float(preds[0][class_index])

    return class_names[class_index], confidence

# ----------------------------
# CLI usage
# ----------------------------
if __name__ == "__main__":
    MODEL_PATH = "model/vgg16_trained_model.h5"
    IMAGE_PATH = "sample.jpg"

    CLASS_NAMES = ["phase1", "phase2", "phase3"]

    label, conf = predict(IMAGE_PATH, MODEL_PATH, CLASS_NAMES)
    print(f"Prediction: {label} (confidence: {conf:.2f})")
