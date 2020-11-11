# Utility functions
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image
from io import BytesIO

SAMPLE_DIR = "./assets/samples"
TARGET_SIZE = (224,224)
DECODE = {0:"Cat", 1:"Dog"}
INV_MAP = {"Cat": 0, "Dog": 1}


def preprocess(im):
    #im = img_to_array(load_img(os.path.join(SAMPLE_DIR,filename),target_size = TARGET_SIZE))
    x = keras_image.img_to_array(im)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    return x


def predict(model, processed_im):
    preds = model.predict(processed_im)
    idx = preds.argmax()

    res = [idx, preds.max()]

    return res

def predict_5(model, processed_im):
    top_n = 5
    preds = model.predict(processed_im)
    top_pred_n = decode_predictions(preds, top=top_n)[0]
    classes = np.argsort(preds[0])[-top_n:][::-1]
        
    P = imagenet_utils.decode_predictions(preds)
    idx = preds.argmax()
    return idx, preds.max(), classes, top_pred_n

def array2bytes(im_arr, fmt='png'):
    img = Image.fromarray(im_arr, mode='RGB')
    f = BytesIO()
    img.save(f, fmt)

    return f.getvalue()
