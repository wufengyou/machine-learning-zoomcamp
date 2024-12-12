#!/usr/bin/env python
# coding: utf-8

import tflite_runtime.interpreter as tflite
from io import BytesIO
from urllib import request
from PIL import Image
import numpy as np
import os

MODEL_PATH = os.path.join('/var/task', 'model_2024_hairstyle_v2.tflite')


def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img


def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return img


interpreter = tflite.Interpreter(model_path=MODEL_PATH)

interpreter.allocate_tensors()
input_index = interpreter.get_input_details()[0]['index']
output_index = interpreter.get_output_details()[0]['index']
def predict(url):
    img = download_image(url)
    pic_file = prepare_image(img, (200, 200))
    img_array = np.array(pic_file)
    img_preprocessed = img_array / 255.0
    X = img_preprocessed.reshape(1, 200, 200, 3)
    interpreter.set_tensor(input_index, X.astype(np.float32))
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_index)
    return prediction[0][0]


# url = 'http://bit.ly/mlbookcamp-pants'

def lambda_handler(event, context):
    try:
        url = event['url']
        result = predict(url)
        return {
            'statusCode': 200,
            'body': float(result)   
        }
    except Exception as e:
        print(f"Error: {str(e)}")   
        return {
            'statusCode': 500,
            'body': str(e)
        }
