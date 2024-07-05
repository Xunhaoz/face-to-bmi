from flask import Flask, request, jsonify, render_template
import os
import urllib.request
from PIL import Image
from matplotlib import pyplot as plt
from tqdm import tqdm
from uuid import uuid4

import torch
from torchvision.transforms import ToTensor

from models import get_model
from loader import vit_transforms

app = Flask(__name__)


def test_and_show(img_dir, weight_dir):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # open and transform image for vit
    image = Image.open(img_dir)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = ToTensor()(image)
    image_vit = vit_transforms(image)
    image_vit = image_vit.unsqueeze(0)
    image_vit = image_vit.to(device)

    # get model and predict
    model = get_model()
    model = model.to(device)
    model.load_state_dict(torch.load(weight_dir, map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(image_vit)

    return pred.item()


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    height = float(request.form['height'])
    weight = float(request.form['weight'])

    if height > 100:
        height = height / 100

    bmi = round(weight / height ** 2, 2)

    file = request.files['file']
    file_path = f'./static/{bmi}-{uuid4()}-{file.filename}'
    file.save(file_path)
    pred = test_and_show(file_path, './weights/aug_epoch_7.pt')

    return {'prediction': pred, 'true': bmi}


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5006)
