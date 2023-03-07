import os
from flask import Flask, request, Response
from flask_cors import CORS
import tensorflow as tf 
from PIL import Image
import numpy as np
import uuid
import csv
from datetime import datetime
from pathlib import Path

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('model')
cached_images = {}

@app.get("/")
def getIndex():
    return 'Method {}'.format(request.method)

@app.post("/recognize")
def parseImage():
    data = request.files['data']
    image = Image.open(data)
    image = image.resize((28,28))
    image = np.asarray(image)[:,:,0].reshape(1,784)
    imageId = uuid.uuid4()
    cached_images[imageId] = {
        "x": image,
        "y": ""
    }
    image = image / 255.
    prediction = np.argmax(tf.nn.softmax(model.predict(image, verbose=0)))
    return {
        "prediction": str(prediction),
        "backend_id": imageId
    }

@app.put("/train")
def trainModel():
    data = request.json
    imageId = uuid.UUID(data['backend_id'])
    correctLabel = data['label']
    print(imageId, correctLabel)
    if imageId and cached_images[imageId]:
        with open('newExamples.csv', 'a') as f:
            w = csv.DictWriter(f, ['x', 'y', 'trained', 'datetimestamp'])
            w.writerow({
                "x": cached_images[imageId]["x"],
                "y": correctLabel,
                "trained": 0,
                "datetimestamp": str(datetime.now())
            })
            del cached_images[imageId]
            f.close()
    return Response("", status=200)