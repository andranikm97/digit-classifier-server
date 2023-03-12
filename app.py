import os
from flask import Flask, request, Response, make_response
from flask_cors import CORS
import tensorflow as tf 
from PIL import Image
import numpy as np
import uuid
import csv
from datetime import datetime
from pathlib import Path
from db.controllers import store_new_example

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
        "image": image,
        "label": ""
    }
    image = image / 255.
    prediction = np.argmax(tf.nn.softmax(model.predict(image, verbose=0)))
    response = make_response({
        "prediction": str(prediction),
        "backend_id": imageId
    })
    return response

@app.put("/train")
def trainModel():
    data = request.json
    imageId = uuid.UUID(data['backend_id'])
    correctLabel = data['label']
    if imageId and cached_images[imageId]:
        store_new_example(cached_images[imageId]['image'], correctLabel)
        del cached_images[imageId]
    return Response("", status=200)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
