import os

import torch
from flask import Flask, Response, make_response, request
from flask_cors import CORS
from PIL import Image

import model.mnist
from model.mnist import predict, train_model
from utils import Cache, prepareImage

app = Flask(__name__)
CORS(app)

cache = Cache()


@app.get("/")
def getIndex():
    return "Method {}".format(request.method)

@app.post("/recognize")
def parseImage():
    data = request.files["data"]
    image = prepareImage(Image.open(data))
    imageCacheId = cache.addImage(image)

    try:
        prediction = predict(image)
    except Exception as e:
        prediction = -1
        print("Prediction failed:", e)

    response = make_response(
        {"prediction": str(prediction), "backend_id": imageCacheId}
    )

    return response


@app.put("/train")
def trainModel():
    data = request.json
    imageId = data["backend_id"]
    label = data["label"]
    image = cache.getImage(imageId)

    image = torch.tensor(image).to(torch.float32)
    label = torch.tensor([int(label)])

    train_model(image, label)
    return Response("", status=200)


@app.post("/clear_image")
def clearImage():
    data = request.json
    id = data["imageId"]
    cache.removeImage(id)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
