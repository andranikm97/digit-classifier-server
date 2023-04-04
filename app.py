import os
import uuid

from flask import Flask, Response, make_response, request
from flask_cors import CORS
from PIL import Image

from model.tools import initModel, predict
from utils import Cache, prepareImage

app = Flask(__name__)
CORS(app)

model = initModel()
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
        prediction = predict(model, image)
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
    imageId = uuid.UUID(data["backend_id"])
    # cache.addImage(ima)
    #     correctLabel = data['label']
    #     # if imageId and cached_images[imageId]:
    #         # store_new_example(cached_images[imageId]['image'], correctLabel)
    #         # del cached_images[imageId]
    return Response("", status=200)


@app.post("/clear_image")
def clearImage():
    data = request.json
    id = data["imageId"]
    cache.removeImage(id)


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
