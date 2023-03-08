import uuid

import numpy as np


def prepareImage(image):
    image = image.resize((28, 28))
    image = np.asarray(image)[:, :, 0].reshape(1, 784)
    image = image / 255.
    return image


class Cache():
    def __init__(self) -> None:
        self.cache = {}

    def addImage(self, image):
        id = uuid.uuid4()
        self.cache[id] = {
            "image": image,
            "label": ""
        }
        return id

    def removeImage(self, id):
        id = uuid.UUID(id)
        if id in self.cache:
            del self.cache[id]
