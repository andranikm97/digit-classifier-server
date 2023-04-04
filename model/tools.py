import torch
from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=2
        )
        self.conv2 = nn.Conv2d(16, 32, 5, 1, 2)
        self.maxPool = nn.MaxPool2d(2)
        self.linear = nn.Linear(32 * 7 * 7, 10)

    def forward(self, x):
        h1 = self.maxPool(nn.functional.relu(self.conv1(x)))
        h2 = self.maxPool(nn.functional.relu(self.conv2(h1)))
        h3 = h2.view(h2.size(0), -1)
        out = self.linear(h3)
        return out


def initModel():
    checkpoint = torch.load("model/checkpoint.pt", map_location=torch.device("cpu"))
    model = NeuralNetwork()
    model_state_dict = checkpoint["model_state_dict"]
    model.load_state_dict(model_state_dict)
    optimizer_state_dict = checkpoint["optimizer_state_dict"]

    return model


def predict(model, example):
    example = torch.tensor(example)
    example = example / 255.0
    if example.dtype == torch.float64:
        example = example.to(torch.float32)
    model.eval()
    with torch.no_grad():
        output = model(example)
    model.train()
    output = torch.argmax(torch.nn.Softmax(dim=1)(output)).item()
    return output


initModel()
