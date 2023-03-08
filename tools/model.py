import torch
from torch import nn

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.l1 = nn.Linear(28*28, 64)
    self.l2 = nn.Linear(64, 64)
    self.l3 = nn.Linear(64, 10)
    self.dropout = nn.Dropout(0.1)
  
  def forward(self, x):
    h1 = nn.functional.relu(self.l1(x))
    h2 = nn.functional.relu(self.l2(h1))
    h3 = self.dropout(h2 + h1)
    logits = self.l3(h3)
    return logits


def predict(model,  example):
  example = torch.tensor(example)
  print(example.shape)
  if (example.dtype == torch.float64):
    example = example.to(torch.float32)
  model.eval()
  with torch.no_grad():
    output = model(example)
  model.train()
  output = torch.argmax(torch.nn.Softmax(dim=1)(output)).item()
  return  output