import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class FakeImageModel(nn.Module):
    def __init__(self):
        super(FakeImageModel, self).__init__()
        self.model = models.resnet18(pretrained=True)

        # replace layer for binary classification
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)

    def forward(self, x):
        return self.model(x)
    
# initialize model
model = FakeImageModel()
print(model)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model = FakeImageModel().to(device)  # Move model to the selected device