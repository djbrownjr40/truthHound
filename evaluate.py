import torch
from model import FakeImageModel
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Set up device
device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# Load dataset
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_data = datasets.ImageFolder(root='C:/Users/d.Brown/Desktop/d.brown/portfolio0/data', transform=transform)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# initialize model

model = FakeImageModel().to(device)
model.load_state_dict(torch.load('model.pth')) # load trained model

# evaluate model
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total * 100
print(f"Test accuracy: {test_accuracy:.2f}%")