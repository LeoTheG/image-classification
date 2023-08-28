import torch
from neural_net import Net  # Ensure this is the model for CIFAR-10
from torchvision.transforms import transforms
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description='Predict class for a given CIFAR-10 image using a pre-trained model.')
parser.add_argument('image_path', type=str, help='Path to the input image.')
args = parser.parse_args()
image_path = args.image_path

model = Net()

# Load the saved model parameters
model.load_state_dict(torch.load('cifar10_model.pth'))

# Set the model in evaluation mode
model.eval()

# CIFAR-10 normalization constants
transform = transforms.Compose([
    transforms.Resize((32, 32)),  # Ensure the image is resized to 32x32
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

# convert to RGB to get rid of the alpha channel
image = Image.open(image_path).convert('RGB')
image = transform(image)

image = image.unsqueeze(0)  # Add batch dimension. image shape becomes [1, 3, 32, 32]

with torch.no_grad():
    output = model(image)
    predicted_class = torch.argmax(output, dim=1).item()

# For CIFAR-10, the classes are:
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

print(f"The model predicts the image as: {classes[predicted_class]}")
