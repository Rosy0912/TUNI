from torchvision import models, transforms
from scipy.spatial.distance import euclidean
import torch
import os
from PIL import Image

# Define paths
model_path = '/home/rosych0912/.conda/mia/resnet50_ft_weight.pt'
optimized_images_dir = '/home/rosych0912/.conda/mia/optimized_images'
real_images_dir = '/home/rosych0912/.conda/mia/real_images'

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Instantiate the model
model = models.resnet50(pretrained=False)
# Adjust the final layer to match the checkpoint's layer
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, 8631)  # Adjust the number of output features to 8631

# Load pre-trained model state dict
state_dict = torch.load(model_path, map_location=device)
model.load_state_dict(state_dict)
model.eval()  # Now this should work as the model instance has an eval() method
model.to(device)  # Move model to the appropriate device

# Define preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to extract features from an image
def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(image).to(device)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        features = model(input_batch)
        features = features.cpu().numpy().flatten()
    return features

# Get image paths from directories
optimized_image_paths = [os.path.join(optimized_images_dir, filename) for filename in os.listdir(optimized_images_dir)]
real_image_paths = [os.path.join(real_images_dir, filename) for filename in os.listdir(real_images_dir)]

# Extract features for each pair of images and compute Euclidean distance
total_distance = 0.0
for optimized_image_path in optimized_image_paths:
    for real_image_path in real_image_paths:
        optimized_features = extract_features(optimized_image_path)
        real_features = extract_features(real_image_path)
        distance = euclidean(optimized_features, real_features)
        total_distance += distance

print("Total Euclidean distance between the images in the two folders:", total_distance)
