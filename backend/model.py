import torch
import torch.nn as nn
import timm
from torchvision import transforms
from PIL import Image

# Load the model with correct classifier head
def load_model(num_classes=2):
    model = timm.create_model("legacy_xception", pretrained=True)
    
    in_features = model.get_classifier().in_features
    model.reset_classifier(num_classes=num_classes)

    model.eval()
    return model

# Define transform for Xception
transform = transforms.Compose([
    transforms.Resize((299, 299)),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])

# Preprocess an image file
def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  
    return img_tensor

# Inference function
def predict(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)
        confidence, pred_class = torch.max(probs, dim=1)

    label = "Real" if pred_class.item() == 0 else "Fake"
    return label, confidence.item()
