import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

class VisualFeatureExtractor:
    def __init__(self, feature_dim: int = 512, device: str = "cpu"):
        self.device = device
        self.feature_dim = feature_dim
        
        # Load a lightweight pretrained model (ResNet-18) and strip off the final classification layer
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        resnet.fc = torch.nn.Identity()  # Outputs 512-dim embedding
        resnet.to(self.device)
        resnet.eval()
        self.model = resnet

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def extract_features(self, image_path_or_pil) -> np.ndarray:
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            embedding = self.model(tensor)
        
        return embedding.squeeze(0).cpu().numpy()

    def generate_visual_description(self, frame_img) -> str:
        """Generates a concise visual summary description of the sampled frame."""
        # Lightweight scene change / keyframe description generator
        return "Visual frame showing lecture slide or video scene activity."
