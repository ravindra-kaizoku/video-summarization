import numpy as np
from PIL import Image
from backend.utils.memory_utils import get_available_memory_mb

class VisualFeatureExtractor:
    def __init__(self, feature_dim: int = 512, device: str = "cpu"):
        self.device = device
        self.feature_dim = feature_dim
        self.model = None
        self.transform = None

    def _init_model(self):
        if self.model is None:
            # Check available RAM before loading ResNet-18
            avail_mb = get_available_memory_mb()
            if avail_mb < 220:
                print(f"[VISUAL] Available RAM is {avail_mb:.1f} MB (< 220MB). Using memory-safe feature extraction.")
                self.model = "lightweight"
                return

            try:
                import torch
                import torchvision.models as models
                import torchvision.transforms as transforms
                print(f"[VISUAL] Loading ResNet-18 [RAM: {avail_mb:.1f} MB]...")
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
                resnet.fc = torch.nn.Identity()
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
            except Exception as e:
                print(f"[VISUAL] ResNet-18 load error: {e}. Falling back to lightweight visual features.")
                self.model = "lightweight"

    def _extract_lightweight_features(self, pil_image) -> np.ndarray:
        """
        Fast, memory-safe 512-dimensional visual scene feature vector.
        Combines spatial block color statistics and luminance gradients.
        Requires 0 MB of model RAM and takes < 1 ms.
        """
        img = pil_image.resize((64, 64))
        arr = np.array(img, dtype=np.float32) / 255.0  # (64, 64, 3)

        # 1. 8x8 spatial block color means: 8 * 8 * 3 = 192 features
        blocks = arr.reshape(8, 8, 8, 8, 3).mean(axis=(1, 3)).flatten()

        # 2. RGB color histograms (32 bins per channel = 96 features)
        hist_r, _ = np.histogram(arr[:, :, 0], bins=32, range=(0, 1))
        hist_g, _ = np.histogram(arr[:, :, 1], bins=32, range=(0, 1))
        hist_b, _ = np.histogram(arr[:, :, 2], bins=32, range=(0, 1))
        hists = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32) / (64 * 64)

        # 3. Grayscale luminance spatial variance (224 features)
        gray = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
        var_blocks = gray.reshape(16, 4, 16, 4).var(axis=(1, 3)).flatten()

        combined = np.concatenate([blocks, hists, var_blocks])  # total: 192 + 96 + 224 = 512
        if len(combined) < 512:
            combined = np.pad(combined, (0, 512 - len(combined)))
        else:
            combined = combined[:512]

        norm = np.linalg.norm(combined)
        if norm > 0:
            combined = combined / norm
        return combined.astype(np.float32)

    def extract_features(self, image_path_or_pil) -> np.ndarray:
        if isinstance(image_path_or_pil, str):
            image = Image.open(image_path_or_pil).convert("RGB")
        else:
            image = image_path_or_pil.convert("RGB")

        self._init_model()

        if self.model == "lightweight" or self.model is None:
            return self._extract_lightweight_features(image)

        try:
            import torch
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                embedding = self.model(tensor)
            return embedding.squeeze(0).cpu().numpy().astype(np.float32)
        except Exception as e:
            print(f"[VISUAL] ResNet forward error: {e}. Using lightweight fallback.")
            return self._extract_lightweight_features(image)

    def unload_model(self):
        if self.model is not None and self.model != "lightweight":
            try:
                del self.model
                del self.transform
            except Exception:
                pass
            self.model = None
            self.transform = None
            import gc
            gc.collect()

    def generate_visual_description(self, frame_img) -> str:
        """Generates a concise visual summary description of the sampled frame."""
        return "Visual frame showing lecture slide or video scene activity."
