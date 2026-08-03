"""Image analysis module using OpenCLIP for mood, scene, and color detection."""

from typing import Any

import numpy as np
import open_clip
import torch
from PIL import Image

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"

MOOD_LABELS: list[str] = [
    "happy and energetic",
    "sad and melancholic",
    "calm and peaceful",
    "dark and moody",
    "romantic and intimate",
    "adventurous and exciting",
    "nostalgic and wistful",
    "angry and intense",
    "dreamy and ethereal",
    "playful and fun",
    "mysterious and eerie",
    "powerful and triumphant",
]

SCENE_LABELS: list[str] = [
    "beach sunset",
    "city skyline at night",
    "forest or nature",
    "party or concert",
    "cozy indoor",
    "rainy day",
    "mountain landscape",
    "urban street",
    "ocean waves",
    "starry night sky",
    "golden hour field",
    "snowy winter",
]

COLOR_LABELS: list[str] = [
    "warm golden tones",
    "cool blue tones",
    "vibrant saturated colors",
    "muted pastel colors",
    "dark shadows and contrast",
    "bright and overexposed",
    "earthy natural tones",
    "neon and electric colors",
]


class ImageAnalyzer:
    """Analyzes images using CLIP to detect mood, scene, and color characteristics.

    Uses zero-shot classification by comparing image embeddings against
    predefined text label embeddings to determine the dominant attributes.
    """

    def __init__(self) -> None:
        """Initialize the CLIP model and preprocessing pipeline."""
        self.device: str = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            MODEL_NAME, pretrained=PRETRAINED
        )
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(MODEL_NAME)

    def _get_similarities(self, image: Image.Image, text_labels: list[str]) -> np.ndarray:
        """Compute cosine similarities between an image and a set of text labels.

        Args:
            image: A PIL Image in RGB mode.
            text_labels: Descriptive text labels to compare against the image.

        Returns:
            A NumPy array of similarity scores, one per label.
        """
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_tokens = self.tokenizer(text_labels).to(self.device)

        with torch.no_grad():
            image_features = self.model.encode_image(image_input)
            text_features = self.model.encode_text(text_tokens)

            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            similarities = (image_features @ text_features.T).squeeze(0)

        return similarities.cpu().numpy()

    def analyze(self, image_path: str) -> dict[str, list[dict[str, Any]]]:
        """Analyze an image and return its top mood, scene, and color attributes.

        Args:
            image_path: Path to the image file on disk.

        Returns:
            A dictionary with keys 'moods', 'scenes', and 'colors', each
            containing a ranked list of labels with their similarity scores.
        """
        image = Image.open(image_path).convert("RGB")

        mood_scores = self._get_similarities(image, MOOD_LABELS)
        scene_scores = self._get_similarities(image, SCENE_LABELS)
        color_scores = self._get_similarities(image, COLOR_LABELS)

        top_moods = sorted(
            zip(MOOD_LABELS, mood_scores), key=lambda x: x[1], reverse=True
        )[:3]
        top_scenes = sorted(
            zip(SCENE_LABELS, scene_scores), key=lambda x: x[1], reverse=True
        )[:2]
        top_colors = sorted(
            zip(COLOR_LABELS, color_scores), key=lambda x: x[1], reverse=True
        )[:2]

        return {
            "moods": [{"label": label, "score": float(score)} for label, score in top_moods],
            "scenes": [{"label": label, "score": float(score)} for label, score in top_scenes],
            "colors": [{"label": label, "score": float(score)} for label, score in top_colors],
        }
