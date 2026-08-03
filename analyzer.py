import os
import base64
import requests
from PIL import Image
from io import BytesIO

HF_API_URL = "https://api-inference.huggingface.co/models/openai/clip-vit-base-patch32"

MOOD_LABELS = [
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

SCENE_LABELS = [
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

COLOR_LABELS = [
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
    def __init__(self):
        self.api_token = os.getenv("HF_API_TOKEN", "")
        self.headers = {}
        if self.api_token:
            self.headers["Authorization"] = f"Bearer {self.api_token}"

    def _classify(self, image_bytes, candidate_labels):
        response = requests.post(
            HF_API_URL,
            headers=self.headers,
            json={
                "inputs": {
                    "image": base64.b64encode(image_bytes).decode("utf-8"),
                },
                "parameters": {
                    "candidate_labels": candidate_labels,
                },
            },
            timeout=30,
        )

        if response.status_code == 503:
            # Model is loading, retry once
            import time
            time.sleep(5)
            response = requests.post(
                HF_API_URL,
                headers=self.headers,
                json={
                    "inputs": {
                        "image": base64.b64encode(image_bytes).decode("utf-8"),
                    },
                    "parameters": {
                        "candidate_labels": candidate_labels,
                    },
                },
                timeout=60,
            )

        response.raise_for_status()
        results = response.json()

        scores = {}
        for item in results:
            scores[item["label"]] = item["score"]

        return scores

    def analyze(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img.thumbnail((512, 512))

        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()

        mood_scores = self._classify(image_bytes, MOOD_LABELS)
        scene_scores = self._classify(image_bytes, SCENE_LABELS)
        color_scores = self._classify(image_bytes, COLOR_LABELS)

        top_moods = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_scenes = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        top_colors = sorted(color_scores.items(), key=lambda x: x[1], reverse=True)[:2]

        return {
            "moods": [{"label": label, "score": score} for label, score in top_moods],
            "scenes": [{"label": label, "score": score} for label, score in top_scenes],
            "colors": [{"label": label, "score": score} for label, score in top_colors],
        }
