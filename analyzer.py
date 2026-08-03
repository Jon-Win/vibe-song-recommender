import numpy as np
from PIL import Image
from io import BytesIO


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

# Maps color properties to mood/scene scores
# This uses computer vision fundamentals: color analysis, brightness, saturation
MOOD_COLOR_MAP = {
    "happy and energetic": {"brightness": (0.6, 1.0), "saturation": (0.5, 1.0), "warmth": (0.5, 1.0)},
    "sad and melancholic": {"brightness": (0.1, 0.4), "saturation": (0.0, 0.3), "warmth": (0.2, 0.5)},
    "calm and peaceful": {"brightness": (0.4, 0.7), "saturation": (0.2, 0.5), "warmth": (0.4, 0.6)},
    "dark and moody": {"brightness": (0.0, 0.3), "saturation": (0.2, 0.6), "warmth": (0.1, 0.4)},
    "romantic and intimate": {"brightness": (0.3, 0.6), "saturation": (0.3, 0.7), "warmth": (0.6, 0.9)},
    "adventurous and exciting": {"brightness": (0.5, 0.9), "saturation": (0.5, 1.0), "warmth": (0.4, 0.7)},
    "nostalgic and wistful": {"brightness": (0.3, 0.6), "saturation": (0.1, 0.4), "warmth": (0.5, 0.8)},
    "angry and intense": {"brightness": (0.2, 0.5), "saturation": (0.6, 1.0), "warmth": (0.6, 1.0)},
    "dreamy and ethereal": {"brightness": (0.5, 0.9), "saturation": (0.1, 0.4), "warmth": (0.3, 0.6)},
    "playful and fun": {"brightness": (0.6, 1.0), "saturation": (0.6, 1.0), "warmth": (0.3, 0.7)},
    "mysterious and eerie": {"brightness": (0.1, 0.35), "saturation": (0.1, 0.5), "warmth": (0.0, 0.4)},
    "powerful and triumphant": {"brightness": (0.4, 0.8), "saturation": (0.5, 1.0), "warmth": (0.5, 0.9)},
}

SCENE_COLOR_MAP = {
    "beach sunset": {"brightness": (0.5, 0.8), "saturation": (0.5, 1.0), "warmth": (0.7, 1.0), "sky_ratio": (0.3, 0.7)},
    "city skyline at night": {"brightness": (0.1, 0.35), "saturation": (0.3, 0.7), "warmth": (0.2, 0.5), "contrast": (0.5, 1.0)},
    "forest or nature": {"brightness": (0.3, 0.6), "saturation": (0.3, 0.7), "warmth": (0.2, 0.5), "green_ratio": (0.3, 1.0)},
    "party or concert": {"brightness": (0.2, 0.5), "saturation": (0.5, 1.0), "warmth": (0.4, 0.8), "contrast": (0.5, 1.0)},
    "cozy indoor": {"brightness": (0.3, 0.6), "saturation": (0.2, 0.5), "warmth": (0.6, 0.9), "contrast": (0.1, 0.4)},
    "rainy day": {"brightness": (0.2, 0.5), "saturation": (0.0, 0.3), "warmth": (0.2, 0.5), "contrast": (0.1, 0.3)},
    "mountain landscape": {"brightness": (0.4, 0.8), "saturation": (0.2, 0.6), "warmth": (0.2, 0.5), "sky_ratio": (0.3, 0.6)},
    "urban street": {"brightness": (0.3, 0.6), "saturation": (0.2, 0.5), "warmth": (0.3, 0.6), "contrast": (0.3, 0.7)},
    "ocean waves": {"brightness": (0.4, 0.7), "saturation": (0.3, 0.7), "warmth": (0.1, 0.4), "blue_ratio": (0.3, 1.0)},
    "starry night sky": {"brightness": (0.0, 0.2), "saturation": (0.1, 0.4), "warmth": (0.1, 0.4), "contrast": (0.3, 0.8)},
    "golden hour field": {"brightness": (0.5, 0.8), "saturation": (0.4, 0.8), "warmth": (0.7, 1.0), "sky_ratio": (0.2, 0.5)},
    "snowy winter": {"brightness": (0.6, 1.0), "saturation": (0.0, 0.2), "warmth": (0.2, 0.5), "contrast": (0.1, 0.4)},
}


class ImageAnalyzer:
    def __init__(self):
        print("Image analyzer ready (color-based CV analysis)")

    def _extract_features(self, image_path):
        img = Image.open(image_path).convert("RGB")
        img = img.resize((256, 256))
        pixels = np.array(img, dtype=np.float32) / 255.0

        # Overall brightness
        brightness = np.mean(pixels)

        # Saturation (using HSV-like calculation)
        max_c = np.max(pixels, axis=2)
        min_c = np.min(pixels, axis=2)
        saturation = np.mean((max_c - min_c) / (max_c + 1e-7))

        # Warmth (red/yellow vs blue)
        warmth = np.mean(pixels[:, :, 0]) - np.mean(pixels[:, :, 2])
        warmth = (warmth + 1) / 2  # normalize to 0-1

        # Contrast
        contrast = np.std(pixels)

        # Color channel ratios
        red_ratio = np.mean(pixels[:, :, 0]) / (np.mean(pixels) + 1e-7)
        green_ratio = np.mean(pixels[:, :, 1]) / (np.mean(pixels) + 1e-7)
        blue_ratio = np.mean(pixels[:, :, 2]) / (np.mean(pixels) + 1e-7)

        # Top portion of image (sky detection)
        top_quarter = pixels[:64, :, :]
        sky_brightness = np.mean(top_quarter)
        sky_blue = np.mean(top_quarter[:, :, 2])
        sky_ratio = sky_brightness * 0.5 + (sky_blue / (np.mean(top_quarter) + 1e-7)) * 0.5

        # Dominant color temperature
        r_mean = np.mean(pixels[:, :, 0])
        g_mean = np.mean(pixels[:, :, 1])
        b_mean = np.mean(pixels[:, :, 2])

        return {
            "brightness": float(np.clip(brightness, 0, 1)),
            "saturation": float(np.clip(saturation, 0, 1)),
            "warmth": float(np.clip(warmth, 0, 1)),
            "contrast": float(np.clip(contrast * 3, 0, 1)),
            "red_ratio": float(red_ratio),
            "green_ratio": float(green_ratio),
            "blue_ratio": float(blue_ratio),
            "sky_ratio": float(np.clip(sky_ratio, 0, 1)),
            "r_mean": float(r_mean),
            "g_mean": float(g_mean),
            "b_mean": float(b_mean),
        }

    def _score_mood(self, features):
        scores = {}
        for mood, ranges in MOOD_COLOR_MAP.items():
            score = 0.0
            for feature_name, (low, high) in ranges.items():
                val = features.get(feature_name, 0.5)
                if low <= val <= high:
                    # How centered is the value in the ideal range
                    mid = (low + high) / 2
                    dist = abs(val - mid) / ((high - low) / 2 + 1e-7)
                    score += 1.0 - dist * 0.5
                else:
                    # How far outside the range
                    if val < low:
                        score -= (low - val) * 2
                    else:
                        score -= (val - high) * 2
            scores[mood] = max(0.01, score / len(ranges))

        # Normalize to sum to 1
        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()}

    def _score_scene(self, features):
        scores = {}
        for scene, ranges in SCENE_COLOR_MAP.items():
            score = 0.0
            for feature_name, (low, high) in ranges.items():
                val = features.get(feature_name, 0.5)
                if low <= val <= high:
                    mid = (low + high) / 2
                    dist = abs(val - mid) / ((high - low) / 2 + 1e-7)
                    score += 1.0 - dist * 0.5
                else:
                    if val < low:
                        score -= (low - val) * 2
                    else:
                        score -= (val - high) * 2
            scores[scene] = max(0.01, score / len(ranges))

        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()}

    def _score_colors(self, features):
        scores = {}
        brightness = features["brightness"]
        saturation = features["saturation"]
        warmth = features["warmth"]
        contrast = features["contrast"]

        scores["warm golden tones"] = warmth * 0.6 + brightness * 0.4
        scores["cool blue tones"] = (1 - warmth) * 0.6 + features["blue_ratio"] * 0.4
        scores["vibrant saturated colors"] = saturation * 0.8 + contrast * 0.2
        scores["muted pastel colors"] = (1 - saturation) * 0.5 + brightness * 0.5
        scores["dark shadows and contrast"] = (1 - brightness) * 0.5 + contrast * 0.5
        scores["bright and overexposed"] = brightness * 0.7 + (1 - contrast) * 0.3
        scores["earthy natural tones"] = warmth * 0.4 + features["green_ratio"] * 0.3 + (1 - saturation) * 0.3
        scores["neon and electric colors"] = saturation * 0.6 + contrast * 0.4

        total = sum(scores.values())
        return {k: v / total for k, v in scores.items()}

    def analyze(self, image_path):
        features = self._extract_features(image_path)

        mood_scores = self._score_mood(features)
        scene_scores = self._score_scene(features)
        color_scores = self._score_colors(features)

        top_moods = sorted(mood_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        top_scenes = sorted(scene_scores.items(), key=lambda x: x[1], reverse=True)[:2]
        top_colors = sorted(color_scores.items(), key=lambda x: x[1], reverse=True)[:2]

        return {
            "moods": [{"label": label, "score": score} for label, score in top_moods],
            "scenes": [{"label": label, "score": score} for label, score in top_scenes],
            "colors": [{"label": label, "score": score} for label, score in top_colors],
        }
