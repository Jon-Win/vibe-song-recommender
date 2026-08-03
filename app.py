"""Flask application for the Vibe Song Recommender service."""

import logging
import os
import uuid

from flask import Flask, Request, jsonify, render_template, request
from dotenv import load_dotenv
from werkzeug.exceptions import RequestEntityTooLarge

from analyzer import ImageAnalyzer
from recommender import SongRecommender

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

ALLOWED_EXTENSIONS: set[str] = {"png", "jpg", "jpeg", "webp", "heic"}

logger.info("Loading CLIP model (first run downloads ~400MB)...")
analyzer = ImageAnalyzer()
logger.info("CLIP model loaded successfully.")

recommender = SongRecommender(
    client_id=os.getenv("SPOTIFY_CLIENT_ID", ""),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET", ""),
)


def allowed_file(filename: str) -> bool:
    """Check whether the uploaded file has a permitted extension."""
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index() -> str:
    """Serve the main upload page."""
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze() -> tuple:
    """Analyze an uploaded image and return mood-matched song recommendations.

    Returns:
        A JSON response containing the image analysis and recommended songs,
        or an error message with the appropriate HTTP status code.
    """
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename or ""):
        return jsonify({"error": "File type not allowed"}), 400

    extension = file.filename.rsplit(".", 1)[1].lower()  # type: ignore[union-attr]
    filename = f"{uuid.uuid4()}.{extension}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        logger.info("Analyzing image: %s", filename)
        analysis = analyzer.analyze(filepath)
        songs = recommender.recommend(analysis)
        logger.info("Analysis complete. Returning %d songs.", len(songs))

        return jsonify({
            "analysis": analysis,
            "songs": songs,
        })
    except FileNotFoundError:
        logger.exception("Image file not found after saving.")
        return jsonify({"error": "Image processing failed"}), 500
    except RuntimeError as exc:
        logger.exception("Analysis failed.")
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(exc: RequestEntityTooLarge) -> tuple:
    """Return a friendly error when the upload exceeds the size limit."""
    return jsonify({"error": "File too large. Maximum size is 16 MB."}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
