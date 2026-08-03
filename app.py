import os
import uuid
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from analyzer import ImageAnalyzer
from recommender import SongRecommender

load_dotenv()

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

print("Loading CLIP model (first run downloads ~400MB)...")
analyzer = ImageAnalyzer()
print("Model loaded!")

recommender = SongRecommender(
    client_id=os.getenv("SPOTIFY_CLIENT_ID"),
    client_secret=os.getenv("SPOTIFY_CLIENT_SECRET"),
)

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg", "webp", "heic"}


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    filename = f"{uuid.uuid4()}.{file.filename.rsplit('.', 1)[1].lower()}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        analysis = analyzer.analyze(filepath)
        songs = recommender.recommend(analysis)

        return jsonify({
            "analysis": analysis,
            "songs": songs,
        })
    finally:
        os.remove(filepath)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
