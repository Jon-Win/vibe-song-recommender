# Vibe Song Recommender

A web application that analyzes the mood, scene, and color palette of uploaded photos using AI vision, then recommends Spotify songs that match the detected vibe.

## Architecture

```
Browser (index.html)
    |
    | POST /analyze (multipart image)
    v
Flask App (app.py)
    |
    |---> ImageAnalyzer (analyzer.py)
    |         Uses OpenCLIP (ViT-B-32) to compute cosine similarity
    |         between the image embedding and predefined mood/scene/color
    |         text label embeddings (zero-shot classification).
    |
    |---> SongRecommender (recommender.py)
              Maps the top detected mood to curated search queries,
              then retrieves matching tracks from the Spotify Web API.
```

## Setup

### 1. Get Spotify API Credentials

1. Go to <https://developer.spotify.com/dashboard>
2. Create a new application
3. Copy your **Client ID** and **Client Secret**

### 2. Configure Environment

```bash
cp .env.example .env
# Edit .env and add your Spotify credentials
```

### 3. Install Dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run the Application

```bash
python app.py
```

Open <http://localhost:5000> in your browser.

The first run downloads the CLIP model (~400 MB). Subsequent startups are fast.

## How It Works

1. **Upload** -- The user uploads a photo through the web interface.
2. **Analyze** -- OpenCLIP encodes the image and computes cosine similarity against three sets of text labels (mood, scene, color). The top matches are returned.
3. **Recommend** -- The primary detected mood is mapped to a curated list of artist/song search queries. The Spotify API returns full track metadata for the matched songs.
4. **Display** -- The frontend renders the detected vibe tags and song cards with album art and Spotify links.

## API Endpoints

| Method | Path       | Description                                      |
|--------|------------|--------------------------------------------------|
| GET    | `/`        | Serves the upload UI                             |
| POST   | `/analyze` | Accepts a multipart image, returns analysis JSON |

### POST /analyze

**Request:** `multipart/form-data` with an `image` field (PNG, JPG, WEBP, or HEIC; max 16 MB).

**Response (200):**
```json
{
  "analysis": {
    "moods": [{"label": "calm and peaceful", "score": 0.31}],
    "scenes": [{"label": "beach sunset", "score": 0.28}],
    "colors": [{"label": "warm golden tones", "score": 0.26}]
  },
  "songs": [
    {
      "name": "Holocene",
      "artist": "Bon Iver",
      "album": "Bon Iver, Bon Iver",
      "preview_url": "https://...",
      "spotify_url": "https://open.spotify.com/track/...",
      "album_art": "https://i.scdn.co/image/..."
    }
  ]
}
```

## Tech Stack

- **AI Vision**: [OpenCLIP](https://github.com/mlfoundations/open_clip) (ViT-B-32, LAION-2B pretrained)
- **Music API**: [Spotify Web API](https://developer.spotify.com/documentation/web-api) via [spotipy](https://spotipy.readthedocs.io/)
- **Backend**: [Flask](https://flask.palletsprojects.com/)
- **Frontend**: Vanilla HTML / CSS / JavaScript
