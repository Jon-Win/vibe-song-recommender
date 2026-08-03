# Vibe Song Recommender

Upload a photo and get song recommendations that match its mood, colors, and scene.

Uses CLIP (AI vision model) to understand the vibe of your image, then maps that to Spotify's audio features to find matching songs.

## Setup

### 1. Get Spotify API credentials

1. Go to https://developer.spotify.com/dashboard
2. Create a new app
3. Copy your Client ID and Client Secret

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your Spotify credentials
```

### 3. Install dependencies

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 4. Run

```bash
python app.py
```

Open http://localhost:5000 in your browser.

First run will download the CLIP model (~400MB). After that, startup is fast.

## How it works

1. **CLIP** analyzes your photo against mood, scene, and color descriptors
2. Top matches are mapped to Spotify audio features (valence, energy, tempo)
3. Spotify's recommendation API returns songs matching those features

## Tech stack

- **AI**: OpenCLIP (ViT-B-32) for image understanding
- **Music**: Spotify Web API for recommendations
- **Backend**: Flask
- **Frontend**: Vanilla HTML/CSS/JS
