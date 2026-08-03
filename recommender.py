import spotipy
from spotipy.oauth2 import SpotifyClientCredentials

MOOD_TO_SPOTIFY = {
    "happy and energetic": {
        "valence": (0.7, 1.0),
        "energy": (0.7, 1.0),
        "tempo": (120, 180),
        "genres": ["pop", "dance", "happy"],
    },
    "sad and melancholic": {
        "valence": (0.0, 0.3),
        "energy": (0.1, 0.4),
        "tempo": (60, 100),
        "genres": ["sad", "indie", "acoustic"],
    },
    "calm and peaceful": {
        "valence": (0.4, 0.7),
        "energy": (0.1, 0.3),
        "tempo": (60, 100),
        "genres": ["ambient", "chill", "acoustic"],
    },
    "dark and moody": {
        "valence": (0.0, 0.3),
        "energy": (0.3, 0.6),
        "tempo": (80, 120),
        "genres": ["dark", "electronic", "trip-hop"],
    },
    "romantic and intimate": {
        "valence": (0.4, 0.7),
        "energy": (0.2, 0.5),
        "tempo": (70, 110),
        "genres": ["r-n-b", "soul", "romance"],
    },
    "adventurous and exciting": {
        "valence": (0.5, 0.9),
        "energy": (0.7, 1.0),
        "tempo": (120, 160),
        "genres": ["rock", "indie", "alternative"],
    },
    "nostalgic and wistful": {
        "valence": (0.3, 0.5),
        "energy": (0.2, 0.5),
        "tempo": (80, 120),
        "genres": ["indie", "folk", "singer-songwriter"],
    },
    "angry and intense": {
        "valence": (0.1, 0.4),
        "energy": (0.8, 1.0),
        "tempo": (130, 200),
        "genres": ["metal", "punk", "hard-rock"],
    },
    "dreamy and ethereal": {
        "valence": (0.3, 0.6),
        "energy": (0.1, 0.4),
        "tempo": (70, 110),
        "genres": ["dream-pop", "shoegaze", "ambient"],
    },
    "playful and fun": {
        "valence": (0.7, 1.0),
        "energy": (0.6, 0.9),
        "tempo": (110, 150),
        "genres": ["pop", "funk", "disco"],
    },
    "mysterious and eerie": {
        "valence": (0.1, 0.3),
        "energy": (0.2, 0.5),
        "tempo": (70, 110),
        "genres": ["dark-ambient", "electronic", "industrial"],
    },
    "powerful and triumphant": {
        "valence": (0.6, 0.9),
        "energy": (0.8, 1.0),
        "tempo": (120, 160),
        "genres": ["epic", "rock", "orchestral"],
    },
}


class SongRecommender:
    def __init__(self, client_id, client_secret):
        auth_manager = SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret
        )
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

    def recommend(self, analysis, limit=5):
        primary_mood = analysis["moods"][0]["label"]
        mood_params = MOOD_TO_SPOTIFY.get(primary_mood, MOOD_TO_SPOTIFY["calm and peaceful"])

        target_valence = sum(mood_params["valence"]) / 2
        target_energy = sum(mood_params["energy"]) / 2
        target_tempo = sum(mood_params["tempo"]) / 2

        seed_genres = mood_params["genres"][:2]

        available = self.sp.recommendation_genre_seeds()["genres"]
        seed_genres = [g for g in seed_genres if g in available]
        if not seed_genres:
            seed_genres = ["pop"]

        results = self.sp.recommendations(
            seed_genres=seed_genres,
            limit=limit,
            target_valence=target_valence,
            target_energy=target_energy,
            target_tempo=target_tempo,
            min_valence=mood_params["valence"][0],
            max_valence=mood_params["valence"][1],
            min_energy=mood_params["energy"][0],
            max_energy=mood_params["energy"][1],
        )

        songs = []
        for track in results["tracks"]:
            songs.append({
                "name": track["name"],
                "artist": ", ".join(a["name"] for a in track["artists"]),
                "album": track["album"]["name"],
                "preview_url": track["preview_url"],
                "spotify_url": track["external_urls"]["spotify"],
                "album_art": track["album"]["images"][0]["url"] if track["album"]["images"] else None,
            })

        return songs
