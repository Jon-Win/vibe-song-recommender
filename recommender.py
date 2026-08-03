"""Song recommendation module that maps image mood analysis to Spotify tracks."""

import logging
import random
from typing import Any

import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from spotipy.exceptions import SpotifyException

logger = logging.getLogger(__name__)

MOOD_TO_SEARCH: dict[str, list[str]] = {
    "happy and energetic": [
        "Pharrell Williams Happy", "Dua Lipa Levitating", "Lizzo Good As Hell",
        "Mark Ronson Uptown Funk", "Carly Rae Jepsen Call Me Maybe",
        "Doja Cat Say So", "Harry Styles Watermelon Sugar",
    ],
    "sad and melancholic": [
        "Adele Someone Like You", "Billie Eilish When The Party's Over",
        "Sam Smith Stay With Me", "Lewis Capaldi Someone You Loved",
        "Lana Del Rey Summertime Sadness", "Bon Iver Skinny Love",
    ],
    "calm and peaceful": [
        "Bon Iver Holocene", "Iron & Wine Flightless Bird",
        "Norah Jones Don't Know Why", "Jack Johnson Better Together",
        "Khruangbin Time You and I", "Mac DeMarco Chamber of Reflection",
    ],
    "dark and moody": [
        "The Weeknd After Hours", "Portishead Glory Box",
        "Massive Attack Teardrop", "Radiohead Everything In Its Right Place",
        "FKA Twigs Cellophane", "James Blake Retrograde",
    ],
    "romantic and intimate": [
        "Frank Ocean Thinkin Bout You", "Daniel Caesar Best Part",
        "Sza Good Days", "John Legend All of Me",
        "Alina Baraz Electric", "Giveon Heartbreak Anniversary",
    ],
    "adventurous and exciting": [
        "Arcade Fire Wake Up", "M83 Midnight City",
        "Foster The People Pumped Up Kicks", "MGMT Electric Feel",
        "Tame Impala The Less I Know The Better", "Empire of the Sun Walking On A Dream",
    ],
    "nostalgic and wistful": [
        "Fleetwood Mac Dreams", "The Cranberries Linger",
        "Oasis Wonderwall", "Coldplay The Scientist",
        "The Smiths There Is A Light", "Mazzy Star Fade Into You",
    ],
    "angry and intense": [
        "Rage Against The Machine Killing In The Name",
        "System of a Down Chop Suey", "Linkin Park In The End",
        "Metallica Enter Sandman", "Bring Me The Horizon Throne",
        "Nine Inch Nails Head Like A Hole",
    ],
    "dreamy and ethereal": [
        "Cocteau Twins Cherry Coloured Funk", "Beach House Space Song",
        "Slowdive When The Sun Hits", "Cigarettes After Sex Apocalypse",
        "Alvvays In Undertow", "Washed Out Feel It All Around",
    ],
    "playful and fun": [
        "Earth Wind and Fire September", "Daft Punk Get Lucky",
        "Bruno Mars 24K Magic", "Outkast Hey Ya",
        "Lizzo Juice", "Cardi B I Like It",
    ],
    "mysterious and eerie": [
        "Radiohead Idioteque", "Massive Attack Angel",
        "Portishead Wandering Star", "Bjork Army of Me",
        "Nine Inch Nails A Warm Place", "Aphex Twin Avril 14th",
    ],
    "powerful and triumphant": [
        "Queen We Will Rock You", "Imagine Dragons Believer",
        "Muse Uprising", "Florence and the Machine Dog Days Are Over",
        "Kanye West Stronger", "Two Steps From Hell Heart of Courage",
    ],
}


class SongRecommender:
    """Recommends Spotify tracks based on detected image mood.

    Maps mood labels from the image analyzer to curated search queries,
    then retrieves matching tracks from the Spotify Web API.
    """

    def __init__(self, client_id: str, client_secret: str) -> None:
        """Initialize the Spotify API client.

        Args:
            client_id: Spotify application client ID.
            client_secret: Spotify application client secret.
        """
        auth_manager = SpotifyClientCredentials(
            client_id=client_id, client_secret=client_secret
        )
        self.sp = spotipy.Spotify(auth_manager=auth_manager)

    def recommend(self, analysis: dict[str, Any], limit: int = 5) -> list[dict[str, Any]]:
        """Return song recommendations matching the primary mood from an analysis.

        Args:
            analysis: The output from ImageAnalyzer.analyze(), containing
                a 'moods' key with ranked mood labels.
            limit: Maximum number of songs to return.

        Returns:
            A list of song dictionaries with track metadata and Spotify URLs.
        """
        primary_mood = analysis["moods"][0]["label"]
        search_terms = MOOD_TO_SEARCH.get(primary_mood, MOOD_TO_SEARCH["calm and peaceful"])

        random.shuffle(search_terms)

        songs: list[dict[str, Any]] = []
        seen_ids: set[str] = set()

        for term in search_terms:
            if len(songs) >= limit:
                break

            try:
                results = self.sp.search(q=term, type="track", limit=3)

                for track in results["tracks"]["items"]:
                    if track["id"] in seen_ids:
                        continue
                    seen_ids.add(track["id"])

                    songs.append({
                        "name": track["name"],
                        "artist": ", ".join(a["name"] for a in track["artists"]),
                        "album": track["album"]["name"],
                        "preview_url": track.get("preview_url"),
                        "spotify_url": track["external_urls"]["spotify"],
                        "album_art": (
                            track["album"]["images"][0]["url"]
                            if track["album"]["images"]
                            else None
                        ),
                    })

                    if len(songs) >= limit:
                        break

            except SpotifyException as exc:
                logger.warning("Spotify API error for '%s': %s", term, exc)
                continue
            except KeyError as exc:
                logger.warning("Unexpected response format for '%s': %s", term, exc)
                continue

        return songs
