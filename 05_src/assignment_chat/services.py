import os
import re
import json
import html
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import time
from urllib.parse import urlparse, parse_qs, unquote

import requests
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from utils.logger import get_logger

_logs = get_logger(__name__)


# Weather + AQI

OPEN_METEO_GEOCODE = "https://geocoding-api.open-meteo.com/v1/search"
OPEN_METEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_AIR_QUALITY = "https://air-quality-api.open-meteo.com/v1/air-quality"
NOMINATIM_SEARCH = "https://nominatim.openstreetmap.org/search"

# Offline fallback coords for common cities
FALLBACK_CITIES = {
    "paris": (48.8566, 2.3522, "Paris, France"),
    "london": (51.5074, -0.1278, "London, United Kingdom"),
    "tokyo": (35.6762, 139.6503, "Tokyo, Japan"),
    "toronto": (43.6532, -79.3832, "Toronto, Canada"),
    "sydney": (-33.8688, 151.2093, "Sydney, Australia"),
    "berlin": (52.5200, 13.4050, "Berlin, Germany"),
    "new york": (40.7128, -74.0060, "New York, USA"),
    "singapore": (1.3521, 103.8198, "Singapore"),
    "dubai": (25.2048, 55.2708, "Dubai, United Arab Emirates"),
    "bangkok": (13.7563, 100.5018, "Bangkok, Thailand"),
}


@dataclass
class WeatherResult:
    city: str
    temp_min_min: Optional[float]
    temp_max_max: Optional[float]
    avg_min: Optional[float]
    avg_max: Optional[float]
    rain_days: int
    aqi_peak: Optional[int]
    notes: Dict[str, str]


def _get_json(
    url: str,
    params: Dict,
    timeout: int = 10,
    retries: int = 2,
    headers: Optional[Dict[str, str]] = None,
) -> Optional[Dict]:
    """
    Fetches JSON from a URL with simple retry logic.
    Returns None when all attempts fail.
    """
    for attempt in range(retries):
        try:
            r = requests.get(url, params=params, timeout=timeout, headers=headers)
            if r.status_code == 200:
                return r.json()
        except Exception as e:
            if attempt < retries - 1:
                _logs.debug(f"Retrying {url} after error: {e}")
                time.sleep(0.5)
    return None


def _geocode_city(city: str) -> Optional[Tuple[float, float, str]]:
    """
    Attempts to geocode a city name.
    Order: offline fallback -> Open-Meteo geocoder -> OpenStreetMap Nominatim.
    """
    city_lower = city.lower().strip()
    
    # Try offline fallback first
    for key, coords in FALLBACK_CITIES.items():
        if key in city_lower or city_lower in key:
            return coords
    
    # Try Open-Meteo
    params = {"name": city, "count": 1, "language": "en", "format": "json"}
    data = _get_json(OPEN_METEO_GEOCODE, params=params, timeout=10, retries=1)
    if data:
        res = data.get("results") or []
        if res:
            item = res[0]
            display = item.get("name") or city
            if item.get("country"):
                display = f"{display}, {item['country']}"
            return (item["latitude"], item["longitude"], display)
    
    # Try Nominatim
    params = {"q": city, "format": "json", "limit": 1}
    headers = {"User-Agent": "Travel-Lite-Guide/1.0 assignment"}
    data = _get_json(NOMINATIM_SEARCH, params=params, timeout=10, retries=1, headers=headers)
    if data and isinstance(data, list) and len(data) > 0:
        item = data[0]
        display = city
        if "address" in item and "country" in item["address"]:
            display = f"{city}, {item['address']['country']}"
        return (float(item["lat"]), float(item["lon"]), display)
    
    return None


def _fetch_forecast(lat: float, lon: float) -> Dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max",
        "timezone": "auto",
    }
    data = _get_json(OPEN_METEO_FORECAST, params=params, timeout=12, retries=2)
    return data or {}


def _fetch_aqi(lat: float, lon: float) -> Dict:
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "us_aqi,pm2_5",
        "timezone": "auto",
    }
    data = _get_json(OPEN_METEO_AIR_QUALITY, params=params, timeout=12, retries=2)
    return data or {}


def service_weather(city: str) -> WeatherResult:
    """
    Retrieves weather and air-quality info for the next week for a given city.
    Returns a WeatherResult dataclass ready for rewriting.
    """
    geo = _geocode_city(city)
    if not geo:
        raise RuntimeError("Could not find that city. Try a broader city name.")
    lat, lon, display_city = geo
    _logs.debug(f"Geocode -> {display_city}: ({lat}, {lon})")

    forecast = _fetch_forecast(lat, lon)
    daily = forecast.get("daily") or {}
    raw_tmax = daily.get("temperature_2m_max") or []
    raw_tmin = daily.get("temperature_2m_min") or []
    raw_precip = daily.get("precipitation_probability_max") or []
    if not daily:
        _logs.debug("Forecast daily payload is empty; proceeding with generic guidance.")

    # Filter to numeric values only
    tmax = [v for v in raw_tmax if isinstance(v, (int, float))]
    tmin = [v for v in raw_tmin if isinstance(v, (int, float))]
    precip_prob = [v for v in raw_precip if isinstance(v, (int, float))]

    temp_min_min = min(tmin) if tmin else None
    temp_max_max = max(tmax) if tmax else None
    avg_min = sum(tmin) / len(tmin) if tmin else None
    avg_max = sum(tmax) / len(tmax) if tmax else None
    rain_days = sum(1 for p in precip_prob if p >= 50) if precip_prob else 0

    aqi_data = _fetch_aqi(lat, lon)
    hourly = aqi_data.get("hourly") or {}
    aqi_list = hourly.get("us_aqi") or []
    aqi_numeric = [v for v in aqi_list if isinstance(v, (int, float))]
    aqi_peak = max(aqi_numeric) if aqi_numeric else None

    _logs.debug(f"Derived -> avg_min={avg_min}, avg_max={avg_max}, min_min={temp_min_min}, max_max={temp_max_max}, rain_days={rain_days}, aqi_peak={aqi_peak}")

    return WeatherResult(
        city=display_city,
        temp_min_min=temp_min_min,
        temp_max_max=temp_max_max,
        avg_min=avg_min,
        avg_max=avg_max,
        rain_days=rain_days,
        aqi_peak=aqi_peak,
        notes={},
    )


def rewrite_weather(result: WeatherResult, prefs: Dict[str, bool]) -> str:
    """
    Converts a WeatherResult into a compact human-facing summary.
    Applies subtle preference nudges when available.
    """
    city = result.city
    trend = []
    if result.avg_max is not None:
        if result.avg_max >= 32:
            trend.append("hot days")
        elif result.avg_max >= 25:
            trend.append("warm days")
        elif result.avg_max >= 17:
            trend.append("mild days")
        else:
            trend.append("cool days")
    if result.avg_min is not None:
        if result.avg_min < 8:
            trend.append("chilly nights")
        elif result.avg_min < 15:
            trend.append("cool evenings")
    line1 = " and ".join(trend).capitalize() + "." if trend else "Typical seasonal temperatures."

    if result.rain_days >= 3:
        line2 = "Showers likely several days; keep plans flexible."
    elif result.rain_days >= 1:
        line2 = "A couple of damp spells possible mid-week."
    else:
        line2 = "Mostly dry across the week."

    bullets: List[str] = []
    bullets.append("Light layers for shifting temps.")
    if result.rain_days >= 1:
        bullets.append("Compact rain shell and quick-dry shoes.")
    else:
        bullets.append("Comfy walking shoes for cobbles and parks.")
    if result.avg_max and result.avg_max >= 28:
        bullets.append("Sun hat, SPF, and refillable bottle.")
    elif result.avg_min and result.avg_min <= 10:
        bullets.append("Warm layer for evenings and early starts.")
    else:
        bullets.append("Evening layer; breeze can pick up.")

    pref_bullets = []
    if prefs.get("museums"):
        pref_bullets.append("Reserve one gallery slot; check late openings.")
    if prefs.get("hiking"):
        pref_bullets.append("Plan park strolls for the clearest morning light.")
    if prefs.get("street_food"):
        pref_bullets.append("Sample night markets; carry small cash and wipes.")
    if prefs.get("budget"):
        pref_bullets.append("Use daily transit caps; cluster sights by neighborhood.")
    if pref_bullets:
        bullets[-1] = pref_bullets[0]
        result.notes["preference_nudge"] = pref_bullets[0]
    else:
        result.notes.pop("preference_nudge", None)

    caution: Optional[str] = None
    if (result.aqi_peak and result.aqi_peak >= 100):
        caution = "Air quality may dip at times - take gentler walks if sensitive."
    elif result.temp_max_max and result.temp_max_max >= 35:
        caution = "Avoid strenuous midday climbs; aim for mornings or late afternoons."
    elif result.temp_min_min is not None and result.temp_min_min <= 2:
        caution = "Frosty starts possible - start later and keep hands warm."

    extras: List[str] = []
    seen: set[str] = set()

    def add_extra(idea: Optional[str]) -> None:
        if not idea:
            return
        if idea not in seen:
            extras.append(idea)
            seen.add(idea)

    if result.avg_max and result.avg_max >= 24:
        add_extra("Book patios or waterfront brunches before midday warmth builds.")
    elif result.avg_max and result.avg_max < 18:
        add_extra("Use cooler afternoons for indoor tastings or cafe hopping.")
    else:
        add_extra("Balance one slower afternoon for neighbourhood wandering.")

    if result.rain_days >= 2:
        add_extra("Stack indoor stops on likely wet days; keep dinner plans flexible.")
    else:
        add_extra("Plan one golden-hour lookout walk for skyline views.")

    has_gallery_tip = any("gallery" in idea.lower() for idea in extras)
    if prefs.get("museums") and not has_gallery_tip:
        add_extra("Reserve an evening gallery or museum late session.")
    if prefs.get("street_food"):
        add_extra("Map night-market stalls near transit for quick bites.")

    out = []
    out.append(f"This week in {city}")
    out.append(f"{line1} {line2}")
    out.append("")
    out.append("Pack & Plan:")
    for b in bullets[:3]:
        out.append(f"- {b}")
    if caution:
        out.append("")
        out.append(caution)
    out.append("")
    out.append("Out & About:")
    for idea in extras[:3]:
        out.append(f"- {idea}")
    return "\n".join(out)


# Semantic search (Chroma)

CHROMA_DIR = os.path.join(os.path.dirname(__file__), "chroma_db")
CORPUS_DIR = os.path.join(os.path.dirname(__file__), "corpus")
COLLECTION_NAME = "assignment_city_corpus"

_embedding_model = None

def _get_embedder():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedding_model


def _ensure_chroma() -> chromadb.ClientAPI:
    os.makedirs(CHROMA_DIR, exist_ok=True)
    client = chromadb.Client(Settings(persist_directory=CHROMA_DIR, is_persistent=True))
    return client


def _load_corpus_files() -> List[Tuple[str, str]]:
    """Returns list of (title, content) from .txt files in CORPUS_DIR."""
    if not os.path.isdir(CORPUS_DIR):
        return []
    docs = []
    for fname in os.listdir(CORPUS_DIR):
        if not fname.lower().endswith(".txt"):
            continue
        path = os.path.join(CORPUS_DIR, fname)
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()
            title = os.path.splitext(fname)[0].replace("_", " ").strip()
            docs.append((title, content))
        except Exception:
            continue
    return docs


def _ensure_collection_indexed():
    client = _ensure_chroma()
    embedder = _get_embedder()
    collection = client.get_or_create_collection(name=COLLECTION_NAME)
    if collection.count() > 0:
        return collection
    docs = _load_corpus_files()
    if not docs:
        return collection
    ids = []
    texts = []
    metadatas = []
    for idx, (title, content) in enumerate(docs):
        ids.append(f"doc-{idx}")
        texts.append(content)
        metadatas.append({"title": title})
    embeddings = embedder.encode(texts, normalize_embeddings=True).tolist()
    collection.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)
    return collection


def service_semantic(query: str, city_hint: Optional[str] = None, k: int = 3) -> Dict:
    collection = _ensure_collection_indexed()
    embedder = _get_embedder()
    q_embedding = embedder.encode([query], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=q_embedding, n_results=k)
    # Normalize
    docs = []
    for i in range(len(results.get("ids", [[]])[0])):
        title = (results.get("metadatas") or [[{}]])[0][i].get("title", "Untitled")
        content = (results.get("documents") or [[""]])[0][i]
        score = (results.get("distances") or [[None]])[0][i]
        docs.append({"title": title, "content": content, "score": score})
    return {"hits": docs}


def synthesize_semantic(hits: List[Dict], prefs: Dict[str, bool]) -> str:
    # Extract 3 concise key points
    bullets = []
    for h in hits[:3]:
        content = h.get("content", "")
        title = h.get("title", "Untitled")
        # Take first sentence-ish
        snippet = re.split(r"[.\n]", content.strip(), maxsplit=1)[0]
        snippet = snippet[:120].rstrip()
        if len(snippet) < 20:
            snippet = f"{title}: practical details for short stays"
        bullets.append(f"{title}: {snippet}")

    # Preference nudges
    nudge_bits = []
    if prefs.get("museums"):
        nudge_bits.append("include one small museum stop")
    if prefs.get("hiking"):
        nudge_bits.append("add a park viewpoint")
    if prefs.get("street_food"):
        nudge_bits.append("sample a local street-food stall")
    if prefs.get("budget"):
        nudge_bits.append("favor budget-friendly options")
    nudge = ""
    if nudge_bits:
        nudge = " Tip: " + ", ".join(nudge_bits[:2]) + "."

    sources = "; ".join([h.get("title", "Untitled") for h in hits[:2]])

    out = []
    out.append("Key points:")
    for b in bullets[:3]:
        out.append(f"- {b}")
    out.append(f"Summary: A compact base near transit saves time; pick one or two nearby sights and explore on foot.{nudge}")
    out.append("Summary: Book flexible tickets and skim neighborhood notes before you go.")
    out.append(f"Sources: {sources}")
    return "\n".join(out)

# Web search (lightweight)
DUCKDUCKGO_HTML = "https://duckduckgo.com/html/"

MONTH_ALIASES = {
    "january": ("january", "jan"),
    "february": ("february", "feb"),
    "march": ("march", "mar"),
    "april": ("april", "apr"),
    "may": ("may",),
    "june": ("june",),
    "july": ("july",),
    "august": ("august", "aug"),
    "september": ("september", "sept"),
    "october": ("october", "oct"),
    "november": ("november", "nov"),
    "december": ("december", "dec"),
}


def _to_ascii(text: str) -> str:
    return text.encode("ascii", "ignore").decode("ascii")


def _ddg_search(query: str, n: int = 10) -> List[Dict[str, str]]:
    try:
        r = requests.get(
            DUCKDUCKGO_HTML,
            params={"q": query, "kl": "us-en"},
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if r.status_code != 200:
            return []
        page = r.text
        anchor_pattern = re.compile(
            r'result__a" href="(?P<href>[^"]+)"[^>]*>(?P<title>.*?)</a>',
            flags=re.IGNORECASE | re.DOTALL,
        )
        results: List[Dict[str, str]] = []
        matches = list(anchor_pattern.finditer(page))
        for idx, match in enumerate(matches):
            href = html.unescape(match.group("href"))
            parsed_href = urlparse(href)
            if parsed_href.netloc.lower().endswith("duckduckgo.com"):
                query_params = parse_qs(parsed_href.query)
                redirect = query_params.get("uddg")
                if redirect:
                    href = unquote(redirect[0])

            title_raw = match.group("title")
            title = html.unescape(re.sub("<.*?>", "", title_raw)).strip()
            title = _to_ascii(title)
            if not title:
                continue

            # Limit snippet search to the block before the next anchor
            next_start = matches[idx + 1].start() if idx + 1 < len(matches) else len(page)
            block = page[match.end():next_start]
            snippet_match = re.search(
                r'result__snippet.*?>(?P<snippet>.*?)</',
                block,
                flags=re.IGNORECASE | re.DOTALL,
            )
            snippet = ""
            if snippet_match:
                snippet = html.unescape(re.sub("<.*?>", "", snippet_match.group("snippet"))).strip()
                snippet = _to_ascii(snippet)

            results.append({"title": title, "url": href, "snippet": snippet})
            if len(results) >= n:
                break
        return results
    except Exception as exc:
        _logs.debug(f"DuckDuckGo search failed: {exc}")
        return []


def service_web(query: str) -> Dict:
    items = _ddg_search(query, n=10)
    return {"items": items, "query": query}


def _find_months(text: str) -> List[str]:
    found = []
    text_lower = text.lower()
    for month, aliases in MONTH_ALIASES.items():
        if any(alias in text_lower for alias in aliases):
            found.append(month)
    return found


def summarize_events(items: List[Dict[str, str]], query: str) -> str:
    lines: List[str] = []
    if not items:
        lines.append("I could not surface timely listings this round.")
        lines.append("Plan tip: Try broader keywords or add the city name.")
        lines.append("Verify: Always confirm on official sites.")
        return "\n".join(lines)

    months_in_query = _find_months(query)
    years_in_query = re.findall(r"20\d{2}", query)

    scored: List[Tuple[int, int, Dict[str, str], Optional[str]]] = []
    for idx, item in enumerate(items):
        title = item.get("title", "")
        url = item.get("url", "")
        snippet = item.get("snippet", "")
        text_lower = f"{title} {snippet}".lower()
        score = 0
        month_hit = None
        for month, aliases in MONTH_ALIASES.items():
            if any(alias in text_lower for alias in aliases):
                month_hit = month
                score += 2
                if month in months_in_query:
                    score += 2
        for year in years_in_query:
            if year in text_lower:
                score += 1
        if "free" in text_lower:
            score += 1
        scored.append((score, idx, item, month_hit))

    scored.sort(key=lambda entry: (-entry[0], entry[1]))

    seen_domains: set[str] = set()
    selected: List[Tuple[Dict[str, str], Optional[str]]] = []
    for score, _idx, item, month_hit in scored:
        url = item.get("url", "")
        domain = urlparse(url).netloc.lower().replace("www.", "") if url else ""
        if domain and domain in seen_domains:
            continue
        if domain:
            seen_domains.add(domain)
        selected.append((item, month_hit))
        if len(selected) >= 5:
            break

    if not selected:
        lines.append("I could not surface timely listings this round.")
        lines.append("Plan tip: Try broader keywords or add the city name.")
        lines.append("Verify: Always confirm on official sites.")
        return "\n".join(lines)

    focus_intro = f"Results for \"{query}\" highlight seasonal ideas from local guides."
    if months_in_query:
        focus = ", ".join(m.title() for m in months_in_query)
        focus_intro = f"{focus} plans for \"{query}\" highlight seasonal ideas from local guides."
    elif years_in_query:
        year_list = ", ".join(sorted(set(years_in_query)))
        focus_intro = f"Results for \"{query}\" surface {year_list} event roundups from trusted guides."
    lines.append(focus_intro.rstrip(".") + ".")
    lines.append("")

    for item, month_hit in selected[:3]:
        title = item.get("title", "Seasonal highlights")
        url = item.get("url", "")
        snippet = item.get("snippet", "")
        domain = urlparse(url).netloc.lower().replace("www.", "") if url else ""
        source = domain if domain else "Local guides"
        detail_parts = []
        if month_hit:
            detail_parts.append(f"{month_hit.title()} picks")
        elif months_in_query:
            focus = ", ".join(m.title() for m in months_in_query)
            detail_parts.append(f"Matches {focus} plans")
        if years_in_query and any(year in title.lower() or year in snippet.lower() for year in years_in_query):
            detail_parts.append("Includes " + ", ".join(sorted(set(years_in_query))))
        if "free" in title.lower() or "free" in snippet.lower():
            detail_parts.append("Includes free options")
        if snippet:
            trimmed_snippet = snippet[:90].rstrip()
            detail_parts.append(trimmed_snippet + ("..." if len(snippet) > 90 else ""))
        if not detail_parts:
            detail_parts.append("Seasonal picks worth bookmarking")
        friendly_name = domain if domain else "local guides"
        label = title
        detail_sentence = ". ".join(detail_parts).rstrip(".") + "."
        lines.append(f"- From {friendly_name}: {label}. {detail_sentence}")

    if months_in_query:
        focus = ", ".join(m.title() for m in months_in_query)
        lines.append(f"Plan tip: Lock key events early for {focus} weekends.")
    elif years_in_query:
        lines.append("Plan tip: Watch ticket release dates; popular slots go fast each season.")
    else:
        lines.append("Plan tip: Favor evening shows and prebook popular picks.")
    lines.append("Verify: Always confirm on official sites.")
    return "\n".join(lines)
