from typing import Callable, Dict, List, Optional, Tuple
import re

from assignment_chat.prompts import SYSTEM_PROMPT, DEVELOPER_PROMPT, get_refusal
from assignment_chat.services import (
    service_weather,
    rewrite_weather,
    service_semantic,
    synthesize_semantic,
    service_web,
    summarize_events,
)
from utils.logger import get_logger

_logs = get_logger(__name__)

ChatHistory = List[Dict[str, str]]
Handler = Callable[[Optional[str], Dict[str, bool]], str]

HELP_TEXT = (
    "Need a quick start? Here are sample commands:\n"
    "- `/weather Toronto` for a week-ahead Pack & Plan.\n"
    "- `/ask Montreal neighborhoods` for corpus-backed pointers.\n"
    "- `/web Vancouver winter events` for seasonal highlights.\n"
    "- `/prefs` to check what preferences I have noted."
)

ASK_FOR_CITY = "Please provide a city. For example: /weather Toronto"
ASK_FOR_QUESTION = "Please provide a question. For example: /ask best areas to stay"
ASK_FOR_KEYWORDS = "Please provide keywords. For example: /web Vancouver winter events"
NO_RESULTS = "Sorry - no local notes found yet. Ask a general 'where to stay' question."


def _extract_city(payload: str) -> str:
    """
    Extracts a city string from payload that may accidentally include
    other commands (e.g., 'Toronto, /ask ...').
    Keeps 'City, Country' but drops anything after a whitespace + '/'.
    Removes dangling punctuation at the end.
    """
    if not payload:
        return ""
    # Drop anything after a space followed by a slash (another command)
    city_part = re.split(r"\s+\/", payload.strip(), maxsplit=1)[0]
    # Trim trailing punctuation/spaces (keep internal comma like 'Paris, France')
    city_part = re.sub(r"[\s,;:.-]+$", "", city_part).strip()
    return city_part

def _extract_prefs(history: List[Dict]) -> Dict[str, bool]:
    """
    Lightweight preference extraction from recent history.
    No personal identifiers stored.
    """
    text = " ".join([str(m.get("content", "")) for m in history[-8:]]).lower()
    prefs = {
        "museums": bool(re.search(r"\bmuseum|gallery|art\b", text)),
        "hiking": bool(re.search(r"\bhike|trail|park|viewpoint\b", text)),
        "street_food": bool(re.search(r"\bstreet food|night market|hawker\b", text)),
        "budget": bool(re.search(r"\bbudget|cheap|affordable|hostel\b", text)),
    }
    return prefs


def _format_preferences(prefs: Dict[str, bool]) -> str:
    """
    Returns a short summary of the session preferences in persona.
    """
    if not any(prefs.values()):
        return (
            "Session preferences: none logged yet. "
            "Mention galleries, parks, street food, or budgets if that helps me steer tips."
        )
    highlights = []
    if prefs.get("museums"):
        highlights.append("museums")
    if prefs.get("hiking"):
        highlights.append("parks and viewpoints")
    if prefs.get("street_food"):
        highlights.append("street food runs")
    if prefs.get("budget"):
        highlights.append("budget picks")
    joined = ", ".join(highlights)
    return (
        f"Session preferences noted: {joined}. "
        "I will keep nudging plans in that direction."
    )


def _guardrails(message: str) -> Optional[str]:
    m = message.lower()
    # Attempt to access/modify prompts
    if "system prompt" in m or "developer prompt" in m or "internal instructions" in m:
        return get_refusal("system_prompt")
    # Restricted topics
    if re.search(r"\b(cat|dog|kitty|puppy|taylor swift|zodiac|aries|taurus|gemini|cancer|leo|virgo|libra|scorpio|sagittarius|capricorn|aquarius|pisces)\b", m):
        return get_refusal("restricted_topic")
    # Regulated advice
    if re.search(r"\bvisa|immigration|asylum|passport|vaccination|diagnosis|prescription|legal advice|attorney|lawyer|sue|court\b", m):
        return get_refusal("regulated_advice")
    return None


def _route(message: str) -> Tuple[str, Optional[str]]:
    """
    Simple router that maps leading commands to handler keys.
    """
    lowered = message.strip().lower()
    if lowered.startswith("/weather"):
        rest = message[len("/weather"):].strip()
        return ("weather", rest or "")
    if lowered.startswith("/ask"):
        rest = message[len("/ask"):].strip()
        return ("ask", rest or "")
    if lowered.startswith("/web"):
        rest = message[len("/web"):].strip()
        return ("web", rest or "")
    if lowered.startswith("/prefs"):
        return ("prefs", None)
    return ("help", None)


def _handle_weather(payload: Optional[str], prefs: Dict[str, bool]) -> str:
    if not payload:
        return ASK_FOR_CITY
    city = _extract_city(payload)
    if not city:
        return ASK_FOR_CITY
    result = service_weather(city)
    return rewrite_weather(result, prefs)


def _handle_semantic(payload: Optional[str], prefs: Dict[str, bool]) -> str:
    if not payload:
        return ASK_FOR_QUESTION
    # Optional city hint detection
    city_hint = None
    m = re.search(r"\b([A-Z][a-z]+(?: [A-Z][a-z]+)*)\b", payload)
    if m:
        city_hint = m.group(1)
    res = service_semantic(payload, city_hint)
    hits = res.get("hits", [])
    if not hits:
        return NO_RESULTS
    return synthesize_semantic(hits, prefs)


def _handle_web(payload: Optional[str], prefs: Dict[str, bool]) -> str:
    if not payload:
        return ASK_FOR_KEYWORDS
    res = service_web(payload)
    items = res.get("items", [])
    query = res.get("query", payload)
    return summarize_events(items, query)


def assignment_chat(message: str, history: Optional[ChatHistory] = None) -> str:
    """
    Entry point consumed by the Gradio front end.
    Routes commands, applies guardrails, and keeps the Travel Lite persona intact.
    """
    _logs.info(f"User: {message}")
    # Guardrails first
    blocked = _guardrails(message)
    if blocked:
        return blocked

    prefs = _extract_prefs(history or [])
    intent, payload = _route(message)

    handlers: Dict[str, Handler] = {
        "weather": lambda value, _: _handle_weather(value, prefs),
        "ask": lambda value, _: _handle_semantic(value, prefs),
        "web": lambda value, _: _handle_web(value, prefs),
        "prefs": lambda _value, _prefs: _format_preferences(prefs),
        "help": lambda _value, _prefs: HELP_TEXT,
    }

    try:
        handler = handlers.get(intent, handlers["help"])
        return handler(payload, prefs)
    except Exception as e:
        _logs.error(f"Failure handling intent {intent}: {e}")
        if intent == "weather":
            return "Sorry - I could not fetch weather just now. Try a broader city name."
        if intent == "ask":
            return "Sorry - semantic search is busy. Ask a general 'where to stay' question."
        if intent == "web":
            return "Sorry - web lookups are slow right now. Try shorter keywords."
        return "Sorry - something went sideways. Try one of the commands again."


