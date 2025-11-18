# Travel Lite Guide

A Gradio-hosted, light-concierge chat assistant specialised in planning short trips with minimal fuss.

## Nature of the chat client
- Persona: Travel Lite Guide sounds like a well-traveled Canadian friend. Responses stay concise, practical, and avoid fluff.
- Short-term memory: The agent scans the latest turns for preferences such as museums, parks, street food, or budget focus. `/prefs` reveals the current summaries so the user can confirm or adjust.
- Guardrails: The router blocks requests for the system prompt, plus restricted topics and regulated advice (visa, legal, medical). Refusals use the supplied templates.
- Interface: A single Gradio `ChatInterface` keeps history intact. The front end is intentionally plain so attention stays on travel guidance.

## Services that it provides
1. **Weather and AQI guidance (`/weather <city>`):**  
   - Fetches a 7-day forecast and air quality data from the Open-Meteo APIs.  
   - Rewrites the raw numbers into a human-friendly briefing: two headline sentences, three "Pack & Plan" bullets, and a new "Out & About" section suggesting extra outings based on temperature, rainfall, and inferred preferences.  
   - Adds caution lines for poor air, heat, or frost, encouraging users to verify conditions.

2. **Semantic neighbourhood tips (`/ask <question>`):**  
   - Runs semantic search over a persistent Chroma collection populated from the bundled corpus files.  
   - Returns three key points, a two-sentence summary, and source citations ("Sources: ...").  
   - Works with optional city hints and gracefully apologises when no hits are found.

3. **Seasonal event lookups (`/web <keywords>`):**  
   - Performs a lightweight DuckDuckGo HTML search.  
   - Summarises the top items into 3-5 one-liners, adds a planning tip, and finishes with "Verify: Always confirm on official sites."  
   - Acts as the open-ended third service using a web search tool.

Support command: **`/prefs`** reports the active preference flags and nudges the user if none have been captured yet.

## Key implementation decisions
- **Offline-aware geocoding:** Start with a curated list of common cities before hitting Open-Meteo and Nominatim; this keeps `/weather` usable even if public APIs are blocked.
- **Preference-driven advice:** Store lightweight preference flags in memory; swap the third bullet and add "Out & About" ideas that match those flags.
- **Persistent vector store:** Chroma lives under `assignment_chat/chroma_db/`, so embeddings are reused between runs without extra setup.
- **ASCII-only output:** All strings avoid smart quotes and em dashes to prevent encoding issues on Windows terminals.
- **Simple, inspectable UI:** Gradio is kept minimal (no custom JS, no icons) for fast startup and easy grading.

## Running the app
```bash
python -m assignment_chat.app
```

The assistant launches at `http://127.0.0.1:7860`; try:
- `/weather Toronto`
- `/ask Montreal neighbourhood tips`
- `/web Vancouver winter festivals`
- `/prefs`
