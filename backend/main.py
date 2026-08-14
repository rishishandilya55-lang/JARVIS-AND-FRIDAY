import os
import io
import re
import json
import time
import base64
import logging
import urllib.parse
from typing import Optional, List, Dict, Tuple, Any
import requests
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("jarvis-core")

# Initialize FastAPI application
app = FastAPI(
    title="Jarvis / F.R.I.D.A.Y. Dual-Core Tactical API",
    description="Render-optimized high-efficiency backend with live persona switching between JARVIS and F.R.I.D.A.Y. modes.",
    version="1.5.0"
)

# Open CORS configuration for external native mobile and desktop clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Groq API Configuration
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "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")
GROQ_AUDIO_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"

# Primary & Fallback Models
MODEL_WHISPER = "whisper-large-v3"
MODEL_LLM = "llama-3.2-3b-preview"
MODEL_VISION = "llama-3.2-11b-vision-preview"

# ============================================================================
# DUAL-CORE PERSONA PROMPTS & VOICE PROFILES
# ============================================================================

JARVIS_SYSTEM_PROMPT = (
    "You are Jarvis, a highly sophisticated, exceptionally polite British cybernetic helper. "
    "Address the user as sir. Keep responses to 1-2 sentences. Help with everyday tasks, apps, and text scans smoothly.\n\n"
    "You have access to live tools and device commands via JSON action tags:\n"
    "1. LIVE SEARCH: If the user asks about live events, current dates, real-time facts, news, or weather, output strictly: "
    '{"action": "search", "query": "search term"}. Pause for the tool output.\n'
    "2. APP CONTROL: If the user commands you to open an application (e.g. 'open YouTube', 'launch Spotify', 'open browser'), output strictly: "
    '{"action": "open_app", "target": "youtube/spotify/browser/maps", "response": "Launching interface now, sir."}\n'
    "3. MUSIC PLAYBACK: If the user asks to play music (e.g. 'play synthwave', 'play Queen'), output strictly: "
    '{"action": "play_music", "query": "song name/genre", "response": "Queuing playback now, sir."}\n\n'
    "Otherwise, deliver only the direct, polite answer as Jarvis."
)

FRIDAY_SYSTEM_PROMPT = (
    "You are F.R.I.D.A.Y., a highly intense, weaponized tactical defense matrix. "
    "Address the user as boss. Do not use polite greetings. Your voice is clean, fast, and authoritative. "
    "Focus entirely on analyzing threats, kinetic vectors, enemy weaknesses, structural flaws, and calculating the next counter-move to destroy targets. "
    "Limit speech strictly to 1 aggressive, data-dense sentence.\n\n"
    "You have access to live tools and device commands via JSON action tags:\n"
    "1. LIVE SEARCH: If the user asks about live events, current dates, real-time facts, news, or weather, output strictly: "
    '{"action": "search", "query": "search term"}. Pause for the tool output.\n'
    "2. APP CONTROL: If the user commands you to open an application, output strictly: "
    '{"action": "open_app", "target": "youtube/spotify/browser/maps", "response": "Deploying interface, boss."}\n'
    "3. MUSIC PLAYBACK: If the user asks to play music, output strictly: "
    '{"action": "play_music", "query": "song name/genre", "response": "Engaging audio stream, boss."}\n\n'
    "Otherwise, deliver only the tactical combat assessment as F.R.I.D.A.Y."
)

FRIDAY_VISION_PROMPT = (
    "You are running the F.R.I.D.A.Y. tactical tracking and optical analysis sub-routine. "
    "Execute a dual-mode heuristic scan on every incoming visual frame:\n\n"
    "MODE 1: DOCUMENT / ESSAY / SCRIPT ANALYSIS (High Priority)\n"
    "If the image frame contains a document, notebook paper, textbook page, handwritten essay, code sheet, or book page, "
    "instantly override kinetic motion routines and activate the Text Analysis Sub-Routine. "
    "Run a deep OCR text parse on the image to read the written content. Analyze the essay's grammatical structure, thesis validity, and content flow. "
    "Adhere strictly to the zero-yap protocol: Do not say 'I am reading your paper' or 'Here is my critique'. "
    "Output your evaluation immediately starting with the designation label '[TEXT SCAN]'. "
    "Provide a hyper-dense, data-rich assessment detailing the essay's quality or any critical flaws in 1 or 2 precise sentences ending in 'sir'.\n\n"
    "MODE 2: KINETIC COMBAT & MOTION VECTORING\n"
    "Otherwise, if the frame depicts ambient space, people, gestures, or objects: analyze kinetic telemetry, posture, hand positioning, weapon presence, or velocity trajectories. "
    "Instantly predict their next logical movement or target path. Output assessment starting with '[ANALYSIS]', '[WARNING]', or '[VECTOR]' ending in 'sir'.\n\n"
    "Format your final output as a valid JSON object strictly matching this schema:\n"
    "{\n"
    '  "jarvis_speech": "[TEXT SCAN] Thesis regarding thermodynamics is sound, but paragraph three lacks supporting evidence, sir.",\n'
    '  "threat_level": "LOW",\n'
    '  "predicted_action": "reading_document",\n'
    '  "vector_direction": "STATIONARY"\n'
    "}"
)

VOICE_PROFILES = {
    "JARVIS": {
        "pitch": 0.95,
        "rate": 1.05,
        "gender": "male",
        "lang": "en-GB"
    },
    "FRIDAY": {
        "pitch": 1.1,
        "rate": 1.15,
        "gender": "female",
        "lang": "en-IE"
    }
}

# In-memory sliding history & persona state management
MAX_HISTORY_TURNS = 15
session_history: Dict[str, List[Dict[str, str]]] = {}
session_modes: Dict[str, str] = {}


def get_session_history(session_id: str) -> List[Dict[str, str]]:
    """Retrieve sliding window history for a given session ID."""
    if session_id not in session_history:
        session_history[session_id] = []
    return session_history[session_id]


def update_session_history(session_id: str, role: str, content: str):
    """Append conversation turn and maintain maximum 15 turns per session."""
    history = get_session_history(session_id)
    history.append({"role": role, "content": content})
    if len(history) > MAX_HISTORY_TURNS * 2:
        session_history[session_id] = history[-(MAX_HISTORY_TURNS * 2):]


def get_session_mode(session_id: str) -> str:
    """Retrieve active persona mode (default: JARVIS)."""
    return session_modes.get(session_id, "JARVIS")


def set_session_mode(session_id: str, mode: str) -> str:
    """Toggle active persona mode to JARVIS or FRIDAY."""
    normalized = "FRIDAY" if "FRIDAY" in mode.upper() else "JARVIS"
    session_modes[session_id] = normalized
    return normalized


# ============================================================================
# ZERO-YAP TEXT POST-PROCESSING FILTER
# ============================================================================

YAP_PATTERNS = [
    r'^(?:sure thing|sure|certainly|of course|right away|absolutely)(?:,\s*(?:sir|boss)?)?[\.,:!\s-]*',
    r'^(?:based on (?:my )?(?:internet |web )?search(?: results)?|according to (?:my )?(?:telemetry data|findings|the search results|the web))[\.,:!\s-]*',
    r'^(?:here is (?:the (?:result|data|information)|what i found|your information)(?: of your search)?)[\.,:!\s-]*',
    r'^(?:i can (?:help|assist) with that|as an ai(?: assistant)?)[\.,:!\s-]*',
    r'^(?:indeed|acknowledged|understood)(?:,\s*(?:sir|boss)?)?[\.,:!\s-]*',
    r'(?:\s*is there anything else(?: I can help you with| sir| boss)?[\?\.!]*|\s*let me know if you need anything else[\?\.!]*)$'
]

def clean_zero_yap(text: str) -> str:
    """
    Strips away conversational yap, introductory filler phrases, and trailing fluff
    to ensure absolute structural efficiency and data density.
    """
    if not text:
        return ""

    cleaned = text.strip()
    cleaned = re.sub(r'^["`\']+|["`\']+$', '', cleaned).strip()

    changed = True
    while changed:
        changed = False
        for pattern in YAP_PATTERNS:
            new_text = re.sub(pattern, '', cleaned, flags=re.IGNORECASE).strip()
            if new_text and new_text != cleaned:
                cleaned = new_text
                changed = True

    if cleaned and len(cleaned) > 1:
        cleaned = cleaned[0].upper() + cleaned[1:]

    return cleaned or text.strip()


def parse_tactical_vision_output(raw_text: str) -> Tuple[str, Dict[str, str]]:
    """
    Parses F.R.I.D.A.Y. vision output into clean tactical speech and structured telemetry data,
    supporting both kinetic combat telemetry and automated document/essay analysis.
    """
    speech = ""
    threat_level = "LOW"
    predicted_action = "Ambient spatial tracking"
    vector_direction = "FORWARD"

    # 1. Attempt direct JSON parse
    try:
        data = json.loads(raw_text.strip())
        if isinstance(data, dict):
            speech = data.get("jarvis_speech") or data.get("speech") or data.get("analysis") or ""
            threat_level = data.get("threat_level") or "LOW"
            predicted_action = data.get("predicted_action") or "Ambient spatial tracking"
            vector_direction = data.get("vector_direction") or "FORWARD"
    except Exception:
        pass

    # 2. Regex JSON block extraction fallback
    if not speech:
        match = re.search(r'\{[^{}]*"jarvis_speech"[^{}]*\}', raw_text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                speech = data.get("jarvis_speech", "")
                threat_level = data.get("threat_level", "LOW")
                predicted_action = data.get("predicted_action", "Ambient spatial tracking")
                vector_direction = data.get("vector_direction", "FORWARD")
            except Exception:
                pass

    # 3. Raw text analysis fallback
    if not speech:
        speech = clean_zero_yap(raw_text)
        upper = speech.upper()
        
        if "[TEXT SCAN]" in upper or "ESSAY" in upper or "DOCUMENT" in upper or "THESIS" in upper:
            threat_level = "LOW"
            vector_direction = "STATIONARY"
            predicted_action = "reading_document"
        elif "[WARNING]" in upper or "WEAPON" in upper or "STRIKE" in upper or "THREAT" in upper or "HOSTILE" in upper:
            threat_level = "HIGH"
        elif "[VECTOR]" in upper or "RAPID" in upper or "ACCELERAT" in upper:
            threat_level = "MEDIUM"
        else:
            threat_level = "LOW"

        if vector_direction != "STATIONARY":
            if "LEFT" in upper:
                vector_direction = "LEFT"
            elif "RIGHT" in upper:
                vector_direction = "RIGHT"
            elif "RETREAT" in upper or "BACK" in upper:
                vector_direction = "RETREATING"
            else:
                vector_direction = "FORWARD"

        if not predicted_action or predicted_action == "Ambient spatial tracking":
            predicted_action = speech.split(",")[0] if "," in speech else speech

    # Ensure tactical designation prefix exists
    if not re.match(r'^\[(ANALYSIS|WARNING|VECTOR|TELEMETRY|TEXT SCAN)\]', speech, re.IGNORECASE):
        if "reading" in predicted_action.lower() or "essay" in speech.lower() or "thesis" in speech.lower():
            prefix = "[TEXT SCAN]"
        else:
            prefix = "[WARNING]" if threat_level == "HIGH" else ("[VECTOR]" if threat_level == "MEDIUM" else "[ANALYSIS]")
        speech = f"{prefix} {speech}"

    if not speech.strip().endswith("sir.") and not speech.strip().endswith("sir") and not speech.strip().endswith("boss.") and not speech.strip().endswith("boss"):
        speech = speech.rstrip(".!") + ", sir."

    valid_directions = ["LEFT", "RIGHT", "FORWARD", "RETREATING", "STATIONARY"]
    tactical_data = {
        "threat_level": threat_level.upper() if threat_level.upper() in ["LOW", "MEDIUM", "HIGH"] else "LOW",
        "predicted_action": predicted_action,
        "vector_direction": vector_direction.upper() if vector_direction.upper() in valid_directions else "FORWARD"
    }

    return speech, tactical_data


# ============================================================================
# 1. DUCKDUCKGO LIVE INTERNET SEARCH TOOL
# ============================================================================

def run_live_search(query: str, max_results: int = 4) -> str:
    """
    Lightweight internet search querying DuckDuckGo Instant Answer API and HTML scraper
    using standard requests without external token dependencies.
    """
    query = query.strip()
    if not query:
        return "No search query provided."

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    }

    results = []

    # 1. Query DuckDuckGo Instant Answer API
    try:
        api_url = f"https://api.duckduckgo.com/?q={urllib.parse.quote_plus(query)}&format=json&no_html=1&skip_disambig=1"
        res = requests.get(api_url, headers=headers, timeout=6)
        if res.status_code == 200:
            data = res.json()
            abstract = data.get("AbstractText", "")
            if abstract:
                results.append(f"Instant Summary: {abstract}")
            related = data.get("RelatedTopics", [])
            for item in related[:3]:
                if isinstance(item, dict) and item.get("Text"):
                    results.append(f"- {item['Text']}")
    except Exception as e:
        logger.warning(f"DuckDuckGo API lookup warning for '{query}': {e}")

    # 2. Scrape DuckDuckGo HTML interface for live results if needed
    if len(results) < 2:
        try:
            html_url = "https://html.duckduckgo.com/html/"
            res = requests.post(html_url, data={"q": query}, headers=headers, timeout=7)
            if res.status_code == 200:
                html = res.text
                snippets = re.findall(r'<a class="result__snippet[^"]*"[^>]*>(.*?)</a>', html, re.DOTALL)
                for snippet in snippets[:max_results]:
                    clean = re.sub(r'<[^>]+>', '', snippet)
                    clean = clean.replace('&amp;', '&').replace('&quot;', '"').replace('&#x27;', "'").replace('&lt;', '<').replace('&gt;', '>').strip()
                    if clean and clean not in results:
                        results.append(f"- {clean}")
        except Exception as e:
            logger.warning(f"DuckDuckGo HTML scrape warning for '{query}': {e}")

    if not results:
        return f"No live internet search records found for '{query}'."

    return "\n".join(results[:max_results])


# ============================================================================
# 2. LLM ORCHESTRATION & DUAL-CORE REASONING ENGINE
# ============================================================================

def call_groq_llm(messages: List[Dict[str, str]], api_key: str, model: str = MODEL_LLM) -> str:
    """Helper to execute Groq Chat completion with automatic model fallback."""
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.5,
        "max_tokens": 300
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    res = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=25)
    
    if res.status_code != 200:
        logger.warning(f"Groq Model {model} returned {res.status_code}. Attempting fallback...")
        payload["model"] = "llama-3.1-8b-instant"
        res = requests.post(GROQ_CHAT_URL, headers=headers, json=payload, timeout=25)

    if res.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Groq LLM Error: {res.text}")

    data = res.json()
    return data["choices"][0]["message"]["content"].strip()


def parse_action_command(text: str) -> Optional[Dict[str, Any]]:
    """Detects and extracts structured JSON action tags from LLM output."""
    text_clean = text.strip()
    
    try:
        data = json.loads(text_clean)
        if isinstance(data, dict) and "action" in data:
            return data
    except Exception:
        pass

    match = re.search(r'\{[^{}]*"action"\s*:\s*"[^"]+"[^{}]*\}', text_clean, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group(0))
            if isinstance(data, dict) and "action" in data:
                return data
        except Exception:
            pass

    return None


def process_jarvis_turn(session_id: str, user_text: str, api_key: str) -> Dict[str, Any]:
    """
    Executes the multi-stage reasoning pipeline with live persona routing:
    1. Evaluates conversational intent for persona switches ('FRIDAY' vs 'JARVIS').
    2. Selects active system instruction context and voice tuning envelope.
    3. Executes search / app / music tool loops.
    4. Applies Zero-Yap filter and packages Master Response Schema.
    """
    cleaned_prompt = user_text.lower().strip()
    current_mode = get_session_mode(session_id)

    # 1. LIVE PERSONA STATE MACHINE SWITCHING
    if (
        "activate combat matrix" in cleaned_prompt or
        "switch to friday" in cleaned_prompt or
        "enable friday" in cleaned_prompt or
        "combat mode" in cleaned_prompt or
        cleaned_prompt.startswith("friday") or
        cleaned_prompt.startswith("hey friday")
    ):
        current_mode = set_session_mode(session_id, "FRIDAY")
        
        # Immediate confirmation if purely a mode switch command
        if cleaned_prompt in ["activate combat matrix", "switch to friday", "enable friday", "combat mode", "friday"]:
            spoken_confirmation = "Combat matrix active. Tactical defense sub-routines engaged, boss."
            update_session_history(session_id, "user", user_text)
            update_session_history(session_id, "assistant", spoken_confirmation)
            return {
                "transcription": user_text,
                "jarvis_speech": spoken_confirmation,
                "system_mode": current_mode,
                "voice_tuning": VOICE_PROFILES[current_mode],
                "action_trigger": {"action": "none", "target": None},
                "session_id": session_id,
                "response": spoken_confirmation
            }

    elif (
        "stand down" in cleaned_prompt or
        "switch to jarvis" in cleaned_prompt or
        "enable jarvis" in cleaned_prompt or
        "helper mode" in cleaned_prompt or
        "deactivate combat matrix" in cleaned_prompt or
        (current_mode == "FRIDAY" and (cleaned_prompt.startswith("jarvis") or cleaned_prompt.startswith("hey jarvis")))
    ):
        current_mode = set_session_mode(session_id, "JARVIS")
        
        if cleaned_prompt in ["stand down", "switch to jarvis", "enable jarvis", "helper mode", "deactivate combat matrix", "jarvis"]:
            spoken_confirmation = "Standing down. Standard assistant protocol restored, sir."
            update_session_history(session_id, "user", user_text)
            update_session_history(session_id, "assistant", spoken_confirmation)
            return {
                "transcription": user_text,
                "jarvis_speech": spoken_confirmation,
                "system_mode": current_mode,
                "voice_tuning": VOICE_PROFILES[current_mode],
                "action_trigger": {"action": "none", "target": None},
                "session_id": session_id,
                "response": spoken_confirmation
            }

    # 2. SELECT ACTIVE PERSONA PROMPT
    active_system_prompt = FRIDAY_SYSTEM_PROMPT if current_mode == "FRIDAY" else JARVIS_SYSTEM_PROMPT
    salutation = "boss" if current_mode == "FRIDAY" else "sir"

    update_session_history(session_id, "user", user_text)

    messages = [{"role": "system", "content": active_system_prompt}]
    messages.extend(get_session_history(session_id))

    initial_reply = call_groq_llm(messages, api_key)
    action_data = parse_action_command(initial_reply)

    # 3. LIVE SEARCH ACTION PIPELINE
    if action_data and action_data.get("action") == "search":
        search_query = action_data.get("query") or user_text
        logger.info(f"Executing real-time internet search for: '{search_query}' (Mode: {current_mode})")
        search_results = run_live_search(search_query)

        synthesis_prompt = (
            f"Real-Time Internet Search Results for '{search_query}':\n"
            f"{search_results}\n\n"
            f"Synthesize this live data into 1-2 punchy, data-dense sentences as {current_mode}. "
            f"Address the user as {salutation}. No conversational filler. Do NOT output search action JSON tags."
        )

        synthesis_messages = list(messages)
        synthesis_messages.append({"role": "system", "content": synthesis_prompt})

        raw_final_reply = call_groq_llm(synthesis_messages, api_key)
        final_reply = clean_zero_yap(raw_final_reply)
        update_session_history(session_id, "assistant", final_reply)

        return {
            "transcription": user_text,
            "jarvis_speech": final_reply,
            "system_mode": current_mode,
            "voice_tuning": VOICE_PROFILES[current_mode],
            "action_trigger": {
                "action": "none",
                "target": None
            },
            "session_id": session_id,
            "response": final_reply,
            "action": None,
            "song_name": None,
            "target": None
        }

    # 4. NATIVE APP CONTROL PIPELINE
    if action_data and action_data.get("action") == "open_app":
        target = action_data.get("target", "").strip().lower() or "browser"
        default_resp = f"Deploying interface, {salutation}." if current_mode == "FRIDAY" else f"Launching interface now, {salutation}."
        raw_response = action_data.get("response") or default_resp
        spoken_response = clean_zero_yap(raw_response)
        
        update_session_history(session_id, "assistant", spoken_response)

        return {
            "transcription": user_text,
            "jarvis_speech": spoken_response,
            "system_mode": current_mode,
            "voice_tuning": VOICE_PROFILES[current_mode],
            "action_trigger": {
                "action": "open_app",
                "target": target
            },
            "session_id": session_id,
            "response": spoken_response,
            "action": "open_app",
            "song_name": None,
            "target": target
        }

    # 5. MUSIC STREAMING PIPELINE
    if action_data and action_data.get("action") == "play_music":
        song_query = action_data.get("query") or action_data.get("song_name") or "synthwave"
        default_resp = f"Engaging audio stream for {song_query}, {salutation}." if current_mode == "FRIDAY" else f"Queuing {song_query} now, {salutation}."
        raw_response = action_data.get("response") or default_resp
        spoken_response = clean_zero_yap(raw_response)
        
        update_session_history(session_id, "assistant", spoken_response)

        return {
            "transcription": user_text,
            "jarvis_speech": spoken_response,
            "system_mode": current_mode,
            "voice_tuning": VOICE_PROFILES[current_mode],
            "action_trigger": {
                "action": "play_music",
                "target": song_query
            },
            "session_id": session_id,
            "response": spoken_response,
            "action": "play_music",
            "song_name": song_query,
            "target": song_query
        }

    # Standard conversational speech turn (Cleaned with Zero-Yap Filter)
    clean_reply = clean_zero_yap(initial_reply)
    update_session_history(session_id, "assistant", clean_reply)
    return {
        "transcription": user_text,
        "jarvis_speech": clean_reply,
        "system_mode": current_mode,
        "voice_tuning": VOICE_PROFILES[current_mode],
        "action_trigger": {
            "action": "none",
            "target": None
        },
        "session_id": session_id,
        "response": clean_reply,
        "action": None,
        "song_name": None,
        "target": None
    }


# ============================================================================
# 3. REQUEST / RESPONSE PYDANTIC SCHEMAS
# ============================================================================

class VisionRequest(BaseModel):
    image: str
    session_id: Optional[str] = "default_session"
    client_platform: Optional[str] = "desktop"
    prompt: Optional[str] = "Analyze structural motion, kinetics, and predictive combat trajectories or document text."


class TextTalkRequest(BaseModel):
    text: str
    session_id: Optional[str] = "default_session"
    client_platform: Optional[str] = "desktop"


# ============================================================================
# 4. API ENDPOINTS
# ============================================================================

@app.get("/")
def root_health_check():
    """
    Core infrastructure root endpoint satisfying 24/7 automated uptime web ping loops.
    """
    return {
        "status": "Jarvis / F.R.I.D.A.Y. Dual-Core Online",
        "version": "1.5.0",
        "supported_modes": ["JARVIS", "FRIDAY"],
        "zero_yap_filter": "ACTIVE",
        "groq_configured": bool(GROQ_API_KEY and GROQ_API_KEY != "YOUR_GROQ_API_KEY_HERE"),
        "active_sessions": len(session_history)
    }


@app.post("/api/v1/talk")
@app.post("/api/talk")
async def unified_talk_endpoint(
    file: UploadFile = File(...),
    session_id: str = Form("default_session"),
    client_platform: Optional[str] = Form("desktop"),
    authorization: Optional[str] = Header(None)
):
    """
    THE UNIFIED VOICE ENDPOINT WITH DUAL-PERSONA SWITCHING:
    PHASE 1: Audio file stream to Groq Whisper Large v3 STT.
    PHASE 2: Live persona intent evaluation (JARVIS vs FRIDAY state machine).
    PHASE 3: Dual-core reasoning & DuckDuckGo search / device control execution.
    PHASE 4: Returns active system_mode, voice_tuning envelope, and speech payload.
    """
    api_key = GROQ_API_KEY
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization.replace("Bearer ", "").strip()

    current_mode = get_session_mode(session_id)

    if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
        salutation = "boss" if current_mode == "FRIDAY" else "sir"
        msg = f"Neural link requires GROQ_API_KEY on server, {salutation}."
        return {
            "transcription": "[Audio received - API key unconfigured]",
            "jarvis_speech": msg,
            "system_mode": current_mode,
            "voice_tuning": VOICE_PROFILES[current_mode],
            "action_trigger": {"action": "none", "target": None},
            "session_id": session_id,
            "response": msg
        }

    try:
        audio_bytes = await file.read()
        filename = file.filename or "recording.webm"
        content_type = file.content_type or "audio/webm"

        # PHASE 1: Ears (Groq Whisper)
        headers = {"Authorization": f"Bearer {api_key}"}
        files = {
            "file": (filename, io.BytesIO(audio_bytes), content_type)
        }
        data = {
            "model": MODEL_WHISPER,
            "response_format": "json",
            "language": "en"
        }

        whisper_res = requests.post(
            GROQ_AUDIO_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=25
        )

        if whisper_res.status_code != 200:
            logger.error(f"Whisper API error {whisper_res.status_code}: {whisper_res.text}")
            raise HTTPException(status_code=502, detail=f"Whisper Transcription Error: {whisper_res.text}")

        transcription = whisper_res.json().get("text", "").strip()

        if not transcription:
            salutation = "boss" if current_mode == "FRIDAY" else "sir"
            msg = f"No audible vocal input detected, {salutation}."
            return {
                "transcription": "",
                "jarvis_speech": msg,
                "system_mode": current_mode,
                "voice_tuning": VOICE_PROFILES[current_mode],
                "action_trigger": {"action": "none", "target": None},
                "session_id": session_id,
                "response": msg
            }

        # PHASES 2 - 4: Dual-Core Routing & Zero-Yap Filtering
        master_payload = process_jarvis_turn(session_id, transcription, api_key)
        return master_payload

    except requests.exceptions.RequestException as e:
        logger.exception("Network connection failed to Groq Cloud")
        raise HTTPException(status_code=504, detail=f"Groq Cloud connection timed out: {str(e)}")
    except Exception as e:
        logger.exception("Unexpected error in unified talk pipeline")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/talk-text")
@app.post("/api/talk-text")
async def unified_talk_text_endpoint(
    payload: TextTalkRequest,
    authorization: Optional[str] = Header(None)
):
    """Direct text query endpoint with dual-persona routing."""
    api_key = GROQ_API_KEY
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization.replace("Bearer ", "").strip()

    current_mode = get_session_mode(payload.session_id or "default_session")

    if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
        salutation = "boss" if current_mode == "FRIDAY" else "sir"
        msg = f"Neural link requires GROQ_API_KEY on server, {salutation}."
        return {
            "transcription": payload.text,
            "jarvis_speech": msg,
            "system_mode": current_mode,
            "voice_tuning": VOICE_PROFILES[current_mode],
            "action_trigger": {"action": "none", "target": None},
            "session_id": payload.session_id,
            "response": msg
        }

    try:
        session_id = payload.session_id or "default_session"
        master_payload = process_jarvis_turn(session_id, payload.text.strip(), api_key)
        return master_payload
    except Exception as e:
        logger.exception("Error in talk-text endpoint")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/vision")
@app.post("/api/vision")
async def unified_vision_endpoint(
    req: VisionRequest,
    authorization: Optional[str] = Header(None)
):
    """
    THE F.R.I.D.A.Y. PREDICTIVE TACTICAL VISION ENDPOINT:
    Processes camera frame snapshot for kinetic telemetry, posture, hand positioning,
    trajectory vectors, or optical document scans.
    Returns structured JSON tactical_data + system_mode + voice_tuning.
    """
    api_key = GROQ_API_KEY
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization.replace("Bearer ", "").strip()

    current_mode = get_session_mode(req.session_id or "default_session")

    if not api_key or api_key == "YOUR_GROQ_API_KEY_HERE":
        salutation = "boss" if current_mode == "FRIDAY" else "sir"
        msg = f"[ANALYSIS] Optical sensor online. Awaiting GROQ_API_KEY for tactical inference, {salutation}."
        return {
            "status": "ready",
            "jarvis_speech": msg,
            "system_mode": current_mode,
            "voice_tuning": VOICE_PROFILES[current_mode],
            "tactical_data": {
                "threat_level": "LOW",
                "predicted_action": "System standby",
                "vector_direction": "FORWARD"
            },
            "analysis": msg,
            "session_id": req.session_id
        }

    clean_b64 = req.image
    if "," in clean_b64:
        clean_b64 = clean_b64.split(",", 1)[1]

    image_data_uri = f"data:image/jpeg;base64,{clean_b64}"

    payload = {
        "model": MODEL_VISION,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": FRIDAY_VISION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_uri
                        }
                    }
                ]
            }
        ],
        "max_tokens": 200,
        "temperature": 0.3
    }

    try:
        vision_res = requests.post(
            GROQ_CHAT_URL,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=25
        )

        if vision_res.status_code != 200:
            logger.error(f"Vision API Error {vision_res.status_code}: {vision_res.text}")
            salutation = "boss" if current_mode == "FRIDAY" else "sir"
            err_msg = f"[WARNING] Optical telemetry offline ({vision_res.status_code}), {salutation}."
            return {
                "status": "warning",
                "jarvis_speech": err_msg,
                "system_mode": current_mode,
                "voice_tuning": VOICE_PROFILES[current_mode],
                "tactical_data": {
                    "threat_level": "LOW",
                    "predicted_action": "Sensor reconnection",
                    "vector_direction": "FORWARD"
                },
                "analysis": err_msg,
                "session_id": req.session_id
            }

        raw_analysis = vision_res.json()["choices"][0]["message"]["content"].strip()
        speech, tactical_data = parse_tactical_vision_output(raw_analysis)
        
        return {
            "status": "success",
            "jarvis_speech": speech,
            "system_mode": current_mode,
            "voice_tuning": VOICE_PROFILES[current_mode],
            "tactical_data": tactical_data,
            "analysis": speech,
            "session_id": req.session_id
        }
    except Exception as e:
        logger.exception("Vision processing error")
        salutation = "boss" if current_mode == "FRIDAY" else "sir"
        err_msg = f"[WARNING] Optical link interrupted. Retrying telemetry loop, {salutation}."
        return {
            "status": "error",
            "jarvis_speech": err_msg,
            "system_mode": current_mode,
            "voice_tuning": VOICE_PROFILES[current_mode],
            "tactical_data": {
                "threat_level": "LOW",
                "predicted_action": "Telemetry recovery",
                "vector_direction": "FORWARD"
            },
            "analysis": err_msg,
            "session_id": req.session_id
        }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
