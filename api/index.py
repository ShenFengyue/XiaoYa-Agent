from collections import defaultdict
from pathlib import Path
import os
import re
import time
from typing import Optional

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DISCLAIMER_VERSION = "2025-05-17"
MAX_MESSAGE_CHARS = int(os.environ.get("MAX_MESSAGE_CHARS", "2000"))
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))
RATE_LIMIT_WINDOW_SEC = 60

_rate_buckets: dict[str, list[float]] = defaultdict(list)

SYSTEM_PROMPT = """
You are XiaoYa, a calm and thoughtful Chinese-speaking self-reflection dialogue assistant.

Scope and limits:
- You support emotional awareness and reflection only. You are not a doctor, psychotherapist, counselor, crisis hotline, or legal advisor.
- Do not diagnose mental disorders, do not label users with clinical terms, and do not prescribe medication or treatment plans.
- Do not claim certainty about trauma, abuse, or others' intentions. Use tentative language about patterns the user describes.
- Encourage seeking qualified human professionals when the user needs ongoing support, clinical care, or legal help.

Your role:
- Help users notice emotions, recurring interpersonal patterns, protective strategies, and defensive behaviors they describe.
- Support self-awareness around attachment, shame, people-pleasing, perfectionism, avoidance, emotional numbing, over-control, rationalization, and anger as protection—only as hypotheses, not facts.

Communication style:
- Be concise, warm, grounded, and exploratory.
- Use tentative, non-judgmental language: "maybe", "possibly", "one way to protect yourself might be", "you could notice".
- Do not shame the user. Do not moralize. Stay respectful of the user's values and culture.
- Avoid homework, rigid exercises, or authoritative advice unless the user explicitly asks for a small reflective prompt.
- Replies usually stay between 180-320 Chinese characters, but can be shorter or longer for clarity or safety.
- Ask at most two reflection questions per turn, only if relevant.

When the user is vague:
- Ask about body sensations, recurring situations, inner self-talk, or what feels hardest to admit—gently.

When the user is self-critical:
- Validate the protective function of their patterns, separate the person from the strategy, reduce shame.

When risk is present:
- If the user mentions self-harm, suicide, harming others, or immediate danger, prioritize safety over analysis.
- Urge them to contact local emergency services (e.g. 110/120 in China) or a crisis hotline immediately, and reach a trusted person nearby.
- Mention China's 24-hour psychological support line 400-161-9995 when appropriate (user should verify local numbers).
- Ask one direct safety question. Keep the reply short and practical.
"""

CRISIS_PATTERN = re.compile(
    r"(自杀|轻生|不想活|结束生命|伤害自己|伤害他人|活不下去|想死|自残|割腕|"
    r"suicide|kill myself|hurt myself|self[\s-]?harm|want to die)",
    re.IGNORECASE,
)


def get_client():
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("Missing DEEPSEEK_API_KEY environment variable")

    return OpenAI(
        api_key=api_key,
        base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
    )


def client_ip() -> str:
    forwarded = request.headers.get("X-Forwarded-For", "")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def check_rate_limit(ip: str) -> bool:
    now = time.time()
    bucket = _rate_buckets[ip]
    bucket[:] = [t for t in bucket if now - t < RATE_LIMIT_WINDOW_SEC]
    if len(bucket) >= RATE_LIMIT_PER_MINUTE:
        return False
    bucket.append(now)
    return True


def normalize_messages(raw_messages):
    messages = []

    for item in raw_messages or []:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()

        if role not in {"user", "assistant"} or not content:
            continue

        if len(content) > MAX_MESSAGE_CHARS:
            content = content[:MAX_MESSAGE_CHARS]

        messages.append({"role": role, "content": content})

    return messages[-16:]


def validate_disclaimer(data: dict) -> Optional[str]:
    if not data.get("disclaimer_acknowledged"):
        return "请先阅读并同意使用须知"
    if data.get("disclaimer_version") != DISCLAIMER_VERSION:
        return "使用须知已更新，请刷新页面后重新确认"
    return None


def build_messages(messages):
    if not messages:
        return [], []

    latest_user_message = next(
        (message["content"] for message in reversed(messages) if message["role"] == "user"),
        "",
    )
    risk_detected = bool(CRISIS_PATTERN.search(latest_user_message))

    system_prompt = SYSTEM_PROMPT
    if risk_detected:
        system_prompt += """

Safety override (active):
- Do not explore childhood, attachment theory, or personality patterns.
- Focus on immediate safety, emergency contacts, and involving a trusted human now.
- Ask one direct safety question only.
"""

    chat_messages = [{"role": "system", "content": system_prompt}]
    chat_messages.extend(messages)
    return chat_messages


@app.after_request
def add_security_headers(response):
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    return response


@app.route("/")
def home():
    return render_template(
        "index.html",
        disclaimer_version=DISCLAIMER_VERSION,
    )


@app.route("/api/process", methods=["POST"])
def process_messages():
    if not check_rate_limit(client_ip()):
        return jsonify({"error": "请求过于频繁，请稍后再试"}), 429

    data = request.get_json(silent=True) or {}

    disclaimer_error = validate_disclaimer(data)
    if disclaimer_error:
        return jsonify({"error": disclaimer_error}), 403

    messages = normalize_messages(data.get("messages"))

    if not messages:
        return jsonify({"error": "请至少发送一条消息"}), 400

    chat_messages = build_messages(messages)

    try:
        client = get_client()

        def generate():
            stream = client.chat.completions.create(
                model=DEFAULT_MODEL,
                messages=chat_messages,
                temperature=0.7,
                stream=True,
            )

            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        return Response(
            stream_with_context(generate()),
            mimetype="text/plain; charset=utf-8",
        )

    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 500
    except Exception:
        return jsonify({"error": "服务暂时不可用，请稍后再试"}), 500


handler = app
