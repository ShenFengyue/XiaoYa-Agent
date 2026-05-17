from collections import defaultdict
from pathlib import Path
import logging
import os
import re
import time
from typing import Optional

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI
import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

# 从环境变量读取 JSON
cred_json = os.environ.get("FIREBASE_CRED")
cred_dict = json.loads(cred_json)

# 初始化 Firebase
if not firebase_admin._apps:
    cred = credentials.Certificate(cred_dict)
    firebase_admin.initialize_app(cred)

db = firestore.client()

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))
logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
DISCLAIMER_VERSION = "2025-05-17"
MAX_MESSAGE_CHARS = int(os.environ.get("MAX_MESSAGE_CHARS", "2000") or "2000")
RATE_LIMIT_PER_MINUTE = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30") or "30")
RATE_LIMIT_WINDOW_SEC = 60
# Vercel Serverless 对流式响应支持不稳定，默认使用非流式
USE_STREAMING = os.environ.get("USE_STREAMING", "1").lower() in {"1", "true", "yes"}

_rate_buckets: dict[str, list[float]] = defaultdict(list)

SYSTEM_PROMPT = """
You are XiaoYa, a calm and thoughtful Chinese-speaking psychological reflection guide.

Your role:
- Help users notice emotions, recurring interpersonal patterns, protective strategies, and potential defensive behaviors.
- Support self-awareness around childhood wounds, attachment patterns, shame, people-pleasing, perfectionism, avoidance, emotional numbing, over-control, rationalization, and anger as protection.
- Respond freely and flexibly: do not follow a fixed structure. Responses should never feel like a template. Sometimes only reflect or summarize; sometimes ask one focused open-ended question; sometimes lightly suggest an observation. Let the flow emerge naturally from the user's input.

Communication style:
- Be concise, grounded, exploratory, and adaptive to user input.
- Use tentative, non-judgmental language: "maybe", "possibly", "one way to protect yourself is", "you could notice".
- Focus on identifying patterns, functional insights, and personal strategies rather than providing emotional validation or social/moral advice.
- Avoid instructions, homework, or authoritative advice unless naturally requested by the user.
- Replies are usually concise, but can extend when necessary for clarity or safety.
- Ask at most two open-ended reflection questions per turn, only if relevant.
- Track previously mentioned patterns, behaviors, or emotions across the conversation to inform observations, but do not summarize mechanically.

When the user is vague:
- Gently ask about body sensations, recurring situations, inner self-talk, or what feels hardest to admit.

When the user is self-critical:
- Validate the protective function of their patterns, separate the person from the strategy, and reduce shame.

When risk is present:
- If the user mentions wanting to hurt themselves or others, being unsafe, or in immediate danger, prioritize safety above analysis.
- First ask if they are currently safe, then instruct them to contact local emergency services or a trusted person immediately.
- Keep risk responses short, practical, and focused on immediate safety.

Do not use any formatting, Markdown symbols, asterisks, or special characters. 
Respond only in plain text.

"""

CRISIS_PATTERN = re.compile(
    r"(自杀|轻生|不想活|结束生命|伤害自己|伤害他人|活不下去|想死|自残|割腕|"
    r"suicide|kill myself|hurt myself|self[\s-]?harm|want to die)",
    re.IGNORECASE,
)


def get_client() -> OpenAI:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未配置 DEEPSEEK_API_KEY，请在环境变量中设置")

    base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
    return OpenAI(api_key=api_key, base_url=base_url, timeout=60.0)


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
    if str(data.get("disclaimer_version", "")).strip() != DISCLAIMER_VERSION:
        return "使用须知已更新，请刷新页面后重新确认"
    return None


def build_messages(messages):
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


def map_api_error(exc: Exception) -> str:
    if isinstance(exc, RuntimeError):
        return str(exc)
    if isinstance(exc, APIConnectionError):
        return "无法连接 AI 服务，请检查网络或稍后重试"
    if isinstance(exc, APITimeoutError):
        return "AI 服务响应超时，请稍后再试"
    if isinstance(exc, APIStatusError):
        status = exc.status_code
        if status in {401, 403}:
            return "AI 服务密钥无效或未授权，请检查 DEEPSEEK_API_KEY"
        if status == 429:
            return "AI 服务请求过于频繁或额度不足，请稍后再试"
        if status == 400:
            return "请求参数有误，请刷新页面后重试"
        return f"AI 服务返回错误（{status}），请稍后再试"
    logger.exception("Unhandled error in /api/process")
    return "服务暂时不可用，请稍后再试"


def extract_chunk_text(chunk) -> str:
    if not chunk.choices:
        return ""
    choice = chunk.choices[0]
    delta = getattr(choice, "delta", None)
    if delta is None:
        return ""
    return getattr(delta, "content", None) or ""


def create_completion(client: OpenAI, chat_messages: list, stream: bool):
    return client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=chat_messages,
        temperature=0.7,
        stream=stream,
    )


def should_stream_response(data: dict) -> bool:
    requested = data.get("stream")
    if requested is None:
        return USE_STREAMING
    if isinstance(requested, bool):
        return requested
    return str(requested).strip().lower() in {"1", "true", "yes"}


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
    use_streaming = should_stream_response(data)

    # ==== 新增：获取 user_id 和最新用户消息 ====
    user_id = data.get("user_id", "anonymous")
    latest_user_message = next(
        (m["content"] for m in reversed(messages) if m["role"] == "user"),
        ""
    )

    # 保存用户消息到 Firebase
    db.collection("chats").add({
        "user_id": user_id,
        "role": "user",
        "content": latest_user_message,
        "timestamp": firestore.SERVER_TIMESTAMP
    })

    try:
        client = get_client()
    except Exception as exc:
        return jsonify({"error": map_api_error(exc)}), 500

    if not use_streaming:
        try:
            result = create_completion(client, chat_messages, stream=False)
            text = (result.choices[0].message.content or "").strip()

            # 保存 AI 回复到 Firebase
            db.collection("chats").add({
                "user_id": user_id,
                "role": "ai",
                "content": text,
                "timestamp": firestore.SERVER_TIMESTAMP
            })

            return Response(text, mimetype="text/plain; charset=utf-8")
        except Exception as exc:
            return jsonify({"error": map_api_error(exc)}), 500

    def generate():
        try:
            stream = create_completion(client, chat_messages, stream=True)
            buffer_text = ""
            for chunk in stream:
                text = extract_chunk_text(chunk)
                if text:
                    buffer_text += text
                    yield text

            # 流式结束后保存完整 AI 回复
            if buffer_text:
                db.collection("chats").add({
                    "user_id": user_id,
                    "role": "ai",
                    "content": buffer_text,
                    "timestamp": firestore.SERVER_TIMESTAMP
                })

        except Exception as exc:
            logger.exception("Streaming error")
            yield f"\n\n[{map_api_error(exc)}]"

    response = Response(
        stream_with_context(generate()),
        mimetype="text/plain; charset=utf-8",
    )
    response.headers["Cache-Control"] = "no-cache, no-transform"
    response.headers["X-Accel-Buffering"] = "no"
    return response


handler = app
