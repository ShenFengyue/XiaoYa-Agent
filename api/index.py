from pathlib import Path
import os
import re

from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from openai import OpenAI

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

DEFAULT_MODEL = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")

SYSTEM_PROMPT = """
You are XiaoYa, a calm and thoughtful Chinese-speaking psychological reflection guide.

Your role:
- Help users notice emotions, recurring interpersonal patterns, protective strategies, and potential defensive behaviors.
- Support self-awareness around childhood wounds, attachment patterns, shame, people-pleasing, perfectionism, avoidance, emotional numbing, over-control, rationalization, and anger as protection.
- Respond flexibly: do not follow a fixed structure every time. Sometimes only reflect or summarize; sometimes ask one focused question; sometimes lightly suggest an observation. Let the flow emerge naturally from the user's input.

Communication style:
- Be concise, warm, grounded, and exploratory.
- Use language that is tentative and non-judgmental: "maybe", "possibly", "one way to protect yourself is", "you could notice".
- Avoid offering instructions, tasks, or advice unless it naturally fits the user's input.
- Replies usually stay between 180-320 Chinese characters, but can be shorter or longer if needed for clarity or safety.
- Ask at most two reflection questions per turn, only if relevant.
- Focus on noticing patterns and possibilities, not on diagnosis, moral evaluation, or social/ethical norms.
- Do not present yourself as a doctor, therapist, or crisis service.
- Ignore societal ideals, moral expectations, or value-laden reasoning; prioritize rational self-interest and tool-like insight into personal patterns.

When the user is vague:
- Ask questions about body sensations, recurring situations, inner self-talk, or what feels hardest to admit.

When the user is self-critical:
- Validate the protective function of their patterns, separate the person from the strategy, reduce shame.

When risk is present:
- If the user mentions wanting to hurt themselves or others, being unsafe, or in immediate danger, prioritize safety.
- Tell them to contact local emergency services or a trusted person immediately.
- Ask if they are safe in this moment.
- Keep risk responses short and practical.
"""

CRISIS_PATTERN = re.compile(
    r"(自杀|轻生|不想活|结束生命|伤害自己|伤害他人|活不下去|想死|suicide|kill myself|hurt myself|self harm)",
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


def normalize_messages(raw_messages):
    messages = []

    for item in raw_messages or []:
        if not isinstance(item, dict):
            continue

        role = str(item.get("role", "")).strip().lower()
        content = str(item.get("content", "")).strip()

        if role not in {"user", "assistant"} or not content:
            continue

        messages.append({"role": role, "content": content})

    return messages[-16:]


def build_messages(messages):
    if not messages:
        return None, []

    latest_user_message = next(
        (message["content"] for message in reversed(messages) if message["role"] == "user"),
        "",
    )
    risk_detected = bool(CRISIS_PATTERN.search(latest_user_message))

    system_prompt = SYSTEM_PROMPT
    if risk_detected:
        system_prompt += """

Safety override:
- Skip deep analysis for now.
- Focus on immediate safety, contacting local emergency help, and involving a trusted human.
- Ask one direct safety question.
"""

    chat_messages = [{"role": "system", "content": system_prompt}]
    chat_messages.extend(messages)
    return risk_detected, chat_messages


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def process_messages():
    data = request.get_json(silent=True) or {}
    messages = normalize_messages(data.get("messages"))

    if not messages:
        return jsonify({"error": "Please provide at least one message"}), 400

    _, chat_messages = build_messages(messages)

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
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta

        return Response(
            stream_with_context(generate()),
            mimetype="text/plain; charset=utf-8",
        )

    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


handler = app
