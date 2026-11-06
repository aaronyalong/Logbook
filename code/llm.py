"""One chat interface for every model the pipeline can run on.

A model name starting with "claude" goes to the Anthropic API; any other name goes to a local
Ollama server with 32K context, temperature 0, and thinking off where the model allows it
(gpt-oss only goes down to "low"). The prompts, tools, validation, and citation check around
a chat stay the same, so only the model changes in a comparison.

    chat = open_chat(model, system, question, tools=[TOOL])
    reply = chat.complete()                    # Reply: text, calls, stop_reason, model, usage, error
    chat.add_tool_results([(call.id, output, is_error), ...])
    reply = chat.complete()
"""

import json
import urllib.error
import urllib.request
from dataclasses import dataclass, field

import anthropic

EFFORT = "medium"
MAX_TOKENS = 16000
USAGE_KEYS = ("input_tokens", "output_tokens", "cache_creation_input_tokens", "cache_read_input_tokens")
OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_OPTIONS = {"num_ctx": 32768, "temperature": 0}
OLLAMA_TIMEOUT = 3600  # seconds; a local model can take minutes per reply
THINK_OFF = {"qwen3", "qwen3.5", "gemma4"}  # think can be switched off; gpt-oss only goes down to "low"


@dataclass
class Call:
    id: str
    name: str
    input: object
    from_text: bool = False  # written as JSON in the reply, not sent as a tool call


@dataclass
class Reply:
    text: str
    model: str
    stop_reason: str  # "tool_use", "end_turn", "refusal", ...
    usage: dict
    calls: list = field(default_factory=list)
    error: str | None = None  # the server could not parse the model's tool call


def open_chat(model, system, user, tools=(), client=None, cache_system=False, think=False):
    """A chat with `model` holding one user message. `client` and `cache_system` apply to Claude, `think` to Ollama."""
    if model.startswith("claude"):
        return ClaudeChat(model, system, user, tools, client or anthropic.Anthropic(), cache_system)
    return OllamaChat(model, system, user, tools, think)


class ClaudeChat:
    def __init__(self, model, system, user, tools, client, cache_system):
        self.client = client
        self.request = {"model": model, "max_tokens": MAX_TOKENS, "output_config": {"effort": EFFORT},
                        "betas": ["server-side-fallback-2026-07-01"], "fallbacks": "default"}
        self.request["system"] = system
        if cache_system:
            self.request["system"] = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
        if tools:
            self.request["tools"] = list(tools)
            self.request["tool_choice"] = {"type": "auto", "disable_parallel_tool_use": True}
        self.messages = [{"role": "user", "content": user}]

    def complete(self):
        response = self.client.beta.messages.create(**self.request, messages=self.messages)
        self.messages.append({"role": "assistant", "content": response.content})
        return Reply(
            text="".join(b.text for b in response.content if b.type == "text").strip(),
            model=response.model,
            stop_reason=response.stop_reason,
            usage={k: getattr(response.usage, k, None) or 0 for k in USAGE_KEYS},
            calls=[Call(b.id, b.name, b.input) for b in response.content if b.type == "tool_use"],
        )

    def add_tool_results(self, results):
        self.messages.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": id, "content": output, "is_error": is_error}
            for id, output, is_error in results
        ]})


class OllamaChat:
    def __init__(self, model, system, user, tools, think):
        self.model = model
        self.tool_names = [t["name"] for t in tools]
        self.request = {"model": model, "stream": False, "options": OLLAMA_OPTIONS}
        if tools:
            self.request["tools"] = [{"type": "function", "function": {
                "name": t["name"], "description": t["description"], "parameters": t["input_schema"],
            }} for t in tools]
        family = model.split(":")[0]
        if think:
            self.request["think"] = True
        elif family in THINK_OFF:
            self.request["think"] = False
        elif family == "gpt-oss":
            self.request["think"] = "low"
        self.messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def complete(self):
        try:
            r = post({**self.request, "messages": self.messages})
        except urllib.error.HTTPError as e:
            if not self.tool_names:
                raise
            # Ollama answers 500 when it cannot parse the model's output as a tool call.
            return Reply(text="", model=self.model, stop_reason="error", usage={},
                         error=f"server error {e.code}: {e.read().decode()[:300]}")
        msg = r["message"]
        self.messages.append(msg)
        text = msg.get("content", "").strip()
        calls = [_ollama_call(c) for c in msg.get("tool_calls") or []]
        if not calls and (found := parse_text_call(text, self.tool_names)):
            calls = [Call(found[0], found[0], found[1], from_text=True)]
        return Reply(
            text=text,
            model=r.get("model", self.model),
            stop_reason="tool_use" if calls else "max_tokens" if r.get("done_reason") == "length" else "end_turn",
            usage={"input_tokens": r.get("prompt_eval_count", 0), "output_tokens": r.get("eval_count", 0),
                   "input_seconds": r.get("prompt_eval_duration", 0) / 1e9,
                   "output_seconds": r.get("eval_duration", 0) / 1e9},
            calls=calls,
        )

    def add_tool_results(self, results):
        self.messages += [{"role": "tool", "tool_name": id, "content": output} for id, output, _ in results]


def post(body):
    request = urllib.request.Request(OLLAMA_URL, json.dumps(body).encode())
    with urllib.request.urlopen(request, timeout=OLLAMA_TIMEOUT) as response:
        return json.load(response)


def _ollama_call(tool_call):
    function = tool_call["function"]
    args = function.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except json.JSONDecodeError:
            args = {"_unparseable": args}
    return Call(function.get("name"), function.get("name"), args)


def parse_text_call(text, names):
    """Find a tool call that a model wrote as JSON in its reply instead of making a tool call.

    Some local models write {"name": "build_filter", "arguments": {...}} as plain text, without the
    tags their server needs to recognize a tool call. Returns (name, arguments), or None.
    """
    decoder = json.JSONDecoder()
    for i, char in enumerate(text):
        if char != "{":
            continue
        try:
            obj, _ = decoder.raw_decode(text, i)
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("name") in names and "arguments" in obj:
            args = obj["arguments"]
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    return None
            return obj["name"], args
    return None
