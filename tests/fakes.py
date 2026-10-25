"""Scripted stand-in for the Anthropic client, shared by the LLM-step tests (no API calls)."""

from types import SimpleNamespace

import build_filter as bf


def text(t):
    return SimpleNamespace(type="text", text=t)


def tool_use(filters, id="t1"):
    return SimpleNamespace(type="tool_use", id=id, name="build_filter", input={"filters": filters})


def response(*content, stop_reason=None):
    has_tool = any(b.type == "tool_use" for b in content)
    return SimpleNamespace(
        content=[SimpleNamespace(type="thinking", thinking=""), *content],
        stop_reason=stop_reason or ("tool_use" if has_tool else "end_turn"),
        model=bf.MODEL,
        usage=SimpleNamespace(input_tokens=10, output_tokens=5, cache_creation_input_tokens=0,
                              cache_read_input_tokens=100),
    )


class FakeClient:
    """Returns the scripted responses in order and records every request."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})
        return self.responses.pop(0)
