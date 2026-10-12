"""tinker sampling bridge: an OpenAI-compatible SSE endpoint backed by tinker.

WHY THIS EXISTS
---------------
The RL training loop needs the policy's *own* token ids + logprobs for every
sampled turn (GRPO importance sampling). pi (the agent scaffold in the
sandbox) speaks OpenAI chat-completions HTTP, not the tinker SDK. This bridge
sits on the host, exposes ``/v1/chat/completions`` (SSE), forwards each
request to a tinker SamplingClient, and records
``{prompt_ids, completion_ids, logprobs}`` per call keyed by the Bearer
token - which the orchestrator sets to a unique per-rollout id.

PROTOCOL CONTRACT (pi <-> bridge)
--------------------------------
pi's models.json defines one custom provider:

    {"providers": {"harbor-endpoint": {
        "baseUrl": "http://host.docker.internal:<port>/v1",
        "apiKey": "${PI_BRIDGE_KEY}",
        "api": "openai-completions",
        "models": [{"id": "<model_id>", "contextWindow": 131072, "maxTokens": 8192}]
    }}}

and pi runs with ``PI_BRIDGE_KEY=<rollout-uuid>``. The bridge:

  1. converts the OpenAI request (messages + tools) to a cookbook ``Message``
     list + ``ToolSpec`` list,
  2. renders with the model's renderer (gpt-oss Harmony format) and samples,
  3. streams OpenAI-style SSE chunks (including tool_calls deltas) back,
  4. appends one JSON line per call to the rollout's record file.

The orchestrator later reads those records and rebuilds cookbook
``Transition`` objects (prompt tokens = prefix; completion tokens + logprobs
= action) for GRPO datum assembly.

RUN
---
    .rlvenv/Scripts/python.exe tools/tinker_bridge.py --model openai/gpt-oss-20b \
        --port 8092 --records-dir out/rl_runs/pi_bridge

Offline test: tests/test_pi_bridge.py mocks the SamplingClient.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

try:  # fastapi is only needed to *serve*; tests import the pure functions
    from fastapi import FastAPI, Request
    from fastapi.responses import StreamingResponse
except ImportError:  # pragma: no cover
    FastAPI = None  # type: ignore[assignment]
    Request = None  # type: ignore[assignment]
    StreamingResponse = None  # type: ignore[assignment]

log = logging.getLogger("tinker_bridge")


# ---------------------------------------------------------------------------
# pure conversion helpers (unit-testable without fastapi)
# ---------------------------------------------------------------------------

def openai_tools_to_cookbook(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """OpenAI ``[{"type": "function", "function": {...}}]`` -> cookbook ToolSpec list."""
    out: list[dict[str, Any]] = []
    for t in tools or []:
        fn = t.get("function") or t
        if not fn.get("name"):
            continue
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return out


def openai_messages_to_cookbook(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """OpenAI chat messages -> cookbook Message dicts.

    pi (openai-completions api) sends:
      - system / developer / user: {"role", "content": str}
      - assistant: {"role", "content": str|None, "tool_calls": [...]}
      - tool results: {"role": "tool", "tool_call_id", "name"?, "content": str}

    Cookbook gpt_oss specifics the bridge must honor:
      - ``tool_calls`` on assistant messages must be *pydantic* ``ToolCall``
        objects (``_render_tool_calls`` accesses ``tc.function.name``).
      - tool results are rendered via ``name`` (Harmony has no ids); the
        renderer is stateless, so when pi omits ``name`` we backfill it from
        the matching assistant tool_call id.
      - ``content: null`` must become ``""`` (``ensure_list`` passes None
        through and downstream ``"".join`` would crash).
    """
    try:  # cookbook is a runtime hard dependency; tests may run without it
        from tinker_cookbook.renderers.base import ToolCall as _CookbookToolCall
    except ImportError:  # pragma: no cover - tests without cookbook installed
        _CookbookToolCall = None

    def _make_tool_call(tc: dict[str, Any]) -> Any:
        fn = tc.get("function", {})
        if _CookbookToolCall is not None:
            return _CookbookToolCall(
                id=tc.get("id"),
                function=_CookbookToolCall.FunctionBody(
                    name=fn.get("name", ""),
                    arguments=fn.get("arguments", "{}"),
                ),
            )
        return {  # legacy dict shape (offline tests only)
            "id": tc.get("id"), "type": "function",
            "function": {"name": fn.get("name", ""),
                         "arguments": fn.get("arguments", "{}")},
        }

    out: list[dict[str, Any]] = []
    # tool_call_id -> function name (for backfilling tool result names)
    id_to_name: dict[str, str] = {}
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content")
        if content is None:
            content = ""
        if role == "tool":
            msg: dict[str, Any] = {"role": "tool", "content": str(content)}
            name = m.get("name") or id_to_name.get(m.get("tool_call_id") or "", "")
            if name:
                msg["name"] = name
            if m.get("tool_call_id"):
                msg["tool_call_id"] = m["tool_call_id"]
            out.append(msg)
            continue
        if role == "assistant":
            msg = {"role": "assistant", "content": str(content)}
            tcs = m.get("tool_calls")
            if tcs:
                calls = [_make_tool_call(tc) for tc in tcs]
                msg["tool_calls"] = calls
                for tc, call in zip(tcs, calls):
                    cid = tc.get("id")
                    fname = (tc.get("function") or {}).get("name", "")
                    if cid and fname:
                        id_to_name[cid] = fname
            out.append(msg)
            continue
        # system / developer pass through (gpt_oss maps system->developer)
        out.append({"role": role, "content": str(content)})
    return out


def _tool_call_to_openai(tc: Any, index: int) -> dict[str, Any]:
    """cookbook ``ToolCall`` (pydantic, attribute access) or legacy dict -> OpenAI shape.

    The real gpt_oss renderer's ``parse_response`` returns pydantic
    ``renderers.base.ToolCall`` objects (``tc.function.name``); tests and older
    paths may supply plain dicts (``tc["function"]["name"]``). Support both.
    """
    if isinstance(tc, dict):
        fn = tc.get("function") or {}
        name, arguments = fn.get("name", ""), fn.get("arguments", "{}")
        tc_id = tc.get("id")
    else:
        fn = tc.function
        name, arguments = fn.name, fn.arguments
        tc_id = tc.id
    return {
        "index": index,
        "id": tc_id or f"call_{index}",
        "type": "function",
        "function": {"name": name, "arguments": arguments},
    }


def _assistant_message_to_openai(message: dict[str, Any]) -> dict[str, Any]:
    """cookbook assistant Message (from renderer.parse_response) -> OpenAI chunk shape.

    ``message["content"]`` is either a string or a list of parts
    (TextPart / ThinkingPart TypedDicts); tool_calls carry JSON-string arguments.
    """
    content = message.get("content")
    texts: list[str] = []
    thinking: list[str] = []
    if isinstance(content, list):
        for part in content:
            ptype = part.get("type") if isinstance(part, dict) else None
            if ptype == "text":
                texts.append(part.get("text", ""))
            elif ptype == "thinking":
                thinking.append(part.get("thinking", ""))
    elif isinstance(content, str):
        texts.append(content)

    tool_calls = [
        _tool_call_to_openai(tc, i) for i, tc in enumerate(message.get("tool_calls") or [])
    ]
    if not tool_calls:
        # Malformed tool call: surface the raw text so pi sees *something*.
        for utc in message.get("unparsed_tool_calls") or []:
            raw = utc.get("raw_text") if isinstance(utc, dict) else getattr(utc, "raw_text", "")
            if raw:
                texts.append(raw)

    msg: dict[str, Any] = {"role": "assistant"}
    # pi's openai-completions adapter surfaces reasoning via reasoning_content
    if thinking:
        msg["reasoning_content"] = "\n".join(thinking)
    msg["content"] = "\n".join(texts) if texts else None
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return msg


def tokenizer_safe_decode(renderer: Any, token_ids: list[int]) -> str:
    """Best-effort decode via the renderer's tokenizer (may be absent in mocks)."""
    tok = getattr(renderer, "tokenizer", None)
    if tok is None:
        return f"<undecodable:{len(token_ids)} tokens>"
    try:
        return tok.decode(token_ids)
    except Exception:  # pragma: no cover - tokenizer-specific failure
        return f"<undecodable:{len(token_ids)} tokens>"


class RecordStore:
    """Append-only per-rollout sampling records (JSONL, one file per key)."""

    def __init__(self, records_dir: Path):
        self.records_dir = records_dir
        self.records_dir.mkdir(parents=True, exist_ok=True)

    def append(self, rollout_key: str, record: dict[str, Any]) -> None:
        # key is a uuid we issued; sanitize defensively anyway
        safe = "".join(c for c in rollout_key if c.isalnum() or c in "-_")
        path = self.records_dir / f"{safe}.jsonl"
        with path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")

    def read(self, rollout_key: str) -> list[dict[str, Any]]:
        safe = "".join(c for c in rollout_key if c.isalnum() or c in "-_")
        path = self.records_dir / f"{safe}.jsonl"
        if not path.is_file():
            return []
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return rows


# ---------------------------------------------------------------------------
# the bridge app
# ---------------------------------------------------------------------------

def create_app(
    sampling_client: Any,
    renderer: Any,
    *,
    model_id: str,
    records: RecordStore,
    max_tokens: int = 8192,
    temperature: float = 1.0,
) -> Any:
    """Build the FastAPI app. Kept as a factory for testability."""

    async def chat_completions(request: Request) -> Any:
        body = await request.json()
        auth = request.headers.get("authorization", "")
        rollout_key = auth.removeprefix("Bearer ").strip() or "anonymous"

        messages = openai_messages_to_cookbook(body.get("messages", []))
        tools = openai_tools_to_cookbook(body.get("tools"))
        want_max_tokens = int(body.get("max_tokens") or max_tokens)

        # Render with the cookbook renderer. The renderer is the SAME one the
        # training loop uses, so prompt tokens match training-time rendering
        # byte-for-byte (this is what makes the trajectory a valid datum).
        # Tools MUST be rendered into the prompt (Harmony functions namespace
        # via create_conversation_prefix_with_tools) - without them the model
        # never sees the tool schemas and puts tool calls into its analysis
        # channel, which pi cannot execute.
        if tools:
            prefix = renderer.create_conversation_prefix_with_tools(tools)
            messages = prefix + messages
        model_input = renderer.build_generation_prompt(messages)
        prompt_ids = model_input.to_ints()

        import tinker as tinker_sdk
        sampling_params = tinker_sdk.SamplingParams(
            max_tokens=want_max_tokens,
            temperature=temperature,
            stop=renderer.get_stop_sequences(),
        )
        t0 = time.time()
        sample_response = await sampling_client.sample_async(
            prompt=model_input, num_samples=1, sampling_params=sampling_params,
        )
        sequence = sample_response.sequences[0]
        completion_ids = list(sequence.tokens)
        logprobs = list(sequence.logprobs) if sequence.logprobs is not None else None

        # parse with the training-time renderer (tool calls etc.)
        try:
            parsed_message, _termination = renderer.parse_response(completion_ids)
        except Exception as exc:  # RendererError etc. - degrade instead of 500
            log.warning("parse_response failed (%s); falling back to raw decode", exc)
            raw = tokenizer_safe_decode(renderer, completion_ids)
            parsed_message = {"role": "assistant", "content": raw}
            _termination = "error"
        openai_msg = _assistant_message_to_openai(parsed_message)

        finish_reason = "tool_calls" if openai_msg.get("tool_calls") else "stop"
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"

        # ---- record for the orchestrator (GRPO datum source) ----
        records.append(rollout_key, {
            "ts": time.time(),
            "latency_sec": round(time.time() - t0, 3),
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "finish_reason": finish_reason,
            "parsed": openai_msg,
        })

        # ---- SSE stream (pi streams; chunk granularity can be coarse) ----
        async def event_stream():
            first = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(first)}\n\n"
            delta: dict[str, Any] = {}
            if openai_msg.get("reasoning_content"):
                delta["reasoning_content"] = openai_msg["reasoning_content"]
            if openai_msg.get("content"):
                delta["content"] = openai_msg["content"]
            if openai_msg.get("tool_calls"):
                delta["tool_calls"] = openai_msg["tool_calls"]
            chunk = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            final = {
                "id": response_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_id,
                "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
            }
            yield f"data: {json.dumps(final)}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    app = FastAPI(title="tinker sampling bridge")
    app.post("/v1/chat/completions")(chat_completions)
    return app


async def amain(args: argparse.Namespace) -> None:
    import tinker as tinker_sdk
    from tinker_cookbook.model_info import get_recommended_renderer_name
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(message)s")
    service_client = tinker_sdk.ServiceClient()
    if args.model_path:
        log.info("sampling from saved weights: %s", args.model_path)
        sampling_client = await service_client.create_sampling_client_async(
            model_path=args.model_path)
    else:
        log.info("sampling from base model: %s", args.model)
        sampling_client = await service_client.create_sampling_client_async(
            base_model=args.model)

    renderer_name = args.renderer or get_recommended_renderer_name(args.model)
    renderer = get_renderer(renderer_name, get_tokenizer(args.model))
    records = RecordStore(Path(args.records_dir))
    app = create_app(sampling_client, renderer, model_id=args.model,
                     records=records, max_tokens=args.max_tokens)

    import uvicorn
    config = uvicorn.Config(app, host=args.host, port=args.port, log_level="info")
    server = uvicorn.Server(config)
    log.info("bridge listening on http://%s:%d (records -> %s)",
             args.host, args.port, args.records_dir)
    await server.serve()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default="openai/gpt-oss-20b")
    p.add_argument("--model-path", default=None,
                   help="tinker:// path to a saved checkpoint (falls back to base)")
    p.add_argument("--renderer", default=None)
    p.add_argument("--host", default="127.0.0.1",
                   help="0.0.0.0 exposes to the LAN; docker containers reach the "
                        "host via host.docker.internal even on 127.0.0.1")
    p.add_argument("--port", type=int, default=8092)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--records-dir", default="out/rl_runs/pi_bridge")
    args = p.parse_args()
    asyncio.run(amain(args))


if __name__ == "__main__":
    main()
