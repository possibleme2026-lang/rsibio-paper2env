"""Offline tests for the pi-GRPO sampling bridge + orchestrator (task #28).

Covers:
  - tinker_bridge pure conversions (tools/messages, assistant message shape)
  - RecordStore append/read + key sanitization
  - the SSE endpoint end-to-end via httpx ASGITransport with a MOCK
    SamplingClient + MOCK renderer (no tinker service, no network)
  - pi_grpo_train: reward parsing, records -> Trajectory, the full GRPO
    assembly chain (groups -> advantages -> datums), and a dry-run
    train_step with the bridge subprocess patched out.

Run under .rlvenv (needs fastapi/httpx/tinker_cookbook, all present).
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "tools"))

# The CI suite is stdlib-only by contract (see .github/workflows/ci.yml: no
# install step). This module drives the tinker SDK, so it skips where that SDK
# is absent -- a missing optional dependency is not a regression.
try:
    import tinker as tinker_sdk  # noqa: E402

    from tinker_bridge import (  # noqa: E402
        RecordStore,
        _assistant_message_to_openai,
        create_app,
        openai_messages_to_cookbook,
        openai_tools_to_cookbook,
    )
except ImportError as _exc:  # pragma: no cover
    raise unittest.SkipTest(f"tinker SDK not installed: {_exc}")

try:  # cookbook types are needed for the trajectory/datum tests
    import tinker_cookbook  # noqa: F401
    HAS_COOKBOOK = True
except ImportError:  # pragma: no cover
    HAS_COOKBOOK = False


# ---------------------------------------------------------------------------
# pure conversion helpers
# ---------------------------------------------------------------------------

class TestOpenAIToolsToCookbook(unittest.TestCase):
    def test_nested_openai_format(self):
        tools = [{
            "type": "function",
            "function": {
                "name": "bash",
                "description": "run a shell command",
                "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}},
            },
        }]
        out = openai_tools_to_cookbook(tools)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["name"], "bash")
        self.assertEqual(out[0]["parameters"]["properties"]["cmd"]["type"], "string")

    def test_flat_form_tolerated(self):
        out = openai_tools_to_cookbook([{"name": "read", "description": "d"}])
        self.assertEqual(out[0]["name"], "read")

    def test_missing_name_skipped(self):
        self.assertEqual(openai_tools_to_cookbook([{"description": "no name"}]), [])

    def test_none_and_empty(self):
        self.assertEqual(openai_tools_to_cookbook(None), [])
        self.assertEqual(openai_tools_to_cookbook([]), [])

    def test_params_default(self):
        out = openai_tools_to_cookbook([{"name": "x"}])
        self.assertEqual(out[0]["parameters"], {"type": "object", "properties": {}})


class TestOpenAIMessagesToCookbook(unittest.TestCase):
    def test_system_user_passthrough(self):
        msgs = [
            {"role": "system", "content": "you are a bio agent"},
            {"role": "user", "content": "do the task"},
        ]
        out = openai_messages_to_cookbook(msgs)
        self.assertEqual(out, msgs)

    def test_assistant_null_content_with_tool_calls(self):
        msgs = [{
            "role": "assistant",
            "content": None,
            "tool_calls": [{
                "id": "call_9",
                "type": "function",
                "function": {"name": "bash", "arguments": "{\"cmd\": \"ls\"}"},
            }],
        }]
        out = openai_messages_to_cookbook(msgs)
        self.assertEqual(out[0]["content"], "")
        tc = out[0]["tool_calls"][0]
        # cookbook installed -> pydantic ToolCall (attribute access);
        # cookbook missing -> legacy dict. Support both in assertions.
        if isinstance(tc, dict):
            self.assertEqual(tc["id"], "call_9")
            self.assertEqual(tc["function"]["name"], "bash")
        else:
            self.assertEqual(tc.id, "call_9")
            self.assertEqual(tc.function.name, "bash")

    def test_tool_result_keeps_name_and_id(self):
        msgs = [{"role": "tool", "tool_call_id": "call_9", "name": "bash", "content": "out"}]
        out = openai_messages_to_cookbook(msgs)
        self.assertEqual(out[0], {"role": "tool", "content": "out", "name": "bash",
                                  "tool_call_id": "call_9"})

    def test_tool_result_without_name(self):
        msgs = [{"role": "tool", "tool_call_id": "c1", "content": "x"}]
        out = openai_messages_to_cookbook(msgs)
        self.assertNotIn("name", out[0])
        self.assertEqual(out[0]["tool_call_id"], "c1")

    def test_tool_result_name_backfilled_from_assistant_history(self):
        """Regression (smoke_005): pi sends tool results without "name"; the
        stateless gpt_oss renderer requires it. The bridge must backfill from
        the matching assistant tool_call id."""
        msgs = [
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_0", "type": "function",
                 "function": {"name": "read", "arguments": "{\"path\": \"/app\"}"}},
            ]},
            {"role": "tool", "tool_call_id": "call_0", "content": "file contents"},
        ]
        out = openai_messages_to_cookbook(msgs)
        self.assertEqual(out[1].get("name"), "read")
        self.assertEqual(out[1]["tool_call_id"], "call_0")

    def test_assistant_toolcalls_are_pydantic_when_cookbook_available(self):
        """Regression (smoke_005): the gpt_oss renderer's _render_tool_calls
        accesses tc.function.name on assistant history, so bridge-produced
        tool_calls must be pydantic objects, not dicts."""
        try:
            from tinker_cookbook.renderers.base import ToolCall
        except ImportError:
            self.skipTest("tinker_cookbook not installed")
        msgs = [{"role": "assistant", "content": "", "tool_calls": [
            {"id": "c0", "function": {"name": "bash", "arguments": "{}"}},
        ]}]
        out = openai_messages_to_cookbook(msgs)
        self.assertIsInstance(out[0]["tool_calls"][0], ToolCall)


class TestAssistantMessageToOpenAI(unittest.TestCase):
    def test_parts_text_and_thinking(self):
        msg = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "need to inspect data"},
                {"type": "text", "text": "Running QC"},
            ],
        }
        out = _assistant_message_to_openai(msg)
        self.assertEqual(out["reasoning_content"], "need to inspect data")
        self.assertEqual(out["content"], "Running QC")
        self.assertNotIn("tool_calls", out)

    def test_plain_string_content(self):
        out = _assistant_message_to_openai({"role": "assistant", "content": "hello"})
        self.assertEqual(out["content"], "hello")
        self.assertNotIn("reasoning_content", out)

    def test_tool_calls_indexed_and_default_ids(self):
        msg = {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"function": {"name": "bash", "arguments": "{}"}},
                {"id": "call_x", "function": {"name": "read", "arguments": "{}"}},
            ],
        }
        out = _assistant_message_to_openai(msg)
        self.assertEqual(out["tool_calls"][0]["index"], 0)
        self.assertEqual(out["tool_calls"][0]["id"], "call_0")
        self.assertEqual(out["tool_calls"][1]["id"], "call_x")

    def test_empty_content_becomes_null(self):
        out = _assistant_message_to_openai({"role": "assistant", "content": []})
        self.assertIsNone(out["content"])

    def test_pydantic_toolcall_objects_real_renderer_shape(self):
        """Regression: real gpt_oss parse_response returns pydantic ToolCall
        objects (attribute access), not dicts. This shape caused 500s in
        smoke_004 because the bridge called .get() on them."""
        try:
            from tinker_cookbook.renderers.base import ToolCall
        except ImportError:
            self.skipTest("tinker_cookbook not installed")
        msg = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "analyze"},
                {"type": "text", "text": "running"},
            ],
            "tool_calls": [
                ToolCall(
                    id="call_py1",
                    function=ToolCall.FunctionBody(
                        name="bash", arguments='{"command": "ls /app"}'),
                ),
            ],
        }
        out = _assistant_message_to_openai(msg)
        self.assertEqual(out["tool_calls"][0]["id"], "call_py1")
        self.assertEqual(out["tool_calls"][0]["function"]["name"], "bash")
        self.assertEqual(out["tool_calls"][0]["function"]["arguments"],
                         '{"command": "ls /app"}')
        self.assertEqual(out["content"], "running")

    def test_unparsed_tool_calls_surface_raw_text(self):
        msg = {
            "role": "assistant",
            "content": [{"type": "text", "text": ""}],
            "unparsed_tool_calls": [{"raw_text": "<tool_call>garbage</tool_call>",
                                     "error": "Invalid JSON"}],
        }
        out = _assistant_message_to_openai(msg)
        self.assertIn("<tool_call>garbage</tool_call>", out["content"])


# ---------------------------------------------------------------------------
# RecordStore
# ---------------------------------------------------------------------------

class TestRecordStore(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            store = RecordStore(Path(td))
            store.append("key-1", {"prompt_ids": [1], "completion_ids": [2]})
            store.append("key-1", {"prompt_ids": [3], "completion_ids": [4]})
            rows = store.read("key-1")
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["prompt_ids"], [1])
            self.assertEqual(rows[1]["completion_ids"], [4])

    def test_key_sanitization_is_consistent(self):
        with tempfile.TemporaryDirectory() as td:
            store = RecordStore(Path(td))
            store.append("roll/abc 123", {"x": 1})
            # same sanitized key -> same file, readable both ways
            self.assertEqual(len(store.read("roll/abc 123")), 1)
            self.assertEqual(len(store.read("rollabc123")), 1)

    def test_read_missing_returns_empty(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(RecordStore(Path(td)).read("nope"), [])

    def test_jsonl_one_line_per_record(self):
        with tempfile.TemporaryDirectory() as td:
            store = RecordStore(Path(td))
            store.append("k", {"a": 1})
            store.append("k", {"a": 2})
            text = (Path(td) / "k.jsonl").read_text(encoding="utf-8")
            self.assertEqual(text.count("\n"), 2)  # LF endings, one per record


# ---------------------------------------------------------------------------
# SSE endpoint with mock sampling client + mock renderer
# ---------------------------------------------------------------------------

class _Seq:
    def __init__(self, tokens, logprobs):
        self.tokens = tokens
        self.logprobs = logprobs


class _SampleResponse:
    def __init__(self, tokens, logprobs):
        self.sequences = [_Seq(tokens, logprobs)]


class MockSamplingClient:
    def __init__(self, tokens=(7, 8, 9), logprobs=(-0.1, -0.2, -0.3)):
        self.tokens, self.logprobs = list(tokens), (
            list(logprobs) if logprobs is not None else None)
        self.calls: list[dict] = []

    async def sample_async(self, prompt, num_samples, sampling_params):
        self.calls.append({"prompt": prompt, "num_samples": num_samples,
                           "params": sampling_params})
        return _SampleResponse(self.tokens, self.logprobs)


class MockRenderer:
    """Same methods the bridge uses from the real gpt_oss renderer."""

    def __init__(self, parsed_message):
        self.parsed_message = parsed_message
        self.last_messages = None
        self.last_tools = None
        self.stop_sequences = ["<|return|>"]

    def create_conversation_prefix_with_tools(self, tools, system_prompt=""):
        self.last_tools = tools
        return [{"role": "system", "content": "tool-routing-prefix"}]

    def build_generation_prompt(self, messages):
        self.last_messages = messages
        return tinker_sdk.ModelInput.from_ints([1, 2, 3])

    def get_stop_sequences(self):
        return list(self.stop_sequences)

    def parse_response(self, tokens):
        return self.parsed_message, "stop"


TOOLCALL_MSG = {
    "role": "assistant",
    "content": [
        {"type": "thinking", "thinking": "check the data first"},
        {"type": "text", "text": "Let me check."},
    ],
    "tool_calls": [{"id": "call_1", "type": "function",
                    "function": {"name": "bash", "arguments": "{\"cmd\": \"ls\"}"}}],
}

PLAIN_MSG = {"role": "assistant", "content": [{"type": "text", "text": "All done."}]}


def _sse_events(text: str) -> list[str]:
    return [ln[len("data: "):] for ln in text.split("\n\n") if ln.startswith("data: ")]


class TestSSEEndpoint(unittest.TestCase):
    def _make(self, parsed_msg, **kw):
        sampling = MockSamplingClient(**kw)
        renderer = MockRenderer(parsed_msg)
        td = tempfile.TemporaryDirectory()
        self.addCleanup(td.cleanup)
        records = RecordStore(Path(td.name))
        app = create_app(sampling, renderer, model_id="openai/gpt-oss-20b",
                         records=records)
        return sampling, renderer, app, records

    def _post(self, app, body, headers=None):
        import httpx

        async def go():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
                return await c.post("/v1/chat/completions", json=body, headers=headers or {})

        return asyncio.run(go())

    BODY = {
        "model": "openai/gpt-oss-20b",
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c9", "function": {"name": "bash", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "c9", "name": "bash", "content": "out"},
        ],
        "tools": [{"type": "function", "function": {"name": "bash", "description": "d"}}],
        "max_tokens": 64,
    }

    def test_full_toolcall_stream_and_record(self):
        sampling, renderer, app, records = self._make(TOOLCALL_MSG)
        r = self._post(app, self.BODY,
                       headers={"Authorization": "Bearer rollout-abc/1"})
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/event-stream", r.headers["content-type"])

        events = _sse_events(r.text)
        self.assertEqual(events[-1], "[DONE]")
        chunks = [json.loads(e) for e in events[:-1]]
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0]["choices"][0]["delta"], {"role": "assistant"})
        delta = chunks[1]["choices"][0]["delta"]
        self.assertEqual(delta["content"], "Let me check.")
        self.assertEqual(delta["reasoning_content"], "check the data first")
        self.assertEqual(delta["tool_calls"][0]["function"]["name"], "bash")
        self.assertEqual(chunks[2]["choices"][0]["delta"], {})
        self.assertEqual(chunks[2]["choices"][0]["finish_reason"], "tool_calls")
        self.assertTrue(all(c["object"] == "chat.completion.chunk" for c in chunks))

        # renderer got the converted cookbook messages + tools
        self.assertEqual(renderer.last_messages[-1]["name"], "bash")
        # TOOLS MUST reach the prompt via the Harmony tools prefix
        # (regression guard: they were silently dropped once)
        self.assertEqual(renderer.last_messages[0]["content"], "tool-routing-prefix")
        self.assertEqual([t["name"] for t in renderer.last_tools], ["bash"])
        # sampling params came from the request + renderer stops
        call = sampling.calls[0]
        self.assertEqual(call["num_samples"], 1)
        self.assertEqual(call["params"].max_tokens, 64)
        self.assertEqual(call["params"].stop, ["<|return|>"])
        self.assertEqual(call["prompt"].to_ints(), [1, 2, 3])

        # one record keyed by the Bearer token, with GRPO-grade fidelity
        rows = records.read("rollout-abc/1")
        self.assertEqual(len(rows), 1)
        rec = rows[0]
        self.assertEqual(rec["prompt_ids"], [1, 2, 3])
        self.assertEqual(rec["completion_ids"], [7, 8, 9])
        self.assertEqual(rec["logprobs"], [-0.1, -0.2, -0.3])
        self.assertEqual(rec["finish_reason"], "tool_calls")
        self.assertEqual(rec["parsed"]["tool_calls"][0]["function"]["name"], "bash")

    def test_plain_stop_stream(self):
        sampling, _, app, records = self._make(PLAIN_MSG, tokens=(5,), logprobs=None)
        r = self._post(app, {"messages": [{"role": "user", "content": "hi"}]})
        events = _sse_events(r.text)
        chunks = [json.loads(e) for e in events[:-1]]
        delta = chunks[1]["choices"][0]["delta"]
        self.assertEqual(delta["content"], "All done.")
        self.assertNotIn("reasoning_content", delta)
        self.assertNotIn("tool_calls", delta)
        self.assertEqual(chunks[2]["choices"][0]["finish_reason"], "stop")
        # logprobs=None propagates into the record verbatim
        rec = records.read("anonymous")[0]
        self.assertIsNone(rec["logprobs"])
        self.assertEqual(rec["completion_ids"], [5])
        self.assertEqual(sampling.calls[0]["params"].max_tokens, 8192)  # bridge default

    def test_missing_auth_falls_back_to_anonymous(self):
        _, _, app, records = self._make(PLAIN_MSG)
        r = self._post(app, {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(records.read("anonymous")), 1)


# ---------------------------------------------------------------------------
# orchestrator: reward parsing, records -> Trajectory, GRPO assembly
# ---------------------------------------------------------------------------

def _fake_bridge_records(seed: int) -> list[dict]:
    return [
        {"prompt_ids": list(range(10, 20 + seed)),
         "completion_ids": list(range(100, 108 + seed)),
         "logprobs": [-0.1 * i for i in range(8 + seed)],
         "finish_reason": "tool_calls"},
        {"prompt_ids": list(range(30, 40 + seed)),
         "completion_ids": list(range(200, 204 + seed)),
         "logprobs": None,
         "finish_reason": "stop"},
    ]


@unittest.skipUnless(HAS_COOKBOOK, "tinker_cookbook not installed")
class TestTrajectoryAssembly(unittest.TestCase):
    def test_records_to_trajectory_shape(self):
        from pi_grpo_train import records_to_trajectory

        recs = _fake_bridge_records(0)
        recs[1]["logprobs"] = [-0.05 * i for i in range(4)]  # must match its 4 tokens
        traj = records_to_trajectory(recs, reward=0.75)
        self.assertIsNotNone(traj)
        self.assertEqual(len(traj.transitions), 2)
        self.assertEqual([t.reward for t in traj.transitions], [0.0, 0.75])
        self.assertEqual([t.episode_done for t in traj.transitions], [False, True])
        self.assertEqual(traj.transitions[0].ob.to_ints(), recs[0]["prompt_ids"])
        self.assertEqual(traj.transitions[0].ac.tokens, recs[0]["completion_ids"])
        self.assertEqual(traj.transitions[0].ac.maybe_logprobs, recs[0]["logprobs"])
        self.assertEqual(traj.transitions[0].ac.stop_reason, "stop")  # tool_calls -> stop
        self.assertEqual(traj.stop_reason, "max_turns")
        self.assertEqual(list(traj.final_ob.to_ints()), [])

    def test_records_without_logprobs_are_dropped(self):
        """assemble_training_data hard-requires per-token logprobs; a sampling
        call without them must be dropped, not crash the whole training step."""
        from pi_grpo_train import records_to_trajectory

        recs = _fake_bridge_records(0)          # rec[1] has logprobs=None
        traj = records_to_trajectory(recs, 1.0)
        self.assertIsNotNone(traj)
        self.assertEqual(len(traj.transitions), 1)
        self.assertEqual(traj.transitions[0].ac.maybe_logprobs, recs[0]["logprobs"])
        self.assertTrue(traj.transitions[0].episode_done)  # survivor is the last

        self.assertIsNone(records_to_trajectory(
            [dict(recs[0], logprobs=None)], 1.0))

    def test_length_finish_reason_maps(self):
        from pi_grpo_train import records_to_trajectory

        recs = [dict(_fake_bridge_records(0)[0], finish_reason="length")]
        traj = records_to_trajectory(recs, 1.0)
        self.assertEqual(traj.transitions[0].ac.stop_reason, "length")

    def test_empty_records_give_none(self):
        from pi_grpo_train import records_to_trajectory

        self.assertIsNone(records_to_trajectory([], 1.0))


class TestParseReward(unittest.TestCase):
    def test_valid(self):
        from pi_grpo_train import _parse_reward

        with tempfile.TemporaryDirectory() as td:
            rj = Path(td) / "result.json"
            rj.write_text(json.dumps(
                {"verifier_result": {"rewards": {"reward": 0.75}}}), encoding="utf-8")
            self.assertEqual(_parse_reward(Path(td)), 0.75)

    def test_missing_and_malformed(self):
        from pi_grpo_train import _parse_reward

        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(_parse_reward(Path(td)), 0.0)
            (Path(td) / "result.json").write_text("{not json", encoding="utf-8")
            self.assertEqual(_parse_reward(Path(td)), 0.0)

    def test_null_reward(self):
        from pi_grpo_train import _parse_reward

        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "result.json").write_text(
                json.dumps({"verifier_result": {"rewards": {"reward": None}}}),
                encoding="utf-8")
            self.assertEqual(_parse_reward(Path(td)), 0.0)

    def test_null_verifier_result(self):
        """Regression (full-12tasks step0): some results carry an explicit
        null verifier_result; .get(k, {}) returns None for a null value and
        previously crashed with AttributeError instead of scoring 0."""
        from pi_grpo_train import _parse_reward

        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "result.json").write_text(
                json.dumps({"verifier_result": None}), encoding="utf-8")
            self.assertEqual(_parse_reward(Path(td)), 0.0)
            (Path(td) / "result.json").write_text("null", encoding="utf-8")
            self.assertEqual(_parse_reward(Path(td)), 0.0)

    def test_job_vs_trial_result_json(self):
        """Regression (smoke_007): harbor writes result.json at BOTH the job
        level (no verifier rewards) and the trial level (with rewards); the
        finder must pick the trial one regardless of sort order."""
        from pi_grpo_train import _find_result_json, _parse_reward

        with tempfile.TemporaryDirectory() as td:
            job = Path(td) / "2026-10-10__21-55-51"
            trial = job / "accessible-bioinformatics-report__YzAFYs9"
            trial.mkdir(parents=True)
            # job-level: no rewards (sorted() finds this first)
            (job / "result.json").write_text(
                json.dumps({"verifier_result": {"rewards": None}}), encoding="utf-8")
            # trial-level: real reward
            (trial / "result.json").write_text(
                json.dumps({"verifier_result": {"rewards": {"score": 1.0}}}),
                encoding="utf-8")
            picked = _find_result_json(Path(td))
            self.assertIsNotNone(picked)
            self.assertEqual(picked.parent, trial)
            self.assertEqual(_parse_reward(Path(td)), 1.0)


@unittest.skipUnless(HAS_COOKBOOK, "tinker_cookbook not installed")
class TestGRPOAssemblyChain(unittest.TestCase):
    """records -> Trajectory -> TrajectoryGroup -> advantages -> datums."""

    def _group(self, rewards):
        from tinker_cookbook.rl.types import TrajectoryGroup
        from pi_grpo_train import records_to_trajectory

        trajs = [records_to_trajectory(_fake_bridge_records(i), r)
                 for i, r in enumerate(rewards)]
        return TrajectoryGroup(
            trajectories_G=trajs,
            final_rewards_G=[t.transitions[-1].reward for t in trajs],
            metrics_G=[{} for _ in trajs],
        )

    def test_chain_shapes(self):
        from tinker_cookbook.rl.data_processing import (
            assemble_training_data,
            compute_advantages,
            remove_constant_reward_groups,
        )

        groups = [self._group([0.2, 0.8]),   # varying -> kept
                  self._group([0.5, 0.5])]   # constant -> removed
        groups = remove_constant_reward_groups(groups)
        self.assertEqual(len(groups), 1)

        advantages = compute_advantages(groups)  # list of per-group (G,) tensors
        self.assertEqual(len(advantages), 1)
        vals = [float(v) for v in advantages[0].view(-1)]
        self.assertEqual(len(vals), 2)
        # group-centered: the two members must be symmetric around 0
        self.assertAlmostEqual(sum(vals), 0.0, places=5)

        data_D, meta_D = assemble_training_data(groups, advantages)
        self.assertGreater(len(data_D), 0)
        self.assertEqual(len(meta_D), len(data_D))
        for d in data_D:
            self.assertIn("advantages", d.loss_fn_inputs)
            self.assertIn("target_tokens", d.loss_fn_inputs)
        # full token coverage: every sampled action token in the surviving
        # group must appear in the training data
        total_tokens = sum(
            int(d.loss_fn_inputs["target_tokens"].shape[0]) for d in data_D)
        total_actions = sum(
            len(traj.transitions[i].ac.tokens)
            for traj in groups[0].trajectories_G
            for i in range(len(traj.transitions)))
        self.assertGreaterEqual(total_tokens, total_actions)

    def test_dry_run_fake_records_are_wellformed(self):
        from pi_grpo_train import _fake_records, records_to_trajectory

        recs = _fake_records({"key": "deadbeef"})
        self.assertEqual(len(recs), 2)
        for rec in recs:
            self.assertEqual(len(rec["completion_ids"]), len(rec["logprobs"]))
        traj = records_to_trajectory(recs, 0.5)
        self.assertEqual(len(traj.transitions), 2)


class TestModelsJsonTemplate(unittest.TestCase):
    def test_renders_valid_json(self):
        from rl_pi_agent import MODELS_JSON

        rendered = MODELS_JSON.format(bridge_port=8092, max_tokens=8192)
        cfg = json.loads(rendered)  # would raise on brace-escaping regressions
        prov = cfg["providers"]["openai"]
        # apiKey is an ENV VAR NAME: pi 0.73.1 resolveConfigValue = env lookup
        # first, so the Bearer token = PI_BRIDGE_KEY = rollout uuid
        self.assertEqual(prov["apiKey"], "PI_BRIDGE_KEY")
        model = prov["models"][0]
        self.assertEqual(model["id"], "gpt-oss-rl")
        self.assertEqual(
            model["baseUrl"], "http://host.docker.internal:8092/v1")
        self.assertEqual(model["api"], "openai-completions")
        self.assertEqual(model["compat"], {"requiresToolResultName": True})

    def test_harbor_rollout_cmd_uses_import_path(self):
        """The rollout command must target our BridgePi subclass, not the
        stock pi agent (whose provider whitelist rejects unknown providers)."""
        import inspect

        from pi_grpo_train import run_one_rollout

        src = inspect.getsource(run_one_rollout)
        self.assertIn("rl_pi_agent:BridgePi", src)
        self.assertIn("--agent-import-path", src)
        self.assertIn("--agent-env", src)
        self.assertIn("PI_BRIDGE_KEY", src)


@unittest.skipUnless(HAS_COOKBOOK, "tinker_cookbook not installed")
class TestDryRunTrainStep(unittest.TestCase):
    """train_step(dry_run=True) with the bridge subprocess patched out."""

    def _trainer(self, task_dir: Path, records_dir: Path, log_dir: Path):
        import argparse

        from pi_grpo_train import PiGRPOTrainer

        args = argparse.Namespace(
            task_dir=task_dir, model="openai/gpt-oss-20b", steps=2, group_size=4,
            max_turns=30, lr=1e-5, lora_rank=32, max_tokens=8192,
            tasks_per_batch=2, rollout_timeout=1800, bridge_port=8092,
            harbor_bin="harbor-does-not-exist", records_dir=str(records_dir),
            log_path=str(log_dir), run_name="pi-grpo-test", save_every=5,
            include=None, dry_run=True,
        )
        trainer = PiGRPOTrainer(args)

        async def fake_restart() -> int:
            return 8092

        trainer._restart_bridge = fake_restart  # type: ignore[method-assign]
        return trainer

    def test_two_steps_end_to_end(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            task_dir = root / "tasks"
            for name in ("task-a", "task-b"):
                (task_dir / name).mkdir(parents=True)
            trainer = self._trainer(task_dir, root / "records", root / "logs")

            async def go():
                return [await trainer.train_step(step, trainer and []) or None
                        for step in ()]  # placeholder, replaced below

            # run the trainer loop (2 steps) with the real run()
            asyncio.run(trainer.run())

            hist = json.loads((root / "logs" / "history.json").read_text(encoding="utf-8"))
            self.assertEqual(len(hist["steps"]), 2)
            for i, s in enumerate(hist["steps"]):
                self.assertEqual(s["step"], i)
                self.assertIsNotNone(s["mean_reward"])
                self.assertEqual(s["n_groups"], 2)  # one group per task
            self.assertEqual(hist["checkpoints"], [])  # dry run saves nothing
            del go

    def test_reward_written_and_parsed_back(self):
        from pi_grpo_train import RolloutSpec, _parse_reward, run_one_rollout

        async def go():
            spec = RolloutSpec(task_dir=Path("."), task_name="t", group_idx=0,
                               rollout_key="k"*8)
            out = await run_one_rollout(spec, bridge_port=1, model_id="m",
                                        max_turns=1, timeout_sec=1,
                                        harbor_bin="x", dry_run=True)
            self.assertEqual(out["reward"], _parse_reward(Path(out["result_dir"])))
            self.assertGreater(out["reward"], 0.0)

        asyncio.run(go())


if __name__ == "__main__":
    unittest.main()
