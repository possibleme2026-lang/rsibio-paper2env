import json
import sys

d = json.load(open("/tmp/req.json", encoding="utf-8"))
print("top keys:", ",".join(sorted(d.keys())))
print("model:", d.get("model"))
print("stream:", d.get("stream"), "stream_options:", json.dumps(d.get("stream_options")))
print("temperature:", d.get("temperature"), "max_tokens:", d.get("max_tokens"))
tools = d.get("tools") or []
print("n_tools:", len(tools))
for t in tools[:25]:
    fn = t.get("function") or {}
    print("  tool:", fn.get("name"))
print("n_messages:", len(d.get("messages") or []))
for m in (d.get("messages") or [])[:4]:
    c = m.get("content")
    print("  msg:", m.get("role"), "|", (str(c)[:60] if c is not None else None))
