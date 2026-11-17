import json, collections
p = "Mini Openclaw/turns.jsonl"
rows = []
with open(p, encoding="utf-8") as f:
    for i, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append((i, json.loads(line)))
        except Exception as e:
            print("BAD LINE", i, e)
print("total turns:", len(rows))
c_incomplete = [r for r in rows if "I could not complete the request" in (r[1].get("final_reply") or "")]
c_limit = [r for r in rows if "Agent round limit reached" in (r[1].get("final_reply") or "")]
print("contains 'I could not complete the request':", len(c_incomplete))
print("contains 'Agent round limit reached':", len(c_limit))
print()
print("=== Agent-round-limit rows: agent_rounds ===")
for ln, r in c_limit:
    print("line %d  agent_rounds=%s  failure_type=%s  tools=%s  chat=%s  msg=%r" % (
        ln, r.get("agent_rounds"), r.get("failure_type"),
        [t.get("name") for t in (r.get("tool_calls") or [])], r.get("chat_id"), (r.get("user_message") or "")[:60]))
print()
print("=== 'could not complete' rows: agent_rounds ===")
for ln, r in c_incomplete:
    print("line %d  agent_rounds=%s  failure_type=%s  chat=%s  msg=%r" % (
        ln, r.get("agent_rounds"), r.get("failure_type"), r.get("chat_id"), (r.get("user_message") or "")[:60]))
print()
print("=== agent_rounds distribution (all rows) ===")
dist = collections.Counter(r[1].get("agent_rounds") for r in rows)
for k in sorted(dist, key=lambda x: (x is None, x)):
    print("  agent_rounds=%s -> %d turns" % (k, dist[k]))
print()
print("=== MAX_AGENT_ROUNDS from config ===")
import re
cfg = open("Mini Openclaw/config.py", encoding="utf-8").read()
m = re.search(r'MAX_AGENT_ROUNDS = int\(os\.environ\.get\("MAX_AGENT_ROUNDS", "(\d+)"\)\)', cfg)
print("  MAX_AGENT_ROUNDS =", m.group(1) if m else "NOT FOUND")
m2 = re.search(r'MAX_TOOL_CALLS_PER_TURN = int\(os\.environ\.get\("MAX_TOOL_CALLS_PER_TURN", "(\d+)"\)\)', cfg)
print("  MAX_TOOL_CALLS_PER_TURN =", m2.group(1) if m2 else "NOT FOUND")
m3 = re.search(r'MAX_HISTORY_TURNS = int\(os\.environ\.get\("MAX_HISTORY_TURNS", "(\d+)"\)\)', cfg)
print("  MAX_HISTORY_TURNS =", m3.group(1) if m3 else "NOT FOUND")
