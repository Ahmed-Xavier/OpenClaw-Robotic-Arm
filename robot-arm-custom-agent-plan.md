# Custom Fine-Tuned Robot Arm Agent — Project Plan

## Goal

Replace OpenClaw with a small, purpose-built program for controlling the SO-100 robot arm,
powered by a model fine-tuned specifically on the arm's 8 API endpoints. No general-purpose
agent framework, no sandbox/approval fights, no wasted overhead — just:

```
Telegram message → simple script → fine-tuned model → Flask arm server → reply
```

---

## Why

OpenClaw is built to handle many tools, many chat platforms, and many safety scenarios.
This project only needs 8 fixed commands and one chat channel (Telegram). Most of the
friction hit so far (exec approvals, sandboxing, tool search overhead, slow workspace
loading) comes from features this project doesn't use.

A fine-tuned small model doesn't need a big "tool search" system to figure out what's
available — it already knows the 8 commands cold. That removes the need for most of
OpenClaw's machinery.

---

## Phase 1 — Confirm the base model fits (in progress)

- [x] Identify target model: `Qwen/Qwen3-4B-Instruct-2507` (pure text, no vision overhead)
- [x] Download full-precision weights (8.06GB) for training — done, saved at
      `C:\Users\pc\Qwen3-4B-Instruct`
- [ ] Load model in 4-bit (QLoRA) and confirm actual VRAM usage on the RTX 4050 (6GB)
- [ ] Confirm training fits without CPU/GPU splitting (small batch size, gradient
      checkpointing, short sequences)

## Phase 2 — Build the training dataset

- [ ] List every real command the arm supports (from `robot-arm` skill):
      `pick`, `place`, `move_to`, `gripper`, `state`, `camera`, `reset_home`, `scenario`
- [ ] Write example (user instruction → correct tool call) pairs covering:
  - Direct commands ("pick up the cube")
  - Positional commands ("put it on the left/right")
  - Ambiguous/status commands ("what's the arm doing", "where's the cube")
  - Multi-step commands ("pick it up and place it on the right")
  - Commands with **no valid tool call** (so the model learns to say so instead of
    guessing or inventing an endpoint)
  - Edge cases: missing info, bad requests, cube not found
- [ ] Optional shortcut: use the existing bigger model (`qwen3.5:4b` or similar) as a
      "teacher" — log its correct tool calls across varied prompts, use those logs to
      bootstrap the training set instead of writing every example by hand
- [ ] Decide output format: raw JSON tool calls (recommended) vs. shell/curl syntax
      directly. JSON + a fixed wrapper that turns JSON into the actual API call is more
      robust than training the model to produce shell syntax.

## Phase 3 — Fine-tune

- [ ] Method: **QLoRA** (4-bit base + LoRA adapters) — fits the 4B model in 6GB VRAM
- [ ] Tooling: Hugging Face `transformers` + `peft` + `bitsandbytes`
- [ ] Small batch size (1–2), gradient checkpointing, paged optimizer
- [ ] Train, then evaluate: does it call the right tool, with the right arguments,
      every time? Does it avoid inventing tools/endpoints that don't exist?
- [ ] Iterate on the dataset if specific failure patterns show up (e.g. bad coordinate
      extraction, wrong tool on ambiguous phrasing)

## Phase 4 — Package the fine-tuned model for local use

- [ ] Merge LoRA adapters into the base model (or keep them separate, loaded at runtime)
- [ ] Convert to GGUF and quantize (e.g. Q4_K_M) for fast local inference via Ollama —
      same as the current `qwen3.5:4b` setup, just trained on your exact tools

## Phase 5 — Build the lightweight custom control program

Replaces OpenClaw entirely for this project.

- [ ] Small Python script using `python-telegram-bot` (or similar) to receive messages
- [ ] Script sends the message to the fine-tuned model via Ollama's local API
- [ ] Model replies with a structured tool call (e.g. `{"tool": "place", "args": {...}}`)
- [ ] Script parses the JSON and calls the Flask arm server (`server.py`) directly —
      no sandbox, no exec-approval system, since the script is trusted code you wrote
- [ ] Script sends a short confirmation back to Telegram
- [ ] (Optional) Add simple safety checks in the script itself — e.g. reject obviously
      malformed coordinates before sending them to the arm

## What this setup gains

- No sandbox/approval fights — the script calls the Flask server directly
- Much faster — no 25+ second workspace loading, no scanning 50 tools per message
- Full transparency — every part of the pipeline is code you wrote and understand
- More reliable tool calls — behavior is baked into the model's weights, not dependent
  on a skill prompt being read correctly every time

## What this setup gives up

- OpenClaw's plugin ecosystem (web search, other integrations)
- Multi-channel support beyond Telegram
- Heartbeats / subagents / session-memory system
- General-purpose flexibility for tasks outside the robot arm

None of these are currently used for this project, so the trade-off favors the custom build.

---

## Open questions to settle before Phase 5

- Exact JSON schema for tool calls (tool name + argument names/types)
- How to handle multi-step commands (chain calls in the script, or let the model emit
  a list of calls in one response?)
- What happens on a malformed/uncertain model response — retry, ask for clarification
  via Telegram, or fail safely?
