
#!/usr/bin/env python3
"""
XavierClaw - Nemotron automatic live benchmark.

Place this file anywhere inside the repository.
It automatically finds launcher.py in a parent directory.

Run:
    python quick_probe.py

Requirements:
    pip install requests

Each configuration gets a fresh launcher/server/MuJoCo instance.
The MuJoCo viewer remains visible during every test.
"""

from __future__ import annotations

import itertools
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


# ============================================================
# SETTINGS
# ============================================================

MODEL = "nemotron-3-nano:4b"

OLLAMA_URL = "http://127.0.0.1:11434"
ROBOT_URL = "http://127.0.0.1:8765"

# Automatic benchmark matrix.
THINKING_VALUES = [False, True]
CONTEXT_VALUES = [4096, 8192]
ROUND_VALUES = [4, 8]

MAX_PICK_ATTEMPTS = 2

OLLAMA_TIMEOUT = 180
ROBOT_TIMEOUT = 120
LAUNCH_TIMEOUT = 180

# A fresh MuJoCo instance is launched for every configuration.
PAUSE_BETWEEN_TESTS = 2


# ============================================================
# FIND REPOSITORY ROOT
# ============================================================

def find_repository_root() -> Path:
    """
    Walk upward from this script until launcher.py and the
    repository's actual tools_schema.json are found.
    """

    for directory in Path(__file__).resolve().parents:
        launcher = directory / "launcher.py"
        schema = directory / "Mini Openclaw" / "tools_schema.json"

        if launcher.is_file() and schema.is_file():
            return directory

    raise FileNotFoundError(
        "Could not locate repository root.\n"
        "Expected launcher.py and Mini Openclaw/tools_schema.json "
        "in the same repository."
    )


ROOT = find_repository_root()

LAUNCHER = ROOT / "launcher.py"
TOOLS_FILE = ROOT / "Mini Openclaw" / "tools_schema.json"

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")

LOG_DIR = ROOT / "benchmark_logs" / f"nemotron_{RUN_ID}"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = LOG_DIR / "summary.json"


# ============================================================
# GLOBAL STATE
# ============================================================

http = requests.Session()

ollama_process: subprocess.Popen | None = None
launcher_process: subprocess.Popen | None = None

launcher_log_handle = None
ollama_log_handle = None

started_ollama = False
started_launcher = False

current_events: list[dict[str, Any]] = []

current_pick_attempts = 0
current_round = 0
last_pick_round = -1

results: list[dict[str, Any]] = []


# ============================================================
# LOGGING
# ============================================================

def log(event: str, **data: Any) -> None:
    record = {
        "timestamp": datetime.now().isoformat(timespec="milliseconds"),
        "event": event,
        **data,
    }

    current_events.append(record)

    print(f"\n[{record['timestamp']}] {event}")

    if data:
        print(
            json.dumps(
                data,
                indent=2,
                ensure_ascii=False,
                default=str,
            )
        )


def save_test_log(
    test_number: int,
    config: dict[str, Any],
    extra: dict[str, Any],
) -> Path:

    path = LOG_DIR / f"test_{test_number:02d}.json"

    data = {
        "test_number": test_number,
        "model": MODEL,
        "configuration": config,
        "events": current_events,
        **extra,
    }

    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    return path


def save_summary() -> None:
    SUMMARY_FILE.write_text(
        json.dumps(
            {
                "model": MODEL,
                "run_id": RUN_ID,
                "repository": str(ROOT),
                "number_of_tests": len(results),
                "results": results,
            },
            indent=2,
            ensure_ascii=False,
            default=str,
        ),
        encoding="utf-8",
    )


# ============================================================
# CONNECTION CHECKS
# ============================================================

def ollama_reachable() -> bool:
    try:
        response = http.get(
            f"{OLLAMA_URL}/api/tags",
            timeout=3,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def robot_reachable() -> bool:
    try:
        response = http.get(
            f"{ROBOT_URL}/state",
            timeout=3,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


def get_models() -> list[str]:
    response = http.get(
        f"{OLLAMA_URL}/api/tags",
        timeout=10,
    )
    response.raise_for_status()

    return [
        item.get("name", "")
        for item in response.json().get("models", [])
    ]


def wait_for(
    condition,
    description: str,
    timeout: int,
) -> bool:

    deadline = time.time() + timeout

    print(f"\nWaiting for {description}...")

    while time.time() < deadline:
        if condition():
            print(f"{description}: READY")
            return True

        time.sleep(1)

    return False


# ============================================================
# OLLAMA STARTUP
# ============================================================

def start_ollama() -> None:
    global ollama_process
    global ollama_log_handle
    global started_ollama

    if ollama_reachable():
        print("Ollama is already running.")
        return

    executable = shutil.which("ollama")

    if not executable:
        raise RuntimeError(
            "Ollama is not running and 'ollama' was not found on PATH."
        )

    print("Starting Ollama...")

    ollama_log_handle = open(
        LOG_DIR / "ollama.log",
        "a",
        encoding="utf-8",
    )

    ollama_process = subprocess.Popen(
        [executable, "serve"],
        cwd=str(ROOT),
        stdin=subprocess.DEVNULL,
        stdout=ollama_log_handle,
        stderr=subprocess.STDOUT,
    )

    started_ollama = True

    if not wait_for(ollama_reachable, "Ollama", 40):
        raise RuntimeError("Ollama startup timed out.")


def check_model() -> dict[str, Any]:
    models = get_models()

    if MODEL not in models:
        raise RuntimeError(
            f"Model {MODEL} is not installed.\n"
            f"Installed models: {models}\n"
            f"Run: ollama pull {MODEL}"
        )

    response = http.post(
        f"{OLLAMA_URL}/api/show",
        json={"model": MODEL},
        timeout=30,
    )
    response.raise_for_status()

    info = response.json()
    thinking = info.get("thinking", {})

    print("\nModel thinking support:")
    print(json.dumps(thinking, indent=2))

    values = thinking.get("values", [])

    if not all(value in values for value in [False, True]):
        raise RuntimeError(
            "The model does not report support for both "
            "thinking=false and thinking=true."
        )

    return info


# ============================================================
# LAUNCHER MANAGEMENT
# ============================================================

def start_launcher(test_number: int) -> None:
    """
    Starts the repository's existing launcher.py.

    launcher.py starts Flask and the visible MuJoCo viewer.
    """

    global launcher_process
    global launcher_log_handle
    global started_launcher

    if robot_reachable():
        raise RuntimeError(
            "A robot API is already running on port 8765.\n"
            "Stop your existing launcher first. This benchmark needs "
            "to own the launcher lifecycle so every test starts "
            "with a clean simulation."
        )

    print("\nStarting launcher.py and MuJoCo...")

    log_path = LOG_DIR / f"launcher_test_{test_number:02d}.log"

    launcher_log_handle = open(
        log_path,
        "a",
        encoding="utf-8",
    )

    env = os.environ.copy()

    env["OLLAMA_MODEL"] = MODEL
    env["OLLAMA_URL"] = OLLAMA_URL
    env["FLASK_URL"] = ROBOT_URL

    # Prevent a second agent from controlling the same arm.
    env["TELEGRAM_BOT_TOKEN"] = ""

    launcher_process = subprocess.Popen(
        [sys.executable, str(LAUNCHER)],
        cwd=str(ROOT),
        env=env,
        stdin=subprocess.PIPE,
        stdout=launcher_log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    started_launcher = True

    deadline = time.time() + LAUNCH_TIMEOUT

    while time.time() < deadline:

        if launcher_process.poll() is not None:
            raise RuntimeError(
                f"launcher.py exited unexpectedly. See {log_path}"
            )

        if robot_reachable():
            print("Flask API is ready.")
            print("MuJoCo viewer should be visible.")

            # Check that the initial scene is accessible.
            state = robot_get("state")

            if not isinstance(state, dict) or "cube" not in state:
                raise RuntimeError(
                    f"Unexpected initial robot state: {state}"
                )

            return

        time.sleep(1)

    raise RuntimeError(
        f"Robot API did not start within {LAUNCH_TIMEOUT}s. "
        f"See {log_path}"
    )


def stop_launcher() -> None:
    global launcher_process
    global launcher_log_handle
    global started_launcher

    if launcher_process is not None:

        if launcher_process.poll() is None:
            print("\nStopping launcher and closing MuJoCo...")

            try:
                if launcher_process.stdin:
                    launcher_process.stdin.write("/exit\n")
                    launcher_process.stdin.flush()

                launcher_process.wait(timeout=15)

            except Exception:
                launcher_process.terminate()

                try:
                    launcher_process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    launcher_process.kill()

        launcher_process = None

    if launcher_log_handle:
        launcher_log_handle.close()
        launcher_log_handle = None

    started_launcher = False


def cleanup() -> None:
    global ollama_process
    global ollama_log_handle

    stop_launcher()

    # Only stop Ollama if this benchmark started it.
    if ollama_process is not None:

        if ollama_process.poll() is None:
            print("Stopping Ollama process started by benchmark...")
            ollama_process.terminate()

            try:
                ollama_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                ollama_process.kill()

        ollama_process = None

    if ollama_log_handle:
        ollama_log_handle.close()
        ollama_log_handle = None


# ============================================================
# ROBOT API
# ============================================================

def robot_get(endpoint: str) -> dict[str, Any]:
    response = http.get(
        f"{ROBOT_URL}/{endpoint}",
        timeout=ROBOT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


def robot_post(
    endpoint: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:

    response = http.post(
        f"{ROBOT_URL}/{endpoint}",
        json=arguments,
        timeout=ROBOT_TIMEOUT,
    )
    response.raise_for_status()
    return response.json()


# ============================================================
# TOOLS
# ============================================================

def load_tools() -> list[dict[str, Any]]:
    schema = json.loads(
        TOOLS_FILE.read_text(encoding="utf-8")
    )

    # These are the only tools needed for this experiment.
    allowed = {
        "state",
        "collisions",
        "pick",
    }

    return [
        tool
        for tool in schema
        if tool["function"]["name"] in allowed
    ]


def execute_tool(
    name: str,
    arguments: dict[str, Any],
) -> dict[str, Any]:

    global current_pick_attempts
    global last_pick_round

    log(
        "tool_call",
        name=name,
        arguments=arguments,
        round=current_round,
    )

    if name == "state":
        result = robot_get("state")

    elif name == "collisions":
        result = robot_get("collisions")

    elif name == "pick":

        # Require a new reasoning round between grasp attempts.
        if last_pick_round == current_round:
            result = {
                "success": False,
                "error": (
                    "A second grasp cannot be attempted in the same "
                    "reasoning round. Inspect the previous result first."
                ),
            }

            log("retry_blocked", result=result)
            return result

        if current_pick_attempts >= MAX_PICK_ATTEMPTS:
            result = {
                "success": False,
                "error": "Maximum grasp attempts reached.",
            }

            log("pick_limit_reached", result=result)
            return result

        target = str(
            arguments.get("target", "cube_2")
        ).strip().lower()

        # Enforce the experiment's target.
        if target not in {
            "cube_2",
            "green_cube",
            "green cube",
        }:
            result = {
                "success": False,
                "error": (
                    "This experiment only permits picking the green "
                    "cube (cube_2)."
                ),
            }

            log("wrong_target_blocked", target=target)
            return result

        current_pick_attempts += 1
        last_pick_round = current_round

        result = robot_post(
            "pick",
            {"target": "cube_2"},
        )

    else:
        result = {
            "success": False,
            "error": f"Unknown tool: {name}",
        }

    log(
        "tool_result",
        name=name,
        result=result,
    )

    return result


# ============================================================
# OLLAMA CHAT
# ============================================================

def call_model(
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:

    payload = {
        "model": MODEL,
        "messages": messages,
        "tools": tools,
        "stream": False,
        "think": config["thinking"],
        "options": {
            "temperature": 0.2,
            "num_ctx": config["num_ctx"],
            "num_predict": 256,
        },
    }

    start = time.perf_counter()

    response = http.post(
        f"{OLLAMA_URL}/api/chat",
        json=payload,
        timeout=OLLAMA_TIMEOUT,
    )

    response.raise_for_status()

    elapsed = time.perf_counter() - start
    data = response.json()

    message = data.get("message", {})

    log(
        "model_response",
        elapsed_seconds=round(elapsed, 3),
        prompt_eval_count=data.get("prompt_eval_count"),
        eval_count=data.get("eval_count"),
        done_reason=data.get("done_reason"),
        thinking=message.get("thinking", ""),
        content=message.get("content", ""),
        tool_calls=message.get("tool_calls", []),
    )

    return message


# ============================================================
# AGENT
# ============================================================

def run_agent(
    initial_state: dict[str, Any],
    tools: list[dict[str, Any]],
    config: dict[str, Any],
) -> str:

    global current_round

    system_prompt = f"""
You are controlling a MuJoCo robotic arm through its real robot API.

TASK:
Pick up the GREEN CUBE, whose target identifier is cube_2.

Rules:
1. Inspect the state before acting.
2. Inspect collisions if useful.
3. Use the pick tool with target cube_2.
4. The pick operation is a complete deterministic grasp routine.
   Do not attempt to manually reproduce its internal movements.
5. Check the tool result and inspect the state after every attempt.
6. If the grasp fails, reason about why it failed.
7. Inspect the cube's current position and collisions before retrying.
8. You have at most {MAX_PICK_ATTEMPTS} grasp attempts.
9. Never repeat the same failed action blindly.
10. Verify holding=true and held_object=cube_2 before reporting success.
11. Do not claim success if the cube was not physically lifted.
12. Do not attempt to pick another object.
13. Stop if the state is unclear.

Be concise, but explain your decisions.
"""

    messages = [
        {
            "role": "system",
            "content": system_prompt,
        },
        {
            "role": "user",
            "content": (
                "Pick up the green cube. If your first attempt fails, "
                "inspect the result and decide whether a recovery attempt "
                "is justified.\n\nInitial state:\n"
                + json.dumps(initial_state, indent=2)
            ),
        },
    ]

    final_reply = ""

    for round_number in range(1, config["max_rounds"] + 1):

        current_round = round_number

        log(
            "agent_round_start",
            round=round_number,
            max_rounds=config["max_rounds"],
        )

        message = call_model(messages, tools, config)

        messages.append(message)

        tool_calls = message.get("tool_calls", [])

        if not tool_calls:
            final_reply = message.get("content", "")

            log(
                "agent_finished",
                round=round_number,
                response=final_reply,
            )
            break

        for tool_call in tool_calls:

            function = tool_call.get("function", {})
            name = function.get("name", "")
            arguments = function.get("arguments", {})

            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            if not isinstance(arguments, dict):
                arguments = {}

            result = execute_tool(name, arguments)

            messages.append({
                "role": "tool",
                "tool_name": name,
                "content": json.dumps(
                    result,
                    ensure_ascii=False,
                    default=str,
                ),
            })

        log(
            "agent_round_complete",
            round=round_number,
            pick_attempts=current_pick_attempts,
        )

    else:
        final_reply = "Maximum reasoning rounds reached."

        log(
            "max_rounds_reached",
            max_rounds=config["max_rounds"],
        )

    return final_reply


# ============================================================
# ONE BENCHMARK TEST
# ============================================================

def run_one_test(
    test_number: int,
    config: dict[str, Any],
    tools: list[dict[str, Any]],
) -> dict[str, Any]:

    global current_events
    global current_pick_attempts
    global current_round
    global last_pick_round

    current_events = []
    current_pick_attempts = 0
    current_round = 0
    last_pick_round = -1

    print("\n" + "=" * 64)
    print(f"TEST {test_number}/8")
    print("=" * 64)

    print(
        f"Thinking: {'ON' if config['thinking'] else 'OFF'}\n"
        f"Context:  {config['num_ctx']}\n"
        f"Rounds:   {config['max_rounds']}"
    )

    start_time = time.perf_counter()

    test_result = {
        "test_number": test_number,
        "configuration": config,
        "status": "started",
    }

    try:
        # Every test gets a new MuJoCo world.
        start_launcher(test_number)

        initial_state = robot_get("state")

        log(
            "initial_state",
            state=initial_state,
        )

        cube = initial_state.get("objects", {}).get("cube_2")

        if not cube:
            raise RuntimeError(
                "cube_2 was not found in the initial robot state."
            )

        if initial_state.get("holding"):
            raise RuntimeError(
                "Robot unexpectedly starts while holding an object."
            )

        final_reply = run_agent(
            initial_state,
            tools,
            config,
        )

        final_state = robot_get("state")
        final_collisions = robot_get("collisions")

        verified_success = (
            final_state.get("holding") is True
            and final_state.get("held_object") == "cube_2"
        )

        elapsed = time.perf_counter() - start_time

        test_result.update({
            "status": "completed",
            "elapsed_seconds": round(elapsed, 2),
            "pick_attempts": current_pick_attempts,
            "verified_success": verified_success,
            "initial_state": initial_state,
            "final_state": final_state,
            "final_collisions": final_collisions,
            "final_reply": final_reply,
        })

        log(
            "test_complete",
            verified_success=verified_success,
            pick_attempts=current_pick_attempts,
            elapsed_seconds=round(elapsed, 2),
            final_state=final_state,
        )

    except Exception as exc:

        test_result.update({
            "status": "error",
            "error": str(exc),
        })

        log("test_error", error=str(exc))

    finally:
        test_result["log_file"] = str(
            save_test_log(
                test_number,
                config,
                test_result.copy(),
            )
        )

        stop_launcher()

    return test_result


# ============================================================
# BENCHMARK SUMMARY
# ============================================================

def print_summary() -> None:

    print("\n\n" + "=" * 72)
    print("NEMOTRON BENCHMARK SUMMARY")
    print("=" * 72)

    print(
        f"{'#':<4}"
        f"{'Think':<10}"
        f"{'Context':<10}"
        f"{'Rounds':<9}"
        f"{'Success':<12}"
        f"{'Picks':<8}"
        f"{'Time':<10}"
    )

    print("-" * 72)

    for result in results:

        config = result["configuration"]

        think = "ON" if config["thinking"] else "OFF"

        success = (
            "YES"
            if result.get("verified_success")
            else "NO"
        )

        print(
            f"{result['test_number']:<4}"
            f"{think:<10}"
            f"{config['num_ctx']:<10}"
            f"{config['max_rounds']:<9}"
            f"{success:<12}"
            f"{result.get('pick_attempts', '-'):<8}"
            f"{result.get('elapsed_seconds', '-'):<10}"
        )

    completed = [
        result
        for result in results
        if result.get("status") == "completed"
    ]

    successful = [
        result
        for result in completed
        if result.get("verified_success")
    ]

    print("-" * 72)
    print(f"Completed tests: {len(completed)}/8")
    print(f"Successful grasps: {len(successful)}/{len(completed)}")
    print(f"\nDetailed results: {LOG_DIR}")
    print(f"Summary: {SUMMARY_FILE}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("\n" + "=" * 64)
    print("XAVIERCLAW — NEMOTRON AUTOMATIC LIVE BENCHMARK")
    print("=" * 64)

    print(f"\nRepository: {ROOT}")
    print(f"Launcher:   {LAUNCHER}")
    print(f"Model:      {MODEL}")
    print(f"Results:    {LOG_DIR}")

    print("\nStarting prerequisites...")

    start_ollama()
    check_model()

    if robot_reachable():
        raise RuntimeError(
            "A robot API is already running.\n"
            "Stop your existing launcher before starting the benchmark. "
            "The benchmark must own the simulation lifecycle to reset "
            "the green cube between tests."
        )

    tools = load_tools()

    print("\nAvailable benchmark tools:")
    for tool in tools:
        print(" -", tool["function"]["name"])

    configurations = [
        {
            "thinking": thinking,
            "num_ctx": context,
            "max_rounds": rounds,
        }
        for thinking, context, rounds in itertools.product(
            THINKING_VALUES,
            CONTEXT_VALUES,
            ROUND_VALUES,
        )
    ]

    print(f"\nTotal tests: {len(configurations)}")

    print("\nThe benchmark will:")
    print("1. Start launcher.py.")
    print("2. Open the visible MuJoCo simulation.")
    print("3. Run one Nemotron configuration.")
    print("4. Save the result.")
    print("5. Close that simulation.")
    print("6. Restart with a fresh world for the next test.")

    input("\nPress ENTER to begin all 8 tests...")

    for index, config in enumerate(configurations, start=1):

        result = run_one_test(
            index,
            config,
            tools,
        )

        results.append(result)
        save_summary()

        print_summary()

        if index < len(configurations):
            time.sleep(PAUSE_BETWEEN_TESTS)

    print("\nAll benchmark configurations have finished.")


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:
        print("\nBenchmark interrupted.")

    except Exception as exc:
        print(f"\nFATAL ERROR: {exc}")

    finally:
        cleanup()
        save_summary()

        print("\nResults directory:")
        print(LOG_DIR)
