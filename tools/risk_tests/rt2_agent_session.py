"""Builds a coding-agent session for the routing traces (M1 risk test 2), in Nemotron 3 Nano's own chat
format: a system prompt with tools, a task, and a loop of tool calls and tool results. The tool
results are real files and command output from this repo, so the token mix (mostly code) is what an
agent session reads; the assistant turns are short, as an agent's are.

  python tools/risk_tests/rt2_agent_session.py OUT.jsonl
"""
import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]

TOOLS = """# Tools

You have access to the following functions:

<tools>
<function>
<name>read_file</name>
<description>Read a file in the repository.</description>
<parameters>
<parameter><name>path</name><type>string</type></parameter>
</parameters>
</function>
<function>
<name>run</name>
<description>Run a shell command in the repository and return its output.</description>
<parameters>
<parameter><name>command</name><type>string</type></parameter>
</parameters>
</function>
<function>
<name>edit_file</name>
<description>Replace text in a file.</description>
<parameters>
<parameter><name>path</name><type>string</type></parameter>
<parameter><name>old</name><type>string</type></parameter>
<parameter><name>new</name><type>string</type></parameter>
</parameters>
</function>
</tools>"""


def call(fn, **args):
    body = "".join(f"<parameter={k}>\n{v}\n</parameter>\n" for k, v in args.items())
    return f"<tool_call>\n<function={fn}>\n{body}</function>\n</tool_call>\n"


def turn(role, text):
    return f"<|im_start|>{role}\n{text}<|im_end|>\n"


def main():
    out = pathlib.Path(sys.argv[1])
    steps = [
        ("I'll start by looking at how the SSD speed test works.", "read_file", {"path": "src/setup/measure_disk.cpp"}),
        ("The grid search is in measure_drive. Let me see how the results are explained.", "read_file", {"path": "src/setup/explain.cpp"}),
        ("Now the profile format, since the planner reads it.", "read_file", {"path": "src/setup/profile.cpp"}),
        ("Let me check the I/O layer the test uses on Linux.", "read_file", {"path": "src/io/io_posix_common.h"}),
        ("And the thread pool behind it.", "read_file", {"path": "src/io/io_pool.cpp"}),
        ("Let me look at the controller too, since its choices depend on the drive speed.", "read_file", {"path": "src/spec/controller.cpp"}),
        ("Let me run the tests to see the current state.", "run", {"command": "git log --stat -5"}),
    ]
    text = turn("system", "You are a coding agent working in the Kera repository. Use the tools to read code, run "
                          "commands and make changes. Think before acting, keep changes small, and run the tests.\n\n" + TOOLS)
    text += turn("user", "The SSD speed test sometimes reports a speed that is too high on drives with a large cache. "
                         "Find where the measurement happens, explain why it can overstate the speed, and propose a fix "
                         "with a test.")
    for i, (thought, fn, args) in enumerate(steps):
        text += turn("assistant", f"<think>\n{thought}\n</think>\n" + call(fn, **args))
        if fn == "read_file":
            result = (ROOT / args["path"]).read_text()
        else:
            result = subprocess.run(args["command"], shell=True, cwd=ROOT, capture_output=True, text=True).stdout
        text += turn("user", f"<tool_response>\n{result}\n</tool_response>")
    text += turn("assistant", "<think>\nThe confirm run re-measures the best cell for several seconds with fresh random "
                              "data, so a drive's write cache can't serve it. The remaining risk is a short test file "
                              "on drives with a large read cache; the test file should be bigger than the cache.\n</think>\n"
                              "The measurement is in `measure_drive` in `src/setup/measure_disk.cpp`. It writes a fresh "
                              "random test file, searches read sizes and parallel reads, then confirms the best setting "
                              "with a sustained run. Two things can still overstate the speed: a test file smaller than "
                              "the drive's read cache, and a file system that ignores direct reads. I propose sizing the "
                              "file from the drive's model where known, and flagging any result above 30 GB/s, which is "
                              "already done. Here is the change and a test.\n"
                 + call("edit_file", path="src/setup/measure_disk.cpp", old="const double margin_gb = 2.0;",
                        new="const double margin_gb = 2.0; // keep this much free after the test file"))
    text += turn("user", "<tool_response>\nThe file was edited.\n</tool_response>")
    text += turn("assistant", "<think>\nNow run the unit tests.\n</think>\n" + call("run", command="ctest --test-dir build"))
    with open(out, "w") as f:
        f.write(json.dumps({"id": "agent-session", "text": text}) + "\n")
    print(f"agent session: {len(text)} characters -> {out}")


if __name__ == "__main__":
    main()
