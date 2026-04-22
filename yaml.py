from __future__ import annotations

import json
import subprocess
from typing import Any


class YAMLError(RuntimeError):
    """Raised when the local YAML shim cannot parse or emit YAML."""


def _read_stream(stream: Any) -> str:
    if hasattr(stream, "read"):
        return str(stream.read())
    return str(stream)


def safe_load(stream: Any) -> Any:
    text = _read_stream(stream)
    if not text.strip():
        return None

    command = [
        "ruby",
        "-ryaml",
        "-rjson",
        "-e",
        (
            "input = STDIN.read\n"
            "obj = YAML.safe_load(input, aliases: false)\n"
            "puts JSON.generate(obj)\n"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            input=text,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise YAMLError(exc.stderr.strip() or "YAML safe_load failed.") from exc

    return json.loads(completed.stdout)


def safe_dump(data: Any, *_, **__) -> str:
    payload = json.dumps(data)
    command = [
        "ruby",
        "-ryaml",
        "-rjson",
        "-e",
        (
            "obj = JSON.parse(STDIN.read)\n"
            "print YAML.dump(obj)\n"
        ),
    ]
    try:
        completed = subprocess.run(
            command,
            input=payload,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        raise YAMLError(exc.stderr.strip() or "YAML safe_dump failed.") from exc

    return completed.stdout


load = safe_load
dump = safe_dump
