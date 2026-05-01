#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.cluster1_batadal import DEFAULT_CONFIG_PATH, prepare_cluster1_batadal  # noqa: E402


def main() -> None:
    result = prepare_cluster1_batadal(DEFAULT_CONFIG_PATH)
    print(f"Cluster 1 BATADAL profile: {result['paths']['data_profile']}")
    print(f"Cluster 1 BATADAL clients: {result['paths']['client_metadata']}")
    print(f"Cluster 1 BATADAL memberships: {result['paths']['membership']}")
    print(f"Cluster 1 BATADAL validation summary: {result['paths']['validation_summary']}")


if __name__ == "__main__":
    main()
