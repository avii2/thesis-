from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.cluster1_balanced import prepare_cluster1_balanced_train


def main() -> None:
    result = prepare_cluster1_balanced_train()
    if result["status"] != "ok":
        print("Cluster 1 balanced variant generation stopped.")
        print(result["status"])
        return

    summary = result["summary"]
    print(f"Raw audit: {summary['raw_audit_path']}")
    print(f"Window audit: {summary['window_audit_path']}")
    print(f"Variant root: {summary['variant_root']}")
    print(f"Ratios generated: {summary['ratios_generated']}")
    if summary["ratios_skipped"]:
        print(f"Ratios skipped: {summary['ratios_skipped']}")


if __name__ == "__main__":
    main()
