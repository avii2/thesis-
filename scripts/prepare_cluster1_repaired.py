from __future__ import annotations

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.cluster1_repaired import DEFAULT_CONFIG_PATH, prepare_cluster1_repaired


def main() -> None:
    result = prepare_cluster1_repaired(DEFAULT_CONFIG_PATH)
    profile = result["profile"]
    print(f"Cluster 1 repaired profile: {result['paths']['data_profile']}")
    print(f"Cluster 1 repaired clients: {result['paths']['client_metadata']}")
    print(f"Cluster 1 repaired memberships: {result['paths']['membership']}")
    print(f"Cluster 1 repaired validation summary: {result['paths']['validation_summary']}")
    print(
        "Clients with positive train/validation windows: "
        f"{profile['clients_with_positive_train_or_validation']}/{profile['num_leaf_clients']}"
    )
    print(f"Held-out test window counts: {profile['heldout_test_window_counts']}")


if __name__ == "__main__":
    main()
