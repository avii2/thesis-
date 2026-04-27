# Cluster 1 P_C1 Tuning Summary

This tuning output is restricted to Cluster 1 proposed FCFL only. The run keeps the architecture fixed as TCN + FedBN + weighted non-BN aggregation and reuses the frozen agglomerative membership file.

## Search Status

- Search mode: `smoke`
- Total configurations in exact search space: `6144`
- Trials attempted in this invocation: `2`
- Completed trials recorded: `1`
- Fair full-budget comparison to current results: `NO`

## Current References

- A_C1 test F1: `0.5622`
- B_C1 test F1: `0.5622`
- P_C1 test F1: `0.4932`

## Best Trial

- Trial: `trial_0001`
- Best validation F1: `0.4578`
- Test F1: `0.3427`
- Test FPR: `0.0028`
- Wall-clock seconds: `33.538`
- Beats current P_C1: `False`
- Beats A_C1: `False`
- Beats B_C1: `False`

## Best Config

- learning_rate: `0.003`
- batch_size: `128`
- local_epochs: `1`
- window_length: `32`
- stride: `8`
- block_channels: `[32, 64, 64]`
- hidden_dim: `32`
- dropout: `0.1`
- positive_class_weight_scale: `0.75`

## Notes

- The primary selection rule is validation F1; test F1, test FPR, and wall-clock time are tie-breakers only.
- Main `outputs/runs/P_C1/` and `outputs/metrics/P_C1_metrics.csv` are not overwritten by this script.
- This invocation used a smoke or partial budget, so the beat/not-beat flags are diagnostic and should not be treated as paper-level evidence.
