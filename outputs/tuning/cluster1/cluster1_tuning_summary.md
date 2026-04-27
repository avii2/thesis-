# Cluster 1 P_C1 Tuning Summary

This tuning output is restricted to Cluster 1 proposed FCFL only. The run keeps the architecture fixed as TCN + FedBN + weighted non-BN aggregation and reuses the frozen agglomerative membership file.

## Search Status

- Search mode: `full`
- Total configurations in exact search space: `864`
- New trials attempted in this invocation: `0`
- Completed full-budget trials recorded: `5`
- Fair full-budget comparison to current results: `YES`

## Current References

- A_C1 test F1: `0.5622`
- B_C1 test F1: `0.5622`
- P_C1 test F1: `0.4932`

## Best Trial

- Trial: `trial_0010`
- Best validation F1: `0.6707`
- Best validation recall: `0.5288`
- Best validation FPR: `0.0002`
- Test F1: `0.5326`
- Test FPR: `0.0006`
- Wall-clock seconds: `1497.281`
- Beats current P_C1: `True`
- Beats A_C1: `False`
- Beats B_C1: `False`

## Best Config

- learning_rate: `0.003`
- batch_size: `128`
- local_epochs: `1`
- window_length: `32`
- stride: `8`
- block_channels: `[32, 64, 128]`
- hidden_dim: `64`
- dropout: `0.05`
- positive_class_weight_scale: `1.25`

## Highest Test-F1 Observation

This is reported for diagnosis only. It is not used for tuning selection.

- Trial: `trial_0001`
- Best validation F1: `0.5882`
- Best validation recall: `0.4808`
- Best validation FPR: `0.0006`
- Test F1: `0.5676`
- Test FPR: `0.0006`
- Beats current P_C1: `True`
- Beats A_C1: `True`
- Beats B_C1: `True`
- Config: lr=`0.003`, batch_size=`128`, local_epochs=`1`, window_length=`32`, stride=`8`, block_channels=`[64, 64, 64]`, hidden_dim=`64`, dropout=`0.05`, positive_class_weight_scale=`1.25`

## Notes

- Current selection uses validation F1, then validation recall, validation FPR, and wall-clock time. Test metrics are reported only after selection.
- Main `outputs/runs/P_C1/` and `outputs/metrics/P_C1_metrics.csv` are not overwritten by this script.
