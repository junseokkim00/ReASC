# `ReASC`: Reliability-Aware Adaptive Self-Consistency for Efficient Sampling in LLM Reasoning 

The Official code for [Reliability-Aware Adaptive Self-Consistency for Efficient Sampling in LLM Reasoning ](https://arxiv.org/abs/2601.02970)

1. generate conda environment
```bash
$ conda env create -f environment.yaml
$ conda activate reasc
```
2. run `run_self_certainty_calibration.sh` to get result for calibration set in offline settings
3. edit `run_sc_self_certainty.sh` and run it for the result of standard Self Consistency.
4. run each notebook within `./notebooks` for the result of [`ASC`](./asc_code.ipynb), [`ESC`](./esc_code.ipynb), [`ReASC Offline`](./confsc_offline_code.ipynb) and [`ReASC Online`](./confsc_online_code.ipynb)