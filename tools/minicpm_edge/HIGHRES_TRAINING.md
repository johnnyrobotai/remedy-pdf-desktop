# MiniCPM High-Resolution Training Note

## Finding

`4x/36` high-resolution generation works for MiniCPM-V-4.6 on the Remedy page
renders, but the original training path failed because `downsample_mode` was
only passed to `processor.apply_chat_template`. It was not passed into the model
forward call made by `Trainer`.

MiniCPM-V-4.6 uses `downsample_mode` twice:

- processor side: controls image placeholder count in the tokenized prompt;
- model side: controls whether the vision tower emits 4x or 16x visual tokens.

If the processor uses `4x` and forward silently defaults to `16x`, the image
placeholder count and visual feature count diverge. That matches the observed
H100 failures:

- `RuntimeError: shape '[21, 1023, 1152]' is invalid for input of size 24754176`
- `ValueError: Multimodal features and tokens do not match, tokens: 252, features: 63`

## Fix

`train_lora_minicpm.py` now carries `downsample_mode` in the collator output so
`Trainer` passes it into `model.forward(...)`:

```python
out = {
    "input_ids": torch.stack(input_ids),
    "attention_mask": torch.stack(attention_mask),
    "labels": torch.stack(labels),
    "downsample_mode": self.downsample_mode,
}
```

`max_slice_nums` remains processor-only.

## Verification

Local non-GPU verification confirms the collator sends:

- processor kwargs: `{"downsample_mode": "4x", "max_slice_nums": 36}`
- forward batch: `downsample_mode="4x"`

GPU verification on the Heidi L4 also passed:

- artifact: `tools/minicpm_edge/eval_runs/highres_training_probe_l4/summary.json`
- fixed path: `4x/36` forward passed with loss `0.641650915145874`
- omitted-forward path: reproduced the original shape error

Full optimizer verification then passed on a RunPod H100 SXM:

- mode: bf16 LoRA rank 8, batch 1, gradient accumulation 1, `4x/36`
- result: 3/3 forward, backward, and optimizer steps completed
- losses: `0.8761`, `0.6406`, `0.8317`
- runtime: `56.84s`
- adapter load: passed in a fresh MiniCPM/PEFT process
- artifact: `tools/minicpm_edge/outputs/_smoke_heading_highres_4x36_h100`
- log: `tools/minicpm_edge/eval_runs/highres_optimizer_smoke_h100/highres_train_smoke_h100.log`

The same bf16 and 4-bit QLoRA smoke runs reached backward on the 24 GB Heidi
L4 but ran out of memory while requesting another `5.71 GiB`. Use the L4 for
high-resolution forward/eval work and an H100-class GPU for `4x/36` training.

Run this on the next GPU workbench before a high-res training job:

```bash
PYTHONPATH=tools/minicpm_edge python tools/minicpm_edge/probe_training_forward.py \
  --train tools/minicpm_edge/data/tasks/heading_hierarchy/train.jsonl \
  --downsample-mode 4x \
  --max-slice-nums 36
```

To prove the old failure mode is gone for the right reason, this command should
fail or reproduce the mismatch:

```bash
PYTHONPATH=tools/minicpm_edge python tools/minicpm_edge/probe_training_forward.py \
  --train tools/minicpm_edge/data/tasks/heading_hierarchy/train.jsonl \
  --downsample-mode 4x \
  --max-slice-nums 36 \
  --omit-forward-downsample
```

The forward and optimizer gates now pass. High-resolution v3 training can use
this path, while retaining a short optimizer smoke at the start of each fresh
GPU environment.
