# LM_Dec_Attention_Nystromformer

Nystromformer - Attention family, decoder-only byte-level language model, standardised in the Statera Working Group Language Model library.

## Status

- Verification level: **Runnable**
- Failure class: Architecture Unsuitable: not a causal language model (later tokens change earlier outputs, leak 1.3e-01)

## Provenance

- Source repository: LM_2021_Nystrom-transformer
- Upstream / license: vendored from RichardMinsooGo/LM_2021_Nystrom-transformer (see LICENSE in vendor folder if present)

## Differences from the source ([Unified] protocol)

- Nystromformer blocks + embedding/positions/head written here, 64 landmarks

All models share one protocol: 256-byte vocabulary, dim 512, 6 layers, 8 heads x 64, Adam 2e-4 with 200 warm-up steps, gradient clip 0.5, bf16 autocast (automatic fp32 fallback). Numbers rank structures under one budget on one GPU (RTX 3070 Ti 8 GB); they do not reproduce paper results.

## Results

- Parameters: 20.22 M
- Causality check: FAIL (largest change of earlier outputs 1.3e-01)

## Tests

- check: data counts, causality (no look-ahead, outputs use the current byte), parameters
- scaling: peak GPU memory and time vs length 512-8192
- overfit: memorise 4 x 256 bytes below 0.5 nats
- LM-02: enwik8 next-byte prediction (bits per byte)
- LM-05: long-horizon key-value recall with key-removed control (v3: distinct queries)
- LM-06: memory update - latest value vs stale value after a key is reassigned

## Run

```bash
pip install torch==2.5.1 einops numpy
python run.py check nystrom
python run.py lm02 nystrom      # enwik8 is downloaded to data/ on first use
```
