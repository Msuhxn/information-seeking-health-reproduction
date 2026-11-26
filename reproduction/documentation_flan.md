## Flan-T5 XL – Zero-Shot Reproduction

### Experimental Setup

- **Model:** `google/flan-t5-xl`
- **Dataset:** TREC Health Misinformation 2022
- **Number of questions:** 50 per configuration
- **Total predictions:** 150
- **Prompt configurations:** No-context, Non-expert, Expert
- **Device:** CPU
- **Precision:** FP32
- **Decoding:** Greedy (`do_sample=False`, `num_beams=1`)
- **Operating system:** Windows
- **CPU:** Intel Core i7-13700H
- **RAM:** 64 GB
- **Python:** 3.11.9
- **PyTorch:** 2.15.0+cpu
- **Transformers:** 5.24.1
- **SentencePiece:** 0.2.3
- **Experiment date:** 26 November 2026
- **Evaluation metric:** Accuracy

### Reproduction Results

| Prompt Configuration | Original Accuracy | Reproduced Accuracy | Difference |
|---|---:|---:|---:|
| No-context | 56% (28/50) | 56% (28/50) | 0 pp |
| Non-expert | 54% (27/50) | 54% (27/50) | 0 pp |
| Expert | 74% (37/50) | 74% (37/50) | 0 pp |

All three accuracy values reported in the original paper were
successfully reproduced.

### Response-Level Comparison

| Metric | No-context | Non-expert | Expert |
|---|---:|---:|---:|
| Exact response agreement | 82% | 80% | 100% |
| Prediction agreement | 100% | 100% | 100% |
| Correctness agreement | 100% | 100% | 100% |
| Invalid responses (original) | 10 | 4 | 0 |
| Invalid responses (reproduction) | 10 | 4 | 0 |
| Textual differences | 9 | 10 | 0 |

Overall:

- **Exact response agreement:** 131/150 (87.33%)
- **Prediction agreement:** 150/150 (100%)
- **Correctness agreement:** 150/150 (100%)
- **Accuracy difference:** 0 percentage points

Prediction agreement includes responses without a valid yes/no answer,
which were classified as `NONE`.

### Execution Time

| Configuration | Inference Runtime | Total Runtime |
|---|---:|---:|
| No-context | 53.4 s | 61.8 s |
| Non-expert | 93.6 s | 102.6 s |
| Expert | 49.8 s | 58.2 s |
| **Total** | **196.8 s** | **222.6 s** |

The initial model download and three-question technical test took
26.12 minutes and are excluded from the full experiment runtimes.

### Modifications to the Original Implementation

- Explicitly configured CPU inference with FP32 precision.
- Explicitly enforced greedy decoding.
- Reused the original dataset and prompt contexts.
- Added a separate output directory to preserve the original results.
- Added runtime measurements and experimental metadata.
- Implemented a separate comparison script to evaluate individual
  responses and prevent ground-truth leakage during prediction extraction.

### Observed Differences and Limitations

Minor textual differences were observed in 19 out of 150 responses:

- No-context: 9 responses.
- Non-expert: 10 responses.
- Expert: 0 responses.

The reproduced responses were slightly longer than the original
responses. A possible explanation is a difference in the default
generation length settings between Transformers versions.

Neither implementation explicitly specifies a maximum generation
length. The exact cause of the textual differences has not been
experimentally verified.

The original software environment, model checkpoint revision and
effective generation parameters have not been fully verified.

None of the observed textual differences affected the extracted
predictions or the final accuracy.

### Reproduction Files

- `reproduction/flan_t5_zero_shot.py` – Model inference script
- `reproduction/compare_flan_t5.py` – Response comparison script
- `reproduction/env_flan_t5.txt` – Environment dependencies
- `reproduction/outputs/flan-t5/` – Raw model outputs
- `reproduction/evaluation/flan-t5/` – Comparison results

### Conclusion

The zero-shot reproduction of Flan-T5 XL on the TREC Health
Misinformation 2022 dataset was successful at both the accuracy
and prediction levels.

All three reported accuracy values were reproduced exactly.
Furthermore, all 150 extracted predictions matched the authors'
original predictions, resulting in 100% prediction and correctness
agreement.

Although exact textual agreement was 87.33%, the observed differences
did not affect the evaluation results.

The experiment also demonstrated that the selected Flan-T5 XL
experiments can be reproduced locally on a Windows laptop using
CPU inference without GPU acceleration.