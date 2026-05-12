# Emotion Recognition With Contextual Scene Understanding

This repository contains four CAER-S emotion-recognition experiments. The main
question is whether emotion classification improves when facial information is
combined with scene or semantic context.

## Dataset

The final experiments use CAER-S with seven emotion classes:

```text
angry, disgusted, fearful, happy, neutral, sad, surprised
```

Final split:

| Split | Samples |
|---|---:|
| Train | 41,655 |
| Validation | 7,352 |
| Test | 20,992 |

The raw CAER-S images and cached feature folders are not included in this
GitHub-ready copy because they are large dataset artifacts.

## Experiments

Each experiment folder has two architectures:

- `simple_mlp`: baseline classifier over frozen pretrained features.
- `gated_attention`: the stronger architecture used for comparison. For the
  single-modality experiments, this is the token-attention version. For the
  fusion experiments, this is the gated/cross-gated fusion model.

| Folder | Meaning |
|---|---|
| `experiments/face_only_pretrained` | FaceNet512 face embeddings |
| `experiments/scene_only_pretrained` | Places365 ResNet-18 scene embeddings |
| `experiments/fusion_scene_vision` | FaceNet512 + Places365 visual fusion |
| `experiments/fusion_face_caption` | FaceNet512 + BLIP/MiniLM caption fusion |

Each run folder includes:

- `run.py`: training/evaluation code for that architecture.
- `best_model.pt`: best checkpoint selected by validation macro-F1.
- `history.csv`: epoch-by-epoch training history.
- `metrics.json`: validation and test metrics.
- `results.csv`: compact metric table.
- `manifest_*.csv`: split and feature manifest used by that run.

Some final runs also include learning curves and confusion matrices.

## Final Reported Results

The main final comparison uses the best architecture for each experiment.

| Experiment | Final model | Test accuracy | Test macro-F1 |
|---|---|---:|---:|
| Face only | Simple MLP over FaceNet512 | 43.54% | 42.72% |
| Scene only | Simple MLP over Places365 ResNet-18 | 74.51% | 73.91% |
| Face + scene fusion | Cross-gated fusion | 78.20% | 77.73% |
| Face + caption fusion | Cross-gated fusion | 63.31% | 62.67% |

## Saliency

Experiment 3 saliency outputs are in:

```text
docs/saliency_map_experiment3/
```

For the 28 displayed Grad-CAM examples, saliency mass was:

| Region | Mean saliency mass |
|---|---:|
| Face | 24.71% |
| Body/person outside face | 66.02% |
| Background | 9.27% |

## Running

Install dependencies:

```bash
pip install -r requirements.txt
```

Then run a specific experiment from its folder, for example:

```bash
python experiments/scene_only_pretrained/simple_mlp/run.py
```

The scripts expect the same local dataset/feature-cache layout used in the
project. Large raw datasets are intentionally not committed.

