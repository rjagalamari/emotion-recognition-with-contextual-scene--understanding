# Results

All results use CAER-S with seven emotion classes:

```text
angry, disgusted, fearful, happy, neutral, sad, surprised
```

Final split:

| Split | Samples |
|---|---:|
| Train | 41,655 |
| Validation | 7,352 |
| Test | 20,992 |

## Architecture Comparison

| Experiment | Architecture | Best epoch | Val macro-F1 | Test accuracy | Test macro-F1 |
|---|---|---:|---:|---:|---:|
| Face only pretrained | Simple MLP | 30 | 0.4355 | 0.4354 | 0.4272 |
| Face only pretrained | Token attention | 27 | 0.5138 | 0.5103 | 0.5006 |
| Scene only pretrained | Simple MLP | 30 | 0.7405 | 0.7451 | 0.7391 |
| Scene only pretrained | Token attention | 30 | 0.7188 | 0.7281 | 0.7205 |
| Face + scene fusion | Simple MLP | 28 | 0.6945 | 0.6946 | 0.6897 |
| Face + scene fusion | Cross-gated fusion | 30 | 0.7741 | 0.7820 | 0.7773 |
| Face + caption fusion | Simple MLP | 30 | 0.4897 | 0.4911 | 0.4829 |
| Face + caption fusion | Cross-gated fusion | 29 | 0.6342 | 0.6331 | 0.6267 |

## Final Four-Experiment Summary

The final project comparison reports the best selected setup for each experiment:

| Experiment | Final model | Test accuracy | Test macro-F1 |
|---|---|---:|---:|
| Face only | Simple MLP over FaceNet512 | 43.54% | 42.72% |
| Scene only | Simple MLP over Places365 ResNet-18 | 74.51% | 73.91% |
| Face + scene fusion | Cross-gated fusion | 78.20% | 77.73% |
| Face + caption fusion | Cross-gated fusion | 63.31% | 62.67% |

## Saliency Summary

Experiment 3 saliency maps and quantitative mass results are in
`saliency_maps/experiment3/`.

For the 28 Grad-CAM examples shown:

| Region | Mean saliency mass |
|---|---:|
| Face | 24.71% |
| Body/person outside face | 66.02% |
| Background | 9.27% |

