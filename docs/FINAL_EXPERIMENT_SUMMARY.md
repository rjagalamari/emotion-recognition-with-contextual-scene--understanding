# Final Experiment Summary

## Dataset
- Dataset: `CAER-S`
- Total samples used: `69,999`
- Emotion classes: `angry`, `disgusted`, `fearful`, `happy`, `neutral`, `sad`, `surprised`
- Split used in all final reported experiments:
  - `train = 41,655`
  - `val = 7,352`
  - `test = 20,992`
- Validation split: stratified `15%` split carved from original `train/` with `seed = 42`
- Test split: original `CAER-S/test`

## Final Benchmark Table
| Experiment | Description | Model Used | Accuracy | Macro F1 |
|---|---|---|---:|---:|
| Experiment 1 | Face-only | Simple MLP classifier | `43.54%` | `42.72%` |
| Experiment 2 | Scene-only | Simple MLP classifier | `74.51%` | `73.91%` |
| Experiment 3 | Face + Scene Fusion | Cross-gated multi-head attention fusion | `78.20%` | `77.73%` |
| Experiment 4 | Face + Semantic Context Fusion | Cross-gated multi-head attention fusion | `63.31%` | `62.67%` |

## Experiment 1
- Goal: measure how much emotion information can be recovered from face embeddings alone.
- Input feature: DeepFace / Facenet512 face embedding (`512` dimensions)
- Model used: simple MLP classifier
  - `512 -> 256 -> 64 -> 7`
- Hyperparameters:
  - `epochs = 30`
  - `batch_size = 128`
  - `lr = 1e-3`
  - `weight_decay = 1e-4`
  - `dropout = 0.3`
  - `patience = 7`
- Final reported result:
  - Accuracy: `43.54%`
  - Macro F1: `42.72%`
- Visuals:
  - [Confusion Matrix - Validation](./experiment1_sujal_style/confusion_validation.png)
  - [Confusion Matrix - Test](./experiment1_sujal_style/confusion_test.png)
  - [Learning Curve](./experiment1_sujal_style/learning_curve.png)

## Experiment 2
- Goal: measure how much surrounding visual scene context alone explains the emotion label.
- Input feature: Places365 ResNet-18 scene embedding (`512` dimensions)
- Model used: simple MLP classifier
  - `512 -> 256 -> 64 -> 7`
- Hyperparameters:
  - `epochs = 30`
  - `batch_size = 128`
  - `lr = 1e-3`
  - `weight_decay = 1e-4`
  - `dropout = 0.3`
  - `patience = 7`
- Final reported result:
  - Accuracy: `74.51%`
  - Macro F1: `73.91%`
- Visuals:
  - [Confusion Matrix - Validation](./experiment2_scene_only_places365/confusion_validation.png)
  - [Confusion Matrix - Test](./experiment2_scene_only_places365/confusion_test.png)
  - [Learning Curve](./experiment2_scene_only_places365/learning_curve.png)

## Experiment 3
- Goal: test whether combining face and scene features improves performance over using a single branch.
- Input features:
  - Face embedding: DeepFace / Facenet512 (`512`)
  - Scene embedding: Places365 ResNet-18 (`512`)
- Fusion model used: cross-gated multi-head attention fusion
  - each modality is projected into learned tokens
  - bidirectional cross-attention is applied
  - learned gating combines both modalities before classification
- Hyperparameters:
  - `epochs = 30`
  - `batch_size = 128`
  - `lr = 1e-3`
  - `weight_decay = 1e-4`
  - `dropout = 0.3`
  - `patience = 7`
- Final reported result:
  - Accuracy: `78.20%`
  - Macro F1: `77.73%`
- Visuals:
  - [Confusion Matrix - Validation](./experiment3_3_places365_cross_gated/confusion_validation.png)
  - [Confusion Matrix - Test](./experiment3_3_places365_cross_gated/confusion_test.png)
  - [Learning Curve](./experiment3_3_places365_cross_gated/learning_curve.png)

## Experiment 4
- Goal: test whether semantic context from captions improves performance when fused with face features.
- Input features:
  - Face embedding: DeepFace / Facenet512 (`512`)
  - Semantic context embedding: BLIP caption -> MiniLM embedding (`384`)
- Fusion model used: cross-gated multi-head attention fusion
  - each modality is projected into learned tokens
  - bidirectional cross-attention is applied
  - learned gating combines both modalities before classification
- Hyperparameters:
  - `epochs = 30`
  - `batch_size = 128`
  - `lr = 1e-3`
  - `weight_decay = 1e-4`
  - `dropout = 0.3`
  - `patience = 7`
- Final reported result:
  - Accuracy: `63.31%`
  - Macro F1: `62.67%`
- Visuals:
  - [Confusion Matrix - Validation](./experiment4_4_cross_gated/confusion_validation.png)
  - [Confusion Matrix - Test](./experiment4_4_cross_gated/confusion_test.png)
  - [Learning Curve](./experiment4_4_cross_gated/learning_curve.png)
