# Experiment 2 Saliency Maps

This folder contains Grad-CAM saliency maps for Experiment 2.

## Model

- Experiment: scene-only baseline
- Checkpoint: `experiments/scene_only_pretrained/simple_mlp/best_model.pt`
- Scene backbone: Places365 ResNet-18
- Classifier: simple MLP over 512-d scene embeddings
- Best epoch: 30
- Best validation macro-F1: 0.740543

## Files

- `gradcam_e2_grid.png`: Grad-CAM grid with 4 examples per emotion class.
- `gradcam_e2_metadata.json`: Metadata for the saliency run.

Grad-CAM is computed through the Places365 ResNet-18 scene backbone.

