# Experiment 3 Saliency Maps

This folder contains the Experiment 3 Grad-CAM saliency outputs for the final
face + scene fusion model.

## Model

- Experiment: Experiment 3, face + scene fusion
- Checkpoint: `experiments/experiment3_3_places365_cross_gated/best_model.pt`
- Model type: FaceNet512 face embedding + Places365 ResNet-18 scene embedding with cross-gated fusion
- Checkpoint selection: best validation macro-F1 checkpoint
- Best epoch: 30
- Best validation macro-F1: 0.774050

## Data

- Dataset: CAER-S in-domain test split
- Test split size: 20,992 images
- Saliency figure subset: 28 images, 4 per class, seed 42
- Classes: angry, disgusted, fearful, happy, neutral, sad, surprised

## Files

- `gradcam_e3_grid.png`: Experiment 3 Grad-CAM grid, 4 examples per class.
- `gradcam_e3_metadata.json`: Metadata for the Grad-CAM grid.
- `gradcam_e3_mass_per_image.csv`: Face/body/background saliency mass for each of the 28 images.
- `gradcam_e3_mass_summary.csv`: Summary table by class and overall.
- `gradcam_e3_mass_summary.png`: Stacked bar plot of mean saliency mass by class.
- `gradcam_e3_mass_metadata.json`: Metadata for the saliency mass analysis.

## Main Result

For the same 28 images shown in `gradcam_e3_grid.png`, the mean Grad-CAM mass is:

| Region | Mean saliency mass |
|---|---:|
| Face | 24.71% |
| Body/person outside face | 66.02% |
| Background | 9.27% |

Paper-ready wording:

> For the 28 Grad-CAM examples shown for Experiment 3, 24.7% of saliency mass falls on detected face regions, 66.0% on body/person regions outside the face, and 9.3% on background regions. This suggests that the fusion model's scene branch relies heavily on person-level contextual cues such as body posture and gestures, while still incorporating facial and background information.

## Region Definition

- Face: detected face bounding boxes from DeepFace/OpenCV.
- Body: detected COCO person bounding boxes, excluding detected face regions.
- Background: area outside the union of detected person and face boxes.

Grad-CAM is computed through the Places365 scene branch while the cached FaceNet512 embedding is held fixed.
