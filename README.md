# blood_cell_image_CNN_repo
Blood cell image classification using Convolutional Neural Networks (CNNs) implemented in TensorFlow/Keras

Deep Learning, Assignment 1. A convolutional neural network, adapted from the
[TensorFlow CNN tutorial](https://www.tensorflow.org/tutorials/images/cnn), that classifies
microscope images of single white blood cells into four classes:

`EOSINOPHIL` · `LYMPHOCYTE` · `MONOCYTE` · `NEUTROPHIL`

**Author:** HOUT NGUONHENG

---

## Results (single run, seed 42, best checkpoint = epoch 20)

| Evaluation set | Images | Accuracy | Macro F1 | Notes |
|---|---:|---:|---:|---|
| Kaggle TEST (internal) | 2,487 | **87.58%** | 0.879 | Most meaningful number |
| Kaggle originals (`dataset-master`) | 347 | 96.83% | 0.968 | Imbalanced; probably the source of the augmented images |
| LISC_dataset test | 996 | 100.00% | 1.000 | **Not independent**: file names match Kaggle TRAIN |

Per-class F1 on the Kaggle TEST set: lymphocyte 0.987, eosinophil 0.879, monocyte 0.848, neutrophil 0.802.
The model over-predicts neutrophil (precision 0.693, recall 0.954).

Training: loss 1.36 to 0.015, training accuracy 39% to 99.5%. Validation was noisy (accuracy swings
between 25% and 100%), calmer after the learning rate was reduced. Validation accuracy at the best epoch
(100%) is about 12 points above Kaggle test accuracy, because validation images are drawn from the same
augmented pool as the training images. See the report (sections 4.6 and 6.6) for details.

## Model

```
Input 120x160x3 -> augmentation (flip, rotation 0.1, zoom 0.1, contrast 0.1; training only)
4 x [Conv2D(3x3, relu, same) -> BatchNorm -> MaxPool(2x2)]   filters: 32, 64, 128, 128
Flatten (8,960) -> Dense(128, relu) -> Dropout(0.3) -> Dense(4, softmax)
```

1,389,764 parameters (1,389,060 trainable, 704 non-trainable).

| Setting | Value |
|---|---|
| Optimizer | Adam, initial learning rate 0.001 |
| LR schedule | `ReduceLROnPlateau` on `val_loss` (factor 0.5, patience 3, min 1e-5) |
| Loss | Categorical cross-entropy (one-hot labels) |
| Batch size / epochs | 32 / 25 |
| Input | Resized to 120x160 (4:3), pixels divided by 255 |
| Split | Kaggle TRAIN: 85% train / 15% validation, stratified |
| Checkpoint | Lowest validation loss (`best_model.keras`) |

## Datasets

The script expects this layout (edit `BASE` at the top of the script if yours differs):

```
../BloodCell_Project/datasets/
├── dataset-master/dataset-master/          # original images
│   ├── JPEGImages/BloodImage_00000.jpg ...
│   └── labels.csv
└── dataset2-master/
    ├── dataset2-master/images/
    │   ├── TRAIN/{EOSINOPHIL,LYMPHOCYTE,MONOCYTE,NEUTROPHIL}/
    │   └── TEST/{EOSINOPHIL,LYMPHOCYTE,MONOCYTE,NEUTROPHIL}/
    └── LISC_dataset/test/{EOSINOPHIL,LYMPHOCYTE,MONOCYTE,NEUTROPHIL}/
```

- **Kaggle blood cell images** (Paul Mooney): <https://www.kaggle.com/datasets/paultimothymooney/blood-cells>
  - TRAIN 9,957 and TEST 2,487 augmented images (320x240), used for training, validation and internal testing.
  - `dataset-master` originals (640x480) are test only. Only the 347 rows of `labels.csv` with exactly one of the four classes and an existing image are kept.
- **LISC_dataset** (test split only, 996 images, 224x224). Its file names match Kaggle TRAIN, so it is a re-export of Kaggle images and must not be treated as independent data.

Datasets are not included in this repository.

## Setup

Python 3.9+ is recommended. Install the dependencies:

```bash
pip install tensorflow numpy pandas matplotlib scikit-learn
```

The model is saved in the native `.keras` format (Keras 3 / recent TensorFlow 2.x).
A GPU is optional; training runs on CPU but is slower.

## Run

```bash
python blood_cell_image_cnn_tf.py
```

The script runs the full pipeline and writes everything to `results_tf/`. It is organized in cells that
follow the assignment tasks:

| Cell | Task | What it does |
|---|---|---|
| 0 | Setup | Imports, seed 42, determinism flag, paths and hyperparameters |
| 1 | Task 1 | Lists the datasets, prints class counts, saves the dataset overview figure |
| 2 | Task 2 | Stratified split, resize and normalize, one-hot labels, augmentation layer, before/after figure |
| 3 | Task 3 | Builds the CNN and saves the model summary |
| 4 | Tasks 4 and 5 | Compiles (Adam, categorical cross-entropy) and trains with checkpoint and LR schedule |
| 5 | Task 5 | Training curves and the convergence analysis |
| 6 | Task 6 | Evaluates the best checkpoint on the three test sets |
| 7 | Task 6 | Grad-CAM visualisation on the Kaggle test set |

## Output files (`results_tf/`)

| File | Content |
|---|---|
| `1_original_dataset.png` | Random example images per class |
| `2_preprocessing_before_after.png` | Original, resized/normalized, and augmented images |
| `3_model_summary.txt` | `model.summary()` output |
| `training_history_tf.csv` | Loss, accuracy, validation metrics and learning rate per epoch |
| `5_training_curves.png`, `5a_accuracy.png`, `5b_loss.png` | Training vs validation curves |
| `4_convergence_analysis.txt` | Automatic convergence statistics |
| `6_report_Kaggle_test.txt`, `6_report_BCCD_external.txt`, `6_report_LISC_test.txt` | Classification reports |
| `6_summary.json` | Accuracy on the three test sets |
| `6_gradcam.png` | Grad-CAM heatmaps (2 test images per class) |
| `best_model.keras` | Weights of the epoch with the lowest validation loss |

## Reproducibility

The script sets the Python, NumPy and TensorFlow seeds to 42 and enables TensorFlow op determinism, so
results should be repeatable on the same hardware and library versions. Different GPUs, TensorFlow
versions, or file orderings can change the numbers slightly. Results come from a single run; repeating
with several seeds would give a range.

## Limitations

- The Kaggle TRAIN and TEST images are augmented copies of the same source cells, so Kaggle test accuracy is likely optimistic.
- The validation set shares near-duplicates with the training set, so validation accuracy (up to 100%) overstates performance. The checkpoint is chosen on a very noisy validation curve.
- LISC_dataset overlaps the Kaggle training data (its 100% score is not evidence of generalization).
- The 347 original images are strongly imbalanced (20 monocytes, 33 lymphocytes, 206 neutrophils).
- Grad-CAM was checked on only 8 images and sometimes highlights red cells or image corners rather than the white cell.
- No confusion matrix is saved by the script (neutrophil confusions in the report are estimated from precision and recall).
- Not a medical device; for coursework only.

## Possible next steps

- Transfer learning with EfficientNet-B0 (proposed in Task 7 of the report, not implemented).
- Stain/colour augmentation, higher resolution or cell cropping.
- Save a confusion matrix; evaluate on genuinely independent data (other labs or patients).
- Further stabilize training (lower BatchNorm momentum, larger batch, milder augmentation).

## Report

The full write-up is in `BloodCell_CNN_Report_Updated.docx`.
