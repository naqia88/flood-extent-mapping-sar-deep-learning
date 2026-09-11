# Flood Extent Mapping Using SAR Satellite Imagery and Deep Learning Segmentation

## Overview

This project implements an end-to-end deep learning pipeline for automated flood extent mapping using **Sentinel-1 Synthetic Aperture Radar (SAR)** imagery.

The project uses the **Sen1Floods11** benchmark dataset and compares two deep learning segmentation architectures, **U-Net** and **UNet++**, against a traditional SAR intensity thresholding baseline.

The pipeline covers SAR preprocessing, patch extraction, semantic segmentation, model training, quantitative evaluation, and flood-map visualization.

---

## Objectives

* Detect flooded areas from Sentinel-1 SAR imagery
* Perform pixel-level flood segmentation using deep learning
* Compare U-Net and UNet++ segmentation architectures
* Establish a traditional SAR thresholding baseline
* Evaluate models using IoU, F1-score, Precision, and Recall
* Produce visual flood extent maps for disaster-response applications

---

## Dataset

### Sen1Floods11

**Sen1Floods11** is a georeferenced benchmark dataset containing real flood events observed using Sentinel-1 SAR imagery.

* 11 real flood events
* Global geographic coverage
* Sentinel-1 Ground Range Detected (GRD) SAR imagery
* Hand-labeled flood masks
* Official training, validation, and testing splits
* GeoTIFF format
* 10 m spatial resolution

The project uses:

* `S1GRDHand` — Sentinel-1 SAR imagery
* `LabelHand` — hand-labeled flood masks

Dataset source:

https://huggingface.co/datasets/blumenstiel/Sen1Floods11

Original dataset paper:

> Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, E. (2020). *Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1*. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops.

---

## Methodology

```text
Sen1Floods11 Dataset
        |
        v
Sentinel-1 SAR GeoTIFF Images
        |
        v
SAR Preprocessing
(percentile normalization + NaN/Inf handling)
        |
        v
256 × 256 Patch Extraction
        |
        v
Official Train / Validation / Test Splits
        |
        +-----------------------+
        |                       |
        v                       v
     U-Net                  UNet++
  ResNet50 Encoder        ResNet50 Encoder
        |                       |
        +-----------+-----------+
                    |
                    v
              Model Evaluation
                    |
                    v
        IoU / F1 / Precision / Recall
                    |
                    v
          Flood Map Visualization
                    |
                    v
        Comparison with SAR Threshold
```

---

## Models

### 1. U-Net

A U-Net semantic segmentation architecture with a **ResNet50 encoder** pretrained on ImageNet.

U-Net uses encoder-decoder connections to preserve spatial information while learning high-level image features.

### 2. UNet++

UNet++ extends U-Net using **nested and dense skip connections**, allowing improved feature propagation between encoder and decoder stages.

The model uses the same **ResNet50 encoder** for a fair comparison.

### 3. SAR Threshold Baseline

A traditional intensity-based SAR thresholding method is used as a non-deep-learning baseline.

This provides a reference point for evaluating whether deep learning provides a meaningful improvement over conventional SAR flood detection.

---

## Model Configuration

| Model         | Architecture           | Encoder  | Input |
| ------------- | ---------------------- | -------- | ----- |
| U-Net         | Encoder-Decoder        | ResNet50 | SAR   |
| UNet++        | Nested Encoder-Decoder | ResNet50 | SAR   |
| SAR Threshold | Intensity Thresholding | —        | SAR   |

### Training Configuration

| Parameter         | Value                      |
| ----------------- | -------------------------- |
| Patch Size        | 256 × 256                  |
| Loss Function     | BCE + Dice Loss            |
| Optimizer         | AdamW                      |
| Learning Rate     | 1e-4                       |
| Weight Decay      | 1e-4                       |
| Scheduler         | CosineAnnealingLR          |
| Epochs            | 15                         |
| Batch Size        | 8                          |
| Data Augmentation | Horizontal + Vertical Flip |
| Hardware          | Google Colab GPU           |

---

## Results

The current experimental results are:

| Model           |        IoU |         F1 | Precision |     Recall |
| --------------- | ---------: | ---------: | --------: | ---------: |
| SAR Threshold   |     0.2298 |     0.3737 |    0.3050 |     0.4824 |
| U-Net ResNet50  |     0.5706 |     0.7266 |    0.6940 |     0.7625 |
| UNet++ ResNet50 | **0.5727** | **0.7283** |    0.6689 | **0.7992** |

### Key Findings

* **UNet++ achieves the highest IoU and F1-score**, with an IoU of **0.5727** and F1-score of **0.7283**.
* UNet++ substantially outperforms the traditional SAR thresholding baseline.
* UNet++ achieves the highest recall (**0.7992**), indicating strong sensitivity to flooded pixels.
* U-Net and UNet++ produce very similar overall IoU and F1 performance.
* Both deep learning approaches substantially outperform the traditional SAR threshold baseline.
* The results demonstrate the potential of deep learning-based semantic segmentation for automated flood extent mapping.

### Improvement Over Threshold Baseline

UNet++ improves IoU from **0.2298 to 0.5727**, representing an absolute improvement of approximately **0.343 IoU points**.

---

## Qualitative Results

The notebook generates visual comparisons between:

1. Original SAR imagery
2. Ground-truth flood mask
3. Predicted flood mask
4. Flood extent overlay

These visualizations help assess how accurately the models delineate flooded regions and where false positives or false negatives occur.

---

## Limitations

### Multi-temporal Analysis

The current experiment focuses on the labeled Sen1Floods11 flood imagery and does not yet implement a dedicated **pre-event vs post-event SAR change-detection pipeline**.

Future work can incorporate paired pre-flood and post-flood Sentinel-1 observations.

### Explainability

Model explainability has not yet been fully integrated into the current experiment.

Future versions can include methods such as:

* Grad-CAM
* Integrated Gradients
* Attention visualization
* Feature activation maps

These techniques could help disaster-response users understand which SAR regions contribute to model predictions.

### Model Architectures

The current comparison focuses on U-Net and UNet++ with ResNet50 encoders.

Future experiments could investigate:

* Attention U-Net
* EfficientNet encoders
* ConvNeXt
* Swin Transformer
* Vision Transformer-based segmentation
* Transformer-enhanced U-Net architectures

### Test-Time Augmentation

Test-time augmentation could also be investigated to determine whether predictions can be improved through averaging predictions from transformed versions of the same SAR image.

---

## Technologies and Libraries

| Library                     | Purpose                                  |
| --------------------------- | ---------------------------------------- |
| PyTorch                     | Deep learning and model training         |
| segmentation-models-pytorch | U-Net and UNet++                         |
| Rasterio                    | GeoTIFF and geospatial raster processing |
| NumPy                       | Numerical processing                     |
| Pandas                      | Results and data management              |
| Scikit-learn                | Evaluation metrics                       |
| Matplotlib                  | Visualization                            |
| tqdm                        | Training and processing progress         |

---

## Project Structure

```text
Flood-Extent-Mapping/
│
├── Flood_Extent_Mapping_Using_SAR_Satellite_Imagery_and_Deep_Learning_Segmentation.ipynb
│
├── flood_mapping_results.csv
│
├── best_unet_resnet50.pth
│
├── best_unetpp_resnet50.pth
│
└── Sen1Floods11/
    └── sen1floods11_v1.1/
        ├── data/
        ├── splits/
        └── Sen1Floods11_Metadata.geojson
```

---

## How to Run

### 1. Open the Notebook

Open the notebook in **Google Colab**.

A GPU runtime is recommended for model training.

### 2. Install Dependencies

The notebook installs the required Python packages automatically.

### 3. Prepare the Dataset

Download the Sen1Floods11 dataset and place it in the expected directory:

```text
/content/Sen1Floods11/sen1floods11_v1.1
```

### 4. Run the Notebook

Execute the cells from top to bottom.

The notebook performs:

```text
Dataset Loading
      ↓
SAR Preprocessing
      ↓
Patch Extraction
      ↓
Model Training
      ↓
Validation
      ↓
Testing
      ↓
Metric Calculation
      ↓
Visualization
      ↓
Model Comparison
```

### 5. GPU Recommendation

A Google Colab GPU is recommended, particularly for training U-Net and UNet++.

Because the dataset contains many image patches, the notebook uses **lazy loading** to reduce RAM usage during training.

---

## Reproducibility

The experiments use a fixed random seed for reproducibility:

```python
SEED = 42
```

The official Sen1Floods11 split files are used for training, validation, and testing.

---

## Research Significance

Flood mapping is an important component of disaster monitoring and emergency response.

Traditional SAR thresholding methods can be affected by:

* Complex terrain
* Vegetation
* Urban structures
* Surface characteristics
* SAR backscatter variability

Deep learning segmentation models can learn spatial and contextual patterns directly from SAR imagery, potentially providing more robust flood extent predictions.

This project investigates that potential by comparing deep learning segmentation against a traditional SAR thresholding baseline.

---

## Future Work

The next stage of the project will focus on extending the current pipeline toward a more comprehensive research framework:

* Pre-event vs post-event SAR change detection
* Attention-based segmentation models
* Explainable AI for flood segmentation
* Multi-event performance analysis
* Per-event IoU and F1 reporting
* Confidence and uncertainty estimation
* Transformer-based segmentation models
* Geospatial flood-area estimation
* Improved visualization of predicted flood extent
* Publication-ready statistical analysis

---

## References

1. Bonafilia, D., Tellman, B., Anderson, T., & Issenberg, E. (2020). *Sen1Floods11: A Georeferenced Dataset to Train and Test Deep Learning Flood Algorithms for Sentinel-1*. IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops.

2. Sen1Floods11 Dataset:
   https://huggingface.co/datasets/blumenstiel/Sen1Floods11

3. Segmentation Models PyTorch:
   https://github.com/qubvel/segmentation_models.pytorch

---

## Author

**Miss Firdous**

Undergraduate Data Science Student

**Project Area:** Deep Learning • Remote Sensing • Semantic Segmentation • Disaster Mapping

---

## Acknowledgements

This project uses the **Sen1Floods11** dataset developed for flood mapping research using Sentinel-1 SAR imagery.

Special thanks to the researchers who developed and released the dataset for the remote sensing and deep learning research community.
