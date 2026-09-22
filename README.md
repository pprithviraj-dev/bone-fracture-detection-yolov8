# Bone Fracture Detection

An end-to-end computer vision project for detecting and localizing possible bone fracture regions from musculoskeletal X-ray images using YOLOv8s.

The project includes model fine-tuning on the FracAtlas dataset, validation analysis, and a Streamlit-based web application for inference.

## Project Overview

This project uses a YOLOv8s object detection model to identify possible fracture regions in X-ray images.

The complete workflow includes:

- Dataset analysis and preparation
- YOLOv8s fine-tuning
- Validation and performance evaluation
- Fracture localization using bounding boxes
- Streamlit web application for image-based inference

## Dataset

This project uses the **FracAtlas** dataset.

FracAtlas is a musculoskeletal radiograph dataset containing images with fracture annotations for classification, localization, and segmentation tasks.

### Dataset Statistics

- Total X-ray images: **4,083**
- Fractured images: **717**
- Non-Fractured images: **3,366**
- Fracture instances: **922** (paper-reported)
- Anatomical regions: Hand, Shoulder, Leg and Hip

### Dataset Credit

**FracAtlas: A Dataset for Fracture Classification, Localization and Segmentation of Musculoskeletal Radiographs**

Dataset and paper:

https://doi.org/10.1038/s41597-023-02459-5

The original dataset is provided under the **CC BY 4.0** license.

Please refer to the original dataset publication for complete dataset information and attribution requirements.

## Model

The project uses:

- **YOLOv8s**
- Object Detection
- Single detection class: `fractured`

The model was fine-tuned using the official FracAtlas train/validation/test split.

## Fine-Tuning

The complete fine-tuning and dataset preparation workflow is available in:

`fractatlas-prithvi.ipynb`

The notebook contains:

- Dataset loading
- Exploratory data analysis
- Official dataset split
- YOLO dataset preparation
- Preprocessing and augmentation
- YOLOv8s model configuration
- Model training
- Validation
- Performance evaluation

## Validation Results

Final validation performance of the selected YOLOv8s model:

| Metric | Score |
|---|---:|
| Precision | 0.593 |
| Recall | 0.538 |
| mAP@50 | 0.555 |
| mAP@50–95 | 0.248 |

The validation plots and evaluation outputs are available in:

`validation_plots_8x6_300dpi/`

These include the generated validation and metric visualization files from the final model evaluation.

## Streamlit Application

The project includes a Streamlit application for interactive inference.

### Live Demo

**[Open Live Streamlit App](https://bone-fracture-detection-yolov8-qbyc5xv499hrxr4o4csp6z.streamlit.app/)**

### Features

- Upload an X-ray image
- Detect possible fracture regions
- Display bounding-box localization
- Show detection count
- Replace the uploaded image
- Run inference through a simple web interface

### Application Structure

```text
fracture_streamlit_app/
├── app.py
├── best.pt
└── requirements.txt

## Medical Disclaimer

This project is an AI/ML demonstration developed for educational and research purposes.

The model was trained on a limited dataset and should **not** be used for medical diagnosis, treatment decisions, or clinical decision-making.

Results produced by the application should not be considered a substitute for assessment by a qualified medical professional.

## Future Improvements

Possible future improvements include:

- Larger and more diverse training datasets
- Improved fracture localization
- Additional anatomical regions
- Model optimization for deployment
- More extensive external validation

## Author

**Prithvi Raj**

This project was developed as a computer vision and machine learning portfolio project.
