# YOLO11 Face Recognition System for Challenging Conditions

This project presents a comprehensive evaluation of YOLO11 neural networks for face recognition tasks under adverse conditions, including harsh lighting scenarios and facial obstructions. The system implements a two-stage pipeline utilizing separate YOLO11 models for face detection and identity classification. Through systematic experimentation with data augmentation techniques and hyperparameter optimization, I investigate the robustness and performance boundaries of modern architectures when applied to challenging face recognition scenarios.

## System Architecture

The system uses a two-stage approach where each stage has a specialized purpose:

#### Stage 1: Face Detection
- **Model**: Fine-tuned YOLO11 detection model
- **Objective**: Robust face localization under varying conditions
- **Output**: Bounding box coordinates with confidence scores

#### Stage 2: Identity Classification  
- **Model**: Multiple YOLO11 classification models (experimental variants)
- **Objective**: Person identification from detected face regions
- **Output**: Identity predictions with confidence scores

### Experimental Design

#### Model Training Strategy

**Detection Model**:
- Single fine-tuned YOLO11 model optimized for face detection
- Focus on robust detection across lighting and occlusion scenarios

**Classification Models**:
- Multiple experimental variants with different configurations:
  - Baseline model with standard augmentations
  - Enhanced augmentation models (HSV manipulation, geometric transforms)
  - Specialized models for specific challenging conditions
- Systematic hyperparameter exploration for optimal performance

**Processing Flow:**
```
Input Image → [YOLO11 Detection] → Face Bounding Boxes → [Crop Faces] → 
Face Regions → [YOLO11 Classification] → Identity + Confidence → Final Result
```

This separation allows me to:
- Train detection and classification models independently
- Optimize each stage for specific challenging conditions
- Easily swap classification models for comparative experiments

## Project Structure

```
yolo-faceID/
├── yolo_face_recognizer.py           # Main recognition pipeline
├── yolo_config.py                    # Configuration
├── analyze_logs.py                   # Model performance analysis tool
├── analyze_yolo_classification_runs.py  # Training analysis tool
├── models/
│   ├── yolo11_detection_model        # Detection model
│   └── yolo11l_cls_models/           # Classification models
        ├── main/
        └── warmup/         
└── recognition_runs/                 # Experimental results
    ├── model_timestamp-results/
    │   ├── person_folders/
    │   ├── unknown/
    │   ├── multiple_faces/
    │   └── recognition_log.csv
    └── analysis_results/
```

### Core Scripts

**`yolo_face_recognizer.py`** - The main recognition engine
- Implements the two-stage detection and classification pipeline
- Supports both real-time camera processing and batch image analysis
- Automatically organizes results by recognized individuals
- Generates comprehensive CSV logs for each experimental run

**`yolo_config.py`** - Centralized configuration system
- Model paths and identifiers
- Processing parameters and thresholds
- Output formatting and organization settings

**`analyze_logs.py`** - Performance analysis tool
- Compares recognition rates across different model configurations
- Generates statistical summaries and visualizations
- Produces detailed reports on model performance under various conditions

**`analyze_yolo_classification_runs.py`** - Training analysis framework
- Analyzes multiple training experiments with different hyperparameters
- Creates interactive HTML reports comparing training curves
- Helps identify optimal configurations for challenging conditions

## Experimental Models

### Detection Model
I fine-tuned a single YOLO11 detection model specifically for robust face detection across various lighting conditions and partial occlusions.

### Classification Models
I trained multiple YOLO11 classification models with different configurations to handle challenging conditions. Some models were trained using a **warmup strategy** to ensure stable convergence and optimal performance:

- **Baseline Model**: Standard augmentations for controlled conditions
- **Enhanced Lighting Model**: Optimized HSV augmentations for lighting robustness
- **Occlusion-Robust Model**: Special augmentations including random erasing
- **General Robust Model**: Combined augmentation strategies for overall robustness

Key augmentation parameters (example below) I experimented with:
```python
hsv_s: 0.7          # HSV saturation adjustment for lighting robustness  
hsv_v: 0.4          # HSV value adjustment for brightness variations
degrees: 10         # Rotation augmentation for pose variation
scale: 0.5          # Scale variation for distance robustness
shear: 0.1          # Shear transformation for geometric robustness
perspective: 0.0005 # Perspective transformation
fliplr: 0.5         # Horizontal flip probability
erasing: 0.4        # Random erasing for occlusion simulation
```

The warmup training strategy helped achieve better convergence by gradually increasing learning rates during initial epochs, which was particularly important for the challenging condition datasets.

## Quick Setup and Usage

### Installation

```bash
# Install required dependencies
pip install ultralytics opencv-python pandas matplotlib seaborn plotly pyyaml
```

### Configuration

Edit `yolo_config.py` to specify your model paths:
```python
FACE_RECOGNITION_MODEL_NAME = 'yolo_recognizer_final_augmentations'
YOLO_DETECTION_MODEL_PATH = 'models/face_detector_model.pt'
YOLO_RECOGNITION_MODEL_PATH = 'models/yolo11l_recognizer_*/weights/model.pt'
YOLO_RECOGNITION_CONFIDENCE_THRESHOLD = 0.70
```

### Basic Usage

#### Real-Time Testing
```bash
python yolo_face_recognizer.py -s live [-vs CAMERA_INDEX] [-t THRESHOLD]
```

**Parameters**:
- `-vs`: Camera source selection (0=primary, 1=secondary)
- `-t`: Confidence threshold for challenging condition testing (0.5=permissive, 0.9=strict)

**Use Cases**:
- Interactive performance demonstration
- Real-time robustness evaluation
- Qualitative assessment under varying lighting conditions

#### Batch Processing
```bash
python yolo_face_recognizer.py -s folder -i /path/to/test_dataset [-t THRESHOLD]
```

**Output Structure**:
```
recognition_runs/
└── yolo__recognizer_final_augmentations_YYYY-MM-DD_HH-MM-SS-results/
    ├── person_1/                    # Successfully recognized individuals
    ├── person_2/
    ├── unknown/                     # Sub-threshold detections
    ├── multiple_faces/              # Multi-face scenarios
    └── recognition_log.csv          # Detailed performance metrics
```

#### Systematic Threshold Analysis
```bash
# Evaluate performance across confidence thresholds
for threshold in 0.5 0.6 0.7 0.8 0.9; do
    python yolo_face_recognizer.py -s folder -i challenging_dataset/ -t $threshold
done
```

### Performance Analysis and Model Comparison

#### Recognition Performance Analysis
```bash
python analyze_logs.py -i recognition_runs/
```

**Generated Outputs**:
- **`summary_report.txt`**: Comprehensive performance analysis including:
  - Model-level recognition rates and confidence statistics
  - Individual run breakdowns with per-person accuracy
  - Comparative analysis across different experimental conditions

- **`plot_status_comparison.png`**: Visual comparison of recognition statuses
- **`plot_confidence_distribution.png`**: Statistical confidence analysis

#### Training Run Analysis
```bash
python analyze_yolo_classification_runs.py --runs-dir training_experiments/ --out-dir analysis_results/
```

**Generated Outputs**:
- **`report.html`**: Interactive analysis dashboard featuring:
  - Training curve comparisons (accuracy, loss progression)
  - Hyperparameter correlation analysis
  - Performance metric summaries
- **`summary.csv`**: Tabulated experimental results for statistical analysis