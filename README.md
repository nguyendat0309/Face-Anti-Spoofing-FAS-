# Face Anti-Spoofing with Improved Deep Learning Models

## Project Summary

Presentation Attack Detection (PAD), commonly known as Face Anti-Spoofing (FAS), is a critical component for securing biometric facial recognition systems against unauthorized access attempts. This repository implements an end-to-end deep learning framework designed for robust binary classification: **Live** (genuine facial presentations) versus **Spoof** (fraudulent presentation attacks).

The framework specifically addresses diverse and challenging physical presentation attacks encountered in real-world scenarios, including:
- Printed photographs (flat and warped paper prints)
- Replayed videos displayed on mobile devices and tablet screens
- High-resolution screen display attacks
- Paper cut masks with eye/mouth openings
- 3D facial mask presentations

To achieve high discrimination capability while maintaining computational efficiency suitable for real-time deployment, this project investigates and enhances compact convolutional architectures—specifically **ResNet18** and **ConvNeXt-Tiny**. The baseline models are augmented with structural and representation-learning improvements:
1. **Generalized Mean (GeM) Pooling**: Replaces standard global average pooling to emphasize localized discriminative features.
2. **Convolutional Block Attention Module (CBAM)**: Incorporates dual channel and spatial attention mechanisms to focus on spoof-specific visual artifacts while suppressing background clutter.
3. **Supervised Contrastive Learning (SupCon)**: Optimizes the latent embedding space during training to maximize inter-class separation between live and spoof samples.

---

## Repository Structure

The repository is organized into a clean, modular hierarchy separating research experiments from modular inference pipelines:

```text
project/
├── demo/
│   ├── app_gradio.py
│   ├── run_image.py
│   ├── run_video.py
│   ├── config.yaml
│   ├── requirements.txt
│   ├── README.md
│   ├── checkpoints/
│   │   └── resnet18_best.pth
│   ├── src/
│   │   ├── detectors/
│   │   ├── models/
│   │   ├── pipeline/
│   │   └── utils/
│   ├── inputs/
│   └── outputs/
│
├── experiments/
│   ├── baseline/
│   ├── gem/
│   ├── cbam/
│   ├── supcon/
│   └── cbam_supcon/
│
├── assets/
├── archive/
├── .gitignore
└── README.md
```

---

## Dataset

The primary dataset utilized for training, validation, and experimental analysis in this project is a structured subset of the **CelebA-Spoof** dataset. 

Key characteristics of the training subset include:
- **Binary Formulation**: Samples are strictly categorized into **Live** (class 0) and **Spoof** (class 1) presentations.
- **Subject Diversity**: Contains diverse subject identities spanning multiple ethnicities, genders, and age groups to prevent subject-identity overfitting.
- **Environmental Variability**: Incorporates wide ranges of ambient lighting, indoor/outdoor backgrounds, and sensor acquisition noise.
- **Attack Modalities**: Encompasses multiple presentation attack instruments, ranging from print attacks under varying illumination conditions to digital replay attacks with visible screen moiré patterns and reflection artifacts.

---

## Method

Our methodology systematically incorporates structural modules and advanced loss functions into compact backbone architectures to improve feature discrimination against spoof artifacts.

```text
[Input Image (224x224)] ---> [Backbone: ResNet18 / ConvNeXt-Tiny]
                                       |
                                       v
                             [CBAM Attention Module] (Channel & Spatial Refinement)
                                       |
                                       v
                               [GeM Pooling Layer] (Localized Feature Aggregation)
                                       |
                   +-------------------+-------------------+
                   | (Training Only)                       | (Training & Inference)
                   v                                       v
       [SupCon Projection Head]                 [Linear Classifier Head]
                   |                                       |
                   v                                       v
          [Supervised Contrastive Loss]             [Binary Classification: Live vs Spoof]
```

### 1. Backbone Architectures
- **ResNet18**: Provides a compact, highly stable convolutional baseline with residual skip connections that mitigate vanishing gradients during deep feature extraction.
- **ConvNeXt-Tiny**: Investigates modern depthwise-separable convolution dynamics and inverted bottleneck designs for enriched visual representation.

### 2. Generalized Mean (GeM) Pooling
Standard Global Average Pooling (GAP) treats all spatial feature locations equally, which can dilute subtle localized spoof indicators (e.g., screen glare or border inconsistencies). GeM pooling introduces a learnable pooling parameter $p$:

$$f = \left( \frac{1}{|\mathcal{X}|} \sum_{x \in \mathcal{X}} x^p \right)^{\frac{1}{p}}$$

When $p \to 1$, GeM operates as average pooling; as $p \to \infty$, it approaches max pooling. This allows the network to adaptively focus on the most salient localized textural anomalies across facial regions.

### 3. Convolutional Block Attention Module (CBAM)
CBAM applies sequential attention refinement along two distinct dimensions:
- **Channel Attention**: Exploits inter-channel relationships using shared multi-layer perceptrons across average-pooled and max-pooled feature maps to emphasize feature channels sensitive to high-frequency spoof noise.
- **Spatial Attention**: Generates a 2D spatial weighting map to direct the network's focus toward informative facial boundaries and high-texture zones while suppressing irrelevant background interference.

### 4. Supervised Contrastive Learning (SupCon)
To prevent the model from memorizing superficial domain or lighting characteristics, we incorporate Supervised Contrastive Learning (`SupCon`) alongside standard cross-entropy optimization during training. The SupCon loss pulls latent embeddings of samples belonging to the same class (`Live` with `Live`, `Spoof` with `Spoof`) into tight clusters while pushing opposite classes far apart on a normalized hypersphere.

### 5. Inference Architecture Specification
For practical deployment and inference (`demo/`), the active model architecture is exclusively defined as:

$$\text{\textbf{ResNet18 + GeM + CBAM}}$$

> **Architectural Note**: The Supervised Contrastive (`SupCon`) projection head is utilized **strictly during the training phase** (`experiments/cbam_supcon/`) to regularize feature learning. When executing inference or real-time evaluation, the SupCon projection head (`projection_head.*` / `supcon_head.*`) is **completely discarded**. The inference model loads the learned backbone, CBAM attention weights, GeM pooling parameters, and linear classification head (`num_classes = 2`) directly from the checkpoint, ensuring optimal inference speed and minimal memory consumption.

---

## Experiments

All experimental investigations, model ablations, and comparative training runs are organized within the `experiments/` directory, structured into self-contained Jupyter Notebooks:

- `experiments/baseline/`: Implements baseline training and evaluation for standard ResNet18 and ConvNeXt-Tiny models using Cross-Entropy optimization (`resnet18_2class.ipynb`, `convnexttiny_2class.ipynb`).
- `experiments/gem/`: Evaluates the individual contribution of Generalized Mean Pooling replacing standard GAP (`resnet18_gem.ipynb`, `convnexttiny_gem.ipynb`).
- `experiments/cbam/`: Analyzes the impact of integrating dual channel and spatial attention modules on model convergence and feature selectivity (`resnet18_gem_cbam.ipynb`, `convnexttiny_gem_cbam.ipynb`).
- `experiments/supcon/`: Explores two-stage or joint representation learning governed by Supervised Contrastive Loss (`resnet18_gem_supcon.ipynb`, `convnexttiny_gem_supcon.ipynb`).
- `experiments/cbam_supcon/`: Represents the full proposed framework, combining GeM pooling, CBAM attention, and SupCon optimization into a unified training pipeline (`resnet18_gem_cbam_supcon.ipynb`, `convnexttiny_gem_cbam_supcon.ipynb`).

Each notebook includes complete data loading, data augmentation, training loops, validation monitoring, and evaluation routines.

---

## Results

Empirical evaluations across the experimental progression demonstrate consistent and cumulative improvements over standard convolutional baselines:

1. **Baseline vs. Attention Refinement**: Integrating CBAM attention enables the network to localize critical presentation attack regions—such as screen reflections and paper boundaries—reducing false acceptance rates on challenging digital replay and print attacks.
2. **Localized Aggregation via GeM**: Replacing standard GAP with GeM pooling consistently enhances classification stability when evaluating cropped facial regions containing varying amounts of background context or marginal lighting variations.
3. **Representation Quality via SupCon**: Models trained with Supervised Contrastive Learning exhibit superior feature separability in the latent space compared to models trained solely with cross-entropy loss. This structural regularization leads to higher overall classification accuracy, improved F1-scores, and more robust Receiver Operating Characteristic (ROC) curves.

Detailed numerical performance summaries, confusion matrices, loss curves, and ROC evaluations are generated directly upon executing the evaluation blocks inside the respective notebooks in `experiments/cbam_supcon/`.

---

## Demo Application

The `demo/` directory provides a modular, self-contained inference application engineered for immediate validation and demonstration. It supports static images, recorded videos, and interactive web-based experimentation.

### Key Components
- **Preprocessing & Detection**: Uses **SCRFD** as the primary face detection engine, configured with a dynamic fallback mechanism to **MediaPipe Face Detection** when severe pose variations or extreme lighting conditions obscure the primary detection.
- **Bounding Box Processing**: Faces are automatically cropped using an adjustable safety margin (default `margin: 1.35`) to ensure consistent spatial context surrounding the facial boundaries.
- **Inference Pipeline (`fas_predictor.py`)**: Automatically filters out inactive `SupCon` projection head parameters during initialization, runs fast forward pass evaluation (`ResNet18 + GeM + CBAM`), and outputs calibrated probabilities (`prob_live`, `prob_spoof`).
- **Gradio Web UI (`app_gradio.py`)**: Provides an interactive browser interface featuring dedicated tabs for single-image analysis and video frame processing, with real-time sliders for classification thresholds (`live_threshold: 0.35`, `spoof_threshold: 0.65`) and temporal smoothing.
- **Video Temporal Smoothing (`smoothing.py`)**: Implements a sliding-window temporal smoother (`window = 5`) to stabilize frame-by-frame prediction variance during video inference, alongside automated CSV prediction tracking (`frame_idx`, `prob_live`, `prob_spoof`, `pred_name`).

---

## Quick Start

### 1. Environment Setup
Navigate to the `demo/` directory and install the required dependencies:

```bash
cd demo
pip install -r requirements.txt
```

### 2. Verify Model Checkpoint
Ensure the trained model weights file (`resnet18_best.pth`) is placed inside the `demo/checkpoints/` folder:

```bash
ls checkpoints/resnet18_best.pth
```

### 3. Command-Line Image Inference
To run inference on a single test image or a batch directory of images:

```bash
python run_image.py --input inputs/images/test.jpg --output outputs/images/test_result.jpg --config config.yaml
```

### 4. Command-Line Video Inference
To process a video file, draw bounding box annotations, and generate frame-by-frame CSV logs:

```bash
python run_video.py --input inputs/videos/test.mp4 --output outputs/videos/test_result.mp4 --config config.yaml
```

### 5. Launch Gradio Web Interface
To start the interactive web application:

```bash
python app_gradio.py --config config.yaml
```

Upon execution, open the displayed local server URL (default: `http://127.0.0.1:7861`) in a web browser.

---

## Configuration

All inference parameters across CLI scripts and the Web UI are governed centrally by `demo/config.yaml`. Key configurations include:

```yaml
model:
  name: resnet18_gem_cbam
  checkpoint_path: checkpoints/resnet18_best.pth
  num_classes: 2
  image_size: 224
  device: auto

detector:
  name: scrfd
  fallback: mediapipe
  confidence_threshold: 0.3

crop:
  margin: 1.35
  save_crops: true

prediction:
  live_label: 0
  spoof_label: 1
  live_threshold: 0.35
  spoof_threshold: 0.65
  uncertain_enabled: true

video:
  smoothing_window: 5
  sample_fps: 5
  save_frame_log: true

output:
  image_dir: outputs/images
  video_dir: outputs/videos
  crop_dir: outputs/crops
  log_dir: outputs/logs
  save_crops: true
  save_csv: true
```

---

## Checkpoint and Large Files

To maintain repository cleanliness and ensure efficient version control, all large binary files—including model weight checkpoints (`*.pth`, `*.pt`, `*.ckpt`), processed output videos (`*.mp4`, `*.avi`), intermediate image crops, and log data—are excluded from Git tracking via `.gitignore`.

When cloning this repository to a new environment or preparing for project evaluation, you must place the primary fine-tuned model checkpoint at the exact relative path below:

```text
demo/checkpoints/resnet18_best.pth
```

If the checkpoint contains legacy state keys from the contrastive training stage (`projection_head.*` or `supcon_head.*`), the inference loader in `demo/src/pipeline/fas_predictor.py` will automatically strip these keys during model initialization.

---

## Future Work

Future directions to extend the capabilities of this research include:
1. **Lightweight Vision Transformers**: Investigating hybrid convolutional-transformer architectures (e.g., MobileViT, Swin-Tiny) to capture global spatial dependencies while maintaining compatibility with edge devices and mobile processors.
2. **Frequency-Domain Analysis**: Incorporating multi-stream Fourier or Wavelet transform feature extractors to detect subtle digital display moiré patterns and high-frequency generative artifacts that are less apparent in spatial RGB channels.
3. **Temporal Attention Mechanisms**: Extending video inference from frame-wise temporal smoothing to 3D convolutional or transformer-based sequence modeling (e.g., TimeSformer) for exploiting spatio-temporal dynamics across consecutive video frames.

---

## Project Status

The repository has completed restructuring and architectural cleaning. The training and research experiments (`experiments/`) are standardized and reproducible, and the standalone inference pipeline (`demo/`) is fully modular, verified, and operational for academic evaluation and mini-project defense.
