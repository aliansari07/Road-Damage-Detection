# Road Damage Detection on Karachi Roads (CS-4045)

A lightweight Ultralytics YOLO pipeline for detecting and localizing potholes and road cracks in Karachi using smartphone photography. This project evaluates cross-domain generalization between the multi-national RDD2022 dataset and a custom, segment-split dataset of local Karachi streets.

## Overview

Manual road inspection is slow, labor-intensive, and expensive. Developed as a coursework proof-of-concept for DLP CS-4045, this project explores how a compact object detection model (`YOLOv8n`) can serve as a low-cost screening aid to help prioritize road maintenance from ordinary smartphone photos.

The experiments are designed to answer two main questions:

1. **Local Detection Feasibility:** How reliably can a lightweight YOLO model locate potholes and surface cracks in everyday phone imagery?  
2. **Cross-Domain Generalization:** How well does a detector trained on foreign road datasets (`RDD2022`) transfer to the lighting, textures, and surface conditions of Karachi roads?

### Target Classes

To keep annotations consistent and aligned with practical repair workflows, damage types are consolidated into two classes:

* **`0: pothole`** — Surface depressions and cavities (mapped from `D40` in RDD2022).  
* **`1: crack`** — Linear and web-like surface fractures, combining longitudinal (`D00`), transverse (`D10`), and alligator (`D20`) cracks into a single class.

## Team and Task Allocation

**Course:** DLP CS-4045 \| **Section:** BCS-7A \| **Semester:** Fall 2026

| Name | Student ID | Email | Role and Responsibilities |
| :---- | :---- | :---- | :---- |
| **Muhammad Ali Ansari** | 22K-4135 | [k224135@nu.edu.pk](mailto:k224135@nu.edu.pk) | **Group Coordinator** — Data cleaning, local bounding-box annotation (LabelImg/CVAT), intra-annotator IoU consistency checks, model evaluation, error analysis, and presentation. |
| **Ukasha Zulfiqar** | 22K-4485 | [k224485@nu.edu.pk](mailto:k224485@nu.edu.pk) | **Team Member** — Karachi street data collection, RDD2022 preprocessing, YOLOv8 model training, and technical report writing. |

## Datasets and Splitting Strategy

Every model variant is evaluated against the same held-out **Karachi Test Set** to ensure a fair comparison across training regimes.

### 1\. Public Dataset (`RDD2022` Subset)

* **Source:** Multi-national Road Damage Dataset 2022 ([Arya et al., 2022](https://figshare.com/articles/dataset/RDD2022_-_The_multi-national_Road_Damage_Dataset_released_through_CRDDC_2022/21431547)).  
* **Composition:** \~3,000 labeled images drawn primarily from the **India** subset (which shares similar road wear and environmental characteristics with Karachi), supplemented with samples from Japan, the Czech Republic, Norway, the United States, and China.  
* **Split:** 2,400 training images / 600 validation images.

### 2\. Custom Karachi Dataset (`karachi_own`)

* **Collection:** \~400 smartphone photographs captured across 10 to 12 distinct road segments around campus and nearby streets in Karachi.  
* **Conditions and Hard Negatives:** Captured from the roadside or as a passenger under varying lighting, heavy shadows, wet and dry surfaces, and both near and far viewing distances. Deliberately includes hard negatives such as manhole covers, asphalt patches, speed bumps, and faded road markings.  
* **Segment-Based Splitting:** Split strictly **by road segment** rather than random frame shuffling. This prevents near-identical consecutive frames from the same street from leaking across training and evaluation sets:  
  * **Train:** \~240 images  
  * **Validation:** \~60 images  
  * **Test:** \~100 images *(Standard benchmark used to evaluate all models)*

### 3\. Combined Dataset (`combined`)

* Merges the training sets (`2,400 + 240 = 2,640` images) and validation sets (`600 + 60 = 660` images) from both datasets while keeping the 100-image Karachi test set completely isolated.

## Repository Structure

```
CS4045_Road_Damage_Detection/
├── dlp_project.py               # Main Google Colab training and evaluation script
├── experiment_results.csv       # Automated metric logs across all runs
├── README.md                    # Project documentation
├── dataset/
│   ├── public_rdd/              # RDD2022 subset (YOLO format + data.yaml)
│   ├── karachi_own/             # Segment-split Karachi dataset (YOLO format + data.yaml)
│   └── combined/                # Merged Public + Karachi dataset (YOLO format + data.yaml)
└── runs/                        # Ultralytics weights, training curves, and validation outputs
    ├── baseline_coco_eval/
    ├── yolo_public_only/
    └── eval_public_only_on_karachi/
```

## Experimental Pipeline

The training and evaluation workflow in `dlp_project.py` compares four configurations on the **Karachi Test Set** using `YOLOv8n` (`imgsz=640`, `batch=16`):

1. **Zero-Shot Baseline:** Pretrained COCO `yolov8n.pt` evaluated directly on the Karachi test set. Since COCO does not include pothole or crack classes, this establishes a near-zero reference floor.  
2. **Experiment 1 (Public Only):** Fine-tuned exclusively on the `RDD2022` subset (2,400 train / 600 val) to test how well foreign road data transfers to Karachi streets.  
3. **Experiment 2 (Local Only):** Fine-tuned exclusively on the `karachi_own` training split (240 train / 60 val).  
4. **Experiment 3 (Combined):** Fine-tuned on the combined dataset (`public_rdd` \+ `karachi_own`) to evaluate whether adding foreign data improves local generalization.

### Evaluation and Error Analysis

Models are benchmarked using **Precision**, **Recall**, **mAP@50**, and **mAP@50-95** (both overall and per class). Beyond aggregate metrics, predictions are reviewed qualitatively to categorize false positives (e.g., shadows, manhole covers, road patches) and missed detections (e.g., thin hairline cracks, glare on wet asphalt, distant damage) across different lighting and distance conditions.

## Getting Started

1. **Environment Setup:** Open `dlp_project.py` (or its corresponding `.ipynb` notebook) in Google Colab with a GPU runtime enabled (`Runtime > Change runtime type > T4 GPU`).  
2. **Install Dependencies:**

```sh
pip install -q ultralytics roboflow opencv-python-headless pyyaml matplotlib seaborn
```

3. **Prepare Datasets:**  
   * Place your annotated YOLO images and `.txt` label files into `dataset/public_rdd` and `dataset/karachi_own` matching the folder layout above.  
   * Alternatively, run the synthetic data generator cells in `dlp_project.py` to verify that the directory paths, YAML configs, and training loops execute properly before loading real imagery.  
4. **Run Experiments:** Execute the cells sequentially. Evaluation scores are automatically logged to `experiment_results.csv`, and visual predictions are saved inside `runs/`.

## Privacy, Ethics, and Limitations

* **Privacy and Safety:** All local photographs were taken from the roadside or from the passenger seat—never while driving. Any frames containing faces, vehicle license plates, ID cards, or private screens were cropped or deleted, and EXIF location metadata was stripped from all files.  
* **Local Annotation Governance:** Bounding-box annotation was carried out locally using LabelImg/CVAT without uploading private street photos to public cloud platforms or third-party AI tools. A provenance log records the source, capture date, photographer, annotator, and permissions for every image.  
* **Known Limitations:** Because this is a coursework study built with a single smartphone camera, one primary annotator (validated via a 10% re-annotation IoU check after one week), and a \~400-image local dataset, results should be interpreted as an exploratory proof-of-concept rather than a production-ready system.  
* **AI Assistance Disclosure:** AI coding tools were used only for boilerplate setup and debugging; all scripts, data splits, and experimental evaluations were reviewed and verified by the authors.

## References

If you use the public dataset referenced in this repository, please cite the original RDD2022 publication:

```bibtex
@article{arya2024rdd2022,
  title={RDD2022: A multi-national image dataset for automatic road damage detection},
  author={Arya, Deeksha and Maeda, Hiroya and Ghosh, Sanjay Kumar and Toshniwal, Durga and Sekimoto, Yoshihide},
  journal={Geoscience Data Journal},
  volume={11},
  number={4},
  pages={846--862},
  year={2024},
  publisher={Wiley Online Library}
}
```