# Phase 1: Data Exploration & Preprocessing

## Overview
This phase explores the WorldStrat dataset structure, analyzes image properties, and prepares the data pipeline.

---

## Workflow

```
Step 1: Dataset Statistics     → 01_dataset_stats.py
Step 2: Image Analysis         → 02_image_analysis.py
Step 3: Metadata Analysis      → 03_metadata_analysis.py
Step 4: Visualization          → 04_visualize_samples.py
Step 5: Create Data Splits     → 05_create_splits.py
```

---

## Scripts

### 1. Dataset Statistics (`01_dataset_stats.py`)
Counts files, checks structure, validates HR-LR pairs.

```bash
python phase1_data_exploration/01_dataset_stats.py
```

**Output**: `outputs/dataset_stats.json`

---

### 2. Image Analysis (`02_image_analysis.py`)
Analyzes image dimensions, bit depth, pixel distributions.

```bash
python phase1_data_exploration/02_image_analysis.py
```

**Output**: `outputs/image_analysis.json`

---

### 3. Metadata Analysis (`03_metadata_analysis.py`)
Parses metadata.csv, analyzes geographic/temporal distribution.

```bash
python phase1_data_exploration/03_metadata_analysis.py
```

**Output**: `outputs/metadata_analysis.json`

---

### 4. Visualization (`04_visualize_samples.py`)
Creates sample visualizations of HR-LR pairs.

```bash
python phase1_data_exploration/04_visualize_samples.py
```

**Output**: `outputs/visualizations/`

---

### 5. Create Splits (`05_create_splits.py`)
Creates train/val/test splits (80/10/10).

```bash
python phase1_data_exploration/05_create_splits.py
```

**Output**: 
- `outputs/splits/train.txt`
- `outputs/splits/val.txt`
- `outputs/splits/test.txt`

---

## Dataset Structure

```
dataset/
├── hr_dataset/12bit/{location}/
│   ├── {location}_pan.tiff      # Panchromatic (grayscale)
│   ├── {location}_ps.tiff       # Pan-sharpened
│   ├── {location}_rgb.png       # RGB preview
│   └── {location}_rgbn.tiff     # RGB + NIR (4 bands)
│
├── lr_dataset/{location}/L2A/
│   ├── {location}-{n}-L2A_data.tiff    # Sentinel-2 bands
│   ├── {location}-{n}-CLM.tiff         # Cloud mask
│   ├── {location}-{n}-CLP.tiff         # Cloud probability
│   ├── {location}-{n}-dataMask.tiff    # Data mask
│   └── {location}-{n}.metadata         # Acquisition metadata
│
└── metadata.csv                  # Global metadata
```

---

## Expected Results

After running all scripts:
- Total number of HR-LR location pairs
- Image dimension statistics
- Bit depth and value range analysis
- Cloud coverage distribution
- Land cover class distribution
- Train/val/test split files


