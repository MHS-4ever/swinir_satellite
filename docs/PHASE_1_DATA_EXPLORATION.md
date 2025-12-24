# Phase 1: Data Exploration & Preprocessing

## Overview
This phase focuses on understanding the WorldStrat dataset structure, exploring data characteristics, and preparing the preprocessing pipeline for satellite imagery super-resolution.

**Status**: ✅ COMPLETED

---

## Objectives
- [x] Understand dataset structure and organization
- [x] Analyze image properties (resolution, bit depth, spectral bands)
- [x] Identify data quality issues and anomalies
- [x] Create train/validation/test splits

---

## Dataset Information

### Source
- **Dataset**: WorldStrat Dataset
- **URL**: https://www.kaggle.com/datasets/jucor1/worldstrat
- **Description**: Multi-temporal Sentinel-2 (LR) paired with commercial satellite imagery (HR)

### Directory Structure
```
dataset/
├── hr_dataset/12bit/{location}/
│   ├── {location}_pan.tiff      # Panchromatic (1054×1054, 1 band)
│   ├── {location}_ps.tiff       # Pan-sharpened (1054×1054, 4 bands)
│   ├── {location}_rgb.png       # RGB preview (1054×1054, 3 bands)
│   └── {location}_rgbn.tiff     # RGBN native (263×263, 4 bands)
├── lr_dataset/{location}/L2A/
│   ├── {location}-{n}-L2A_data.tiff    # Sentinel-2 (12 bands)
│   ├── {location}-{n}-CLM.tiff         # Cloud mask
│   ├── {location}-{n}-CLP.tiff         # Cloud probability
│   └── {location}-{n}.metadata         # Acquisition metadata
└── metadata.csv
```

---

## Results

### Dataset Statistics

| Metric | Value |
|--------|-------|
| Total HR locations | 3,929 |
| Total LR locations | 3,928 |
| **Matched HR-LR pairs** | **3,928** |
| Orphan HR (no LR) | 1 ("testing on one Amnesty POI") |
| Total HR files | 15,716 (11,787 TIFF + 3,929 PNG) |
| Total LR files | 201,504 (170,080 TIFF + 31,424 metadata) |
| LR acquisitions per location | 8 (consistent) |
| Metadata rows | 62,848 (16 per location) |

### Image Properties

#### High-Resolution Images
| Type | Resolution | Bands | Dtype | Value Range |
|------|------------|-------|-------|-------------|
| pan (panchromatic) | 1054×1054 | 1 | uint16 | 0-17,090 |
| ps (pan-sharpened) | 1054×1054 | 4 (RGBN) | uint16 | 0-20,877 |
| rgbn (native) | 263×263 | 4 (RGBN) | uint16 | 0-13,836 |
| rgb (preview) | 1054×1054 | 3 (RGB) | uint8 | 0-255 |

#### Low-Resolution Images (Sentinel-2)
| Type | Resolution | Bands | Dtype | Value Range |
|------|------------|-------|-------|-------------|
| L2A_data | ~157×159 (varies) | 12 | float32 | 0.0-1.875 |
| CLM (cloud mask) | ~155×161 | 1 | float32 | 0-1 (binary) |
| CLP (cloud prob) | ~155×161 | 1 | float32 | 0-252 |
| dataMask | ~155×161 | 1 | float32 | 1.0 (constant) |

#### Scale Factor Analysis
- HR ps/pan resolution: **1054×1054**
- LR Sentinel-2 resolution: **~157×159**
- **Effective scale factor: ~6.7x** (between 4x and 8x)
- Recommendation: Use **4x** scale (crop HR to 628×628) or **8x** (pad HR to 1256×1256)

### Cloud Coverage Analysis
| Metric | Value |
|--------|-------|
| Minimum | 0.0% |
| Maximum | 99.93% |
| Mean | 7.98% |
| Median | 0.66% |
| **Low cloud (<10%)** | **74.9%** ✅ |
| High cloud (>50%) | 2.4% |

**Recommendation**: Filter samples with cloud_cover > 20% for cleaner training.

### Geographic Distribution
| Metric | Value |
|--------|-------|
| Latitude range | -76.73° to 80.94° |
| Longitude range | -167.24° to 177.88° |
| Mean latitude | 21.94° |
| Mean longitude | 14.12° |

**Coverage**: Global (all continents represented) ✅

### Land Cover Distribution (IPCC Classes)
| Class | Samples | Percentage |
|-------|---------|------------|
| Settlement | 27,200 | 43.3% |
| Forest | 11,744 | 18.7% |
| Agriculture | 11,200 | 17.8% |
| Other | 6,688 | 10.6% |
| Grassland | 3,280 | 5.2% |
| Water | 944 | 1.5% |
| Wetland | 656 | 1.0% |

**Note**: Settlement class is overrepresented. Stratified splits ensure balance.

### Settlement Density (SMOD Classes)
| Class | Samples | Percentage |
|-------|---------|------------|
| Rural: Very Low Density | 35,312 | 56.2% |
| Urban: Centre | 9,312 | 14.8% |
| Rural: Low Density | 5,168 | 8.2% |
| Urban: Suburban | 4,240 | 6.7% |
| Urban: Dense | 3,200 | 5.1% |
| Other | 5,616 | 8.9% |

---

## Data Splits

| Split | Locations | Percentage |
|-------|-----------|------------|
| Train | 3,140 | 79.9% |
| Validation | 390 | 9.9% |
| Test | 398 | 10.1% |

- **Split type**: Stratified by IPCC land cover class
- **Random seed**: 42 (reproducible)
- **Files**: `phase1_data_exploration/outputs/splits/`

---

## Key Findings & Decisions

### ✅ Strengths
1. **Large dataset**: 3,928 usable HR-LR pairs
2. **Low cloud coverage**: 74.9% samples have <10% cloud
3. **Global coverage**: Diverse geographic locations
4. **Multi-temporal**: 8 LR acquisitions per location
5. **Land cover diversity**: 7 IPCC classes represented
6. **Consistent HR resolution**: All 1054×1054

### ⚠️ Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| High scale factor (~6.7x) | Use 4x or 8x SwinIR config with cropping/resizing |
| Band mismatch (LR:12, HR:4) | Extract RGB bands from both (3 channels) |
| LR resolution varies slightly | Resize to fixed size (e.g., 128×128 or 160×160) |
| HR uint16, LR float32 | Normalize both to [0,1] range |
| Multiple temporal acquisitions | Select lowest cloud acquisition or temporal fusion |
| Settlement class imbalance | Already handled with stratified splits |

### 🎯 Training Configuration Decisions

| Parameter | Decision | Rationale |
|-----------|----------|-----------|
| HR target image | `ps` (pan-sharpened) | Highest resolution, 4 bands |
| LR input image | `L2A_data` | 12 Sentinel-2 bands |
| Input bands | RGB (bands 2,3,4 of S2) | Match HR RGB |
| Output bands | RGB (first 3 of ps) | Standard SR output |
| Scale factor | **4x** | Crop HR to 636×636, LR to 159×159 |
| HR patch size | 256×256 (cropped) | Standard for SwinIR |
| LR patch size | 64×64 | 256/4 = 64 |
| Cloud filter | Exclude >20% | Use lowest cloud acquisition |

---

## Output Files

```
phase1_data_exploration/outputs/
├── dataset_stats.json         # File counts, pair validation
├── image_analysis.json        # Image properties analysis
├── metadata_analysis.json     # Geographic, cloud, land cover stats
├── visualizations/            # Sample HR-LR comparison images
│   ├── sample_1_*.png
│   ├── sample_2_*.png
│   └── ...
└── splits/
    ├── train.txt              # 3,140 locations
    ├── val.txt                # 390 locations
    ├── test.txt               # 398 locations
    └── split_info.json        # Split metadata
```

---

## Visualizations

Sample HR-LR pairs visualized showing:
- HR RGB preview
- HR Pan-sharpened (RGBN)
- HR RGBN native resolution
- LR Sentinel-2 RGB composite
- Cloud probability map
- Cloud mask

See: `phase1_data_exploration/outputs/visualizations/`

---

## Next Steps (Phase 2)

1. Implement data loading pipeline with:
   - Band selection (RGB extraction)
   - Normalization (uint16→float, float32→float)
   - Cloud filtering (select best acquisition)
   - Resizing/cropping to fixed sizes
2. Implement data augmentation:
   - Random cropping (256×256 HR patches)
   - Random flipping (horizontal/vertical)
   - Random rotation (90°, 180°, 270°)
3. Create PyTorch Dataset and DataLoader

---

## Notes & Observations

1. The orphan HR location ("testing on one Amnesty POI") appears to be a test file - excluded from training.
2. LR images have varying resolutions (~153-163 pixels) due to geographic projection - need resizing.
3. The `rgbn` files are at native multispectral resolution (263×263) before pan-sharpening.
4. Cloud probability (CLP) ranges 0-252, not 0-100 as expected - may need rescaling.
5. Data mask is constant 1.0 - indicates all pixels are valid in sampled images.

---

## References
- WorldStrat Dataset: https://www.kaggle.com/datasets/jucor1/worldstrat
- Sentinel-2 Bands: https://sentinels.copernicus.eu/web/sentinel/user-guides/sentinel-2-msi

---

**Phase Start Date**: December 24, 2024  
**Phase End Date**: December 24, 2024  
**Completed By**: ___________
