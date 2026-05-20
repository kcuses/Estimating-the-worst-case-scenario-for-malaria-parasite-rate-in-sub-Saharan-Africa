# Estimating the Worst-Case Scenario for Malaria Parasite Rate in sub-Saharan Africa

This repository contains the computational pipeline used to estimate baseline malaria prevalence and incidence across sub-Saharan Africa under a counterfactual scenario of minimal malaria intervention coverage.

The framework combines:

- Landsat-8 satellite imagery
- Self-supervised contrastive learning
- High-dimensional environmental embeddings
- Bayesian ridge regression
- Spatial raster prediction
- Prevalence-to-incidence conversion
- Population-weighted aggregation

The repository accompanies the manuscript:

> *Estimating the worst-case scenario for malaria parasite rate in sub-Saharan Africa*

---

# Repository Structure

```text
├── satellite/
│   └── download_satellite_images.py
│
├── pretraining/
│   ├── contrastive_training.py
│   └── extract_features.py
│
├── modelling/
│   ├── malaria_bayesian_prediction.py
│   ├── baseline_predictions.r
│   ├── prevalence_to_incidence_conversion.R
│   └── summarize_baseline.r
│
├── plotting/
│   └── final_incidence_average_cases_mapping.ipynb
│
├── requirements.txt
├── requirements_r.txt
└── README.md
```

# Overview of the Workflow

The complete workflow consists of seven major stages:

1. Download Landsat-8 imagery from Google Earth Engine
2. Train a self-supervised contrastive learning model
3. Extract environmental feature embeddings
4. Fit Bayesian ridge regression model
5. Generate continent-wide malaria prevalence predictions
6. Convert prevalence estimates to incidence
7. Aggregate and visualise outputs

---

# 1. Download Landsat-8 Satellite Imagery

Script:

```
satellite/download_satellite_images.py
```

This script downloads RGB Landsat-8 image patches centred on DHS/MIS survey locations using Google Earth Engine.

## Notes

- Uses Landsat Collection 2:

```
LANDSAT/LC08/C02/T1_L2
```

- Temporal range used in this study:

```
2013-01-01 to 2022-12-31
```

- Image patches correspond approximately to:

```
10 km × 10 km
```

- Image dimensions:

```
224 × 224 pixels
```

## Authentication

Authenticate Earth Engine before running:

```
earthengine authenticate
```

---

# 2. Train Contrastive Learning Model

Script:

```
pretraining/contrastive_10_04.py
```

This script trains a SimSiam-style self-supervised contrastive learning model on Landsat-8 image patches.

## Model Summary

- Custom convolutional encoder
- SimSiam predictor head
- Two stochastic augmentation pipelines
- Cosine similarity contrastive objective

## Output

The script produces:

- Trained encoder weights
- Trained predictor weights
- Training diagnostics and augmentation visualisations

---

# 3. Extract Learned Feature Embeddings

Script:

```
pretraining/extract_features_full_5k_10k.py
```

This script loads the pretrained encoder and extracts high-dimensional environmental embeddings from each satellite image.

## Output

A CSV file containing:

- image filename
- latitude
- longitude
- learned feature embeddings

These embeddings are subsequently used as predictors in the Bayesian malaria model.

---

# 4. Bayesian Malaria Prevalence Modelling

Script:

```
modelling/malaria_bayesian_prediction.py
```

This script implements the QR-decomposed Bayesian ridge regression model described in Equation 3 of the manuscript.

## Model Specification

```
tau       ~ Exponential(1)beta_hat  ~ Normal(0, tau)sigma     ~ Exponential(1)
```

The model is fitted using:

- NumPyro
- JAX
- NUTS sampling

## Features

- Empirical logit prevalence transformation
- Intervention-based baseline filtering
- QR decomposition for numerical stability
- Posterior uncertainty estimation

## Output

The script generates:

- posterior coefficient samples
- posterior prevalence predictions
- posterior uncertainty estimates

---

# 5. Generate Baseline Prevalence Maps and Realisations

Script:

```
modelling/baseline_predictions.r
```

This script applies posterior coefficient samples to all prediction locations to generate:

- baseline malaria prevalence maps
- posterior prevalence realisations
- uncertainty intervals
- administrative aggregation outputs

## Output

The script produces:

- prevalence raster stacks
- administrative-level prevalence summaries
- uncertainty visualisations

---

# 6. Convert Prevalence to Incidence

Script:

```
modelling/prevalence_to_incidence_conversion.R
```

This script converts posterior malaria prevalence realisations into incidence estimates using the prevalence-to-incidence framework described in:

> Cameron et al. (2015)

## Inputs

- prevalence raster realisations
- population rasters
- endemicity masks

## Output

- incidence raster realisations
- uncertainty estimates

---

# 7. Summarise Incidence Outputs

Script:

```
modelling/summarize_baseline.r
```

This script aggregates incidence realisations to administrative units using population-weighted zonal statistics.

## Outputs

- mean incidence
- median incidence
- lower and upper uncertainty intervals
- population-adjusted summaries

---

# 8. Final Visualisation and Mapping

Notebook:

```
plotting/final_incidence_average_cases_mapping.ipynb
```

This notebook generates the final figures and maps used in the manuscript.

---

# Supplementary Analyses

Additional notebooks used for supplementary analyses:

```
extras/obsvpred.ipynbextras/regression_evi.ipynb
```

These include:

- observed vs predicted prevalence plots
- EVI validation analyses
- additional regression diagnostics

---

# Installation

## Python Environment

Install dependencies using:

```
pip install -r requirements.txt
```

## R Environment

Install required R packages from:

```
requirements_r.txt
```

---

# Computational Requirements

The contrastive learning stage was trained using NVIDIA GPUs.

Recommended environment:

- CUDA-enabled GPU(s)
- ≥24 GB GPU memory
- ≥64 GB RAM
- TensorFlow 2.x
- Python 3.9+

Bayesian inference was performed using:

- NumPyro
- JAX
- CPU-based inference

---

# Data Availability

## Publicly Available Data

### Landsat-8 Imagery

Satellite imagery was accessed through Google Earth Engine:

```
LANDSAT/LC08/C02/T1_L2
```

### Google Earth Engine

[https://earthengine.google.com/](https://earthengine.google.com/)

---

## Restricted-Access Data

### DHS/MIS Survey Data

Malaria prevalence survey data were obtained through:

- Demographic and Health Surveys (DHS)
- Malaria Indicator Surveys (MIS)

These datasets are subject to data-use agreements and cannot be redistributed.

### DHS Program

[https://dhsprogram.com/](https://dhsprogram.com/)

---

## MAP Population and Administrative Data

Population surfaces and administrative rasters were obtained through collaborations and licensed datasets associated with the Malaria Atlas Project (MAP).

Some raster products cannot be publicly redistributed.

---

# Reproducibility Notes

- The original study used Landsat-8 imagery from 2013–2022.
- Google Earth Engine Collection 1 has since been deprecated and replaced with Collection 2.
- Exact image composites may therefore differ slightly due to upstream updates in Earth Engine processing pipelines.
- Exact reproduction of manuscript figures additionally requires access to controlled DHS/MIS datasets and MAP-derived rasters that cannot be publicly redistributed.

---

# Citation

If using this repository, please cite:

```
[Manuscript citation to be added upon publication]
```

---

# Contact

For questions regarding code or data availability, please contact the corresponding authors.
kcuses@gmail.com
