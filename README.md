# Indian Sign Language (ISL) Model

This repository is an iterative build focused on developing a model for Indian Sign Language, building upon previous projects:
- [asl_beta_v1.0](https://github.com/SIDDHARTH-PADIGAR/asl_beta_v1.0)
- [asl_beta_v2.0](https://github.com/SIDDHARTH-PADIGAR/asl_beta_v2.0)

## Overview

The goal of this project is to transition from a single-hand American Sign Language (ASL) model to a more inclusive Indian Sign Language (ISL) model that handles two-hand inputs. The ISL model increases the feature set from 42 (single hand) to 84 (two hands) to capture the complexity of sign language communication.

## Development Stages

### Stage 1: Multi-Hand Feature Handling
- **Objective:** Adapt the model to process and analyze two-hand inputs.
- **Modification:** Expand feature representation from 42 (for one hand) to 84 (for two hands).

### Stage 2: Dataset Enhancement
- **Objective:** Improve the robustness and diversity of the dataset.
- **Modification:** Integrate a mix of personally captured images and curated datasets (e.g., from Kaggle) to expose the model to a wide range of real-world scenarios.
- **Status:** This stage is actively in progress to ensure the dataset is comprehensive and adaptable.

### Stage 3: Training Data Improvement
- **Objective:** Optimize the training process to accurately predict signs from two-hand inputs.
- **Modification:** Refine training data and methodologies to enhance the model’s proficiency in sign prediction.

## Current Status

While the underlying ASL model is fully functional for real-world usage, the ISL model is still under development. Currently, the focus is on Stage 2—enhancing the dataset to improve the model's adaptability and accuracy.

## Future Work

- **Finalize Dataset Enhancements:** Continue incorporating diverse image sources to build a robust dataset.
- **Optimize Training Process:** Complete Stage 3 by fine-tuning the training data to support efficient two-hand sign prediction.
- **Expand Model Capabilities:** Adapt and scale the model to perform reliably in various real-world settings.

---

*Note: This README focuses strictly on the developmental aspects and challenges of the ISL model as outlined above.*
