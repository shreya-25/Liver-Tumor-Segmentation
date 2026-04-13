# Liver Tumor Segmentation

This repository presents a deep learning-based approach for **liver and tumor segmentation from CT scans**. It includes model training, testing, and a synthetic image generation pipeline to support experimentation in medical image analysis and improve segmentation performance on challenging tumor regions.

The project is built in **Python** and focuses on automated segmentation workflows using an **AIM-UNet-based architecture**, along with preprocessing, augmentation, and evaluation steps for volumetric medical imaging data.

## Project Overview

Accurate liver and tumor segmentation plays an important role in computer-aided diagnosis, treatment planning, and disease monitoring. This repository explores a segmentation pipeline designed to identify liver and tumor regions from CT volumes using deep learning techniques.

In addition to the core segmentation model, the project also includes a synthetic image pipeline that can be used to support data augmentation and experimentation when working with limited annotated medical imaging datasets.

## Features

- Liver and tumor segmentation using a deep learning model
- AIM-UNet-inspired architecture for medical image segmentation
- Training pipeline for CT scan volumes and segmentation masks
- Testing pipeline for model evaluation
- Synthetic image generation pipeline for experimentation and augmentation
- Performance tracking using segmentation and classification metrics
- Logging and checkpoint saving during training

## Repository Structure

```bash
Liver-Tumor-Segmentation/
│
├── README.md
├── train_orig_AIMUNetALF.py        # Model training script
├── test_aimunet_orig_65.py         # Model testing/evaluation script
└── Syn_Img_Pipeline_Code.py        # Synthetic image generation pipeline
