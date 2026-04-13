# Liver Tumor Segmentation with Synthetic Tumor Augmentation

This repository presents a deep learning-based pipeline for **liver and tumor segmentation from CT scans**, utilizing **synthetic data generation for medical image augmentation**.

A key part of this project is a **multi-GAN-based 3D synthetic tumor generation pipeline**. Instead of relying only on original tumor scans, the workflow generates **3D synthetic tumor volumes and corresponding masks** using multiple GAN components, and then **inserts these synthetic tumors into healthy liver regions within real CT volumes**. The inserted tumors are blended into the surrounding liver tissue to create more realistic augmented samples for segmentation model training. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}

## Project Overview

Accurate liver and tumor segmentation is important for diagnosis, treatment planning, and disease monitoring. One of the major challenges in medical imaging is the limited availability of diverse annotated tumor data. To address this, this project combines:

- **3D synthetic tumor generation using GANs**
- **Insertion of synthetic tumors into healthy liver CT scans**
- **Creation of augmented liver-tumor datasets for experimentation**
- **Segmentation model training and testing**

## Key Features

- 3D synthetic tumor generation using a **multi-GAN pipeline**
- Use of **DCGAN, WGAN, Aggregator, and Style Transfer modules**
- Generation of synthetic tumor volumes and tumor masks
- Insertion of synthetic tumors into healthy liver regions in real CT scans
- Patch blending and seam reduction for more realistic composite volumes
- Saving of augmented NIfTI volumes and segmentation masks
- Liver and tumor segmentation from CT images
- Training and evaluation workflows for segmentation experiments :contentReference[oaicite:7]{index=7} :contentReference[oaicite:8]{index=8}

## Repository Structure

```bash
Liver-Tumor-Segmentation/
│
├── README.md
├── train_orig_AIMUNetALF.py        # Training script for liver/tumor segmentation
├── test_aimunet_orig_65.py         # Testing/evaluation script
└── Syn_Img_Pipeline_Code.py        # Synthetic 3D tumor generation and insertion pipeline
