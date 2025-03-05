# PREFFECT - PaRaffin Embedded Formalin-FixEd FixEd Cleaning Tool
Release Version 0.1

## Description
PREFFECT (PaRaffin Embedded Formalin-FixEd Cleaning Tool) is a probabilistic framework for the analysis of RNA sequencing data from formalin-fixed paraffin embedded (FFPE) samples. PREFFECT uses generative models to fit distributions to observed expression measurements while adjusting for technical and biological variables. PREFFECT offers a series of models that exploit multiple expression profiles generated from matched
tissues for a single sample, sample-sample adjacency networks and clinicopathological/technical sample variables.

PREFFECT is designed to accomplish the following tasks:
- Accurately denoise RNAseq data from FFPE materials (fRNAseq)
- Impute missing values from the count matrix
- Improve performance of standard analyses

## Authors
**Eliseos John Mucaki -**
*Department of Biochemistry, Western University, London, Canada*

**Wenhan Zhang -**
*Department of Biochemistry, Western University, London, Canada*

**Aryamaan Saha -**
*Department of Biological Engineering and Data Science, Indian Institute of Technology, Madras, India*

**Sabina Trabinjac -**
*Department of Radiation Oncology, Sunnybrook Health Sciences Centre
Department of Laboratory Medicine and Pathobiology, University of Toronto Toronto, Canada*

**Sharon Nofech-Mozes -**
*Department of Anatomic Pathology, Sunnybrook Health Sciences Centre*
*Department of Laboratory Medicine and Pathobiology, University of Toronto, Toronto, Canada*

**Eileen Rakovitch -**
*Department of Radiation Oncology, Sunnybrook Health Sciences Centre
Department of Laboratory Medicine and Pathobiology, University of Toronto Toronto, Canada*

**Vanessa Dumeaux -**
*Department of Anatomy and Cell Biology, Department of Oncology, Western University, London, Canada*

**Michael Hallett -**
*Department of Biochemistry, Department of Oncology, Western University, London, Canada*

Contact: [Mike Hallett](mailto:michael.hallett@uwo.ca)


<p align="center">
  <span style="background-color: white; display: inline-block; padding: 10px;">
  <img src="./assets/logos/western_logo.png" alt="Western_Small" width="80" style="vertical-align: middle; margin-right: 20px;"/>
  <img src="./assets/logos/sunnybrook_logo_2024.png" alt="SunnyBrook" width="280" style="vertical-align: middle; margin-right: 20px;"/>
  <img src="./assets/logos/nserc_2024.png" alt="NSERC" width="130" style="vertical-align: middle;"/>
  <img src="./assets/logos/cihr_color_logo_2024.png" alt="NSERC" width="150" style="vertical-align: middle;"/>
</span>
</p>

# <u>User Guide</u>
PREFFECT allows the user to perform a series of tasks to clean, impute and evaluate RNA sequencing from FFPE materials. Below, we will describe how to set up your input, setup and optimize runs through the __config.py_ file, derive new models (training), retrain previously derived models (pre-train), as well as performing inference and other analyses.

## [Installation](./installation.md)

## [Import and Structure of Data for PREFFECT](./importing.md)

## [Setting PREFFECT configurations](./setting_parameters.md)

## [Training a PREFFECT model](./training.md)

## [Inference with PREFFECT](./inference.md)

## [Imputation](./imputation.md)

## [Clustering](./clustering.md)

## [Transfer Learning](./transfer_learning.md)

## Differential Expression (In Development)

## [Code Reference](https://hallettmiket.github.io/preffect/)

Additional documentation detailing the specific contents (i.e., functions, classes, modules) of PREFFECT. It is meant to serve developers or advanced users who need to understand the code’s capabilities and usage.

https://hallettmiket.github.io/preffect/
