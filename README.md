# PhantomSegmentation

This C++ project implements MRI phantom segmentation using a pre-trained U-Net model. It also provides OpenCV-based segmentation methods for comparison and evaluation. 

## Table of Contents

* [Introduction](#introduction)
* [Features](#features)
* [Dependencies](#dependencies)
* [Building and Running](#building-and-running)
    * [Prerequisites](#prerequisites)
* [File Structure](#file-structure)

## Introduction

This project focuses on segmenting MRI phantoms from input images. It leverages a trained U-Net model (implemented in PyTorch) for accurate segmentation.  It also incorporates OpenCV-based segmentation techniques for comparative analysis and provides visual outputs to demonstrate the performance of each method.

## Features

* **U-Net Segmentation:** Leverages a pre-trained U-Net model to perform segmentation.
* **OpenCV Segmentation:**  Implements threshold-based and denoised threshold-based segmentation using OpenCV.
* **IoU Score Calculation:** Calculates Intersection over Union (IoU) to evaluate segmentation accuracy for both U-Net and OpenCV methods.
* **Visualization:** Generates subplot images showcasing input, ground truth masks, and segmentation results from different methods.
* **Data Handling:** Reads image and mask paths from a CSV file and efficiently loads data using a custom dataset class.
* **Normalization:** Handles different input data types (int16, uint16) and normalizes them to float32.

## Dependencies

* **OpenCV:**  Required for image processing, segmentation, and visualization.
* **SimpleITK:** Used for loading medical images (in this case, assumed to be DICOM).
* **LibTorch:**  Essential for loading and running the pre-trained U-Net model. 

## Building and Running

Please note that this work is a skeleton implementation and cannot be compiled without modifications to the source code !!!! 

### Prerequisites

* C++ Compiler (e.g., g++, Clang) with C++17 support
* OpenCV library
* SimpleITK library 
* LibTorch library
* CMake (for building)

## File Structure

```
├── source
│   ├── main.cpp              # Main program entry point
│   ├── Segmentation.cpp      # Segmentation class implementation
│   └── Dataset.cpp           # Dataset class implementation
├── headers
│   ├── Segmentation.h        # Segmentation class definition
│   └── Dataset.h             # Dataset class definition
└── README.md                 # This file
```

    

