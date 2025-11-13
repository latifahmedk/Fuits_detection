#Fruit Detection using YOLOv8 and Roboflow

## Overview
This project demonstrates how to train a **YOLOv8 object detection model** using a **Fruit Detection dataset** from [Roboflow Universe](https://universe.roboflow.com/).  
The goal is to detect different fruits (such as apples, bananas, and oranges) in images.

---

## Problem Statement
Build a computer vision model that can **detect and classify fruits** in an image using a Roboflow dataset.  
The model is trained, evaluated, and deployed for inference in Google Colab.

---

## Dataset Setup
- **Dataset Source:** Roboflow (Workspace: `mllab`, Project: `fruits-ogc66`)
- **Format:** YOLOv8
- **Classes:** Apple, Banana, Orange
- **Image Size:** 416 × 416  
- **Splits:** Train / Validation / Test provided by Roboflow

**Dataset Import Code:**
```python
from roboflow import Roboflow
rf = Roboflow(api_key="YOUR_API_KEY")
project = rf.workspace("mllab").project("fruits-ogc66")
version = project.version(1)
dataset = version.download("yolov8")
