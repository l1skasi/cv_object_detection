from custom_DPM import DPM
import torch
import os
import time
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import cv2
import pandas as pd
from pprint import pprint

# Function for parsing CSV files
def get_true_bboxes(csv_path, image_dir):
    df = pd.read_csv(csv_path)
    gr_true_bboxxes = []
    grouped = df.groupby('image')
    for img_name, group in grouped:
        boxes = []
        labs = []
        for _, row in group.iterrows():
            x1, y1, x2, y2 = row['x1'], row['y1'], row['x2'], row['y2']
            boxes.append([x1, y1, x2, y2])
            labs.append(1)  # car = 1
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labs = torch.as_tensor(labs, dtype=torch.int64)
        gr_true_bboxxes.append({"boxes": boxes, "labels": labs})
    return gr_true_bboxxes
