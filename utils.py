import numpy as np 
import os
import cv2
import shutil
import csv
from random import shuffle
from pathlib import Path

def crop_image(image_path, bounding_box):
    image = cv2.imread(image_path)
    x_min, y_min, x_max, y_max = bounding_box
    cropped_image = image[y_min:y_max, x_min:x_max]
    return cropped_image


def extract_prefix(filename):
    prefix = filename.split("/")[-1].split("_", 1)[0]
    return prefix


def split_data(csv_file, img_dir, train_ratio=0.7):
    with open(csv_file, 'r', newline='') as file:
        reader = csv.reader(file)
        headers = next(reader)
        data = list(reader)

    shuffle(data)

    split_idx = int(len(data) * train_ratio)
    train_data = data[:split_idx]
    test_data = data[split_idx:]

    train_img_dir = Path(img_dir) / 'train'
    test_img_dir = Path(img_dir) / 'test'
    train_img_dir.mkdir(parents=True, exist_ok=True)
    test_img_dir.mkdir(parents=True, exist_ok=True)

    train_csv = train_img_dir / 'train_labels.csv'
    test_csv = test_img_dir / 'test_labels.csv'

    with open(train_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for row in train_data:
            img_name = row[0]
            src_img_path = Path(img_dir)/"images"/img_name
            dst_img_path = train_img_dir / img_name
            shutil.copy(src_img_path, dst_img_path)
            writer.writerow(row)

    with open(test_csv, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(headers)
        for row in test_data:
            img_name = row[0]
            src_img_path = Path(img_dir)/"images"/img_name
            dst_img_path = test_img_dir / img_name
            shutil.copy(src_img_path, dst_img_path)
            writer.writerow(row)

def load_feature(file):
    image_paths = []
    labels = []
    features = []
    with open(file, "r") as f:
        for line in f:
            parts = line.strip().split(" ")
            if len(parts) < 3:
                continue
            img_path = parts[0]
            label = int(parts[1])
            feature_vector = [float(feature) for feature in parts[2:]]
            image_paths.append(img_path)
            labels.append(label)
            features.append(feature_vector)
    labels = np.array(labels)
    features = np.array(features)
    return image_paths, labels, features

if __name__ == "__main__":
    img_dir = "../traffic-sign-detection/dataset"
    csv_file = "../traffic-sign-detection/dataset/images/data.csv"
    split_data(csv_file, img_dir, train_ratio=0.8)