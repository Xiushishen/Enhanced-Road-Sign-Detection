import os
import numpy as np
import cv2
import csv
import random
from collections import defaultdict

random.seed(50)

classes_name = {"pedestrianCrossing": 0, "stop":1, "signalAhead":2, "speedLimit":3, "keepRight":5, "backGround":6, "stopAhead":7,
			 "doNotPass": 8, "schoolSpeedLimit25": 9, "merge": 10, "laneEnds": 11, "school": 12, "rightLaneMustTurn": 13, "yield": 14, "turnLeft":15,
			 "addedLane": 16, "yieldAhead": 17, "slow": 18, "truckSpeedLimit55": 19, "turnRight": 4}

SIGN_ROOT = "../traffic-sign-detection"
DATA_PATH = os.path.join(SIGN_ROOT, 'data/images')

TRAIN_ROOT = "../traffic-sign-detection/dataset/train/"
TEST_ROOT = "../traffic-sign-detection/dataset/test/"
TRAIN_CSV = "../traffic-sign-detection/dataset/train/train_labels.csv"
TEST_CSV = "../traffic-sign-detection/dataset/test/test_labels.csv"


def parse_csv(csv_file):
    labels = []
    number = defaultdict(int)
    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip the header
        for row in reader:
            if len(row) < 6:
                continue
            filename, tag, xmin, ymin, xmax, ymax = row
            number[tag] += 1  # Increment the count for the tag
            labels.append([str(tag), int(xmin), int(ymin), int(xmax), int(ymax)])
    return labels, number

def get_paths(train=False):
    image_paths = []
    csv_file = TRAIN_CSV if train else TEST_CSV

    with open(csv_file, 'r') as file:
        reader = csv.reader(file)
        next(reader, None)
        for row in reader:
            if len(row) < 6:
                continue
            filename = row[0]
            path = (TRAIN_ROOT if train else TEST_ROOT) + filename
            image_paths.append(path)
    return image_paths


def generate_normal_feature(csv_file, labels, dir_to, feature_set):
	image_paths = get_paths()
	print(len(image_paths))
	# assert(len(image_paths) == len(labels))
	os.makedirs(dir_to, exist_ok=True)
	for i, image_path in enumerate(image_paths):
		img = cv2.imread(image_path)
		cls_name, xmin, ymin, xmax, ymax = labels[i]
		if cls_name != "backGround":
			xmin, xmax = max(0, xmin), min(img.shape[1], np.ceil(xmax))
			ymin, ymax = max(0, ymin), min(img.shape[0], np.ceil(ymax))
			crop_img = img[int(ymin):int(ymax), int(xmin):int(xmax)]
			feature_set[cls_name]+=1
			write_name = f"{cls_name}_{feature_set[cls_name]}.png"
			cv2.imwrite(os.path.join(dir_to, write_name), crop_img)
	return feature_set

def generate_noise_feature(csv_file, labels, dir_to, noise_set):
	image_paths = get_paths()
	print(len(image_paths))
	assert(len(image_paths) == len(labels))
	os.makedirs(dir_to, exist_ok=True)
	for i, image_path in enumerate(image_paths):
		if labels[i][0] == "backGround":
			img = cv2.imread(image_path)
			h, w, _ = img.shape
			crop_w = random.randint(40, min(w, 110))
			crop_h =  random.randint(40, min(h, 110))

			xmin = random.randint(0, w - crop_w)
			ymin = random.randint(0, h - crop_h)
			xmax = xmin + crop_w
			ymax = ymin + crop_h

			crop_img = img[ymin:ymax, xmin:xmax]
			noise_set["backGround"]+=1
			write_name = f'{labels[i][0]}_{noise_set["backGround"]}.png'
			cv2.imwrite(os.path.join(dir_to, write_name), crop_img)
	return noise_set


def generate_features(dir_in, labels, dir_to, train=False):
	feature_set = defaultdict(int)
	noise_set = defaultdict(int)
	total_set = defaultdict(int)
	for cls_name in classes_name:
		total_set[cls_name] = 0
	if train:
		csv_file_name = "train_labels.csv"
	else:
		csv_file_name = "test_labels.csv" 
	csv_file_path = os.path.join(dir_in, csv_file_name)
	normal_feature = generate_normal_feature(csv_file_path, labels, dir_to, feature_set)
	noise_feature = generate_noise_feature(csv_file_path, labels, dir_to, noise_set)
	for key, value in normal_feature.items():
		total_set[key] += value
	for key, value in noise_feature.items():
		total_set[key] += value
	return total_set

if __name__ == "__main__":
	csv_file_train = "../traffic-sign-detection/dataset/train/train_labels.csv"
	csv_file_test = "../traffic-sign-detection/dataset/test/test_labels.csv"
	csv_dir_train = "../traffic-sign-detection/dataset/train/"
	csv_dir_test = "../traffic-sign-detection/dataset/test/"
	save_dir = "../traffic-sign-detection/dataset/train_proposed"
	cropped_image_path_train = "../traffic-sign-detection/dataset/train_feature/"
	cropped_image_path_test = "../traffic-sign-detection/dataset/test_feature/"
	
	classes_name = {"pedestrianCrossing": 0, "stop":1, "signalAhead":2, "speedLimit":3, "keepRight":5, "backGround":6, "stopAhead":7,
			 "doNotPass": 8, "schoolSpeedLimit25": 9, "merge": 10, "laneEnds": 11, "school": 12, "rightLaneMustTurn": 13, "yield": 14, "turnLeft":15,
			 "addedLane": 16, "yieldAhead": 17, "slow": 18, "truckSpeedLimit55": 19, "turnRight": 4}

	# proposal_num = produce_proposals(xml_dir, save_dir, square=True)
	# print("proposal num = ", proposal_num)
	labels, number = parse_csv(csv_file_test)
	print(len(labels))

	feature_set = generate_features(csv_dir_test, labels, cropped_image_path_test)
	# print(feature_set)
