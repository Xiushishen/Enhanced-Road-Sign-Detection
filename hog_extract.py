import os
from skimage.feature import hog
from skimage import exposure
import cv2
import glob
from utils import extract_prefix

class_set = {"pedestrianCrossing": 0, "stop":1, "signalAhead":2, "speedLimit":3, "keepRight":5, "backGround":6, "stopAhead":7,
			 "doNotPass": 8, "schoolSpeedLimit25": 9, "merge": 10, "laneEnds": 11, "school": 12, "rightLaneMustTurn": 13, "yield": 14, "turnLeft":15,
			 "addedLane": 16, "yieldAhead": 17, "slow": 18, "truckSpeedLimit55": 19, "turnRight": 4}

def file_getter(dir, out_file):
    img_paths = glob.glob(os.path.join(dir, '*.png'))
    try:
        os.remove(out_file)
    except FileNotFoundError:
        pass
    return img_paths

def extract_hog_features(image_path):
    image = cv2.imread(image_path)
    resize = (64,64)
    img = cv2.resize(image, resize)
    image_grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features, hog_image = hog(image_grayed, orientations=8, pixels_per_cell=(16,16),
                              cells_per_block=(2,2), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0,10))
    # cv2.imshow('HOG Image', hog_image_rescaled)
    # cv2.waitKey(0)

    return features

def extract_and_save_features(directory, out_file):
    index = 0
    img_paths = file_getter(directory, out_file)

    with open(out_file, "w") as f:
        for img_path in img_paths:
            features = extract_hog_features(img_path)
            name = extract_prefix(img_path)
            number = class_set[name]
            line = f"{img_path} {number} " + " ".join(map(lambda x: f"{x:.2f}", features)) + "\n"
            f.write(line)
            index += 1
            if index % 200 == 0 or index == len(img_paths):
                print(f"Writing into files: {index} / {len(img_paths)}")


if __name__ == "__main__":
    # img_paths = file_getter(img_dir1, out_file)
    # print(len(img_paths))
    # extract_and_save_features(img_dir1, out_file)

    feature_train_dir = "../traffic-sign-detection/dataset/train_feature"
    out_file_train = "../traffic-sign-detection/dataset/hog_feature_train.txt"
    feature_test_dir = "../traffic-sign-detection/dataset/test_feature"
    out_file_test = "../traffic-sign-detection/dataset/hog_feature_test.txt"
    extract_and_save_features(feature_train_dir, out_file_train)
    extract_and_save_features(feature_test_dir, out_file_test)
