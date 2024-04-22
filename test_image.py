import os
import cv2
import joblib
from skimage.feature import hog
from skimage import exposure

classes_name = {"pedestrianCrossing": 0, "stop":1, "signalAhead":2, "speedLimit":3, "turnRight": 4, "keepRight":5, "backGround":6, "stopAhead":7,
			    "doNotPass": 8, "schoolSpeedLimit25": 9, "merge": 10, "laneEnds": 11, "school": 12, "rightLaneMustTurn": 13, "yield": 14, 
                "turnLeft":15, "addedLane": 16, "yieldAhead": 17, "slow": 18, "truckSpeedLimit55": 19}

def extract_hog_features(image, resize=(64, 64)):
    img = cv2.resize(image, resize)
    image_grayed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # print("Grayscale image dimensions:", image_grayed.shape)
    features, hog_image = hog(image_grayed, orientations=8, pixels_per_cell=(16, 16),
                              cells_per_block=(2, 2), visualize=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    # cv2.imshow('HOG Image', hog_image_rescaled)
    # cv2.waitKey(0)
    return features

def sliding_window(image, step_size, window_size):
    """Slide a window across the image and collect windows in a list."""
    windows = []
    for y in range(0, image.shape[0] - window_size[1], int(step_size)):
        for x in range(0, image.shape[1] - window_size[0], int(step_size)):
            window = image[y:y + window_size[1], x:x + window_size[0]]
            windows.append((x, y, window))
    return windows

def process_image(image, step_size, window_size):
    windows = sliding_window(image, step_size, window_size)
    features_list = []
    location = []
    for x, y, window in windows:
        features = extract_hog_features(window)
        location.append((x,y))
        features_list.append(features)
    return features_list, location

if __name__ == "__main__":
    model_path = "../traffic-sign-detection/dataset/svm_model.pkl"
    svm_model = joblib.load(model_path)
    image_path = "../traffic-sign-detection/speedLimit_1323823292.avi_image3.png"
    image = cv2.imread(image_path)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # print(image.shape)

    x, y, w, h = 53, 150, 40, 40 
    cropped_image = image[y:y+h, x:x+w]
    # print(image.shape)
    features = extract_hog_features(cropped_image)
    prediction = svm_model.predict(features.reshape(1, -1))
    print(f"Window predicted as {prediction}")
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    label_text = f"Label: {prediction[0]}"
    cv2.putText(image, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    coords_text = f"Coords: ({x},{y})"
    cv2.putText(image, coords_text, (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.imshow("Image with Detection", image)
    cv2.waitKey(0)
    cv2.imwrite("../traffic-sign-detection/result2.png", image)
    cv2.destroyAllWindows()


    '''
    window_size = (40, 40)
    step_size = 20
    feature_list = []
    locations = []
    features_list, locations = process_image(image, step_size, window_size)
    print(len(features_list))
    predictions = []
    for feature in features_list:
        prediction = svm_model.predict(feature.reshape(1, -1))
        predictions.append(prediction[0])


    for location, prediction in zip(locations, predictions):
        print(f"Window at {location} predicted as {prediction}")
    '''