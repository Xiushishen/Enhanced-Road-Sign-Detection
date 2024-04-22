import os
import cv2
import joblib
from test_image import extract_hog_features

classes_name = {"pedestrianCrossing": 0, "stop":1, "signalAhead":2, "speedLimit":3, "turnRight": 4, "keepRight":5, "backGround":6, "stopAhead":7,
			    "doNotPass": 8, "schoolSpeedLimit25": 9, "merge": 10, "laneEnds": 11, "school": 12, "rightLaneMustTurn": 13, "yield": 14, 
                "turnLeft":15, "addedLane": 16, "yieldAhead": 17, "slow": 18, "truckSpeedLimit55": 19}

if __name__ == "__main__":
    model_path = "../traffic-sign-detection/dataset/svm_model.pkl" # path to trained model
    svm_model = joblib.load(model_path)
    image_path = "../traffic-sign-detection/stop_1323896588.avi_image28.png" # test the image
    image = cv2.imread(image_path)
    # cv2.imshow("image", image)
    # cv2.waitKey(0)
    # print(image.shape)
    x, y, w, h = 475, 175, 50, 50 
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
    save_path = "../traffic-sign-detection/result.png"
    cv2.imwrite(save_path, image)
    cv2.destroyAllWindows()