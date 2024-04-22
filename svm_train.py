import joblib
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from utils import load_feature


def train_svm_model(features, labels, model_path):
    svm_model = SVC(C=1.0, kernel="linear", probability = True)
    svm_model.fit(features, labels)
    joblib.dump(svm_model, model_path)
    print(f"Model saved to {model_path}")

def test_svm_model(features, labels, model_path):
    svm_model = joblib.load(model_path)
    predictions = svm_model.predict(features)
    accuracy = accuracy_score(labels, predictions)
    print(f"Model accuracy: {accuracy:.2f}")
    print("Classification report:")
    print(classification_report(labels, predictions))

if __name__ =="__main__":
    hog_train_txt = "../traffic-sign-detection/dataset/hog_feature_train.txt"
    hog_test_txt = "../traffic-sign-detection/dataset/hog_feature_test.txt"
    model_path = "../traffic-sign-detection/dataset/svm_model.pkl"
    _, labels_train, features_train = load_feature(hog_train_txt)
    _, labels_test, features_test = load_feature(hog_test_txt)
    train_svm_model(features_train, labels_train, model_path)
    # print(features.shape)
    model_path = "../traffic-sign-detection/dataset/svm_model.pkl"
    # train_svm_model(features, labels, model_path)
    test_svm_model(features_test, labels_test, model_path)