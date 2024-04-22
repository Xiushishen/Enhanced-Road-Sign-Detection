# Enhanced-Road-Sign-Detection

How to test a random image:
```
python run.py
```
In side the run.py file, you can change the image path to the one you like and do some tesing.

If you want to train the model from the begining, please follow the steps below:

1. Find the dataset you like and download them from the [here](https://www.kaggle.com/datasets/omkarnadkarni/lisa-traffic-sign).

2. Put all the images and the csv file inside the trassif-sign-detection/dataset/images folder.

3. Run data split code below to get the train set and test set.
```
python utils.py
```
4. Run script below to get the cropped image of positive and negative dataset.
```
python region_proposal_producer.py
```
5. Extract HOG features from both positive and negative dataset.
```
python hog_extract.py
```
6. Now, you can train the SVM model with the extracted HOG features.
```
python svm_train.py
```
7. As soon as you have the trained model, you can test your image.
```
python test_image.py
```
