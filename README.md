
[DEMO LINK](https://colab.research.google.com/drive/1wPSV-k95ii5iDpa2Cr7qyc_7DrBdV4R6?usp=sharing)

# Emotion Recognition from Images and Videos

This project repository provides two scripts for recognizing emotions from facial expressions in images and videos. The project utilizes a Convolutional Neural Network (CNN) built using TensorFlow's Keras API, OpenCV for image processing, and a pre-trained model to predict emotions.

## Overview

The repository includes:
- A CNN model for emotion recognition.
- Scripts for processing images and videos to detect and display emotions.

## Dependencies

To run the code, you need the following Python libraries:
- `numpy`
- `opencv-python` (`cv2`)
- `matplotlib`
- `tensorflow`
- `google-colab` (for `cv2_imshow` in Colab environments)

## Model Architecture

The model consists of multiple convolutional layers followed by max-pooling layers, dropout layers for regularization, and dense layers. It outputs a softmax probability over 7 possible emotions: Angry, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.

## Usage

### Image Emotion Recognition

1. **Model Definition and Weights Loading**: The model is defined using TensorFlow's Keras API. Pre-trained weights are loaded from `model.h5`.

2. **Emotion Recognition Function (`emotion_recog`)**: 
   - Detects faces using OpenCV's Haar Cascade Classifier.
   - Converts detected face regions to grayscale.
   - Resizes and normalizes the face images to 48x48 pixels.
   - Predicts the emotion using the loaded model and labels it on the image.

### Video Emotion Recognition

1. **Initialization**:
   - Capture video using `cv2.VideoCapture('video.mp4')`.
   - Read the first frame to determine video dimensions.

2. **Processing**:
   - For each frame, detect faces, predict emotions, and display the results.
   - Save the processed frames into an output video file `output.avi`.

3. **Display**:
   - Display the frames with detected emotions using `cv2_imshow` (compatible with Google Colab).
