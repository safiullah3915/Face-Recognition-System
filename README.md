Face-Recognition-System

A streamlined Python-based facial recognition application that leverages a pretrained face-embedding model for efficient identification.

Overview

This system detects faces from video or image inputs, extracts embeddings using a FaceNet model (facenet.pb), and matches them against stored encodings for recognition. Embedded facial features are managed in the face_encodings module.

Structure & Files

app.py — Main application script: handles face detection, encoding extraction, and recognition logic.

facenet.pb — Pretrained FaceNet model used for generating facial embeddings.

face_encodings/ — Directory likely containing stored known face encodings and matching functionality.

haarcascade_frontalface_default.xml — Haar Cascade classifier for initial face detection using OpenCV.

Quick Start

Clone the repository

git clone https://github.com/safiullah3915/Face-Recognition-System.git
cd Face-Recognition-System


Install dependencies
(Add requirements.txt if it's not already present; dependencies likely include OpenCV, TensorFlow, NumPy, etc.)

pip install -r requirements.txt


Run the Application

python app.py


Provide an image or video source for real-time facial recognition via webcam or media file.

Usage Details

Face Detection: Uses Haar Cascade (haarcascade_frontalface_default.xml) to locate faces.

Embedding Extraction: Processes detected faces through the FaceNet model (facenet.pb) to generate embeddings.

Recognition: Compares embeddings against stored face profiles in face_encodings to identify matches and label faces accordingly.
