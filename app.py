import cv2
import numpy as np
import mediapipe as mp
import pickle
import tensorflow as tf
import os
import hashlib
import tkinter as tk
from tkinter import messagebox, simpledialog, ttk
from PIL import Image, ImageTk
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Mediapipe face detection
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Directory to store face encodings
ENCODINGS_DIR = "face_encodings"
if not os.path.exists(ENCODINGS_DIR):
    os.makedirs(ENCODINGS_DIR)

# Path to your local FaceNet model file
model_file = "facenet.pb"  # Update this path to your model location

# Load the FaceNet model from .pb file
def load_graph(model_file):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.compat.v1.GraphDef()

        with tf.io.gfile.GFile(model_file, "rb") as f:
            graph_def.ParseFromString(f.read())
            tf.import_graph_def(graph_def, name="")
    return graph

# Load the graph
graph = load_graph(model_file)

# Start a session
sess = tf.compat.v1.Session(graph=graph)

# Define input and output tensor names
input_tensor_name = 'input:0'
output_tensor_name = 'embeddings:0'
phase_train_tensor_name = 'phase_train:0'

def process_frame(frame):
    """Process a video frame to detect faces."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    detections = []

    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = frame.shape
            bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            detections.append(bbox)

    return detections

def encode_face(frame):
    """Encode a face using FaceNet."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face = cv2.resize(rgb_frame, (160, 160))
    face = np.expand_dims(face, axis=0)
    face = (face - 127.5) / 128.0

    # Run the model to get embeddings
    embeddings = sess.run(output_tensor_name, feed_dict={
        input_tensor_name: face,
        phase_train_tensor_name: False
    })
    return embeddings.flatten()

def add_user(encodings, user_name):
    """Saves a new user with averaged face encodings."""
    if encodings:
        average_encoding = np.mean(encodings, axis=0)
        with open(f"{ENCODINGS_DIR}/{user_name}.pkl", 'wb') as f:
            pickle.dump(average_encoding, f)
        return True
    return False

def recognize_face(encoding):
    """Recognizes a face by comparing it with saved encodings."""
    known_encodings = []
    known_names = []

    for filename in os.listdir(ENCODINGS_DIR):
        with open(f"{ENCODINGS_DIR}/{filename}", 'rb') as f:
            known_encodings.append(pickle.load(f))
            known_names.append(filename.split('.')[0])

    # Calculate distances between the encoding and known encodings
    distances = np.linalg.norm(known_encodings - encoding, axis=1)
    min_distance = np.min(distances)
    threshold = 0.7  # Adjust this threshold for your model
    name = "Unknown"

    if min_distance < threshold:
        min_index = np.argmin(distances)
        name = known_names[min_index]

    return name

class CameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Face Recognition App")

        try:
            self.video_capture = cv2.VideoCapture(0)
            if not self.video_capture.isOpened():
                raise IOError("Cannot open webcam")

            self.canvas = tk.Canvas(root, width=640, height=480)
            self.canvas.pack()

            self.btn_frame = ttk.Frame(root)
            self.btn_frame.pack()

            self.add_user_btn = ttk.Button(self.btn_frame, text="Add User", command=self.prepare_to_add_user)
            self.add_user_btn.pack(side=tk.LEFT)

            self.quit_btn = ttk.Button(self.btn_frame, text="Quit", command=self.quit)
            self.quit_btn.pack(side=tk.LEFT)

            self.adding_user_mode = False
            self.num_photos = 5
            self.encodings = []

            self.face_cascade = self.load_cascade()
            self.update_frame()

        except Exception as e:
            logging.error(f"Initialization error: {e}")
            messagebox.showerror("Error", f"Initialization error: {e}")
            self.quit()

    def load_cascade(self):
        try:
            face_cascade_path = 'haarcascade_frontalface_default.xml'
            face_cascade = cv2.CascadeClassifier(face_cascade_path)
            if face_cascade.empty():
                raise IOError(f"Failed to load Haar Cascade classifier from {face_cascade_path}")
            return face_cascade
        except Exception as e:
            logging.error(f"Error loading Haar Cascade: {e}")
            raise

    def update_frame(self):
        try:
            ret, frame = self.video_capture.read()
            if not ret:
                self.root.after(10, self.update_frame)
                return

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray_frame,
                scaleFactor=1.1,
                minNeighbors=7,
                minSize=(100, 100),
                maxSize=(400, 400)
            )

            if self.adding_user_mode:
                for (x, y, w, h) in faces:
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    face_roi = frame[y:y + h, x:x + w]
                    self.encodings.append(encode_face(face_roi))

                    if len(self.encodings) >= self.num_photos:
                        self.complete_add_user()
                        break

            else:
                for (x, y, w, h) in faces:
                    face_roi = frame[y:y + h, x:x + w]
                    encoding = encode_face(face_roi)
                    name = recognize_face(encoding)
                    self.draw_rectangle(frame, (x, y, w, h), name)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)

            self.root.after(10, self.update_frame)

        except Exception as e:
            logging.error(f"Error updating frame: {e}")
            messagebox.showerror("Error", f"Error updating frame: {e}")

    def prepare_to_add_user(self):
        self.adding_user_mode = True
        self.encodings = []
        messagebox.showinfo("Info", "Capturing user images...")

    def complete_add_user(self):
        self.adding_user_mode = False
        user_name = simpledialog.askstring("Input", "Enter user name:", parent=self.root)

        if user_name:
            if add_user(self.encodings, user_name):
                messagebox.showinfo("Success", f"User {user_name} added successfully!")
            else:
                messagebox.showerror("Error", "No faces detected.")
        else:
            messagebox.showwarning("Warning", "No user name entered. User not added.")

    def draw_rectangle(self, img, bbox, name):
        x, y, w, h = bbox
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        cv2.putText(img, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    def quit(self):
        if self.video_capture.isOpened():
            self.video_capture.release()
        self.root.quit()

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = CameraApp(root)
        root.mainloop()
    except Exception as e:
        logging.error(f"Application error: {e}")
