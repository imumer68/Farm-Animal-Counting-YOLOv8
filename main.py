import streamlit as st
from PIL import Image
import numpy as np
import cv2
import tempfile
from ultralytics import YOLO

# Load YOLOv8 modelS
model = YOLO('farm_animal_detector.pt')

st.title("Animal Counting using YOLOv8")
st.write("Upload an image or video to detect and count animals.")

# Image upload and detection
uploaded_file = st.file_uploader("Choose an image or video...", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])


def count_animals(results, model):
    labels = [model.names[int(label)] for label in results.boxes.cls]
    unique_labels, counts = np.unique(labels, return_counts=True)
    return dict(zip(unique_labels, counts))


def draw_animal_counts(image_np, animal_counts):
    overlay = image_np.copy()
    y0, dy = 40, 30  # Starting position and vertical space between lines

    for i, (animal, count) in enumerate(animal_counts.items()):
        y = y0 + i * dy
        text = f"{animal}: {count}"
        text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_COMPLEX, 1, 2)
        text_width, text_height = text_size

        # Drawing the semi-transparent rectangle
        # cv2.rectangle(overlay, (10, y - text_height - 10), (10 + text_width + 10, y + 10), (0, 0, 0), -1)

        # Adding text
        cv2.putText(overlay, text, (8, y), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 0, 0), 2)

    # Apply the overlay
    alpha = 0.8
    cv2.addWeighted(overlay, alpha, image_np, 1 - alpha, 0, image_np)
    return image_np


if uploaded_file is not None:
    # For images
    if uploaded_file.type in ["image/jpeg", "image/png", "image/jpg"]:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        # Perform detection
        results = model(image_np)

        # Count animals
        animal_counts = count_animals(results[0], model)

        # Draw bounding boxes and labels
        for result in results:
            for bbox, score, label in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                x1, y1, x2, y2 = map(int, bbox)
                label = model.names[int(label)]
                confidence = float(score)
                cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image_np, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9,
                            (0, 255, 0), 2)

        # Draw animal counts on the image
        image_np = draw_animal_counts(image_np, animal_counts)

        st.image(image_np, caption="Processed Image with Animal Counts", use_column_width=True)

    # For videos
    else:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        video = cv2.VideoCapture(tfile.name)

        stframe = st.empty()

        animal_counts = {}

        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break

            # Perform detection
            results = model(frame)

            # Count animals
            frame_animal_counts = count_animals(results[0], model)
            for animal, count in frame_animal_counts.items():
                if animal in animal_counts:
                    animal_counts[animal] += count
                else:
                    animal_counts[animal] = count

            # Draw bounding boxes and labels
            for result in results:
                for bbox, score, label in zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls):
                    x1, y1, x2, y2 = map(int, bbox)
                    label = model.names[int(label)]
                    confidence = float(score)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_COMPLEX, 0.9,
                                (0, 255, 0), 2)

            # Draw animal counts on the frame
            frame = draw_animal_counts(frame, animal_counts)

            stframe.image(frame, channels="BGR")

        video.release()

        st.write("Total Animal Counts in Video:", animal_counts)
