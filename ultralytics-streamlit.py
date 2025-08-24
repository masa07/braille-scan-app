# uv add ultralytics
# uv add streamlit
# uv add lap
# uv run streamlit run app/ultralytics-streamlit.py

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time
import tile

# Load YOLO model
model = YOLO("best.pt")

# Streamlit app setup
st.title("Object Detection (YOLOv8)")

# Input source selection
input_source = st.radio("Select input source:", ("Webcam", "Upload Image"))

# Webcam functionality
if input_source == "Webcam":
    start_button = st.button("Start Camera")
    stop_button = st.button("Stop Camera")
    FRAME_WINDOW = st.image([])
    results_placeholder = st.empty() # Placeholder for displaying results

    # Camera setup
    camera = cv2.VideoCapture(0)

    # App state
    if 'running' not in st.session_state:
        st.session_state['running'] = False

    # Button actions
    if start_button:
        st.session_state['running'] = True
    if stop_button:
        st.session_state['running'] = False

    # Real-time object detection
    if st.session_state['running']:
        while True:
            _, frame = camera.read()
            if frame is None:
                st.write("Camera disconnected or unavailable.")
                st.session_state['running'] = False
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform inference every 1 second
            # results = model(frame)
            # annotated_frame = results[0].plot()
            # FRAME_WINDOW.image(annotated_frame, channels="RGB")

            # # Process and display results
            # # You can access detection information like bounding boxes, classes, and confidence scores
            # # Example: Display the detected classes and confidence scores
            # detections = results[0].boxes.data.tolist()
            # results_text = "Detected objects:\n"
            # if detections:
            #     for detection in detections:
            #         x1, y1, x2, y2, confidence, class_id = detection
            #         results_text += f"- Class ID: {int(class_id)}, Confidence: {confidence:.2f}\n"
            # else:
            #     results_text += "No objects detected."
            # results_placeholder.text(results_text) # Display results in the placeholder

            all_results = tile.tile_predect(model, None, tile_size=640, overlap=0.2, image=frame)
            final_detections = tile.nms_predictions(all_results)
            annotated_frame, labels = tile.visualize_detections(None, final_detections, model.names, image=frame)
            FRAME_WINDOW.image(annotated_frame, caption="Detected Objects.", use_container_width=True)
            results_placeholder.text(labels)
            time.sleep(1)

            if not st.session_state['running']:
                break
    else:
        st.write("Camera stopped.")

    camera.release()

# Upload Image functionality
elif input_source == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image file
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Display the uploaded image
        st.image(image, caption="Uploaded Image.", use_container_width=True)

        # # Perform inference
        # results = model(image)
        # annotated_frame = results[0].plot()

        # # Display the annotated image
        # st.image(annotated_frame, caption="Detected Objects.", use_container_width=True)

        # # Process and display results
        # detections = results[0].boxes.data.tolist()
        # results_text = "Detected objects:\n"
        # if detections:
        #     for detection in detections:
        #         x1, y1, x2, y2, confidence, class_id = detection
        #         results_text += f"- Class ID: {int(class_id)}, Confidence: {confidence:.2f}\n"
        #     else:
        #         results_text += "No objects detected."
        # st.text(results_text)

        all_results = tile.tile_predect(model, uploaded_file, tile_size=640, overlap=0.2)
        final_detections = tile.nms_predictions(all_results)
        annotated_frame, labels = tile.visualize_detections(uploaded_file, final_detections, model.names)
        st.image(annotated_frame, caption="Detected Objects.", use_container_width=True)
        st.text(labels)
