from flask import Flask, render_template, request
import os
from ultralytics import YOLO
import numpy as np
import cv2
import base64
from datetime import datetime

app = Flask(__name__)

# Path to YOLO model (put your one-way model here). If missing, server will show an error on POST.
YOLO_MODEL = os.path.join(os.path.dirname(__file__), "yolo_models", "oneway.pt")
model = None
try:
    if os.path.exists(YOLO_MODEL):
        model = YOLO(YOLO_MODEL)
    else:
        # Attempt to load a local yolov8n weight if user has it; otherwise model stays None
        try:
            model = YOLO("yolov8n.pt")
        except Exception:
            model = None
except Exception:
    model = None


@app.route('/', methods=['GET', 'POST'])
def one_way_detection():
    # Render template on GET. On POST, handle upload and detection.
    if request.method == 'GET':
        return render_template('objectdetection.html', message='', result_image='')

    # POST
    if 'file' not in request.files:
        return render_template('objectdeection.html', message='No file uploaded', result_image='')

    file = request.files['file']
    if file.filename == '':
        return render_template('objectdetection.html', message='No file selected', result_image='')

    uploads_dir = os.path.join(app.static_folder, 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)

    # Save original upload
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"upload_{timestamp}_{file.filename}"
    save_path = os.path.join(uploads_dir, filename)
    file.save(save_path)

    if model is None:
        return render_template('objectdetection.html', message=f'Model not loaded. Place your model at {YOLO_MODEL}', result_image=filename)

    # Run inference
    img = cv2.imread(save_path)
    results = model(img, conf=0.5)
    annotated = results[0].plot()

    # Save annotated image
    annotated_name = f"annotated_{timestamp}_{file.filename}.jpg"
    annotated_path = os.path.join(uploads_dir, annotated_name)
    cv2.imwrite(annotated_path, annotated)

    return render_template('objectdetection.html', message='Detection complete', result_image=annotated_name)


if __name__ == '__main__':
    app.run(debug=True,port=5001)