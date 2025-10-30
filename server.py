from flask import Flask, render_template, request, send_from_directory,send_file,jsonify
import numpy as np
import os
from ultralytics import YOLO
from pathlib import Path
import base64
from datetime import datetime
import cv2

model = YOLO("yolo_models/helmet.pt")

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def helmet_detection():
    if "file" not in request.files:
        return jsonify({"error": "No image provided"}), 400

    file = request.files["file"]
    image_bytes = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)

    # Run inference (no saving)
    results = model(img, conf=0.5)

    # Get annotated frame as numpy array
    annotated_frame = results[0].plot()  # Draw boxes, labels, etc.

    # Encode annotated image to base64
    _, buffer = cv2.imencode(".jpg", annotated_frame)
    img_base64 = base64.b64encode(buffer).decode("utf-8")

    # Optional: also return detection info
    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": model.names[int(box.cls)],
            "confidence": float(box.conf),
            "bbox": box.xyxy[0].tolist()
        })

    return jsonify({
        "image": img_base64,
        "detections": detections
    })

    

if __name__ == '__main__':
    app.run(debug=True)


