from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

KOG_IMG_HEIGHT = 128
KOG_IMG_WIDTH = 128
app = Flask(__name__)
CORS(app)
yolo_model = YOLO('yolo_v5.pt')
kog_model = load_model('kog.keras')
class_names = [
    'ace of clubs', 'ace of diamonds', 'ace of hearts', 'ace of spades',
    'eight of clubs', 'eight of diamonds', 'eight of hearts', 'eight of spades',
    'five of clubs', 'five of diamonds', 'five of hearts', 'five of spades',
    'four of clubs', 'four of diamonds', 'four of hearts', 'four of spades',
    'jack of clubs', 'jack of diamonds', 'jack of hearts', 'jack of spades',
    'joker',
    'king of clubs', 'king of diamonds', 'king of hearts', 'king of spades',
    'nine of clubs', 'nine of diamonds', 'nine of hearts', 'nine of spades',
    'queen of clubs', 'queen of diamonds', 'queen of hearts', 'queen of spades',
    'seven of clubs', 'seven of diamonds', 'seven of hearts', 'seven of spades',
    'six of clubs', 'six of diamonds', 'six of hearts', 'six of spades',
    'ten of clubs', 'ten of diamonds', 'ten of hearts', 'ten of spades',
    'three of clubs', 'three of diamonds', 'three of hearts', 'three of spades',
    'two of clubs', 'two of diamonds', 'two of hearts', 'two of spades'
]


def predict_and_annotate(img_rgb, model, conf=0.1):
    results = model.predict(img_rgb, conf=conf)
    annotator = Annotator(img_rgb)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0]
            c = box.cls
            annotator.box_label(b, model.names[int(c)])
    return annotator.result()


@app.route('/detect/yolo', methods=['POST'])
def detect():
    data = request.json
    img_data = data['imgData']
    confidence = data['confidence']
    img_data = base64.b64decode(img_data.split(',')[1])
    img_np = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    annotated_img = predict_and_annotate(img, yolo_model, confidence)
    _, img_encoded = cv2.imencode('.png', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    img_data = base64.b64encode(img_encoded).decode('utf-8')
    return jsonify({'imgData': 'data:image/png;base64,' + img_data})


@app.route('/detect/kog', methods=['POST'])
def detect_kog():
    data = request.json
    img_data = data['imgData']
    img_data = base64.b64decode(img_data.split(',')[1])
    img_np = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (KOG_IMG_HEIGHT, KOG_IMG_WIDTH))
    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, 0)
    prediction = kog_model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    return jsonify({'predictedClass': predicted_class})


@app.route('/detect/yolo-kog', methods=['POST'])
def detect_yolo_kog():
    data = request.json
    img_data = data['imgData']
    confidence = data['confidence']
    img_data = base64.b64decode(img_data.split(',')[1])
    img_np = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Use YOLO model to detect the card
    results = yolo_model.predict(img, conf=confidence)
    annotated_img = img.copy()
    all_boxes = [(box, r) for r in results for box in r.boxes]
    if not all_boxes:
        return jsonify({'error': 'No card detected in the image.'})

    # Sort the boxes by area (smallest first)
    all_boxes.sort(key=lambda x: (x[0].xyxy[0][2] - x[0].xyxy[0][0]) * (x[0].xyxy[0][3] - x[0].xyxy[0][1]))

    # Take the smallest box
    box, r = all_boxes[0]
    b = box.xyxy[0]
    # Annotate the detected card on the original image
    cv2.rectangle(annotated_img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)
    # Crop the image to only include the detected card
    cropped_img = img[int(b[1]):int(b[3]), int(b[0]):int(b[2])]
    # Resize the cropped image to the input size expected by the KOG model
    cropped_img = cv2.resize(cropped_img, (KOG_IMG_HEIGHT, KOG_IMG_WIDTH))
    img_array = img_to_array(cropped_img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, 0)
    # Use KOG model to classify the cropped card
    prediction = kog_model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    # Encode the cropped image and the annotated image to base64 strings
    _, cropped_img_encoded = cv2.imencode('.png', cv2.cvtColor(cropped_img, cv2.COLOR_RGB2BGR))
    cropped_img_data = base64.b64encode(cropped_img_encoded).decode('utf-8')
    _, annotated_img_encoded = cv2.imencode('.png', cv2.cvtColor(annotated_img, cv2.COLOR_RGB2BGR))
    annotated_img_data = base64.b64encode(annotated_img_encoded).decode('utf-8')
    return jsonify({'predictedClass': predicted_class, 'cropped': 'data:image/png;base64,' + cropped_img_data,
                    'detected': 'data:image/png;base64,' + annotated_img_data})


if __name__ == "__main__":
    app.run(port=3005)
