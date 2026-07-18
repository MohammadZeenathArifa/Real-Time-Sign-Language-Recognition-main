from flask import Flask, render_template, jsonify, Response
import cv2
import mediapipe as mp
import torch
import numpy as np
import sys

print("App file executed")

# allow importing model
sys.path.append("Sign Language Recognition")

from CNNModel import CNNModel

app = Flask(__name__)

print("Starting Flask Sign Language App...")

# Load model
model = CNNModel()
model.load_state_dict(torch.load(
    "Sign Language Recognition/CNN_model_alphabet_SIBI.pth",
    map_location="cpu"
))
model.eval()

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands()

# Camera
cap = cv2.VideoCapture(0)

# GLOBAL VARIABLE FOR PREDICTION
current_prediction = "--"


def generate_frames():

    global current_prediction

    while True:

        success, frame = cap.read()

        if not success:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        result = hands.process(rgb)

        if result.multi_hand_landmarks:

            for hand_landmarks in result.multi_hand_landmarks:

                # Draw hand skeleton grid
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                h, w, c = frame.shape

                x_list = []
                y_list = []
                z_list = []

                for lm in hand_landmarks.landmark:

                    x_list.append(lm.x)
                    y_list.append(lm.y)
                    z_list.append(lm.z)

                # Draw square around hand
                x_pixels = [int(x * w) for x in x_list]
                y_pixels = [int(y * h) for y in y_list]

                xmin, xmax = min(x_pixels), max(x_pixels)
                ymin, ymax = min(y_pixels), max(y_pixels)

                cv2.rectangle(
                    frame,
                    (xmin - 20, ymin - 20),
                    (xmax + 20, ymax + 20),
                    (0, 255, 0),
                    2
                )

                # Prepare model input
                data = []

                for i in range(len(x_list)):
                    data.append(x_list[i] - min(x_list))
                    data.append(y_list[i] - min(y_list))
                    data.append(z_list[i] - min(z_list))

                data = np.array(data, dtype=np.float32)

                data = torch.tensor(data).view(1, 63, 1)

                output = model(data)

                pred = torch.argmax(output, dim=1)

                # STORE PREDICTION
                current_prediction = labels[pred.item()]

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)

        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')



@app.route("/")
def home():
    return render_template("index.html")


@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/predict")
def predict():
    global current_prediction
    return jsonify({"prediction": current_prediction})


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)