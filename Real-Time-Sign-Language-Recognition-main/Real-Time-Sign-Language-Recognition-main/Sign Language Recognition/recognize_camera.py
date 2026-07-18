import cv2
import mediapipe as mp
import torch
import numpy as np
from CNNModel import CNNModel

# -----------------------------
# Load trained model
# -----------------------------
model = CNNModel()
model.load_state_dict(
    torch.load("CNN_model_alphabet_SIBI.pth",
               map_location=torch.device("cpu"))
)
model.eval()

print("Model loaded successfully")

# -----------------------------
# Mediapipe setup
# -----------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# -----------------------------
# Camera setup (Windows safe)
# -----------------------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not opened")
    exit()

print("Camera started")

labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# -----------------------------
# Main loop
# -----------------------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            try:

                # collect landmarks
                data = []

                x_list = []
                y_list = []
                z_list = []

                for lm in hand_landmarks.landmark:
                    x_list.append(lm.x)
                    y_list.append(lm.y)
                    z_list.append(lm.z)

                # normalize same as training
                for i in range(len(x_list)):
                    data.append(x_list[i] - min(x_list))
                    data.append(y_list[i] - min(y_list))
                    data.append(z_list[i] - min(z_list))

                data = np.array(data, dtype=np.float32)

                # reshape correctly → (batch, channels, length)
                data = torch.tensor(data).view(1, 63, 1)

                # prediction
                output = model(data)

                pred = torch.argmax(output, dim=1)

                letter = labels[pred.item()]

                cv2.putText(
                    frame,
                    f"Prediction: {letter}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5,
                    (0,255,0),
                    3
                )

            except Exception as e:
                print("Prediction error:", e)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()