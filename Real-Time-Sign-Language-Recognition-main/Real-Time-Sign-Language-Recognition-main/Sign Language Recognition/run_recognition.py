import cv2
import mediapipe as mp
import torch
import numpy as np
from CNNModel import CNNModel

# Load trained alphabet model
model = CNNModel()
model.load_state_dict(torch.load("CNN_model_alphabet_SIBI.pth", map_location=torch.device('cpu')))
model.eval()

# Mediapipe setup
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5
)

mp_draw = mp.solutions.drawing_utils

# Alphabet labels
labels = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Open webcam
cap = cv2.VideoCapture(0)

print("Press ESC to exit")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            data = []

            x_coords = []
            y_coords = []
            z_coords = []

            for lm in hand_landmarks.landmark:

                x_coords.append(lm.x)
                y_coords.append(lm.y)
                z_coords.append(lm.z)

            for lm in hand_landmarks.landmark:

                data.append(lm.x - min(x_coords))
                data.append(lm.y - min(y_coords))
                data.append(lm.z - min(z_coords))

            data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)

            prediction = model(data)

            predicted_class = torch.argmax(prediction).item()

            letter = labels[predicted_class]

            cv2.putText(
                frame,
                letter,
                (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX,
                3,
                (0,255,0),
                5
            )

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()