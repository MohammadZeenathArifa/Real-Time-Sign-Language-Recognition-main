import os
import mediapipe
import cv2
from PIL import Image
import pandas as pd
import numpy as np

# Initialize MediaPipe
handTracker = mediapipe.solutions.hands
handDetector = handTracker.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.2
)

# Folder containing character folders (A, B, C, etc.)
DATA_FOLDER = '../Data'

os.environ["PYTHONIOENCODING"] = "utf-8"

coordinates = []
index = 0

# Loop through each character folder
for file in os.listdir(DATA_FOLDER):

    folder_path = os.path.join(DATA_FOLDER, file)

    # Skip if not a folder
    if not os.path.isdir(folder_path):
        continue

    for imgPath in os.listdir(folder_path):

        fullImgPath = os.path.join(folder_path, imgPath)

        image = Image.open(fullImgPath)
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        if img is None:
            print(f"Failed to load image from {fullImgPath}")
            continue

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgMediapipe = handDetector.process(imgRGB)

        if imgMediapipe.multi_hand_landmarks:

            for handLandmarks in imgMediapipe.multi_hand_landmarks:

                data = {}
                data['CHARACTER'] = file
                data['GROUPVALUE'] = index

                x_coords = []
                y_coords = []
                z_coords = []

                for lm in handLandmarks.landmark:
                    x_coords.append(lm.x)
                    y_coords.append(lm.y)
                    z_coords.append(lm.z)

                for i, landmark in enumerate(handTracker.HandLandmark):

                    lm = handLandmarks.landmark[i]

                    data[f'{landmark.name}_x'] = lm.x - min(x_coords)
                    data[f'{landmark.name}_y'] = lm.y - min(y_coords)
                    data[f'{landmark.name}_z'] = lm.z - min(z_coords)

                coordinates.append(data)

    index += 1

# Convert to DataFrame
df = pd.DataFrame(coordinates)

excel_path = "output_landmarks.xlsx"

df.to_excel(excel_path, index=False)

print("Excel file created successfully!")