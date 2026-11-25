
import cv2
import mediapipe as mp
import numpy as np
import pickle

from pathlib import Path
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model


VERSION = 4
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / f"v{VERSION}"

model = load_model(MODEL_DIR/"hand_gesture_model.keras")
with open(MODEL_DIR/"labels.pickle", "rb") as f:
    labels = pickle.load(f)


labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D',
    4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L',
    12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X',
    24: 'Y', 25: 'Z'
}

app = FastAPI(title="BISINDO Recognition API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    min_detection_confidence=0.3,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

def extract_coordinates(results):

    if not results.multi_hand_landmarks:
        return None, None

    hand_data = {
        "Right": None,
        "Left": None
    }

    all_x = []
    all_y = []

    # Extract landmarks and handedness
    for i, hand_landmarks in enumerate(
        results.multi_hand_landmarks
    ):

        mp_label = (
            results.multi_handedness[i]
            .classification[0].label
        )

        hand_label = (
            "Right" if mp_label == "Left" else "Left"
        )

        coords = []
        for landmark in hand_landmarks.landmark:

            x = landmark.x
            y = landmark.y

            coords.append((x, y))

            all_x.append(x)
            all_y.append(y)

        hand_data[hand_label] = coords

    min_x = min(all_x)
    min_y = min(all_y)

    max_x = max(all_x)
    max_y = max(all_y)

    width = max_x - min_x
    height = max_y - min_y

    scale = max(width, height)

    if scale == 0:
        return None, None

    data_aux = []

    # RIGHT HAND FIRST, LEFT HAND SECOND
    for hand_label in ["Right", "Left"]:

        if hand_data[hand_label] is not None:

            for x, y in hand_data[hand_label]:

                normalized_x = (x - min_x) / scale
                normalized_y = (y - min_y) / scale

                data_aux.append(normalized_x)
                data_aux.append(normalized_y)

        else:
            # Missing hand = 42 zeros
            data_aux.extend([0.0] * 42)

    return data_aux, hand_data

@app.get("/")
def home():
    return {
        "message": "BISINDO API is running"
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    try:
        # Read uploaded webcam image
        contents = await file.read()

        image_array = np.frombuffer(
            contents,
            dtype=np.uint8
        )

        frame = cv2.imdecode(
            image_array,
            cv2.IMREAD_COLOR
        )

        if frame is None:
            return {
                "success": False,
                "message": "Invalid image"
            }

        # Same BGR -> RGB conversion as notebook
        frame_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = hands.process(frame_rgb)
        data_aux, hand_data = extract_coordinates(results)

        # No hand detected
        if data_aux is None:
            return {
                "success": True,
                "detected": False,
                "message": "No hand detected"
            }

        # Validate feature count
        if len(data_aux) != 84:
            return {
                "success": False,
                "message": "Invalid feature count"
            }

        # Predict using existing model
        prediction = model.predict(
            np.asarray(
                [data_aux],
                dtype=np.float32
            ),
            verbose=0
        )

        predicted_class = int( np.argmax(prediction[0]))
        predicted_character = labels_dict[predicted_class]
        confidence = float(np.max(prediction[0]))

        return {
            "success": True,
            "detected": True,
            "prediction": predicted_character,
            "class_id": predicted_class,
            "confidence": confidence,
            "feature_count": len(data_aux),
            "hands_detected": sum(
                hand is not None
                for hand in hand_data.values()
            )
        }

    except Exception as e:

        return {
            "success": False,
            "message": str(e)
        }


@app.on_event("shutdown")
def shutdown():
    hands.close()
