import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter
import paho.mqtt.client as mqtt 

# MQTT setup
broker_address = ""  # Your MQTT broker IP
topic = "sign_language/sentence"
client = mqtt.Client()
client.connect(broker_address, 1883, 60)

# Load model and labels
model = load_model("landmark_model/landmark_model.h5")
with open("landmark_model/labels.txt", "r") as f:
    class_names = f.read().splitlines()

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_drawing = mp.solutions.drawing_utils

# Video capture
cap = cv2.VideoCapture(0)

# Variables for sentence forming
current_sentence = ""
final_sentence = ""
buffer = deque(maxlen=15)
min_consistent_count = 10
last_confirmed = ""
hand_visible = False
hand_missing_frames = 0
max_missing_frames = 15  # Number of frames to wait before printing final sentence

# Cooldown to allow repeated letters
confirmed_letter_timer = 0
confirm_delay_frames = 20  # Minimum gap before repeating same letter

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_visible = True
        hand_missing_frames = 0

        for handLms in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])

            if len(landmarks) == 63:
                prediction = model.predict(np.array([landmarks]), verbose=0)
                class_id = np.argmax(prediction)
                predicted_label = class_names[class_id]
                buffer.append(predicted_label)

                # Stable prediction logic with cooldown
                most_common, count = Counter(buffer).most_common(1)[0]
                if count >= min_consistent_count:
                    if confirmed_letter_timer == 0 or most_common != last_confirmed:
                        if most_common.lower() == "space":
                            current_sentence += " "
                        elif most_common.lower() in ["stop", "fullstop"]:
                            current_sentence += "."
                        else:
                            current_sentence += most_common.upper()

                        last_confirmed = most_common
                        confirmed_letter_timer = confirm_delay_frames

    else:
        if hand_visible:
            hand_missing_frames += 1
            if hand_missing_frames >= max_missing_frames:
                # Finalize sentence
                final_sentence = current_sentence
                print("Final Sentence:", final_sentence)
                if final_sentence.strip() != "":
                    client.publish(topic, final_sentence)
                current_sentence = ""
                last_confirmed = ""
                buffer.clear()
                hand_visible = False
                hand_missing_frames = 0

    # Decrease cooldown timer
    if confirmed_letter_timer > 0:
        confirmed_letter_timer -= 1

    # Overlay current sentence
    cv2.putText(frame, f"Current: {current_sentence}", (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # If finalized, show final sentence
    if final_sentence:
        cv2.putText(frame, f"Final: {final_sentence}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("ASL Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

