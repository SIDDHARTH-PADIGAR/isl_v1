import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

DATA_DIR = './Data'  # Make sure this points to your ISL data directory
data = []
labels = []

for dir_ in os.listdir(DATA_DIR):
    class_folder = os.path.join(DATA_DIR, dir_)
    if not os.path.isdir(class_folder):
        continue

    for img_name in os.listdir(class_folder):
        img_path = os.path.join(class_folder, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue  # skip if file is not an image

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # We want exactly 84 features (42 for hand #1, 42 for hand #2)
        features_2hand = [0.0] * 84  # Fill with zeros initially

        if results.multi_hand_landmarks:
            # We'll process up to two hands max
            hand_count = min(len(results.multi_hand_landmarks), 2)

            for i in range(hand_count):
                hand_landmarks = results.multi_hand_landmarks[i]

                # Collect all x,y for normalizing
                x_ = []
                y_ = []
                for lm_id in range(21):
                    x_.append(hand_landmarks.landmark[lm_id].x)
                    y_.append(hand_landmarks.landmark[lm_id].y)

                # Build a 42-element list for this hand
                hand_features = []
                for lm_id in range(21):
                    # anchor the coords at (min(x_), min(y_))
                    x_val = hand_landmarks.landmark[lm_id].x - min(x_)
                    y_val = hand_landmarks.landmark[lm_id].y - min(y_)
                    hand_features.append(x_val)
                    hand_features.append(y_val)

                # If i == 0 => first 42 slots, if i == 1 => next 42
                start_idx = i * 42
                for j in range(42):
                    features_2hand[start_idx + j] = hand_features[j]

        # Append the 84-feature vector
        data.append(features_2hand)
        labels.append(dir_)

# Finally, save the data
with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)

print("Data creation complete. Saved to data.pickle.")
