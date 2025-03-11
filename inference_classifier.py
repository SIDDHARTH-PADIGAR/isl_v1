import pickle
import cv2
import mediapipe as mp
import numpy as np

# 1) Load your trained model (84-feature, two-hand approach)
model_dict = pickle.load(open('model_84features.p', 'rb'))  # Use the new 84-feature model
model = model_dict['model']

# 2) Initialize camera
cap = cv2.VideoCapture(0)

# 3) Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# For live video with two hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,  # Important: Set to detect up to 2 hands
    min_detection_confidence=0.3
)

# 4) Label dictionary for your ISL classes (adjust based on your classes)
labels_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
    8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
    15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
    22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Mirror the frame for more intuitive interaction
    frame = cv2.flip(frame, 1)

    H, W, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    # We'll create an 84-element vector: 42 for hand #1, 42 for hand #2
    data_2hand = [0.0] * 84
    predicted_char = None
    confidence = None

    if results.multi_hand_landmarks:
        # Up to two hands
        hand_count = min(len(results.multi_hand_landmarks), 2)

        # Track hand positions for visualization
        hand_positions = []

        for i in range(hand_count):
            hand_landmarks = results.multi_hand_landmarks[i]

            # Draw the landmarks on the frame
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Collect x,y for normalizing
            x_list = []
            y_list = []
            for lm_id in range(21):
                x_val = hand_landmarks.landmark[lm_id].x
                y_val = hand_landmarks.landmark[lm_id].y
                x_list.append(x_val)
                y_list.append(y_val)

            # Build 42 features for this hand
            hand_features = []
            for lm_id in range(21):
                x_val = hand_landmarks.landmark[lm_id].x - min(x_list)
                y_val = hand_landmarks.landmark[lm_id].y - min(y_list)
                hand_features.append(x_val)
                hand_features.append(y_val)

            # Place them in the correct segment of data_2hand
            start_idx = i * 42
            for j in range(42):
                data_2hand[start_idx + j] = hand_features[j]

            # Save hand position for visualization
            hand_positions.append((x_list, y_list))

        # Get prediction and probability
        try:
            prediction = model.predict([np.asarray(data_2hand)])
            pred_idx = int(prediction[0])
            predicted_char = labels_dict.get(pred_idx, "?")

            # Get prediction probability if available
            if hasattr(model, "predict_proba"):
                proba = model.predict_proba([np.asarray(data_2hand)])[0]
                confidence = proba[pred_idx] * 100
        except Exception as e:
            print(f"Prediction error: {e}")
            cv2.putText(frame, f"Error: {str(e)[:30]}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display the prediction
        if predicted_char:
            # Display prediction in center top
            text = f"{predicted_char}"
            if confidence is not None:
                text += f" ({confidence:.1f}%)"

            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0]
            text_x = int((W - text_size[0]) / 2)
            cv2.putText(frame, text, (text_x, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3, cv2.LINE_AA)

            # Draw rectangle around each hand
            for x_list, y_list in hand_positions:
                x1 = int(min(x_list) * W) - 10
                y1 = int(min(y_list) * H) - 10
                x2 = int(max(x_list) * W) + 10
                y2 = int(max(y_list) * H) + 10

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Display hand count and instructions
    if results.multi_hand_landmarks:
        hand_text = f"Hands: {len(results.multi_hand_landmarks)}"
    else:
        hand_text = "No hands detected"
    cv2.putText(frame, hand_text, (20, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

    # Display instructions
    cv2.putText(frame, "Press 'q' to quit", (W - 150, H - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imshow('ISL Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()