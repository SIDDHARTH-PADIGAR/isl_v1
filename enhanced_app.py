import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from enum import Enum
import random
import pyttsx3
import threading

class AppMode(Enum):
    ALPHABET_PRACTICE = 1
    WORD_BUILDER = 2
    QUIZ_MODE = 3
    STORY_TIME = 4

class SignLanguageApp:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        # Set static_image_mode=False for live camera
        self.hands = self.mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.3)

        # Load the 2-hand model
        self.model_dict = pickle.load(open('./model.p', 'rb'))
        self.model = self.model_dict['model']

        # If your ISL dataset has custom labels, update these accordingly:
        # e.g., if you have 5 classes, do {0: 'CLASS1', 1: 'CLASS2', ...}
        self.labels_dict = {
            0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H',
            8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O',
            15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V',
            22: 'W', 23: 'X', 24: 'Y', 25: 'Z'
        }

        # Word and story samples (you can change to ISL words if needed)
        self.practice_words = ["HELLO", "THANK", "YOU", "PLEASE", "GOOD", "BAD", "YES", "NO"]
        self.stories = [
            {"text": "I AM HAPPY", "prompt": "Express your feelings"},
            {"text": "HOW ARE YOU", "prompt": "Ask about someone's wellbeing"},
            {"text": "NICE TO MEET", "prompt": "Greet someone new"}
        ]

        # Word builder variables
        self.current_target_word = None
        self.word_builder_progress = 0

        # Story mode variables
        self.current_story = None
        self.story_progress = 0
        self.current_story_prompt = None

        # Text-to-speech engine
        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', 130)
        self.engine.setProperty('volume', 0.9)
        self.speech_thread = None
        self.speech_lock = threading.Lock()

        self.last_letter_time = time.time()

        # Video capture
        self.cap = cv2.VideoCapture(0)

        # App state
        self.current_mode = AppMode.ALPHABET_PRACTICE
        self.scores = {
            'alphabet': {'correct': 0, 'attempts': 0},
            'word': {'correct': 0, 'attempts': 0, 'words_completed': 0},
            'quiz': {'correct': 0, 'attempts': 0, 'streak': 0},
            'story': {'correct': 0, 'attempts': 0, 'stories_completed': 0}
        }
        self.last_prediction_time = time.time()
        self.prediction_buffer = 2.0
        self.attempts = 0
        self.current_word = ""
        self.target_letter = None
        self.collected_letters = []
        self.feedback_timer = 0
        self.show_feedback = False

        self.feedback_text = "Great job! Keep going!"
        # This is overwritten each time you show feedback

    def speak_text(self, text):
        """Speak text in a separate thread to avoid blocking the main application."""
        def speak_worker():
            with self.speech_lock:
                self.engine.say(text)
                self.engine.runAndWait()

        # Cancel any ongoing speech
        if self.speech_thread and self.speech_thread.is_alive():
            self.engine.stop()
            self.speech_thread.join()

        # Start new speech thread
        self.speech_thread = threading.Thread(target=speak_worker)
        self.speech_thread.start()

    def process_hand_landmarks(self, frame):
        """Collect up to 2 hands, each 42 features => total 84 features."""
        H, W, _ = frame.shape
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(frame_rgb)

        data_2hand = [0.0] * 84  # 84 features: 42 per hand
        predicted_character = None

        if results.multi_hand_landmarks:
            # We'll process up to two hands
            hand_count = min(len(results.multi_hand_landmarks), 2)

            for i in range(hand_count):
                hand_landmarks = results.multi_hand_landmarks[i]
                # Draw landmarks on the frame
                self.mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )

                # Collect x,y for normalizing
                x_ = []
                y_ = []
                for lm_id in range(21):
                    x_.append(hand_landmarks.landmark[lm_id].x)
                    y_.append(hand_landmarks.landmark[lm_id].y)

                # Build 42 features for this hand
                hand_features = []
                for lm_id in range(21):
                    x_val = hand_landmarks.landmark[lm_id].x - min(x_)
                    y_val = hand_landmarks.landmark[lm_id].y - min(y_)
                    hand_features.append(x_val)
                    hand_features.append(y_val)

                # Place them in the correct segment of data_2hand
                start_idx = i * 42
                for j in range(42):
                    data_2hand[start_idx + j] = hand_features[j]

            # Predict once we've built data_2hand
            prediction = self.model.predict([np.asarray(data_2hand)])
            pred_idx = int(prediction[0])
            predicted_character = self.labels_dict.get(pred_idx, "?")

            # Optionally draw bounding box for the FIRST hand
            first_hand = results.multi_hand_landmarks[0]
            x_vals = [lm.x for lm in first_hand.landmark]
            y_vals = [lm.y for lm in first_hand.landmark]
            x1, x2 = int(min(x_vals)*W)-10, int(max(x_vals)*W)+10
            y1, y2 = int(min(y_vals)*H)-10, int(max(y_vals)*H)+10

            cv2.rectangle(frame, (x1, y1), (x2, y2), (120, 220, 120), 2)
            cv2.putText(frame, predicted_character, (x1 + 5, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

        return frame, predicted_character

    # ------------------ MODE HANDLERS ------------------
    def handle_alphabet_practice(self, predicted_char):
        if self.target_letter is None:
            self.target_letter = random.choice(list(self.labels_dict.values()))
            self.speak_text(f"Show me the letter {self.target_letter}")

        current_time = time.time()
        if predicted_char and (current_time - self.last_prediction_time) >= self.prediction_buffer:
            if not hasattr(self, 'last_prediction') or predicted_char != self.last_prediction:
                self.scores['alphabet']['attempts'] += 1
                if predicted_char == self.target_letter:
                    self.scores['alphabet']['correct'] += 1
                    self.target_letter = random.choice(list(self.labels_dict.values()))
                    self.feedback_text = "Great job! Keep going!"
                    self.speak_text("Correct! Great job!")
                else:
                    self.feedback_text = f"Try again! Make the letter: {self.target_letter}"
                    self.speak_text("Try again")

                self.last_prediction = predicted_char
            self.last_prediction_time = current_time

    def handle_word_builder(self, predicted_char):
        if self.current_target_word is None:
            self.current_target_word = random.choice(self.practice_words)
            self.collected_letters = []
            self.word_builder_progress = 0
            self.speak_text(f"Spell the word: {' '.join(self.current_target_word)}")

        current_time = time.time()
        if predicted_char and (current_time - self.last_prediction_time) >= self.prediction_buffer:
            try:
                expected_letter = self.current_target_word[self.word_builder_progress]

                if predicted_char == expected_letter:
                    self.collected_letters.append(predicted_char)
                    self.word_builder_progress += 1
                    self.scores['word']['correct'] += 1
                    self.feedback_text = "Correct! Keep going!"

                    if self.word_builder_progress == len(self.current_target_word):
                        self.scores['word']['words_completed'] += 1
                        self.feedback_text = f"Congratulations! You completed the word: {self.current_target_word}"
                        self.speak_text("Word completed! Great job!")
                        # Reset for next word
                        self.current_target_word = random.choice(self.practice_words)
                        self.collected_letters = []
                        self.word_builder_progress = 0
                        self.speak_text(f"Next word: {' '.join(self.current_target_word)}")
                else:
                    self.feedback_text = f"Try again! Next letter should be: {expected_letter}"
                    self.speak_text("Incorrect letter, try again")

                self.scores['word']['attempts'] += 1
                self.last_prediction_time = current_time

            except Exception as e:
                print(f"Word builder error: {e}")
                self.collected_letters = []

    def handle_quiz_mode(self, predicted_char):
        if self.target_letter is None:
            self.target_letter = random.choice(list(self.labels_dict.values()))
            self.speak_text(f"Show me the letter {self.target_letter}")

        current_time = time.time()
        if predicted_char and (current_time - self.last_prediction_time) >= self.prediction_buffer:
            self.scores['quiz']['attempts'] += 1
            if predicted_char == self.target_letter:
                self.scores['quiz']['correct'] += 1
                self.scores['quiz']['streak'] += 1
                self.feedback_text = f"Correct! Streak: {self.scores['quiz']['streak']}"
                self.speak_text("Correct! Great job!")
                new_letter = random.choice(list(self.labels_dict.values()))
                while new_letter == self.target_letter:
                    new_letter = random.choice(list(self.labels_dict.values()))
                self.target_letter = new_letter
                self.speak_text(f"Next letter is {self.target_letter}")
            else:
                self.scores['quiz']['streak'] = 0
                self.feedback_text = f"Try again! Show the letter: {self.target_letter}"
                self.speak_text("Try again")

            self.last_prediction_time = current_time

    def handle_story_mode(self, predicted_char):
        if self.current_story is None:
            self.current_story = random.choice(self.stories)
            self.story_progress = 0
            self.collected_letters = []
            self.current_story_prompt = self.current_story["prompt"]
            self.speak_text(f"{self.current_story_prompt}. Sign: {self.current_story['text']}")

        current_time = time.time()
        if predicted_char and (current_time - self.last_prediction_time) >= self.prediction_buffer:
            try:
                story_words = self.current_story["text"].split()
                current_word = story_words[self.story_progress]

                if len(self.collected_letters) == len(current_word):
                    self.story_progress += 1
                    self.collected_letters = []
                    if self.story_progress >= len(story_words):
                        self.scores['story']['stories_completed'] += 1
                        self.feedback_text = "Story completed! Great job!"
                        self.speak_text("Story completed! Well done!")
                        # Reset for next story
                        self.current_story = random.choice(self.stories)
                        self.story_progress = 0
                        self.collected_letters = []
                        self.current_story_prompt = self.current_story["prompt"]
                        self.speak_text(f"New story! {self.current_story_prompt}")
                        return

                expected_letter = current_word[len(self.collected_letters)]
                if predicted_char == expected_letter:
                    self.collected_letters.append(predicted_char)
                    self.scores['story']['correct'] += 1
                    self.feedback_text = "Correct! Keep going!"
                else:
                    self.feedback_text = f"Try again! Expected letter: {expected_letter}"
                    self.speak_text("Incorrect letter, try again")

                self.scores['story']['attempts'] += 1
                self.last_prediction_time = current_time

            except Exception as e:
                print(f"Story mode error: {e}")
                self.current_story = random.choice(self.stories)
                self.story_progress = 0
                self.collected_letters = []

    # ------------------ DRAWING / UI ------------------
    def draw_mode_info(self, frame):
        height, width = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, 150), (255, 255, 255), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        cv2.putText(frame, f"Mode: {self.current_mode.name}",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if self.current_mode == AppMode.ALPHABET_PRACTICE:
            score_text = f"Score: {self.scores['alphabet']['correct']}/{self.scores['alphabet']['attempts']}"
            cv2.putText(frame, score_text, (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            if self.target_letter:
                cv2.putText(frame, f"Show Letter: {self.target_letter}",
                            (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        elif self.current_mode == AppMode.WORD_BUILDER:
            if self.current_target_word:
                target_text = f"Target Word: {self.current_target_word}"
                current_text = f"Your Progress: {''.join(self.collected_letters)}"
                score_text = (f"Words: {self.scores['word']['words_completed']}  "
                              f"Correct: {self.scores['word']['correct']}/{self.scores['word']['attempts']}")

                cv2.putText(frame, target_text, (20,80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0),2)
                cv2.putText(frame, current_text, (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, score_text, (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        elif self.current_mode == AppMode.QUIZ_MODE:
            if self.target_letter:
                target_text = f"Show Letter: {self.target_letter}"
                score_text = f"Score: {self.scores['quiz']['correct']}/{self.scores['quiz']['attempts']}"
                streak_text = f"Current Streak: {self.scores['quiz']['streak']}"

                cv2.putText(frame, target_text, (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, score_text, (20, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                cv2.putText(frame, streak_text, (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        if hasattr(self, 'feedback_text') and self.feedback_text:
            cv2.putText(frame, self.feedback_text,
                        (20, height - 30), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 100, 0), 2)

    def run(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            frame, predicted_char = self.process_hand_landmarks(frame)

            # Mode logic
            try:
                if self.current_mode == AppMode.ALPHABET_PRACTICE:
                    self.handle_alphabet_practice(predicted_char)
                elif self.current_mode == AppMode.WORD_BUILDER:
                    self.handle_word_builder(predicted_char)
                elif self.current_mode == AppMode.QUIZ_MODE:
                    self.handle_quiz_mode(predicted_char)
                elif self.current_mode == AppMode.STORY_TIME:
                    self.handle_story_mode(predicted_char)
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue

            self.draw_mode_info(frame)
            cv2.imshow('Sign Language Learning System (Two-Hand)', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):  # Reset scores
                self.scores = {
                    'alphabet': {'correct': 0, 'attempts': 0},
                    'word': {'correct': 0, 'attempts': 0, 'words_completed': 0},
                    'quiz': {'correct': 0, 'attempts': 0, 'streak': 0},
                    'story': {'correct': 0, 'attempts': 0, 'stories_completed': 0}
                }
            elif key == ord('1'):
                self.current_mode = AppMode.ALPHABET_PRACTICE
                self.speak_text("Switching to Alphabet Practice mode")
            elif key == ord('2'):
                self.current_mode = AppMode.WORD_BUILDER
                self.speak_text("Switching to Word Builder mode")
            elif key == ord('3'):
                self.current_mode = AppMode.QUIZ_MODE
                self.speak_text("Switching to Quiz mode")
            elif key == ord('4'):
                self.current_mode = AppMode.STORY_TIME
                self.speak_text("Switching to Story Time mode")

        # Cleanup
        if self.speech_thread and self.speech_thread.is_alive():
            self.engine.stop()
            self.speech_thread.join()
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = SignLanguageApp()
    app.run()
