import cv2
import numpy as np
import mediapipe as mp
from colorama import Fore
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

gesture_map = {"Call": 0, "Four": 1, "Rock" : 2, "Three" : 3, "None": 4}
#base_options = python.BaseOptions(model_asset_path="/full_dataset_gesture_recognizer.task")
#/Users/tathagato.roy_int
base_options = python.BaseOptions(model_asset_path="/Users/tathagato.roy_int/Tathagato/pose-estimation/gesture_recognition/rock_call_three_four_none_recognition.task")
#/Users/tathagato.roy_int
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

def plot_landmarks(src_image, hand_recognition_result):
    dst_image = src_image.copy()
    hand_landmarks = hand_recognition_result.hand_landmarks
    if len(hand_landmarks) < 1:
        return dst_image
    hand_landmarks = hand_landmarks[0]
    height, width, _ = src_image.shape
    for landmark in hand_landmarks:
        x, y = int(landmark.x * width), int(landmark.y * height)
        dst_image = cv2.circle(dst_image, (x, y), 3, 0, -1)
    return dst_image


def count_bit_flips(arr):
    if not arr:
        raise ValueError("Array cannot be empty")
    bit_flips_count = sum(bit1 != bit2 for bit1, bit2 in zip(arr, arr[1:]))
    return bit_flips_count


def recognise_gesture(src_image, gesture_recognizer):
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(src_image, cv2.COLOR_BGR2RGB))
    recognition_result = gesture_recognizer.recognize(mp_image)
    gestures = recognition_result.gestures
    if len(gestures) < 1:
        return {"score": None, "gesture_name": None, "category": -1}
    top_gesture = gestures[0][0]
    return {"score": top_gesture.score, "gesture_name": top_gesture.category_name,
            "category": gesture_map.get(top_gesture.category_name, 0)}


capture = cv2.VideoCapture(0)
gesture_count = []
gestures_to_detect = set(np.random.choice([gesture for gesture in gesture_map if gesture != "None"], size=2, replace=False).tolist())
detected_gestures = set()
subject_is_live = False

print(f"For liveness required gestures are: {gestures_to_detect}")

while True:
    result, frame = capture.read()
    frame = cv2.flip(frame, 1)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    try:
        recognition_result = recognizer.recognize(mp_image)
        gestures = recognition_result.gestures
        top_gesture = gestures[0][0]
        dst_image = plot_landmarks(frame, recognition_result)
        height, width, _ = dst_image.shape
        cv2.putText(dst_image, f"{top_gesture.category_name}", (width // 4, height - height // 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5, cv2.LINE_AA)
        is_gesture = 0 if top_gesture.category_name == "None" else 1
        gesture_count.append(is_gesture)
        detected_gestures.add(top_gesture.category_name)
        if detected_gestures.intersection(gestures_to_detect) == gestures_to_detect:
            subject_is_live = True
        # print(f"Recognized gesture: {top_gesture.category_name}")
    except Exception as err:
        # print(f"When recognizing gesture, received exception: {err}")
        dst_image = frame
    cv2.imshow("original_frame", frame)
    cv2.imshow("dst_image", dst_image)
    pressed_key = cv2.waitKey(1)
    if pressed_key == ord("q"):
        break

message = (Fore.RED + "Liveness: False, subject did not perform the required gestures" 
           if subject_is_live is False else Fore.GREEN + "Liveness: True, subject performed the required gestures")
print(message)






