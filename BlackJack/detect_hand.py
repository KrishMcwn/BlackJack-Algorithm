from ultralytics import YOLO
import cv2

# --- CONFIGURATION ---
# Path to your best trained model
MODEL_PATH = '../runs/detect/train12/weights/best.pt'
# Path to an image you want to test
IMAGE_PATH = "dataset/test/test2.png"

# --- LOAD THE MODEL ---
model = YOLO(MODEL_PATH)


# --- CALCULATE HAND VALUE FUNCTION ---
def get_hand_value(results):
    # This dictionary now uses your abbreviated labels as keys
    card_values = {'a': 11, '2': 2, '3': 3, '4': 4, '5': 5,
                   '6': 6, '7': 7, '8': 8, '9': 9, '10': 10,
                   'j': 10, 'q': 10, 'k': 10}

    total_value = 0
    num_aces = 0

    # Get detected class names
    detected_objects = results[0].boxes.data
    class_names = results[0].names

    for obj in detected_objects:
        # obj is a tensor: [x1, y1, x2, y2, confidence, class_id]
        class_id = int(obj[5])
        class_name = class_names[class_id]
        confidence = float(obj[4])

        if confidence > 0.5:  # Filter low-confidence detections
            if class_name == 'a':  # Check for the abbreviated 'a'
                num_aces += 1
            total_value += card_values.get(class_name, 0)

    # Adjust for Aces
    while total_value > 21 and num_aces > 0:
        total_value -= 10
        num_aces -= 1

    return total_value


# --- RUN DETECTION AND CALCULATION ---
results = model(IMAGE_PATH)
hand_value = get_hand_value(results)
print(f"Detected Hand Value: {hand_value}")

# --- VISUALIZE RESULTS (OPTIONAL) ---
# This will show the image with bounding boxes and labels
annotated_frame = results[0].plot()
cv2.imshow("YOLOv8 Detection", annotated_frame)
cv2.waitKey(0)
cv2.destroyAllWindows()