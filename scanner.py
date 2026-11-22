import cv2
import time
import os
from datetime import datetime
from ultralytics import YOLO

# 1. Create a folder to store the defect image records
if not os.path.exists("saved_defects"):
    os.makedirs("saved_defects")

# 2. Load the custom AI model you just trained
model = YOLO("runs/detect/train-2/weights/best.pt")

# 3. Start the webcam
cap = cv2.VideoCapture(0)

# Cooldown timer to prevent saving 30 photos per second
last_save_time = 0  

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run AI detection on the current frame
    results = model(frame, conf=0.5)
    
    # Check if any objects (defects) were detected
    if len(results[0].boxes) > 0:
        # .plot() automatically draws the bounding boxes around the detected areas
        display_frame = results[0].plot() 
        
        # Write "DEFECT FOUND!" in red text
        cv2.putText(display_frame, "DEFECT FOUND!", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        
        # Save the image (maximum once every 2 seconds)
        current_time = time.time()
        if current_time - last_save_time > 2.0:
            # Generate a timestamp for the filename
            timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filepath = os.path.join("saved_defects", f"defect_{timestamp}.jpg")
            
            # Save the frame to the folder
            cv2.imwrite(filepath, display_frame)
            print(f"Log saved: {filepath}") # Prints to the terminal
            
            last_save_time = current_time

    else:
        # No defects found, keep the clean frame
        display_frame = frame.copy()
        
        # Write "GOOD" in green text
        cv2.putText(display_frame, "GOOD", (30, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    # Show the video window
    cv2.imshow("Casting Defect Scanner", display_frame)

    # Press 'q' to close the scanner
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()