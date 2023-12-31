#this is the main script where the core logic of the application resides. The core logic includes the eye tracking, face detection, and distance estimation
#this file is the entry point of the application. 
from flask import Flask, jsonify
import threading
import cv2
import dlib  
import numpy as np
import time
from est_dist import estimate_distance

app = Flask(__name__)

# Initialize Dlib face detector
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("../models/shape_predictor_68_face_landmarks.dat")

# Video from webcam
cap = cv2.VideoCapture(1)

green_color = (0, 255, 0)  # Green for safe distance
yellow_color = (0, 255, 255)  # Yellow for middle distance
red_color = (0, 0, 255)  # Red for too close

current_state = None
last_state_change_time = time.time()
last_message_print_time = time.time()

# Define a route in Flask to provide the current state and distance
@app.route('/get-data')
def get_data():
    return jsonify({"state": current_state, "distance": distance})

# Function to run the Flask app
def run_flask_app():
    app.run(port=5000, debug=False, use_reloader=False)

# Start the Flask server in a separate thread
flask_thread = threading.Thread(target=run_flask_app)
flask_thread.start()

while True:

    #cap.read() captures a frame from the webcam. ret = bool that is true if the frame was read correctly, frame holds the actual image captured from the webcam
    ret, frame = cap.read()

    #if the frame was not read correctly, then exit the loop
    if not ret:
        break

    #converts captured frame to a grayscale image, which simplifies the image, thus reducing amt of data to process
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #faces contains a list of detected faces in the frame
    faces = detector(gray)

    #predictor is called for gray and face, and returns the facial landmarks for that face
    for face in faces:
        landmarks = predictor(gray, face)
        #Get the eye positions for the current frame (x and y coordinates) for both eyes
        #Dlib's facial landmark detector provides 68 landmarks (points 36-41 is left eye)
        #left_eye is a tuple containing 2 lists: one for x coords, one for y coords
        left_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)]
        right_eye = [(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)]


        #estimate distance (define)
        distance = estimate_distance(left_eye, right_eye)

        if distance >= 450:
            new_state = 'red'
            box_color = red_color
        elif 300 <= distance < 450:
            new_state = 'yellow'
            box_color = yellow_color
        else:
            new_state = 'green'
            box_color = green_color

        if new_state != current_state:
            current_state = new_state
            #update time of state change and message print time
            last_state_change_time = time.time()
            last_message_print_time = time.time()

            if current_state == 'red':
                print("Please move further away from the screen")
            elif current_state == 'yellow':
                print("Consider moving back. Extended exposure at this distance is not ideal")
            elif current_state == 'green':
                print("You are currently at a safe viewing distance")

        #get current time, check if current viewing distance (VD) is yellow or red, set the seconds for the repeat interval depending on the current color of the VD
        # if the difference btwn then current time and last print message is greater than the interval time of the current color, then print, and set last_print_msg to current time
        current_time = time.time()
        if current_state in ['yellow', 'red']:
            #2 minute interval for yellow 1 min interval for red (reminder intervals)
            repeat_interval = 120 if current_state == 'yellow' else 60
            if current_time - last_message_print_time >= repeat_interval and current_time - last_state_change_time >= repeat_interval:
                last_message_print_time = current_time
                if current_state == 'red':
                    print("Please move further away from the screen")
                elif current_state == 'yellow':
                    print("Consider moving back. Extended exposure at this distance is not ideal")


        # Draw rectangles around the eyes for visual feedback.
        # Calculate the bounding box for the left eye.
        left_eye_x = [x for x, y in left_eye]
        left_eye_y = [y for x, y in left_eye]
        left_eye_rect = (min(left_eye_x), min(left_eye_y)), (max(left_eye_x), max(left_eye_y))
        
        # Draw a green rectangle around the left eye.
        cv2.rectangle(frame, left_eye_rect[0], left_eye_rect[1], box_color, 2)

        # Calculate the bounding box for the right eye.
        right_eye_x = [x for x, y in right_eye]
        right_eye_y = [y for x, y in right_eye]
        right_eye_rect = (min(right_eye_x), min(right_eye_y)), (max(right_eye_x), max(right_eye_y))
        
        # Draw a green rectangle around the right eye.
        cv2.rectangle(frame, right_eye_rect[0], right_eye_rect[1], box_color, 2)

    #display current frame in a window
    cv2.imshow('Frame', frame)

    #delay of 1 ms before processing the next frame and checks if the key 'q' is pressed
    #if q, the loop breaks
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


#release webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()


