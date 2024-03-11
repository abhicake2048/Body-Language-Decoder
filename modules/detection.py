import cv2
import mediapipe as mp
import numpy as np
import pandas as pd

# Function to get the data point values for different classes

def make_predictions(model , cap_index=0):
    mp_drawing = mp.solutions.drawing_utils
    mp_holistic = mp.solutions.holistic


    cap = cv2.VideoCapture(cap_index)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False

            # Processing the image with holistic
            results = holistic.process(image)

            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Face Detection
            mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                                       mp_drawing.DrawingSpec(color=(0, 255, 123), thickness=1, circle_radius=1),
                                       mp_drawing.DrawingSpec(color=(13, 255, 222), thickness=1, circle_radius=1))

            # Right hand
            mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2),
                                       mp_drawing.DrawingSpec(color=(13, 255, 222), thickness=1, circle_radius=1))
            # Left hand
            mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1, circle_radius=2),
                                       mp_drawing.DrawingSpec(color=(13, 255, 222), thickness=1, circle_radius=1))

            # Pose Detection
            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=(0, 0, 123), thickness=1, circle_radius=1),
                                       mp_drawing.DrawingSpec(color=(13, 255, 222), thickness=1, circle_radius=1))

            # exporting co-ordinates
            try:
                # Extract pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility] for landmark in pose]).flatten())

                # Extract face landmarks
                face = results.face_landmarks.landmark
                face_row = list(np.array([[landmark.x,landmark.y,landmark.z,landmark.visibility] for landmark in face]).flatten())

                row = pose_row + face_row

                # Make detetctions
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]

                # Getting co-ordinates to print the result of the predictions
                coords = tuple(np.multiply(np.array(
                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].x,
                     results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EAR].y)
                ),[640,400]).astype(int))

                # Printing the results of the predicitons
                cv2.rectangle(image,(coords[0],coords[1]+5),(coords[0]+len(body_language_class)*20,coords[1]+30),(245,117,16),-1)

                cv2.putText(image, body_language_class,coords,cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

                # Get status box
                cv2.rectangle(image,(0,0),(250,60),(245,117,16),-1)

                # Display class
                cv2.putText(image,"CLASS",(95,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
                cv2.putText(image,body_language_class.Split(" ")[0],(90,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

                # Display Probability
                cv2.putText(image,"PROB",(15,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
                cv2.putText(image,str(round(body_language_prob[np.argmax(body_language_prob)],2)),(10,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)


            except:
                pass


            cv2.imshow("webcam", image)

            if cv2.waitKey(10) & 0xFF == ord('x'):
                break

    cap.release()
    cv2.destroyAllWindows()

