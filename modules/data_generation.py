import cv2
import mediapipe as mp
import numpy as np
import csv

# Function to get the data point values for different classes

def detect_holistic_pose(classname, filename , cap_index=0):
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

            cv2.putText(image,"CLASS",(95,12),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,0),1,cv2.LINE_AA)
            cv2.putText(image,classname,(90,40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2,cv2.LINE_AA)

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

                row.insert(0,classname)

                # export to csv
                with open(filename,mode='a',newline='') as f:
                    csv_write = csv.writer(f, delimiter=',',quotechar='"',quoting=csv.QUOTE_MINIMAL)
                    csv_write.writerow(row)

            except:
                pass


            cv2.imshow("webcam", image)

            if cv2.waitKey(10) & 0xFF == ord('x'):
                break

    cap.release()
    cv2.destroyAllWindows()

 # Function to add columns to the CSV file   

def get_columns(cap_index):
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

            # Getting landmarks and creating column list
            num_cord = len(results.pose_landmarks.landmark) + len(results.face_landmarks.landmark)
            land = ["class"]
            for val in range(1,num_cord+1):
                land += ['x{}'.format(val),'y{}'.format(val),'z{}'.format(val),'v{}'.format(val)]

            cv2.imshow("webcam", image)

            if cv2.waitKey(10) & 0xFF == ord('x'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return land


if __name__ == "__main__":
    detect_holistic_pose()
