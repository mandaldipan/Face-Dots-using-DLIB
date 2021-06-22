"""
- Image processing using Opencv
  This contains Face Mapping using the Dlib & openCV module.
@author : Dipan Mandal
"""

import cv2
import dlib

cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

while True:
    check, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()
        # RECTANGLING THE FACE OUTLINE
        cv2.rectangle(gray, (x1, y1), (x2, y2), (0, 0, 0), 3)

        landmarks = predictor(gray, face)
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x,y), 3, (255, 6, 0), -1)

    cv2.imshow("Frame", frame)
    cv2.imshow("Grayscale Frame", gray)
    key = cv2.waitKey(1)
    if key == 27:      # Esc key
        break

cap.release()
cv2.destroyAllWindows()
