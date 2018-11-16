import time
import numpy
import cv2
import mss
import pickle
import os
import Trainer


face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
color = [0, 255, 0]
stroke = 2
font = cv2.FONT_HERSHEY_SIMPLEX
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("recognizors/faceTrainer.yml")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
unknown_dir = os.path.join(BASE_DIR, "Unknown")
labels = {"person_name": 1}
with open("pickles/face-labels.pickle", 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

Trainer.start_thread()
# Screen capture
with mss.mss() as sct:
    # Part of the screen to capture
    monitor = {"top": 20, "left": 0, "width": 800, "height": 720}

    while "Screen capturing":
        last_time = time.time()

        # Get raw pixels from the screen, save it to a Numpy array
        img = numpy.array(sct.grab(monitor))
        # From colors to gray
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        for (x, y, w, h) in faces:
            # print(x,y,w,h)
            roi_gray = gray[y:y + h, x:x + w]  # (ycord_start, ycord_end)
            roi_color = img[y:y + h, x:x + w]
            id_, conf = recognizer.predict(roi_gray)
            # recognize? deep learned model predict keras tensorflow pytorch scikit learn
            if conf >= 4 and conf <= 75:
                # print(5: #id_)
                # print(labels[id_])
                name = labels[id_]
                color = (0, 255, 0)
                stroke = 2
                cv2.putText(img, name, (x, y), font, 1, color, stroke, cv2.LINE_AA)
            else:
                print("Prediction is weird :", conf)
                color = (0, 0, 255)
                stroke = 2
                cv2.putText(img, "Who the fuck ?", (x - 20, y), font, 1, color, stroke, cv2.LINE_AA)
            # img_item = "7.png"
            # cv2.imwrite(img_item, roi_color)
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(img, (x, y), (end_cord_x, end_cord_y), color, stroke)

        # Display the picture
        cv2.imshow("OpenCV", img)

        # Display the picture in grayscale
        # cv2.imshow('OpenCV/Numpy grayscale',
        #            cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY))
        # print("fps: {0}".format(1 / (time.time() - last_time)))
        # Press "q" to quit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            break
