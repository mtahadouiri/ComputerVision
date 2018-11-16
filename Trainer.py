import cv2
import os
import numpy as np
from PIL import Image
import pickle
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
face_cascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
current_id = 0
label_ids = {}
y_labels = []
x_train = []
current_file = 0
done = True


# A thread to continuously train a new data-set each x seconds
def create_unknown_thread():
    print("Training unknown data set")
    traindataset(0)


def start_thread():
    global done
    if done:
        done = False
        threading.Thread(target=create_unknown_thread).start()


# Train data-set with name and in path
def traindataset(curr_id):
    image_dir = os.path.join(BASE_DIR, "Unknown")
    print("Training new data set from images in : " + image_dir)
    for root, dirs, files in os.walk(image_dir):
        for file in files:
            if file.endswith("png") or file.endswith("jpg"):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(" ", "-").lower()
                # print(label, path)
                # print(label_ids)
                # y_labels.append(label) # some number
                # x_train.append(path) # verify this image, turn into a NUMPY arrray, GRAY
                final_image = Image.open(path).convert("L")  # grayscale
                # size = (550, 550)
                # final_image = pil_image.resize(size, Image.ANTIALIAS)
                image_array = np.array(final_image, "uint8")
                # print(image_array)
                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.3, minNeighbors=5)
                for (x, y, w, h) in faces:
                    roi = image_array[y:y + h, x:x + w]
                    img_item = (str(curr_id) + "unknown" + ".png")
                    cv2.imwrite(str(root) + "/trainer/" + img_item, roi)
                    x_train.append(roi)
                    y_labels.append(curr_id)
                    curr_id = curr_id + 1
    # print(y_labels)
    # print(x_train)

    with open("pickles/unknown-face-labels.pickle", 'wb') as f:
        pickle.dump(label_ids, f)
    recognizer.train(x_train, np.array(y_labels))
    recognizer.save("recognizors/unknownTrainer.yml")
    print("DONE")
    global done
    done = True
