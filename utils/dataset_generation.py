import cv2
import os
import numpy as np
from PIL import Image
import json



def train_classifier(data_dir):
    path = [os.path.join(data_dir,f) for f in os.listdir(data_dir)]

    faces = []
    names = []

    name_to_id = {}
    current_id = 0

    for image in path:
        img = Image.open(image).convert('L')
        imageNp = np.array(img,'uint8')
        name = (os.path.split(image)[1].split(".")[0])

        if name not in name_to_id:
            name_to_id[name] = current_id
            current_id += 1

        faces.append(imageNp)
        names.append(name_to_id[name])

    names = np.array(names, dtype=np.int32)

    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.train(faces,names)

    clf.write("../models/classifier.xml")
    with open("../models/name_to_id.json", "w") as f:
        json.dump(name_to_id, f)
    print("Training complete. Model saved as 'classifier.xml'.")


def generate_dataset(user_name):
    face_cap = cv2.CascadeClassifier("../models/haarcascade_frontalface_default.xml")
    video_cap = cv2.VideoCapture(0)

    img_id = 0

    def face_cropped(img):
        gray = cv2.cvtColor(video_data, cv2.COLOR_BGR2GRAY)
        faces = face_cap.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )
        if len(faces) == 0:
            return None
        for (x,y,w,h) in faces:
            return img[y:y+h,x:x+w]

    while True:
        ret, video_data = video_cap.read()
        if face_cropped(video_data) is not None:
            img_id += 1
            face = cv2.resize(face_cropped(video_data),(200,200))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            file_name_path=f"../data/pre_stored_images/{user_name}."+str(img_id)+".jpg"
            cv2.imwrite(file_name_path,face)
            cv2.imshow("Collecting your Images", face)
        if (cv2.waitKey(10) & 0xFF == 27) or int(img_id) == 200:
            break

    video_cap.release()
    cv2.destroyAllWindows()
    print(f"Collecting Samples for {user_name} Completed....")


name = input("Enter your name to create your dataset...")
generate_dataset(name)
print("Training Model...")
train_classifier("../data/pre_stored_images")