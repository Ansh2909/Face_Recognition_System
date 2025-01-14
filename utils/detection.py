import streamlit as st
import cv2
import json
from tensorflow.keras.models import model_from_json


@st.cache_resource
def load_models():
    with open("../models/name_to_id.json", "r") as f:
        name_to_id = json.load(f)

    json_file = open('../models/emotion_detector.json','r')
    model_json = json_file.read()
    json_file.close()
    model = model_from_json(model_json)

    model.load_weights('../models/emotion_detector.h5')

    age_net = cv2.dnn.readNetFromCaffe("../models/age_deploy.prototxt","../models/age_net.caffemodel")
    gender_net = cv2.dnn.readNetFromCaffe("../models/gender_deploy.prototxt","../models/gender_net.caffemodel")


    clf = cv2.face.LBPHFaceRecognizer_create()
    clf.read("../models/classifier.xml")


    face_cap = cv2.CascadeClassifier("../models/haarcascade_frontalface_default.xml")

    return name_to_id, model, age_net, gender_net, clf, face_cap

name_to_id, model, age_net, gender_net, clf, face_cap = load_models()


MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
gender_list = ['Male', 'Female']
labels = {0 : 'angry', 1 : 'disgust', 2 : 'fear', 3 : 'happy', 4 : 'neutral', 5 : 'sad', 6 : 'surprise'}

def extract_features(image):
    resized_image = cv2.resize(image, (48, 48))
    if len(resized_image.shape) == 3:
        resized_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    feature = resized_image.reshape(1,48,48,1)
    return feature/255.0


def find_key_value(dictionary,value):
    keys = []
    for key,val in dictionary.items():
        if val == value:
            keys.append(key)
    return keys


def prediction(data):
    gray_video_data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
    faces = face_cap.detectMultiScale(
        gray_video_data,
        scaleFactor=1.1,
        minNeighbors=4,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x,y,w,h) in faces:
        cv2.rectangle(data, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Predicting Name
        id, pred = clf.predict(gray_video_data[y:y+h,x:x+h])
        confidence = int(100*(1-pred/300))

        face_img = data[y:y + h, x:x + w].copy()
        blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        # Predicting Gender
        gender_net.setInput(blob)
        gender_preds = gender_net.forward()
        gender = gender_list[gender_preds[0].argmax()]

        # Predicting Age
        age_net.setInput(blob)
        age_preds = age_net.forward()
        age = age_list[age_preds[0].argmax()]

        image = cv2.resize(face_img, (48, 48))
        img = extract_features(image)
        pred = model.predict(img)
        prediction_label = labels[pred.argmax()]


        overlay_text = f"{find_key_value(name_to_id, id)[0]}, {gender}, {age}, {prediction_label}"

        if confidence>80:
            cv2.putText(data, overlay_text, (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (173, 216, 230), 1, cv2.LINE_AA)
        else:
            overlay_text = f"Unknown,{gender},{age},{prediction_label}"
            cv2.putText(data, overlay_text, (x, y - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)


    return data

st.title("Real-Time Face Recognition")
st.write("This application detects faces, predicts emotions, age, and gender in real-time.")

run = st.checkbox("Start Camera")

if run:
    video_capture = cv2.VideoCapture(0)
    st_frame = st.empty()
    while run:
        ret, frame = video_capture.read()
        if not ret:
            st.error("Failed to read from the camera.")
            break
        processed_frame = prediction(frame)
        st_frame.image(cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB), channels="RGB")
    video_capture.release()
else:
    st.write("Click the checkbox to start the camera.")