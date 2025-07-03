import face_recognition
import os
import pandas as pd

def load_student_encodings(directory):
    known_faces = {}
    for filename in os.listdir(directory):
        if filename.lower().endswith(('png', 'jpg', 'jpeg')):
            path = os.path.join(directory, filename)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                key = filename.rsplit('.', 1)[0]  # Name_RegisterNo
                known_faces[key] = encodings[0]
    return known_faces

def recognize_faces(face_encodings, known_faces):
    names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(list(known_faces.values()), face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = list(known_faces.keys())[first_match_index]
        names.append(name)
    return names

def recognize_faces_from_excel(df, image_folder):
    known_faces = {}
    for _, row in df.iterrows():
        name = str(row['Name']).strip()
        reg = str(row['Register Number']).strip()
        img_file = str(row['Image Filename']).strip()
        img_path = os.path.join(image_folder, img_file)
        if os.path.exists(img_path):
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)
            if encodings:
                key = f"{name}_{reg}"
                known_faces[key] = encodings[0]
    return known_faces
