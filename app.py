import streamlit as st
import os
import cv2
import face_recognition
import pandas as pd
import numpy as np
from datetime import datetime
from PIL import Image
import tempfile

# Load Haar Cascade
CASCADE_PATH = "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + CASCADE_PATH)

# Directories
REG_DIR = "registered_faces"
if not os.path.exists(REG_DIR):
    os.makedirs(REG_DIR)

ATTENDANCE_LOG = "attendance.csv"

# Helper Functions
def get_encoded_faces():
    encoded = []
    names = []
    registers = []
    for file in os.listdir(REG_DIR):
        if file.endswith(".jpg") or file.endswith(".png"):
            path = os.path.join(REG_DIR, file)
            img = face_recognition.load_image_file(path)
            encoding = face_recognition.face_encodings(img)
            if encoding:
                encoded.append(encoding[0])
                name = os.path.splitext(file)[0].split("_")
                names.append(name[0])
                registers.append(name[1])
    return encoded, names, registers

def mark_attendance(name, reg):
    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    if not os.path.exists(ATTENDANCE_LOG):
        with open(ATTENDANCE_LOG, "w") as f:
            f.write("Name,RegisterNumber,Date,Time\n")

    df = pd.read_csv(ATTENDANCE_LOG)
    if not ((df['Name'] == name) & (df['RegisterNumber'] == reg) & (df['Date'] == date)).any():
        with open(ATTENDANCE_LOG, "a") as f:
            f.write(f"{name},{reg},{date},{time}\n")

# UI
st.set_page_config(layout="wide")
st.title("📸 Face Recognition Attendance System")

menu = ["Upload Student Photos", "Take Attendance", "Attendance History"]
choice = st.sidebar.selectbox("Select Option", menu)

if choice == "Upload Student Photos":
    st.subheader("🧑‍💼 Register Students")
    uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
    name = st.text_input("Student Name")
    reg = st.text_input("Register Number")
    if uploaded_file and name and reg:
        img = Image.open(uploaded_file)
        save_path = os.path.join(REG_DIR, f"{name}_{reg}.jpg")
        img.save(save_path)
        st.success(f"✅ {name} registered successfully")

elif choice == "Take Attendance":
    st.subheader("📷 Webcam Attendance")
    start = st.button("Start Camera and Recognize Faces")

    if start:
        known_faces, known_names, known_regs = get_encoded_faces()
        present_students = set()

        cap = cv2.VideoCapture(0)
        stframe = st.empty()
        end_btn = st.button("End Attendance")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            encodings = face_recognition.face_encodings(rgb_frame)
            names_seen = []

            for encoding in encodings:
                matches = face_recognition.compare_faces(known_faces, encoding)
                face_distances = face_recognition.face_distance(known_faces, encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_names[best_match_index]
                    reg = known_regs[best_match_index]
                    names_seen.append((name, reg))
                    mark_attendance(name, reg)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

            stframe.image(frame, channels="BGR")
            if end_btn:
                break
        cap.release()
        st.success("✅ Attendance session ended")

elif choice == "Attendance History":
    st.subheader("📅 Attendance Logs")
    if os.path.exists(ATTENDANCE_LOG):
        df = pd.read_csv(ATTENDANCE_LOG)
        date_filter = st.date_input("Select Date to Filter")
        filtered = df[df['Date'] == date_filter.strftime("%Y-%m-%d")]
        st.dataframe(filtered)

        total = len(set(df[df['Date'] == date_filter.strftime("%Y-%m-%d")]['RegisterNumber']))
        all_regs = set(name.split("_")[1] for name in os.listdir(REG_DIR))
        absent = all_regs - set(filtered['RegisterNumber'])

        st.info(f"✅ Total Present: {len(filtered)}")
        st.warning(f"❌ Total Absent: {len(absent)}")
        if absent:
            st.write("### Absent Students")
            st.write(list(absent))
    else:
        st.warning("No attendance log found. Please take attendance first.")
