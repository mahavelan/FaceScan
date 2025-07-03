# app.py
import streamlit as st
import os
import zipfile
import cv2
import tempfile
import face_recognition
import pandas as pd
from datetime import datetime, timedelta
from face_recognition_utils import load_student_encodings, recognize_faces, recognize_faces_from_excel

st.set_page_config(page_title="Face Recognition Attendance", layout="centered")
st.image("https://img.icons8.com/emoji/96/face-with-monocle.png", width=100)
st.title("FaceScan: AI-Based Attendance System")

if 'attendance' not in st.session_state:
    st.session_state.attendance = {}

st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["📸 Take Attendance", "📂 Upload Data", "📊 View History"])

if page == "📂 Upload Data":
    st.header("Upload Student Database")
    db_zip = st.file_uploader("Upload ZIP file of face images", type="zip")
    db_excel = st.file_uploader("(Optional) Excel file with Name, Register Number, Image Filename", type=["xlsx", "xls", "csv"])

    use_excel = False

    if db_zip:
        with tempfile.TemporaryDirectory() as tmpdir:
            zip_path = os.path.join(tmpdir, "students.zip")
            with open(zip_path, "wb") as f:
                f.write(db_zip.read())
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall("student_database")
            st.success("✅ Student images extracted.")

            if db_excel:
                try:
                    if db_excel.name.endswith(".csv"):
                        df = pd.read_csv(db_excel)
                    else:
                        df = pd.read_excel(db_excel)
                    st.session_state.known_faces = recognize_faces_from_excel(df, "student_database")
                    use_excel = True
                    st.success(f"Loaded {len(st.session_state.known_faces)} students from Excel.")
                except Exception as e:
                    st.error(f"Error loading Excel: {e}")
            else:
                st.session_state.known_faces = load_student_encodings("student_database")
                st.success(f"Loaded {len(st.session_state.known_faces)} students from image filenames.")

elif page == "📸 Take Attendance":
    st.header("Take Attendance via Webcam")
    if st.button("Start Attendance"):
        if 'known_faces' not in st.session_state:
            st.error("Please upload student data first.")
        else:
            cap = cv2.VideoCapture(0)
            stframe = st.empty()
            detected_students = set()

            with st.spinner("Scanning... Show faces to the webcam. Press 'Q' to stop."):
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                    rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                    face_locations = face_recognition.face_locations(rgb_frame)
                    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                    names = recognize_faces(face_encodings, st.session_state.known_faces)

                    for name in names:
                        if name != "Unknown":
                            detected_students.add(name)

                    for (top, right, bottom, left), name in zip(face_locations, names):
                        top *= 4
                        right *= 4
                        bottom *= 4
                        left *= 4
                        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)

                    stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB")

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            cap.release()
            cv2.destroyAllWindows()

            # Attendance Report
            all_students = list(st.session_state.known_faces.keys())
            present = list(detected_students)
            absent = list(set(all_students) - detected_students)

            data = []
            for student in all_students:
                status = "Present" if student in present else "Absent"
                if "_" in student:
                    name, regno = student.split("_", 1)
                else:
                    name, regno = student, ""
                data.append({"Name": name, "Register Number": regno, "Status": status})

            df = pd.DataFrame(data)
            st.header("📋 Attendance Summary")
            st.write(df)
            st.success(f"✅ Total: {len(all_students)}, Present: {len(present)}, Absent: {len(absent)}")

            # Save attendance CSV
            os.makedirs("attendance_records", exist_ok=True)
            date_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            filename = f"attendance_{date_str}.csv"
            df.to_csv(f"attendance_records/{filename}", index=False)
            st.download_button("Download CSV", df.to_csv(index=False), file_name=filename)

elif page == "📊 View History":
    st.header("📊 Attendance History (Last 90 Days)")
    history_path = "attendance_records"
    os.makedirs(history_path, exist_ok=True)
    history_files = sorted(os.listdir(history_path), reverse=True)

    if history_files:
        for file in history_files:
            file_path = os.path.join(history_path, file)
            try:
                file_date = datetime.strptime(file.replace("attendance_", "").replace(".csv", ""), "%Y-%m-%d_%H-%M-%S")
                if datetime.now() - file_date <= timedelta(days=90):
                    st.subheader(f"🗓️ {file_date.strftime('%B %d, %Y %H:%M:%S')}")
                    df = pd.read_csv(file_path)
                    st.dataframe(df)
            except:
                continue
    else:
        st.info("No attendance records found.")
