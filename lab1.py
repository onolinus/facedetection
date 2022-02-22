import face_recognition
import cv2
import numpy as np


# Mendapatkan webcam
video_capture = cv2.VideoCapture(0)

# Load contoh image dan kenali
jokowi_image = face_recognition.load_image_file("face/jokowi.jpeg")
jokowi_face_encoding = face_recognition.face_encodings(jokowi_image)[0]

# Load contoh image dan kenali
prabowo_image = face_recognition.load_image_file("face/prabowo.jpeg")
prabowo_face_encoding = face_recognition.face_encodings(prabowo_image)[0]

# Load contoh image dan kenali
# obama_image = face_recognition.load_image_file("face/obama.jpeg")
# obama_face_encoding = face_recognition.face_encodings(obama_image)[0]


# Membuat encoding wajah dari image
known_face_encodings = [
    jokowi_face_encoding,
    prabowo_face_encoding,
    # obama_face_encoding
]
known_face_names = [
    "Joko Widodo",
    "Prabowo",
    # "Obama"
]

# Inisialisasi wajah
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

while True:
    # Take grab single value
    ret, frame = video_capture.read()

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:
            # Untuk memastikan apakah wajah yang di bandingkan sama
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Tidak Kenal"

            # Membandingkan jarak antara satu wajah dengan wajah yang baru dan menampilkannya
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]

            face_names.append(name)

    process_this_frame = not process_this_frame


    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Mengambar rectangle warna merah pada image
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Mengambarkan nama atau label dibawah image yang sudah direcognize
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Menampilkan image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()