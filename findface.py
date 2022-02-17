import face_recognition
import dlib



foto = "face/obama_jokowi.jpeg"

# # HOG detector dibangun dengan menggunakan dglib class
# pengenal_wajah = dlib.get_frontal_face_detector()
# win = dlib.image_windows()

# # memasukan image ke dalam sebuah array
# gambar = io.imread(foto)

# #run HOG face detector on the image data
# wajah_terdeteksi = face_detector(gambar,1)

# print("wajah yang ditemukan {}".format(len(wajah_terdeteksi),foto))


import face_recognition
image = face_recognition.load_image_file(foto)
face_locations = face_recognition.face_locations(image)



