from PIL import Image
import face_recognition


foto = "face/obama_jokowi.jpeg"

# load image into numpy
image = face_recognition.load_image_file(foto)

lokasi_wajah = face_recognition.face_locations(image)

print("Ditemukan {} wajah di foto ini.".format(len(lokasi_wajah)))

for wajah in lokasi_wajah:

    # cetak lokasi dari wajah
    top, right, bottom, left = wajah
    print("Wajah berada di lokasi pixel Atas: {}, Kiri: {}, Bawah: {}, Kanan: {}".format(top, left, bottom, right))

    # Akses wajah kedalam object window
    face_image = image[top:bottom, left:right]
    pil_image = Image.fromarray(face_image)
    pil_image.show()