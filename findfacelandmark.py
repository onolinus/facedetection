import face_recognition
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw


foto = "face/megawati.jpeg"

#load image
image = face_recognition.load_image_file(foto)

tanda_wajah = face_recognition.face_landmarks(image)
obj_wjh = Image.fromarray(image)


facial_features = [
        'chin',
        'left_eyebrow',
        'right_eyebrow',
        'nose_bridge',
        'nose_tip',
        'left_eye',
        'right_eye',
        'top_lip',
        'bottom_lip']

blue = ImageColor.getcolor('blue', 'RGB')


for face_landmarks in tanda_wajah:
        drawing = ImageDraw.Draw(obj_wjh)
        for facial_feature in facial_features:
            drawing.line(face_landmarks[facial_feature], width=2, fill=blue)

obj_wjh.show()