from tkinter import *
import cv2
from PIL import Image, ImageTk
from pathlib import Path
from keras_facenet import FaceNet
from numpy.linalg import norm


vid = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
width, height = 800, 600
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width)
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

app = Tk()
app.title("Filtry na twarze")
app.bind('<Escape>', lambda e: app.quit())

current_filter = None
m = 40
n = m // 2
p = 80

last_faces = []
last_frame = None

saved_faces = []

label_widget = Label(app)
label_widget.pack()

embedder = FaceNet()
TRESHOLD = 1.2
def embed_face(face_img):
    face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    face_img_gray = cv2.equalizeHist(face_img_gray)
    face_img = cv2.cvtColor(face_img_gray, cv2.COLOR_GRAY2BGR)
    
    face_img = cv2.resize(face_img, (160, 160))
    face_img_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    embedding = embedder.embeddings([face_img_rgb])[0]
    return embedding

def on_click(event):
    global last_faces, last_frame, current_filter

    if current_filter is None:
        return

    x_click = event.x
    y_click = event.y

    for (x, y, w, h) in last_faces:
        if x <= x_click <= x + w and y <= y_click <= y + h:
            face_img = last_frame[y:y+h, x:x+w]
            face_embedding = embed_face(face_img)

            updated = False
            for idx, saved in enumerate(saved_faces):
                dist = norm(face_embedding - saved['embedding'])
                if dist < TRESHOLD:  
                    saved['filter'] = current_filter.replace("\\", "/")
                    updated = True
                    break

            if not updated:
                saved_faces.append({
                    'embedding': face_embedding,
                    'filter': current_filter.replace("\\", "/")
                })
            break


def overlay_image(frame, overlay_img, x, y, w, h):
    overlay_resized = cv2.resize(overlay_img, (w + m, h + m))
    alpha_mask = overlay_resized[:, :, 3] / 255.0
    alpha_mask = cv2.merge([alpha_mask, alpha_mask, alpha_mask])

    y1 = max(0, y - n - p)
    y2 = min(frame.shape[0], y + h + n - p)
    x1 = max(0, x - n)
    x2 = min(frame.shape[1], x + w + n)

    overlay_crop = overlay_resized[0:(y2 - y1), 0:(x2 - x1), :3]
    alpha_crop = alpha_mask[0:(y2 - y1), 0:(x2 - x1)]

    frame_crop = frame[y1:y2, x1:x2]
    blended = (1 - alpha_crop) * frame_crop + alpha_crop * overlay_crop
    frame[y1:y2, x1:x2] = blended.astype('uint8')

    return frame

def recognize_face(face_img):
    embedding = embed_face(face_img)
    for saved in saved_faces:
        dist = norm(embedding - saved['embedding'])
        if dist < TRESHOLD:
            return saved['filter']
    return None

def open_camera():
    global last_faces, last_frame

    ret, frame = vid.read()
    if not ret:
        app.after(10, open_camera)
        return

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces_detected = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8,minSize=(80, 80))

    for (x, y, w, h) in faces_detected:
        face_img = frame[y:y+h, x:x+w]
        matched_filter = recognize_face(face_img)
        if matched_filter:
            filter_img = cv2.imread(matched_filter, cv2.IMREAD_UNCHANGED)
            if filter_img is not None:
                frame = overlay_image(frame, filter_img, x, y, w, h)

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    last_faces = faces_detected
    last_frame = frame.copy()

    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    captured_image = Image.fromarray(opencv_image)
    photo_image = ImageTk.PhotoImage(image=captured_image)
    label_widget.photo_image = photo_image
    label_widget.configure(image=photo_image)

    label_widget.after(10, open_camera)

def resize_image(image_path, w, h):
    img = Image.open(image_path)
    resized_image = img.resize((w, h))
    return ImageTk.PhotoImage(resized_image)

def set_filter(filter_path):
    global current_filter
    current_filter = str(filter_path)
    
def create_button(filter_path, w, h):
    image = resize_image(filter_path, w, h)
    btn = Button(app, image=image, command=lambda path=filter_path: set_filter(path))
    btn.image = image
    btn.pack(side=LEFT, padx=10, pady=10)

# Start kamery i GUI
open_camera()
label_widget.bind("<Button-1>", on_click)

images = Path("filtry").glob("*.png")
for img in images:
    create_button(img, 50, 50)

app.mainloop()
