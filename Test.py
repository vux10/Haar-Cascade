import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np
from cv2 import face

# Biến lưu ảnh đã tải lên
loaded_image = None
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def train_recognizer():
    recognizer = face.LBPHFaceRecognizer_create()
    faces = []
    labels = []

    # Duyệt qua tất cả ảnh trong thư mục 'data' để thêm vào tập huấn luyện
    for i in range(1, 4):  # Ba nhãn 1, 2, và 3
        for j in range(1, 21):  # Giả sử 20 ảnh mỗi nhãn
            img_path = f"data/User.{i}.{j}.jpg"
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:  # Kiểm tra xem ảnh có tồn tại
                faces.append(img)
                labels.append(i)

    recognizer.train(faces, np.array(labels))
    recognizer.save("trained_model.yml")
    print("Đã hoàn thành huấn luyện!")

def open_camera_with_recognition():
    cap = cv2.VideoCapture(0)
    recognizer = face.LBPHFaceRecognizer_create()
    recognizer.read("trained_model.yml")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            face_img = gray[y:y+h, x:x+w]
            label, confidence = recognizer.predict(face_img)

            # Hiển thị nhãn và độ tin cậy trên khung hình
            if confidence < 100:  # Ngưỡng tự điều chỉnh
                label_text = f"User {label} - {round(confidence, 2)}"
            else:
                label_text = "Unknown"

            cv2.putText(frame, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        cv2.imshow("Camera - Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty('Camera - Face Recognition', cv2.WND_PROP_AUTOSIZE) == -1:
            break

    cap.release()
    cv2.destroyAllWindows()

# Hàm thu thập dữ liệu khuôn mặt từ camera.
def collect_training_data(label):
    cap = cv2.VideoCapture(0)
    face_id = label  # Dán nhãn cho khuôn mặt
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            count += 1
            face_img = gray[y:y+h, x:x+w]  # Cắt khuôn mặt
            cv2.imwrite(f"data/User.{face_id}.{count}.jpg", face_img)  # Lưu ảnh khuôn mặt với nhãn

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.imshow("Collecting Training Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q') or count >= 20:  # Thu thập 20 ảnh cho mỗi người
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Đã thu thập dữ liệu cho nhãn {label}")

# Hàm bật camera
def open_camera():
    cap = cv2.VideoCapture(0)  # Mở camera mặc định

    while True:  # Vòng lặp để liên tục lấy và hiển thị hình ảnh từ camera.
        ret, frame = cap.read()  # Đọc một khung hình (frame) từ camera.

        if not ret:
            break

        # Chuyển ảnh thành ảnh xám để dễ nhận diện khuôn mặt
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Phát hiện khuôn mặt
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Vẽ khung hình chữ nhật quanh mỗi khuôn mặt
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Hiển thị hình ảnh từ camera có khung khuôn mặt
        cv2.imshow('Camera - Face Detection', frame)

        # Kiểm tra nếu người dùng nhấn phím hoặc đóng cửa sổ
        key = cv2.waitKey(1) & 0xFF  # Lấy mã phím người dùng nhấn (nếu có).
        if key == ord('q') or cv2.getWindowProperty('Camera - Face Detection', cv2.WND_PROP_AUTOSIZE) == -1:
            break  # Thoát khỏi vòng lặp và tắt camera.

    cap.release()  # Giải phóng camera (tắt camera).
    cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ hiển thị hình ảnh của OpenCV.

# Hàm tải ảnh
def upload_image():
    global loaded_image
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            img = Image.open(file_path)
            img = img.resize((400, 300), Image.LANCZOS)
            img_tk = ImageTk.PhotoImage(img)
            original_img_label.config(image=img_tk)
            original_img_label.image = img_tk
            # Chuyển đổi ảnh sang định dạng BGR để sử dụng với OpenCV
            img_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            loaded_image = img_bgr
        except Exception as e:
            print(f"Không thể đọc ảnh từ đường dẫn đã chọn: {e}")

# Hàm xử lý ảnh
def process_image():
    global loaded_image
    if loaded_image is None:
        print("Chưa có ảnh để xử lý!")
        return

    # Chuyển ảnh sang grayscale (xám) vì bộ phân loại hoạt động tốt hơn với ảnh xám
    gray = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2GRAY)

    # Phát hiện các khuôn mặt
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.03, minNeighbors=4, minSize=(30, 30))

    # Kiểm tra và vẽ khung lên từng khuôn mặt
    if len(faces) == 0:
        print("Không tìm thấy khuôn mặt nào!")
    else:
        print(f"Đã phát hiện {len(faces)} khuôn mặt.")
        for (x, y, w, h) in faces:
            cv2.rectangle(loaded_image, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Chuyển ảnh đã vẽ khung từ BGR sang RGB để hiển thị với Tkinter
    img_rgb = cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_display = ImageTk.PhotoImage(img_pil)
    processed_img_label.config(image=img_display)
    processed_img_label.image = img_display

# Tạo cửa sổ giao diện
root = tk.Tk()
root.title("Nhận Diện Khuôn Mặt Bằng Haar Cascade")
root.configure(bg="lightblue")

# Nhãn tiêu đề
title_label = tk.Label(root, text="Nhận diện khuôn mặt bằng thuật toán Haar Cascade", font=("Arial", 24), bg='yellow')
title_label.grid(row=0, column=0, columnspan=3, pady=10, padx=10, sticky='ew')

# Khung ảnh gốc
original_img_label = tk.Label(root, text="Ảnh gốc", bg="lightblue")
original_img_label.grid(row=1, column=0, padx=2, pady=2)

# Khung ảnh xử lý
processed_img_label = tk.Label(root, text="Ảnh đã xử lý", bg="lightblue")
processed_img_label.grid(row=1, column=1, padx=2, pady=2)

# Nút tải ảnh
upload_button = tk.Button(root, text="Tải ảnh", command=upload_image, bg='orange')
upload_button.grid(row=2, column=0, padx=10, pady=10)

# Nút xử lý ảnh
process_button = tk.Button(root, text="Xử lý ảnh", command=process_image, bg='orange')
process_button.grid(row=2, column=1, padx=10, pady=10)

# Nút mở camera
open_camera_button = tk.Button(root, text="Mở Camera", command=open_camera, bg='orange')
open_camera_button.grid(row=3, column=0, padx=10, pady=10)

# Nút thu thập dữ liệu
collect_data_button = tk.Button(root, text="Thu thập dữ liệu", command=lambda: collect_training_data(1), bg='orange')
collect_data_button.grid(row=3, column=1, padx=10, pady=10)

# Nút huấn luyện mô hình
train_button = tk.Button(root, text="Huấn luyện mô hình", command=train_recognizer, bg='orange')
train_button.grid(row=4, column=0, padx=10, pady=10)

# Nút mở camera với nhận diện
open_camera_recognition_button = tk.Button(root, text="Mở Camera Nhận Diện", command=open_camera_with_recognition, bg='orange')
open_camera_recognition_button.grid(row=4, column=1, padx=10, pady=10)

root.mainloop()
