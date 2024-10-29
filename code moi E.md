import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

# Khởi tạo cascade phát hiện khuôn mặt
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Đường dẫn ảnh đã tải lên
uploaded_img_path = None

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

def upload_image():
    global uploaded_img_path
    uploaded_img_path = filedialog.askopenfilename()  # Hiển thị hộp thoại chọn file và lấy đường dẫn file ảnh.
    if uploaded_img_path:  # Nếu có file được chọn (file_path không rỗng).
        img = Image.open(uploaded_img_path)  # Mở file ảnh bằng thư viện PIL (Pillow).
        img = img.resize((300, 200), Image.LANCZOS)  # Thay đổi kích thước ảnh về 300x200px.
        img = ImageTk.PhotoImage(img)  # Chuyển đổi ảnh từ định dạng PIL sang định dạng mà Tkinter có thể hiển thị.

        original_img_label.config(image=img)  # Hiển thị ảnh trong `Label` của Tkinter (label chứa ảnh gốc).
        original_img_label.image = img  # Lưu lại tham chiếu để ngăn trình thu gom rác xóa đi.

def process_image():
    if uploaded_img_path:
        # Đọc ảnh từ đường dẫn
        img = cv2.imread(uploaded_img_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Nhận diện khuôn mặt
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Vẽ khung hình chữ nhật xung quanh mỗi khuôn mặt
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Chuyển ảnh kết quả về định dạng mà Tkinter có thể hiển thị
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = img.resize((300, 300), Image.LANCZOS)
        img = ImageTk.PhotoImage(img)

        processed_img_label.config(image=img)  # Hiển thị ảnh đã xử lý trên giao diện Tkinter
        processed_img_label.image = img  # Lưu tham chiếu để tránh bị xóa ảnh.

# Tạo cửa sổ giao diện
root = tk.Tk()
root.title("Nhận Diện Khuôn Mặt Bằng Haar Cascade")
root.configure(bg="lightblue")

# Căn chỉnh nhãn tiêu đề
title_label = tk.Label(root, text="Nhận diện khuôn mặt bằng thuật toán Haar Cascade", font=("Arial", 24), bg='yellow')
title_label.grid(row=0, column=0, columnspan=3, pady=10, padx=10, sticky='ew')

# Khung ảnh gốc (Ảnh đầu vào)
original_img_label = tk.Label(root, text="Ảnh gốc", bg="lightblue")
original_img_label.grid(row=1, column=0, padx=2, pady=2)

# Khung ảnh đầu ra (Ảnh xử lý)
processed_img_label = tk.Label(root, text="Ảnh đã xử lý", bg="lightblue")
processed_img_label.grid(row=1, column=2, padx=2, pady=2)

# Các nút chức năng
button_frame = tk.Frame(root, bg="lightblue")
button_frame.grid(row=2, column=1, pady=10)

upload_btn = tk.Button(button_frame, text="Tải ảnh lên", command=upload_image, font=("Arial", 14), bg="green", fg="white")
upload_btn.grid(row=0, column=0, padx=10, pady=5)

camera_btn = tk.Button(button_frame, text="Bật Camera", command=open_camera, font=("Arial", 14), bg="green", fg="white")
camera_btn.grid(row=1, column=0, padx=10, pady=5)

process_btn = tk.Button(button_frame, text="Xử lý hình ảnh", command=process_image, font=("Arial", 14), bg="green", fg="white")
process_btn.grid(row=2, column=0, padx=10, pady=5)

# Căn giữa các cột
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)

root.mainloop()
