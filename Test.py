import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import numpy as np

# Biến lưu ảnh đã tải lên
loaded_image = None
camera_running = False

# Hàm bật camera
def open_camera():
    global loaded_image, camera_running
    cap = cv2.VideoCapture(0)
    camera_running = True

    def update_camera():
        global loaded_image
        ret, frame = cap.read()
        if ret:
            loaded_image = frame  # Lưu ảnh hiện tại từ camera để xử lý sau
            img = Image.fromarray(cv2.cvtColor(loaded_image, cv2.COLOR_BGR2RGB))
            img = img.resize((300, 200), Image.LANCZOS)
            img = ImageTk.PhotoImage(img)
            original_img_label.config(image=img)
            original_img_label.image = img
        if camera_running:
            root.after(10, update_camera)  # Cập nhật camera sau 10ms

    update_camera()

    def stop_camera():
        global camera_running
        camera_running = False
        cap.release()

    camera_stop_btn = tk.Button(button_frame, text="Dừng Camera", command=stop_camera, font=("Arial", 14), bg="red", fg="white")
    camera_stop_btn.grid(row=3, column=0, padx=10, pady=5)

# Hàm tải ảnh
def upload_image():
    global loaded_image
    file_path = filedialog.askopenfilename()
    if file_path:
        try:
            img = Image.open(file_path)
            img = img.resize((300, 200), Image.LANCZOS)
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

    # Tải bộ phân loại haar cascade cho khuôn mặt
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        print("Không thể tải file haarcascade_frontalface_default.xml!")
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
