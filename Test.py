import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2

def open_camera():
    cap = cv2.VideoCapture(0)  # Mở camera mặc định

    while True:  # Vòng lặp để liên tục lấy và hiển thị hình ảnh từ camera.
        ret, frame = cap.read()  # Đọc một khung hình (frame) từ camera.

        if ret:  # Nếu việc đọc thành công (camera hoạt động bình thường).
            cv2.imshow('Camera', frame)  # Hiển thị hình ảnh camera trong cửa sổ tên 'Camera'.

        # Kiểm tra nếu người dùng nhấn phím hoặc đóng cửa sổ
        key = cv2.waitKey(1) & 0xFF  # Lấy mã phím người dùng nhấn (nếu có).

        # Nếu người dùng nhấn 'q' hoặc cửa sổ 'Camera' bị đóng
        if key == ord('q') or cv2.getWindowProperty('Camera', cv2.WND_PROP_AUTOSIZE) == -1:
            break  # Thoát khỏi vòng lặp và tắt camera.

    cap.release()  # Giải phóng camera (tắt camera).
    cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ hiển thị hình ảnh của OpenCV.

def upload_image():
    file_path = filedialog.askopenfilename()  # Hiển thị hộp thoại chọn file và lấy đường dẫn file ảnh.
    if file_path:  # Nếu có file được chọn (file_path không rỗng).
        img = Image.open(file_path)  # Mở file ảnh bằng thư viện PIL (Pillow).
        img = img.resize((300, 200),
                         Image.LANCZOS)  # Thay đổi kích thước ảnh về 300x200px, dùng bộ lọc LANCZOS để giảm chất lượng giảm nhiễu.
        img = ImageTk.PhotoImage(img)  # Chuyển đổi ảnh từ định dạng PIL sang định dạng mà Tkinter có thể hiển thị.

        original_img_label.config(image=img)  # Hiển thị ảnh trong `Label` của Tkinter (label chứa ảnh gốc).
        original_img_label.image = img  # Lưu lại tham chiếu đến ảnh để ngăn trình thu gom rác (garbage collector) xóa đi.

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

process_btn = tk.Button(button_frame, text="Xử lý hình ảnh", font=("Arial", 14), bg="green", fg="white")
process_btn.grid(row=2, column=0, padx=10, pady=5)

# Căn giữa các cột
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_columnconfigure(2, weight=1)

root.mainloop()
