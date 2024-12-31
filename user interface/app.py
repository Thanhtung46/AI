import tkinter as tk
from tkinter import messagebox

# Hàm xử lý khi nhấn nút
def on_button_click():
    user_input = entry.get()  # Lấy giá trị người dùng nhập vào ô entry
    if user_input:
        messagebox.showinfo("Thông báo", f"Chào bạn, {user_input}!")
    else:
        messagebox.showwarning("Cảnh báo", "Vui lòng nhập tên của bạn!")

# Khởi tạo cửa sổ chính
root = tk.Tk()
root.title("Giao diện Người Dùng Đơn Giản")
root.geometry("400x200")  # Kích thước cửa sổ (rộng x cao)

# Thêm nhãn (Label)
label = tk.Label(root, text="Nhập tên của bạn:")
label.pack(pady=10)  # `pady` là khoảng cách dọc

# Thêm ô nhập liệu (Entry)
entry = tk.Entry(root, width=30)
entry.pack(pady=5)

# Thêm nút (Button)
button = tk.Button(root, text="Gửi", command=on_button_click)
button.pack(pady=20)

# Chạy vòng lặp chính của cửa sổ GUI
root.mainloop()
