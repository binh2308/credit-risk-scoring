## Hướng dẫn cài đặt môi trường LaTeX Local

Để có thể chỉnh sửa và biên dịch file báo cáo này trên máy tính cá nhân, bạn cần thiết lập môi trường theo các bước sau:

### 1. Cài đặt MiKTeX (Bản phân phối TeX)

MiKTeX là công cụ quản lý và cung cấp các package cần thiết để biên dịch LaTeX.

* **Tải về:** Truy cập trang chủ [MiKTeX Download](https://miktex.org/download) và tải bộ cài đặt phù hợp với hệ điều hành của bạn (Windows/macOS/Linux).
* **Cài đặt:** Chạy file cài đặt. 
* **Lưu ý quan trọng:** Trong quá trình cài đặt, ở mục **"Install missing packages on-the-fly"**, hãy chọn **"Yes"** hoặc **"Always"**. Điều này giúp MiKTeX tự động tải các package còn thiếu (như font chữ, định dạng bảng biểu) khi bạn biên dịch báo cáo mà không cần hỏi lại.

### 2. Cài đặt Perl (Bắt buộc cho `latexmk` trên Windows)

Công cụ biên dịch `latexmk` được viết bằng ngôn ngữ Perl. Nếu bạn dùng Windows, bạn cần cài đặt Perl để `latexmk` có thể hoạt động.

* Tải và cài đặt [Strawberry Perl](https://strawberryperl.com/).
* (Người dùng macOS/Linux thường đã có sẵn Perl trong hệ thống).

### 3. Cài đặt VS Code và Extension "LaTeX Workshop"

Visual Studio Code (VS Code) kết hợp với extension là môi trường soạn thảo LaTeX trực quan và tiện lợi nhất.

1. Tải và cài đặt [Visual Studio Code](https://code.visualstudio.com/) (nếu chưa có).
2. Mở VS Code, vào mục **Extensions** (phím tắt `Ctrl + Shift + X`).
3. Tìm kiếm và cài đặt extension có tên **LaTeX Workshop** (của tác giả James Yu).

Extension này hỗ trợ highlight cú pháp, tự động gợi ý lệnh, và tích hợp sẵn trình xem PDF trực tiếp ngay trong VS Code.

---

## Hướng dẫn biên dịch (Build PDF)

Dự án này sử dụng `latexmk` để tự động hóa quá trình biên dịch. Báo cáo đồ án thường có mục lục, tài liệu tham khảo và hình ảnh, yêu cầu phải chạy lệnh biên dịch nhiều lần. `latexmk` sẽ tự động lo việc đó cho bạn chỉ với một lần gọi lệnh.

### Cách 1: Biên dịch trực tiếp trên VS Code (Khuyên dùng)

Theo mặc định, LaTeX Workshop đã cấu hình sẵn công cụ biên dịch (recipe) là `latexmk`.

1. Mở thư mục gốc của project bằng VS Code.
2. Mở file `main.tex`.
3. Nhấn vào biểu tượng **Build LaTeX project** (nút Play màu xanh `▶`) ở góc trên cùng bên phải màn hình, hoặc dùng phím tắt `Ctrl + Alt + B` (Windows/Linux) / `Cmd + Option + B` (macOS).
4. Để xem kết quả, nhấn vào biểu tượng **View LaTeX PDF** ở góc trên bên phải (hoặc `Ctrl + Alt + V`). File PDF sẽ hiển thị ở một tab chia đôi màn hình, tự động cập nhật mỗi khi bạn lưu file (`Ctrl + S`).

### Cách 2: Biên dịch bằng Terminal (Command Line)

Nếu bạn muốn chạy thủ công, hãy mở Terminal (hoặc Command Prompt) tại thư mục gốc của project và làm theo các bước sau:

**Lưu ý quan trọng trước khi chạy:** Nếu bạn đang mở file `main.pdf` bằng các phần mềm bên ngoài (như Adobe Acrobat, Foxit Reader,...), **bạn bắt buộc phải tắt file PDF đó đi**. Nếu không tắt, hệ điều hành sẽ khóa file, khiến lệnh biên dịch báo lỗi không thể ghi đè nội dung mới.

**Bước 1: Dọn dẹp file tạm (Khuyên dùng)**
Quá trình biên dịch trước đó thường sinh ra nhiều file phụ (`.aux`, `.log`, `.toc`,...). Để tránh lỗi xung đột cache, hãy dọn dẹp môi trường trước bằng lệnh:
```bash
latexmk -C
```

**Bước 2: Biên dịch tạo file PDF**
Chạy lệnh sau để hệ thống bắt đầu quá trình build:
```bash
latexmk -pdf main.tex
```

**Bước 3: Đọc kết quả (Pass/Fail)**
Quá trình chạy sẽ in ra rất nhiều dòng log trên terminal. Khi terminal dừng lại, hãy xem những dòng cuối cùng:
- **Pass (Thành công):** Bạn sẽ thấy dòng thông báo `Latexmk: All targets (...) are up-to-date` và không có chữ `Error` nào. Lúc này, file `main.pdf` đã được tạo mới hoàn chỉnh trong thư mục.
- **Fail (Thất bại):** Terminal báo lỗi (thường bắt đầu bằng `! Error`) hoặc dừng lại hiển thị dấu `?` chờ bạn nhập lệnh. Lỗi này thường do sai cú pháp LaTeX (thiếu ngoặc, sai tên ảnh,...). Nhấn `Ctrl + C` để thoát tiến trình, mở file `.tex` lên sửa lỗi, rồi quay lại làm từ Bước 1.

**Bước 4: Xem kết quả**
Mở file `main.pdf` trong thư mục project để xem báo cáo đã cập nhật.