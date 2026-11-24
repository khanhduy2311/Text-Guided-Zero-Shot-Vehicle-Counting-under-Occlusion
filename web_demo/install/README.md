# Chạy CountLab trên máy cá nhân

Bản local phục vụ cả bốn nhánh: CountGD++ gốc, A3 + SigLIP2, CountGD-Box + SAM 3 và A3 + SAM 3. Trình duyệt có thể dùng ảnh mẫu hoặc ảnh tải lên, xem khung dự đoán, đổi nhánh và xuất JSON/CSV.

## Cài đặt

Dùng Linux x86-64, Python 3.10 có venv, Git, curl và một GPU NVIDIA với driver tương thích PyTorch CUDA 12.4. Các tệp trọng số và cấu hình tải xuống tổng cộng 8,28 GB. Dành khoảng 50 GB ổ đĩa cho ba môi trường Python, cache cài đặt và model. Các phiên bản Transformers của detector, SigLIP2 và SAM 3 được cài riêng. Các bước sau thực hiện từ thư mục gốc sau khi clone.

```bash
bash web_demo/install/setup.sh
.demo-envs/detector/bin/python -m web_demo.run
```

Mở **http://127.0.0.1:7860**. Có thể đặt `CUDA_VISIBLE_DEVICES=0` để chọn GPU hoặc `COUNTLAB_PORT=7861` để đổi cổng trước khi chạy web. Các lượt suy luận dùng chung GPU theo thứ tự. Mỗi giai đoạn nạp model khi cần và giải phóng sau khi hoàn tất; thời gian hiển thị trên máy cá nhân có thể khác bản server giữ sẵn trọng số.

## Checkpoint

Bộ cài tự tải hai detector đã train và head ngữ nghĩa của dự án từ [GitHub Release countlab-v1](https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion/releases/tag/countlab-v1). Bản xuất giữ nguyên các tensor dùng để suy luận. Model nền CountGD++ chính thức, BERT, SigLIP2 và SAM 3 được tải từ revision Hugging Face đã ghim. Mỗi tệp được kiểm tra SHA-256 trước khi đưa vào thư mục đích. [Danh mục tệp, kích thước và SHA-256](../weights.json).

Cần Internet trong lần cài đầu. Sau khi cài xong có thể chạy inference offline. Có thể chạy lại riêng bước tải và kiểm tra tệp:

```bash
.demo-envs/detector/bin/python -m web_demo.install.weights
```

Bộ cài đưa trọng số dự án vào `checkpoints/countlab/`. CountGD-Box trong demo dùng checkpoint tối ưu AP. Các nhánh SAM 3 dùng cùng bộ trọng số model nền. Tệp huấn luyện, optimizer và ảnh dataset không cần cho web demo.
