# Checkpoint nhóm đã huấn luyện

| Nhánh demo | Checkpoint gốc sau huấn luyện | Bản trọng số do trình cài demo tải |
|---|---|---|
| CountGD++ A3 + SigLIP2 và A3 + SAM 3 | [checkpoint_best.pth](https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion/releases/download/countlab-v1/checkpoint_best.pth) | [a3.pth](https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion/releases/download/countlab-v1/a3.pth) |
| CountGD-Box + SAM 3 (Cubi, bản tối ưu AP) | [checkpoint_best_ap.pth](https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion/releases/download/countlab-v1/checkpoint_best_ap.pth) | [countgd-box.pth](https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion/releases/download/countlab-v1/countgd-box.pth) |

[semantic-head.pt](https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion/releases/download/countlab-v1/semantic-head.pt) là head bổ sung cho nhánh ngữ nghĩa. Demo tự tải các tệp suy luận qua [bộ cài](../web_demo/install/README.md). Hai checkpoint gốc giữ nguyên nội dung tệp từ lần train. Bản suy luận chỉ lấy state dict của detector, với cùng tên, kiểu và giá trị tensor.

Để kiểm tra hai checkpoint gốc sau khi tải, đặt chúng cùng thư mục với [SHA256SUMS](SHA256SUMS) và chạy `sha256sum -c SHA256SUMS`.
