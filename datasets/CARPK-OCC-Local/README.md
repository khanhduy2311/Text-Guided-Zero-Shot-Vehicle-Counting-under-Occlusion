# CARPK-OCC Local v1

Bộ đánh giá của CountLab xuất phát từ 459 ảnh CARPK và 47.500 hộp xe. Mỗi ảnh gốc có ba biến thể với vật che tổng hợp; tổng cộng 1.377 ảnh che và 1.836 mẫu khi tính cả ảnh sạch. Các mức dùng chung nhãn đếm của ảnh gốc, kể cả khi xe bị che hoàn toàn. Bộ này được nhóm tạo cục bộ để đánh giá, độc lập với bộ dữ liệu công bố của CountOCC.

| Tải dữ liệu | Số ảnh | Nội dung |
|---|---:|---|
| [Ảnh sạch (.zip)](https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion/releases/download/carpk-occ-local-v1/CARPK-OCC-Local-clean.zip) | 459 | CARPK gốc, mặt nạ rỗng |
| [Che nhẹ (.zip)](https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion/releases/download/carpk-occ-local-v1/CARPK-OCC-Local-mild.zip) | 459 | Chọn khoảng 35% số xe để đặt vật che |
| [Che vừa (.zip)](https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion/releases/download/carpk-occ-local-v1/CARPK-OCC-Local-moderate.zip) | 459 | Chọn khoảng 65% số xe để đặt vật che |
| [Che nặng (.zip)](https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion/releases/download/carpk-occ-local-v1/CARPK-OCC-Local-heavy.zip) | 459 | Chọn khoảng 85% số xe để đặt vật che |

Từng gói chứa ảnh, mặt nạ PNG nhị phân, danh sách inference và nhãn của mức tương ứng. Để lấy thêm manifest gộp, quy trình tạo dữ liệu và biên bản kiểm tra, tải [gói metadata](https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion/releases/download/carpk-occ-local-v1/CARPK-OCC-Local-metadata.zip).

## Sử dụng

Giải nén vào thư mục gốc của repo để các đường dẫn trong manifest hoạt động. Nếu cần cả bộ, chạy từ thư mục gốc:

```bash
mkdir -p downloads/carpk-occ
gh release download carpk-occ-local-v1 --repo khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion --dir downloads/carpk-occ
(cd downloads/carpk-occ && sha256sum -c SHA256SUMS)
for archive in downloads/carpk-occ/*.zip; do unzip -oq "$archive"; done
```

- **Chạy inference:** dùng `data/carpk_occ_local_v1/inference_heavy.jsonl` hoặc thay `heavy` bằng mức cần chạy.
- **Chấm số đếm:** trường `count` trong `data/carpk_occ_local_v1/heavy.jsonl` là số xe thật.
- **Chấm định vị:** nối `parent_id` với `id` trong `data/carpk_mirror_verified_v1/test.jsonl`. Trường `raw_boxes` của bản mirror dùng tọa độ pixel `[left, top, width, height]`.
- **Đo diện tích che:** mở tệp tại trường `mask` hoặc đọc số liệu `coverage`. Một hình che có thể phủ sang xe bên cạnh.

Ảnh và nhãn cơ sở do Meng-Ru Hsieh, Yen-Liang Lin và Winston H. Hsu công bố tại ICCV 2017. Nhóm CountLab bổ sung vật che, mặt nạ và manifest cho benchmark này. [CARPK của tác giả](https://lafi.github.io/LPN/) · [Bài báo gốc](https://openaccess.thecvf.com/content_ICCV_2017/html/Hsieh_Drone-Based_Object_Counting_ICCV_2017_paper.html).
