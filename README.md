# Text-Guided Zero-Shot Vehicle Counting under Occlusion

Mã nghiên cứu cho hai phương pháp đếm phương tiện theo mô tả văn bản khi xe bị che khuất.

**Checkpoint nhóm đã train** · [CountGD++ A3 — checkpoint_best.pth](https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion/releases/download/countlab-v1/checkpoint_best.pth) · [CountGD-Box của Cubi — checkpoint_best_ap.pth](https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion/releases/download/countlab-v1/checkpoint_best_ap.pth) · [Trọng số demo và SHA-256](checkpoints/README.md)

| Phương pháp | Huấn luyện | Suy luận |
|---|---|---|
| CountGD++ A3 + SigLIP2 | Teacher cố định nhận ảnh sạch. Student học từ ảnh có vật che tổng hợp, loss phát hiện và tín hiệu đặc trưng của Teacher. | CountGD++ đề xuất vị trí xe. SigLIP2 so vùng ảnh với mô tả để lọc theo prompt. |
| CountGD-Box + SAM 3 | Huấn luyện detector với box xe và các cấu hình tối ưu AP hoặc thêm vật che trong thư mục Cubi. | Detector đề xuất khung theo loại xe. SAM 3 nhận mô tả đầy đủ, rồi ghép một–một với các khung đề xuất. |

## Các thành phần

- [Huấn luyện Teacher](scripts/run_countgdpp_a1.py), [huấn luyện Student](scripts/run_countgdpp_step6.py), [khối phục hồi đặc trưng và loss](src/step6_model.py), [bộ tạo che khuất](src/countgdpp_occlusion.py) và [cấu hình A3](configs/countgdpp_a3_v1.json).
- [Suy luận hai lượt](src/countgdpp_inference.py), [lọc SigLIP2 dùng trong benchmark](scripts/run_siglip_semantics.py), [lọc mô tả tự do dùng trong demo](web_demo/open_text.py) và [cấu hình ngưỡng của demo](web_demo/open_text_policy.json).
- [Huấn luyện CountGD-Box tối ưu AP](Cubi/train_countgd_box_ap_optimized.py), [huấn luyện với vật che](Cubi/train_countgd_box_occ.py), [benchmark](Cubi/benchmark_countvimg_suite.py) và [suy luận CountGD-Box + SAM 3](Cubi/infer_sam3.py).

## Chạy web demo đầy đủ trên một máy

```bash
git clone https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion.git
cd Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion
bash web_demo/install/setup.sh
.demo-envs/detector/bin/python -m web_demo.run
```

Mở **http://127.0.0.1:7860**. Bộ cài chuẩn bị backend, tải checkpoint và kiểm tra SHA-256. [Yêu cầu phần cứng, bốn nhánh và cách chạy offline](web_demo/install/README.md).

## Chuẩn bị và chạy

Sau khi clone, lấy hai backend upstream ở các commit đã ghim và áp dụng bản sửa deformable attention dùng khi train A3.

```bash
git submodule update --init --recursive
git -C CountGDPlusPlus apply ../patches/countgdpp-deformable-attention.patch
```

Cài dependencies theo backend tương ứng. Các runner nghiên cứu đọc manifest, nhãn, trọng số và artifact của giai đoạn trước theo cấu hình; người chạy cần chuẩn bị các đầu vào này riêng. [Định danh checkpoint CountGD++ gốc](configs/pretrained_countgdpp_official.json).

Ví dụ với CountGD-Box + SAM 3, chạy lệnh đầu ở môi trường CountGD-Box, lệnh sau ở môi trường Transformers có Sam3VideoModel và Sam3Processor. Thay đường dẫn ví dụ bằng ảnh, checkpoint và thư mục SAM 3 dạng Hugging Face trên máy. Hai lệnh dùng cùng một ảnh.

```bash
python -m Cubi.infer_sam3 proposals --image street.jpg --noun car --checkpoint checkpoints/countgd_box_ap.pth --out proposals.json
python -m Cubi.infer_sam3 verify --image street.jpg --prompt 'red car' --sam3 checkpoints/sam3 --proposals proposals.json
```
