<h1 align="center">CountLab</h1>
<p align="center"><strong>Text-Guided Zero-Shot Vehicle Counting under Occlusion</strong></p>
<p align="center">Đếm phương tiện theo mô tả văn bản trong cảnh có che khuất.</p>

<p align="center">
  <a href="web_demo/install/README.md"><img alt="Python 3.10" src="https://img.shields.io/badge/Python-3.10-3776AB?logo=python&logoColor=white"></a>
  <a href="web_demo/install/README.md"><img alt="CUDA 12.4" src="https://img.shields.io/badge/CUDA-12.4-547D16?logo=nvidia&logoColor=white"></a>
  <a href="checkpoints/README.md"><img alt="Tải checkpoint" src="https://img.shields.io/badge/Checkpoint-GitHub_Release-6848B8?logo=github&logoColor=white"></a>
  <a href="datasets/CARPK-OCC-Local/README.md"><img alt="Dữ liệu CARPK-OCC Local" src="https://img.shields.io/badge/Dataset-CARPK--OCC_Local-147D81"></a>
</p>

## Demo · Đếm xe theo prompt

Nhập **`red car`**, CountLab tìm các khung xe trong ảnh rồi dùng SigLIP2 đối chiếu từng khung với mô tả. Trên ảnh mẫu dưới đây, nhánh **CountGD++ A3 + SigLIP2** tìm được 19 ứng viên và giữ lại **2 xe**. Bạn có thể đổi nhánh trên web để so sánh cùng ảnh và prompt, rồi xuất ảnh kết quả, JSON hoặc CSV.

| Ảnh đầu vào | Kết quả với `red car` |
|:---:|:---:|
| ![Ảnh đường phố dùng làm đầu vào](web_demo/static/assets/street.jpg) | ![Hai xe được mô hình giữ lại, có khung và số thứ tự](docs/images/red-car-result.png) |

## 🚀 Chạy trên máy cá nhân

Cần **Linux x86-64**, **Python 3.10** có `venv`, Git, curl và GPU NVIDIA với driver hỗ trợ CUDA 12.4. Dành khoảng **50 GB** cho môi trường, cache và mô hình.

```bash
git clone https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion.git
cd Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion
bash web_demo/install/setup.sh
.demo-envs/detector/bin/python -m web_demo.run
```

Mở **http://127.0.0.1:7860**. Lần đầu, bộ cài tải các trọng số và kiểm tra SHA-256. Những lần sau chỉ cần chạy lại dòng cuối; backend sẽ nạp model của nhánh bạn chọn khi bấm **Đếm**. [Hướng dẫn cài đặt và chạy offline](web_demo/install/README.md)

## Các nhánh mô hình

| Nhánh trên web | Cách tạo kết quả |
|---|---|
| **CountGD++ gốc** | Đưa trực tiếp cụm mô tả vào detector. Đây là nhánh đối chứng. |
| **CountGD++ A3 + SigLIP2** | Khi train, Teacher nhìn ảnh sạch và hướng dẫn Student học đặc trưng từ ảnh có vật che. Khi suy luận, A3 tìm ứng viên và SigLIP2 kiểm tra từng vùng ảnh với mô tả. |
| **CountGD-Box + SAM 3** | CountGD-Box do Cubi huấn luyện, dùng checkpoint tối ưu AP để tìm xe. SAM 3 nhận prompt đầy đủ, sau đó hệ thống ghép các vùng khớp với khung detector. |
| **CountGD++ A3 + SAM 3** | Dùng A3 để tạo khung ứng viên và SAM 3 để kiểm tra mô tả trên toàn ảnh. |

## 📦 Checkpoint và dữ liệu

| Tài nguyên | Tải xuống |
|---|---|
| **CountGD++ A3** · checkpoint gốc sau huấn luyện | [checkpoint_best.pth](https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion/releases/download/countlab-v1/checkpoint_best.pth) |
| **CountGD-Box của Cubi** · checkpoint tối ưu AP | [checkpoint_best_ap.pth](https://github.com/khanhduy2311/Text-Guided-Zero-Shot-Vehicle-Counting-under-Occlusion/releases/download/countlab-v1/checkpoint_best_ap.pth) |
| **Trọng số suy luận** · các tệp web tự tải | [Danh mục checkpoint và SHA-256](checkpoints/README.md) |
| **CARPK-OCC Local** · dữ liệu đánh giá của nhóm | [Tải ảnh, mặt nạ và nhãn theo mức che](datasets/CARPK-OCC-Local/README.md) |

CARPK-OCC Local gồm **459 ảnh CARPK gốc**, mỗi ảnh có thêm ba phiên bản che nhẹ, vừa và nặng. Tổng cộng có **1.836 mẫu đánh giá từ 459 ảnh gốc**. Các biến thể giữ nguyên số xe của ảnh gốc để đo ảnh hưởng của che khuất tổng hợp lên số đếm.

## Mã nguồn phương pháp

| Thành phần | Điểm bắt đầu |
|---|---|
| Huấn luyện A3 | [Teacher](scripts/run_countgdpp_a1.py) · [Student](scripts/run_countgdpp_step6.py) · [Loss đặc trưng](src/step6_model.py) · [Sinh vật che](src/countgdpp_occlusion.py) · [Cấu hình](configs/countgdpp_a3_v1.json) |
| Đếm theo văn bản | [Suy luận hai lượt](src/countgdpp_inference.py) · [SigLIP2 trong benchmark](scripts/run_siglip_semantics.py) · [Lọc mô tả trên web](web_demo/open_text.py) · [Ngưỡng lọc](web_demo/open_text_policy.json) |
| CountGD-Box và SAM 3 | [Train tối ưu AP](Cubi/train_countgd_box_ap_optimized.py) · [Train với vật che](Cubi/train_countgd_box_occ.py) · [Benchmark](Cubi/benchmark_countvimg_suite.py) · [Suy luận SAM 3](Cubi/infer_sam3.py) |

<details>
<summary><strong>Chạy CountGD-Box + SAM 3 bằng dòng lệnh</strong></summary>

Sau khi chạy bộ cài, có thể dùng cùng ảnh mẫu cho cả hai giai đoạn. Lệnh đầu xuất các khung ứng viên. Lệnh sau kiểm tra chúng với prompt và in số đếm.

```bash
.demo-envs/detector/bin/python -m Cubi.infer_sam3 proposals --image web_demo/static/assets/street.jpg --noun car --checkpoint checkpoints/countlab/countgd-box.pth --out proposals.json
.demo-envs/sam/bin/python -m Cubi.infer_sam3 verify --image web_demo/static/assets/street.jpg --prompt 'red car' --sam3 checkpoints/sam3 --proposals proposals.json
```

</details>

<details>
<summary><strong>Chuẩn bị backend để tái lập huấn luyện A3</strong></summary>

Trên một checkout mới, lấy hai backend upstream ở các commit đã ghim và áp dụng bản sửa deformable attention cho CountGD++. Bộ cài web đã thực hiện bước này nếu bạn đã chạy demo.

```bash
git submodule update --init --recursive
git -C CountGDPlusPlus apply ../patches/countgdpp-deformable-attention.patch
```

Các runner huấn luyện đọc manifest dữ liệu, nhãn và artifact của giai đoạn trước theo cấu hình. [Định danh checkpoint CountGD++ ban đầu](configs/pretrained_countgdpp_official.json)

</details>
