# Tune tự do DeepScaler (được đổi giá trị của code), Colab T4, 04/11/2026

- **Dữ liệu:** `train_burst_20261104_0751` (6 giờ, 30 s, có burst). Chia tập 0,7/0,1/0,2 theo bước được dự đoán.
- **Script:** `experiments/tune_free.py`.
- **Lý do được đổi tham số:** cluster của ta khác báo (1 máy kind c6i.2xlarge so với 8 VM, 144 core; Sec. V-A), nên các giá trị trong code của tác giả không nhất thiết hợp.
- **Cách chọn:** chỉ dựa trên **MAE tập validation**; tập test chỉ để báo cáo.
  - Bước 1: thử từng yếu tố một từ cấu hình gốc, seed 2021.
  - Bước 2: ghép các giá trị tốt nhất, chạy 3 seed.

## Bước 1: thử từng yếu tố (seed 2021)

Cấu hình gốc: 30 epoch × 6 vòng, cửa sổ 80, lr 1e-3, 128 filter, 2 block, kernel 5, batch 64, đồ thị khởi tạo od, ε và δ như code. Các giá trị trong ngoặc ở cột "Thay đổi" là giá trị của code.

| Thay đổi | MAE valid | MAE test | Số cạnh học được |
|---|---|---|---|
| gốc | 0,255 | 0,187 | 7 |
| epoch 15 / **50** | 0,290 / **0,248** | 0,216 / 0,175 | 2 / 1 |
| cửa sổ (step_in) **20** / 40 (code: 80) | **0,138** / 0,178 | 0,143 / 0,170 | 1 / 6 |
| lr 3e-4 / 3e-3 | 0,366 / 0,383 | 0,275 / 0,287 | 3 / 14 |
| 64 filter (code: 128) | 0,315 | 0,276 | 6 |
| block 1 / **3** (code: 2) | 0,364 / **0,224** | 0,245 / 0,167 | 6 / 4 |
| kernel 3 (code: 5) | 0,270 | 0,221 | 4 |
| **batch 32** (code: 64) | **0,218** | 0,149 | 9 |
| đồ thị khởi tạo od+cc | 0,272 | 0,206 | 6 |
| ε 0,01 / **0,1** (code: 0,5/N = 0,05) | 0,246 / **0,227** | 0,173 / 0,186 | 16 / 1 |
| **δ 0,2** (code: 0,05) | **0,229** | 0,214 | 2 |

**Cấu hình được chọn:** 50 epoch × 6 vòng, cửa sổ 20, lr 1e-3, 128 filter, **3 block**, kernel 5, **batch 32**, đồ thị od, **ε 0,1**, **δ 0,2**.

## Bước 2: 3 seed (2021–2023), MAE test (số pod)

| Model | Cửa sổ | MAE test |
|---|---|---|
| DeepScaler, giá trị của code (5/5) | 80 | 0,315 ± 0,028 |
| DeepScaler, cấu hình gốc (30/6) | 80 | 0,197 ± 0,033 |
| **DeepScaler, đã tune** | 20 | **0,106 ± 0,005** |
| DCRNN (≤ 300 epoch, dừng sớm sau 20 epoch) | 80 / 20 | 0,095 ± 0,005 / **0,097 ± 0,005** |
| T-GCN | 80 / 20 | 0,286 / 0,227 ± 0,004 |
| Giữ nguyên số pod | – | **0,066** |

- **Fig. 12:** DeepScaler sau khi tune giảm sai số khoảng 3 lần so với giá trị của code. Nó ngang DCRNN (chênh 0,009, khoảng 2 độ lệch chuẩn) và hơn hẳn T-GCN. Như vậy pattern "DeepScaler và DCRNN ≫ T-GCN" ra được. Nhưng "DeepScaler tốt nhất" như báo thì chưa.
- **Fig. 11:** cấu hình đã tune chỉ học được 1 cạnh; độ giống với OD là 0,39, so với 0,33 của ma trận không có cạnh nào. Phần cải thiện đến từ phần thời gian (cửa sổ ngắn, mạng sâu hơn, train lâu hơn), không đến từ đồ thị học được.
- **Không model nào thắng cách giữ nguyên số pod** khi dự đoán nhãn HPA trước 1 bước 30 s.

## Ghi vào luận văn
- Ghi bảng tham số đã đổi so với code của tác giả và lý do: phần cứng và độ dài dữ liệu khác.
- Ghi cách chọn theo tập validation.
- Báo cáo cả kết quả với giá trị của code lẫn với cấu hình đã tune.
- Model đã tune (cấu hình trên, seed 2022: MAE valid 0,099, test 0,099) dùng được cho chạy online (`guide/RUN_ONLINE.md`). Model mới dùng cửa sổ 20 bước nên chỉ cần 10 phút dữ liệu là bắt đầu dự đoán được.
