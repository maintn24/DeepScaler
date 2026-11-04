# Lưới tune DeepScaler (Colab T4, 02/11/2026)

Dữ liệu: `run2_hpa_5f.csv`, Online Boutique dưới HPA, 612 bước × 30 s, call graph Istio `run2_hpa_od.npy`. Script: `experiments/tune_grid.py`, chạy 3 seed (2021–2023) cho mỗi cấu hình. Đo trên tập test. Persistence là cách giữ nguyên số pod của bước trước.

## Tham số

**Cố định theo code gốc, không thay đổi:**
- `config/train_dataset_speed.yaml`: num_block 2, kernel 5, cheb/time filters 128, K 3, init_feature_num 64, batch 64.
- `config/train_config.yaml`: Adam, lr 1e-3, wd 1e-4, CosineAnnealing (T_max 15, eta_min 1e-5).
- Trong code: ε = 0,5/N (eq. 6, `GraphLearn`, `alt_trainer.py`); δ = 0,05 (eq. 7); EarlyStop(5); max_graph_num 3.
- Dữ liệu: 5 feature (`data_process.py`), step_in 80 (`points_per_hour=80`), horizon 1.
- Đồ thị khởi tạo chỉ là call graph (Sec. IV-B, V-C).

**Code gốc có hai giá trị, nên thử cả hai:**
- Khoảng lấy mẫu: 30 s (báo Sec. V-B) hoặc 60 s (`data_process.py`).
- Epoch/vòng: 1/1 (`main.py`) hoặc 5/5 (`predict_scale.py`).

**Code gốc không đặt, được chọn:** cách chia tập 0,6/0,2/0,2 hoặc 0,7/0,1/0,2 (chia theo bước được dự đoán).

## Kết quả (MAE trên test, trung bình ± độ lệch qua 3 seed)

| Cấu hình | DeepScaler 1/1 | DeepScaler 5/5 | DCRNN | T-GCN | Persistence |
|---|---|---|---|---|---|
| 30 s, 0,6/0,2/0,2 | 3,114 ± 1,017 | 0,253 ± 0,009 | **0,058** ± 0,001 | 0,204 ± 0,014 | 0,040 |
| 30 s, 0,7/0,1/0,2 | 2,510 ± 0,910 | **0,236** ± 0,006 | **0,062** ± 0,002 | 0,190 ± 0,011 | 0,040 |
| 60 s, 0,6/0,2/0,2 | 3,514 ± 1,069 | 0,381 ± 0,018 | 0,132 ± 0,009 | 0,269 ± 0,006 | 0,079 |
| 60 s, 0,7/0,1/0,2 | 5,230 ± 0,965 | 0,355 ± 0,011 | 0,120 ± 0,004 (2 seed) | – | 0,079 |

Fig. 11 (độ giống đồ thị theo `similarity.py` của tác giả, so với OD/CC đã chuẩn hoá theo eq. 3):
- DeepScaler 5/5: OD 0,39–0,41, CC 0,14–0,17, số cạnh học được 2–6.
- **Ma trận đơn vị** (không có cạnh nào) đã đạt OD **0,33**. Trong báo: 0,89 ± 0,07.

## Đối chứng: thay giá trị mà code gốc đã đặt (không dùng làm kết quả chính)

30 s, 0,6/0,2/0,2, 15 epoch × 6 vòng:

| Cấu hình | MAE | Fig. 11 OD |
|---|---|---|
| step_in 80, đồ thị khởi tạo od | 0,135 ± 0,007 | 0,41 |
| step_in 20, od | 0,109 ± 0,003 | 0,39 |
| step_in 20, od + cc | 0,104 ± 0,002 | 0,40 |

## Kết luận

1. **Với các tham số code gốc đã đặt, chỉ chỉnh được cách chia tập và khoảng lấy mẫu.** Cấu hình tốt nhất là 30 s, 0,7/0,1/0,2, 5/5. Nhưng thứ hạng không đổi ở mọi cấu hình: persistence < DCRNN < T-GCN < DeepScaler. **Pattern của Fig. 12 không ra.**
2. **Đồ thị học được gần như không có cạnh.** Độ giống chỉ nhỉnh hơn mức của ma trận đơn vị. **Pattern của Fig. 11 không ra.**
3. Ngay cả khi thay các giá trị của code gốc (15/6, cửa sổ 20, thêm cc), DeepScaler vẫn kém DCRNN và persistence.
4. **Nghẽn nằm ở dữ liệu, không ở tham số.** Persistence chỉ sai 0,04 pod: nhãn gần như không đổi theo thời gian. Fig. 12 của báo có MAE khoảng 0,5 trên Online Boutique, tức số pod trong dữ liệu của tác giả dao động nhiều hơn hẳn. Hai thứ còn được phép thay đổi theo quy tắc của bạn là **lượng và độ đa dạng của dữ liệu** (tải lớn hơn, `SCALE`; thu lâu hơn; nhiều kiểu request như Fig. 11).
