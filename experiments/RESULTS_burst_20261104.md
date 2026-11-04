# DeepScaler trên dữ liệu có burst (Colab T4, 04/11/2026)

Dữ liệu: `train_burst_20261104_0751.csv`, Online Boutique dưới HPA (CPU 50 %, 1–10 pod), 721 bước × 30 s (6 giờ), trace `sendFlow/train-burst-6h.req` (FIFA ×1,0/1,5/0,6/1,8/1,2 + 15 burst). Call graph Istio `_od.npy` (14 cạnh) và `_npas.npy`.

`check_data`:
- Vượt SLA 200 ms: 33,4 % thời gian. Cả 15/15 burst đều gây vi phạm.
- Ngoài burst vẫn vượt SLA 26,8 %, do HPA chạm giới hạn 10 pod ở các vòng tải ×1,5 và ×1,8 (nhãn nhu cầu của frontend lên tới 18 pod).
- MAE khi giữ nguyên số pod: 0,075 (cả lần chạy).

Script `experiments/tune_grid.py --strides 1,2 --splits "0.7,0.1,0.2" --epoch_iters 5/5`, 3 seed (2021–2023). Mọi tham số mà code gốc đã đặt được giữ nguyên (xem `RESULTS_tune_grid_20261102.md`). Chia tập theo bước được dự đoán; tập test là 20 % cuối.

## Kết quả: MAE trên tập test (số pod)

| Lấy mẫu | DeepScaler 5/5 | DCRNN | T-GCN | Giữ nguyên số pod |
|---|---|---|---|---|
| 30 s | 0,350 ± 0,008 | **0,095** ± 0,005 | 0,286 ± 0,001 | **0,066** |
| 60 s | 0,803 ± 0,183 | 0,178 ± 0,009 | 0,49 | 0,122 |

So với dữ liệu cũ `run2_hpa` (30 s, 0,7/0,1/0,2):
- DeepScaler 0,236; DCRNN 0,062; T-GCN 0,190; giữ nguyên số pod 0,040.
- Nhãn trên dữ liệu mới khó dự đoán hơn: sai số của mọi cách đều tăng, cách giữ nguyên số pod tăng 65 %.
- Thứ hạng không đổi: giữ nguyên số pod < DCRNN < T-GCN < DeepScaler.

Fig. 11 (độ giống đồ thị, công thức `similarity.py` của tác giả, so với OD theo eq. 3):
- DeepScaler đạt 0,38–0,41, chỉ học được 2–5 cạnh.
- Ma trận đơn vị (không có cạnh nào) đã đạt 0,333. Trong báo: 0,89.

## Đối chứng: train DeepScaler lâu hơn (20 epoch × 5 vòng, khác giá trị của code gốc)

| | MAE (30 s) | Fig. 11 OD |
|---|---|---|
| DeepScaler 20/5 | 0,204 ± 0,020 (0,200 / 0,181 / 0,230) | 0,39–0,40 |

Train lâu hơn làm sai số giảm 42 % và DeepScaler vượt T-GCN, nhưng vẫn kém DCRNN và cách giữ nguyên số pod. Các baseline thì được train tối đa 300 epoch, dừng sớm sau 20 epoch không cải thiện.

## Kết luận

1. **Dữ liệu tốt hơn nhưng pattern của Fig. 12 vẫn không ra.** Dữ liệu không phải nguyên nhân duy nhất.
2. **Ngân sách train mặc định của code (5/5) quá nhỏ.** Mỗi lần chạy chỉ mất khoảng 30 s. Train ×4 thì sai số giảm 42 %, nhưng thứ hạng so với DCRNN vẫn giữ nguyên.
3. **Bước học đồ thị (AdapGLA, Algorithm 2) không học ra phụ thuộc giữa các service.** Chỉ có 2–5 cạnh, độ giống OD chỉ nhỉnh hơn ma trận đơn vị. Phần "holistic" của báo (Fig. 11) không tái lập được.
4. **Với nhãn HPA dự đoán trước 1 bước 30 s, cách giữ nguyên số pod là đối thủ rất mạnh.** Không model nào thắng nó, vì HPA thay đổi chậm và có thời gian chờ trước khi scale in. Báo không so với cách này.
5. Điểm này ủng hộ hướng của luận văn. Nhãn HPA vừa gần như không đổi, vừa bị chặn ở 10 pod khi HPA bão hoà, nên nó là mục tiêu dự đoán kém; nhãn nhu cầu (O1) và hiệu chỉnh ACI giải quyết đúng chỗ này.
