# Kịch bản tái lập DeepScaler (theo báo, Sec. V–VI)

Tài liệu này mô tả cách chạy từng kịch bản trong báo. Phần cơ bản (SSH, bootstrap, `00`–`01`, profiling μ, đưa file về máy, tắt EC2) nằm trong [`RUN_ONLINE.md`](RUN_ONLINE.md). Mọi lệnh đều chạy trên EC2, trong thư mục `~/DeepScaler`, bên trong `tmux`.

Chọn ứng dụng bằng biến `DS_APP`: `boutique` (mặc định), `bookinfo` hoặc `trainticket`. Biến này áp dụng cho mọi script trong `pipeline/` (xem `pipeline/common.py`).

| Kịch bản | Trong báo | Ứng dụng | Máy EC2 | Thời gian (ước tính) |
|---|---|---|---|---|
| **A** – thu dữ liệu train (HPA điều khiển) | dữ liệu cho mọi hình | Boutique, BookInfo, Train-Ticket | c6i.2xlarge; Train-Ticket: m6i.4xlarge | mỗi app ≥ 6 giờ |
| **B** – nhiều kiểu request | Fig. 11 | Boutique | c6i.2xlarge | 3 × 2 giờ |
| **C** – độ chính xác dự đoán | Fig. 12 | cả 3 | Colab | khoảng 1 giờ |
| **D** – autoscaling online | Table II, Fig. 9–10 | Boutique | c6i.2xlarge (có thể chạy song song nhiều máy) | 4 autoscaler × 3 lần × 75 phút |
| **D2** – burst | đoạn bàn về Fig. 10 | Boutique | như D | 4 × 1 × 75 phút |
| **E** – overhead | Table III | Boutique | trong lúc chạy D | – |

Các lựa chọn mà báo **không nêu** được cố định trước khi chạy, và luận văn phải ghi rõ:
- SLO: p95 tại service vào (entry), báo cáo theo dải 100–300 ms, giá trị chính là 200 ms.
- Định nghĩa CAE.
- Các tham số của trust threshold.
- Cách cài lại hai autoscaler SLO và MMN.
- Ngưỡng T_Ci của eq. 16.

---

## A. Thu dữ liệu train (HPA điều khiển)

### A0. Boutique, có burst và vượt SLA (làm trước tiên)

Lần thu `run2_hpa` (5,1 giờ) không có bước nào vượt SLA (p95 tối đa 117 ms). Số pod gần như không đổi: giữ nguyên số pod chỉ sai 0,039 pod. Với dữ liệu như vậy, DeepScaler không thể hơn DCRNN hay cách giữ nguyên số pod. Báo cũng nêu điều này (Sec. VI): DeepScaler bị giới hạn bởi *"the quality and quantity of available historical data"*.

Trace mới `sendFlow/train-burst-6h.req` dài 6 giờ, sinh bằng `pipeline/make_traces.py train`, seed 2026:
- Trace FIFA của tác giả được lặp lại. Mỗi vòng 75 phút nhân với một mức tải 1,0 / 1,5 / 0,6 / 1,8 / 1,2, để HPA dùng nhiều số pod khác nhau và có cả lúc scale in.
- Có 15 burst đột ngột, mỗi burst dài 1–4 phút, nhân tải ×1,8–3,0, tối đa 220 user. HPA không kịp phản ứng nên sẽ có vi phạm SLA, giống hiện tượng báo mô tả ở Fig. 10 (phút 30–50, Sec. V-B).
- Thời điểm của từng burst nằm trong `sendFlow/train-burst-6h.req.bursts.csv`.

```bash
# 0. Dựng EC2 + cluster + Boutique như RUN_ONLINE.md bước 1–3 (02_boutique.sh: HPA CPU 50 %, 1..10 pod)
python -m pipeline.make_traces train --out sendFlow/train-burst-6h.req --cap 220   # đã có trong repo, chạy lại ra y hệt

# 1. Chạy thử 20 phút (burst đầu tiên ở phút 13), kiểm tra có vượt SLA không
RUN=smoke_$(date +%Y%m%d_%H%M); echo "START $(date '+%Y-%m-%d %H:%M:%S')" | tee Logs/${RUN}_time.txt
TRACE=sendFlow/train-burst-6h.req RUN_TIME=1200 locust -f pipeline/locustfile.py --headless \
    --host http://localhost:31000 --csv Logs/${RUN}_locust > Logs/${RUN}_locust.log 2>&1
python -m pipeline.collect --last 20 --step 30 --out data/raw/${RUN}.csv
python -m pipeline.check_data data/raw/${RUN}.csv --bursts sendFlow/train-burst-6h.req.bursts.csv \
    --start "$(sed -n 's/^START *//p' Logs/${RUN}_time.txt)"
kubectl -n boutique scale deploy --all --replicas=1          # đưa về 1 pod rồi để HPA tự quyết

# 2. Chạy thật 6 giờ (trong tmux)
RUN=train_burst_$(date +%Y%m%d_%H%M); echo "START $(date '+%Y-%m-%d %H:%M:%S')" | tee Logs/${RUN}_time.txt
TRACE=sendFlow/train-burst-6h.req RUN_TIME=21600 locust -f pipeline/locustfile.py --headless \
    --host http://localhost:31000 --csv Logs/${RUN}_locust > Logs/${RUN}_locust.log 2>&1
echo "END   $(date '+%Y-%m-%d %H:%M:%S')" | tee -a Logs/${RUN}_time.txt

# 3. Thu metric + call graph, rồi kiểm tra
START=$(sed -n 's/^START *//p' Logs/${RUN}_time.txt); END=$(sed -n 's/^END *//p' Logs/${RUN}_time.txt)
python -m pipeline.collect --start "$START" --end "$END" --step 30 \
    --out data/raw/${RUN}.csv --adj_out data/raw/${RUN}_od.npy
python -m pipeline.check_data data/raw/${RUN}.csv --bursts sendFlow/train-burst-6h.req.bursts.csv --start "$START"
```

Dữ liệu đạt khi `check_data` cho thấy:

| Chỉ số | Mong muốn | `run2_hpa` cũ |
|---|---|---|
| Tỉ lệ vượt SLA 200 ms (cả lần chạy) | khoảng 5–20 % | 0 % |
| Phần vượt SLA nằm trong burst / ngoài burst | phần lớn nằm trong burst, ngoài burst < 5 % | – |
| Số burst gây vi phạm | ≥ 2/3 | – |
| MAE khi giữ nguyên số pod | ≥ 0,1 pod | 0,039 |
| CPU app tối đa | < khoảng 5 core (máy 8 vCPU) | 1,98 |

Nếu bước 1 chưa đạt, chỉnh như sau rồi chạy thử lại. Ghi giá trị cuối cùng vào luận văn.
- **Không vượt SLA:** tăng burst (`--factor_rng 2.5,4 --cap 260`).
- **Vượt SLA cả ngoài burst** (máy hoặc HPA bão hoà): giảm mức tải (`--scales 0.8,1.2,0.5,1.5,1.0`) hoặc `--cap`.

### A1. BookInfo và Train-Ticket

BookInfo và Train-Ticket dùng **đúng manifest của tác giả** trong `benchmarks/` (không sửa), chỉ thêm Istio sidecar (gắn nhãn namespace) và HPA CPU 50 %. Cluster kind phải được tạo lại bằng `00_kind_cluster.sh` mới, vì cần map thêm cổng 30012 và 32677.

| App | Manifest | Namespace | Service (node) | Cổng vào (Locust) | Bộ sinh tải |
|---|---|---|---|---|---|
| BookInfo | `benchmarks/bookinfo/bookinfo.yaml` (ảnh `tongjingwan/*`) | `bookinfossj` (cố định trong manifest) | productpage, reviews, ratings, details | `http://localhost:30012` | `pipeline/locust_bookinfo.py` (repo tác giả không có bộ cho BookInfo) |
| Train-Ticket | `benchmarks/train-ticket/ts-deployment-part1–3.yml` (ảnh `codewisdom/*:0.0.4`) | `trainticket` | 41 service của part2 (đúng 41 như báo) | `http://localhost:32677` (`ts-ui-dashboard`, nginx) | `pipeline/locust_trainticket.py` = 14 luồng của `sendFlow/load_generator_train.py` (tác giả) |

```bash
# BookInfo (cùng cluster với Boutique được, khác namespace và cổng; nhưng thu riêng từng app)
deploy/02_bookinfo.sh
export DS_APP=bookinfo
RUN=bookinfo_hpa_$(date +%Y%m%d_%H%M)
echo "START $(date '+%Y-%m-%d %H:%M:%S')" | tee Logs/${RUN}_time.txt
RUN_TIME=21600 locust -f pipeline/locust_bookinfo.py --headless --host http://localhost:30012 \
    --csv Logs/${RUN}_locust > Logs/${RUN}_locust.log 2>&1          # 6 giờ
echo "END   $(date '+%Y-%m-%d %H:%M:%S')" | tee -a Logs/${RUN}_time.txt
START=$(sed -n 's/^START *//p' Logs/${RUN}_time.txt); END=$(sed -n 's/^END *//p' Logs/${RUN}_time.txt)
python -m pipeline.collect --start "$START" --end "$END" --step 30 \
    --out data/raw/${RUN}.csv --adj_out data/raw/${RUN}_od.npy
```

**Train-Ticket** cần máy **m6i.4xlarge (16 vCPU, 64 GB)**: `ec2_bootstrap.sh` → `00` → `01`, sau đó:

```bash
deploy/02_trainticket.sh            # 22 DB, rồi 41 service Java + ts-ui-dashboard; khoảng 20–40 phút
export DS_APP=trainticket
# giống BookInfo, đổi locustfile và host:
RUN_TIME=21600 locust -f pipeline/locust_trainticket.py --headless --host http://localhost:32677 ...
```

Với Train-Ticket, độ trễ end-to-end đo ở `ts-ui-dashboard` (không phải 1 trong 41 node). `collect.py` tự thêm các cột `ts-ui-dashboard_p95/_res/_req/_err` vào CSV. HPA giới hạn tối đa 3 pod mỗi service vì RAM (biến `MAXR`); cần ghi điều này vào luận văn.

Nếu pod báo `ImagePullBackOff`, nghĩa là ảnh Docker của tác giả không còn trên Docker Hub. Khi đó dừng lại và báo, đừng tự đổi sang phiên bản khác của ứng dụng.

`collect.py` giờ ghi thêm file `*_npas.npy`: số request trung bình trên mỗi cạnh của call graph. File này dùng để tính đồ thị OD theo eq. 15. Với Train-Ticket, HPA giới hạn tối đa 3 pod mỗi service vì RAM (biến `MAXR`); cần ghi điều này vào luận văn.

## B. Fig. 11: đổi trọng số các loại request (Boutique)

Báo: *"we verify its adaptability to different workload patterns by tuning different weights of user request types"*. Locustfile của Boutique nhận biến `TASK_WEIGHTS`. Trọng số mặc định giống `sendFlow/load_generator.py` của tác giả: `index=1,set_currency=2,browse_product=10,view_cart=2,add_to_cart=3,checkout=1`.

Đề xuất 3 kiểu tải (giống tỉ lệ Buy:Browse trong Fig. 1), mỗi kiểu thu 2 giờ dưới HPA:

| Tên | `TASK_WEIGHTS` |
|---|---|
| browse | `browse_product=10,add_to_cart=1,checkout=1` |
| mixed (mặc định) | để trống |
| buy | `browse_product=4,add_to_cart=6,checkout=4` |

```bash
TASK_WEIGHTS="browse_product=4,add_to_cart=6,checkout=4" RUN_TIME=7200 locust -f pipeline/locustfile.py ...
```

Sau đó, với mỗi kiểu tải: chạy `build_dataset --npas data/raw/<run>_npas.npy`, rồi train. `metrics.json` sẽ có `fig11_sim_od` / `fig11_sim_cc`, là độ giống tính bằng công thức trong `similarity.py` của tác giả, so với OD (eq. 15) và CC (eq. 16) đã chuẩn hoá theo eq. 3. Cột `fig11_identity_baseline_od` là điểm số mà một ma trận đơn vị (không có cạnh nào) đạt được. Phải so độ giống của model với mức này.

## C. Fig. 12: dự đoán trên 3 ứng dụng (Colab)

Dùng notebook `notebooks/DeepScaler_train_colab_kaggle.ipynb` cho từng ứng dụng. Đặt `DS_APP` trong ô cấu hình, upload CSV và `_od.npy` của ứng dụng đó, để `RUN_BASELINES = True`. So MAE/RMSE/MAPE của DeepScaler, DCRNN và T-GCN theo số service: 4 (BookInfo), 10 (Boutique), 41 (Train-Ticket).

## D. Table II / Fig. 9–10: autoscaling online (Boutique)

Cả 4 autoscaler đều dùng cùng trace FIFA, chạy 75 phút, bắt đầu với 1 pod mỗi service. Chỉ đổi lệnh ở **cửa sổ 1** của bước 4b trong `RUN_ONLINE.md`:

| Autoscaler | Cửa sổ 1 |
|---|---|
| **DeepScaler** (có trust threshold, Sec. IV-D) | `LOG=Logs/${RUN}_autoscaler.jsonl MODEL_DIR=model_states/ds_s2021 deploy/04_run_deepscaler.sh` (mặc định `--trust_k 3 --trust_window 10 --trust_cooldown 10 --slo 0.2`) |
| DeepScaler không có trust threshold (ablation) | thêm `AUTOSCALER_ARGS="--trust_k 0"` vào trước lệnh trên |
| **AWS** [47] ≈ HPA CPU 50 % | `deploy/02_boutique.sh` (tạo lại HPA), không có cửa sổ 1 |
| **SLO** [48] | `kubectl -n boutique delete hpa --all; python -m pipeline.rule_autoscalers --policy slo --slo 0.2 --ref_csv data/raw/run2_hpa_5f.csv --log Logs/${RUN}_autoscaler.jsonl` |
| **MMN** [33] | `kubectl -n boutique delete hpa --all; python -m pipeline.rule_autoscalers --policy mmn --slo 0.2 --mu reports/mu.json --ref_csv data/raw/run2_hpa_5f.csv --log Logs/${RUN}_autoscaler.jsonl` |
| HAB [35] | không làm (mạng hàng đợi cân bằng, rất khó cài lại đúng; ghi vào phần hạn chế) |

Mỗi autoscaler chạy **3 lần**, xen kẽ thứ tự. Để rút ngắn thời gian, có thể bật nhiều máy c6i.2xlarge **cùng loại**, mỗi máy một autoscaler. Sau mỗi lần chạy: `collect` (bước 5 của `RUN_ONLINE.md`). Khi đã đủ tất cả các lần chạy:

```bash
python -m pipeline.compare_runs --slo 0.2 --warmup_min 10 --out reports/table2 \
   DeepScaler=data/raw/online_ds_1.csv DeepScaler=data/raw/online_ds_2.csv DeepScaler=data/raw/online_ds_3.csv \
   AWS=data/raw/online_hpa_1.csv ... SLO=... MMN=...
```

Kết quả gồm:
- `table2.csv`: tỉ lệ vi phạm (eq. 1), core-hours (eq. 2) và CAE, dạng mean ± std như Table II.
- `slo_sweep.csv`: tỉ lệ vi phạm theo từng mức SLO.
- `fig10_cumulative.png`: các đường cộng dồn như Fig. 10.

## D2. Burst

Giống D, thêm `TRACE=sendFlow/fifa-burst.req` vào trước lệnh `locust` (`RUN_ONLINE.md`, bước 6b). Mỗi autoscaler chạy 1 lần.

## E. Overhead (Table III)

Trong một lần chạy D, mở thêm cửa sổ tmux:

```bash
pid=$(pgrep -f pipeline.autoscaler); while sleep 30; do ps -o %cpu=,rss= -p $pid; done | tee Logs/overhead_autoscaler.txt
kubectl top pod -n istio-system | grep prometheus          # chi phí monitoring
```

Thời gian train lấy từ `train_time_s` trong `metrics.json`. Lưu ý: báo đo trên 10,64 core, còn ở đây train trên GPU T4 của Colab.
