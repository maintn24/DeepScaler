# Kịch bản tái lập DeepScaler (theo báo, Sec. V–VI)

Tài liệu này mô tả cách chạy từng kịch bản trong báo. Phần cơ bản (SSH, bootstrap, `00`–`01`, profiling μ, đưa file về máy, tắt EC2) nằm trong [`RUN_ONLINE.md`](RUN_ONLINE.md); thu dữ liệu train Boutique có burst: [`COLLECT_DATA.md`](COLLECT_DATA.md). Mọi lệnh đều chạy trên EC2, trong thư mục `~/DeepScaler`, bên trong `tmux`.

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

Hướng dẫn từng bước đầy đủ (EC2, Security group, lỗi thường gặp): [`COLLECT_DATA.md`](COLLECT_DATA.md). Phần dưới đây là bản tóm tắt.


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

Các baseline của Table II (Sec. V-B): AWS [47], MMN [33], HAB [35], SLO [48].
- Mọi autoscaler dùng cùng trace FIFA, chạy 75 phút, bắt đầu với 1 pod mỗi service.
- SLA là 200 ms trên p95 của frontend.
- Bỏ 10 phút đầu khi tính chỉ số.

| Autoscaler | Trong báo | Cách dựng lại ở đây | Lệnh |
|---|---|---|---|
| **DeepScaler** | đề xuất của báo | `pipeline/autoscaler.py`, model `model_states/ds_burst_tuned`, trust threshold (Sec. IV-D) | `deploy/05_online_run.sh ds` |
| **AWS** [47] | *"emulates Amazon Auto-Scaling Services"*, luật theo CPU | Kubernetes HPA, CPU 50 % (target tracking). Không dùng được AWS Auto Scaling thật, vì nó không scale pod trong cluster kind | `deploy/05_online_run.sh hpa` |
| **SLO** [48] | luật theo response time, chia nhỏ SLO | `rule_autoscalers.py --policy slo`: chia SLA end-to-end theo tỉ lệ latency của từng service trong một lần chạy HPA tham chiếu (`REF_CSV`); p95 của service > SLA cục bộ thì +1 pod; 4 chu kỳ liền < 0,5 × SLA cục bộ thì −1 pod | `deploy/05_online_run.sh slo` |
| **MMN** [33] | hàng đợi M/M/n/PS cho từng service, không holistic | `rule_autoscalers.py --policy mmn`: tốc độ phục vụ = μ / 0,5 (`reports/mu.json`); chọn n nhỏ nhất sao cho R(n) ≤ SLA cục bộ và ρ ≤ 0,8. Nếu SLA cục bộ nhỏ hơn thời gian phục vụ, chỉ dùng ρ ≤ 0,8 | `deploy/05_online_run.sh mmn` |
| HAB [35] | mạng hàng đợi cân bằng, holistic | **không làm** (báo không mô tả đủ để dựng lại đúng; ghi vào hạn chế) | – |
| (ablation) DeepScaler không có trust threshold | – | `AUTOSCALER_ARGS="--trust_k 0"` với `deploy/04_run_deepscaler.sh` | chạy tay theo `RUN_ONLINE.md` bước 4b |

### Thứ tự và số lần chạy

Các lần chạy 04/11 và 08/11 cho thấy **lần chạy đầu tiên trên một cluster luôn tốt hơn 3–5 điểm phần trăm**. Mức này lớn hơn khác biệt giữa các autoscaler. Vì vậy:
- **Dựng lại cluster trước mỗi lần chạy.** Đây là mặc định của `deploy/05_online_run.sh` (`FRESH=1`, thêm khoảng 15–20 phút mỗi lần).
- **Đổi thứ tự giữa các đợt.** Ví dụ đợt 1: SLO → MMN; đợt 2: MMN → SLO.
- Mỗi autoscaler chạy ít nhất 2 lần. Báo cáo mean ± std, ghi rõ n.

Mỗi lần `deploy/05_online_run.sh` mất khoảng 1 giờ 40 phút: dựng cluster, chờ 2 phút, Locust 75 phút, collect. Kết quả là `data/raw/online_<autoscaler>_<ngày giờ>.csv` (khoảng 151 dòng), cùng log autoscaler và log Locust trong `Logs/`.

Biến môi trường: `FRESH=0` để dùng lại cluster hiện có; `REF_CSV=...` để chọn lần chạy HPA tham chiếu cho SLO/MMN (mặc định là file `online_hpa_*.csv` mới nhất); `SLO=0.2`; `MODEL_DIR=...`.

### So sánh

```bash
python -m pipeline.compare_runs --slo 0.2 --warmup_min 10 --out reports/table2_full \
   DeepScaler=data/raw/online_dstuned_20261104_1637.csv DeepScaler=data/raw/online_dstuned_20261108_1704.csv \
   AWS=data/raw/online_hpa_20261104_1802.csv AWS=data/raw/online_hpa_20261108_1548.csv \
   SLO=data/raw/online_slo_<1>.csv SLO=data/raw/online_slo_<2>.csv \
   MMN=data/raw/online_mmn_<1>.csv MMN=data/raw/online_mmn_<2>.csv
```

Kết quả gồm:
- `table2.csv`: tỉ lệ vi phạm (eq. 1), core-hours (eq. 2), CAE = Σ|p95 − SLO| và `CAE_over` = Σ max(0, p95 − SLO), dạng mean ± std.
  - CAE như báo (không có công thức trong báo) cũng tăng khi latency thấp hơn SLA nhiều. Vì vậy báo cáo cả `CAE_over`.
- `slo_sweep.csv`: tỉ lệ vi phạm theo từng mức SLO.
- `fig10_cumulative.png`: các đường cộng dồn như Fig. 10.

### Kết quả đã có (đợt 04–09/11/2026, n = 2 mỗi autoscaler)

| Autoscaler | Vi phạm 200 ms | Chi phí (core-h) | CAE (s) | CAE_over (s) | Lần chạy |
|---|---|---|---|---|---|
| **DeepScaler** (ds_burst_tuned) | **13,4 ± 1,1 %** | 5,06 ± 0,22 | **12,8 ± 0,9** | 1,25 ± 0,20 | 04/11 16:37; 08/11 17:04 |
| SLO [48] | 13,7 ± 2,3 % | 5,57 ± 0,02 | 13,0 ± 0,1 | **1,17 ± 0,09** | 08/11 19:16; 08/11 23:46 |
| AWS [47] (HPA CPU 50 %) | 14,1 ± 5,0 % | 5,17 ± 0,01 | 14,4 ± 1,5 | 1,49 ± 0,35 | 04/11 18:02; 08/11 15:48 |
| MMN [33] | 21,0 ± 1,1 % | **4,12 ± 0,01** | 15,0 ± 0,6 | 4,20 ± 0,11 | 08/11 20:37; 08/11 22:26 |
| *(đã loại)* | – | – | – | – | DS 04/11 21:10 (lỗi); HPA 04/11 19:43 (bỏ theo cặp) |

Bảng chi tiết, đường cộng dồn kiểu Fig. 10 và tỉ lệ vi phạm theo ngưỡng SLA nằm trong `reports/table2_full/`.

**So với Table II của báo (không có HAB):**
- **Vi phạm SLA:** DS 13,4 < SLO 13,7 < AWS 14,1 < MMN 21,0. Thứ tự **giống hệt báo** (DS 3,8 < SLO 6,0 < AWS 7,3 < MMN 8,0). Tuy nhiên chênh lệch giữa DS, SLO và AWS nhỏ hơn độ lệch chuẩn. Chỉ khác biệt với MMN là rõ ràng.
- **CAE:** DS thấp nhất và MMN cao nhất, **giống báo**.
- **Chi phí:** DS < AWS < SLO, **giống báo**, SLO đắt nhất. MMN rẻ nhất, khác báo, vì cách dựng lại MMN ở đây cấp ít pod (ρ ≤ 0,8 với tốc độ phục vụ μ/0,5). Vì vậy MMN vi phạm nhiều nhất và có CAE_over gấp khoảng 3,5 lần các autoscaler khác.

**Mức cải thiện trung bình của DeepScaler so với các baseline** (cách tính của báo: trung bình mức cải thiện so với từng baseline):

| | So với AWS | So với SLO | So với MMN | Trung bình | Báo |
|---|---|---|---|---|---|
| Vi phạm | −5 % | −3 % | −36 % | **−15 %** | −41 % |
| Chi phí | −2 % | −9 % | +23 % | **+4 %** | −23 % |
| CAE | −11 % | −2 % | −15 % | **−9 %** | −13 % |

**Kết luận:** thứ tự của các autoscaler và các xu hướng chính (DS dẫn đầu về vi phạm và CAE, SLO đắt nhất, MMN vi phạm nhiều nhất) được tái lập. Mức ưu thế của DeepScaler nhỏ hơn báo nhiều. So với SLO và AWS, chênh lệch nằm trong mức dao động giữa các lần chạy.

**Hạn chế:**
- n = 2 mỗi autoscaler.
- Hai lần DS/AWS ngày 04/11 không chạy trên cluster mới; SLO và MMN đều chạy trên cluster mới.
- Các tham số `target_util` 0,5, `rho_max` 0,8, `down` 0,5 và `down_steps` 4 là do mình chọn; báo không ghi.
- HAB [35] không dựng lại được.

## D2. Burst

Giống D, thêm `TRACE=sendFlow/fifa-burst.req` vào trước lệnh `locust` (`RUN_ONLINE.md`, bước 6b). Mỗi autoscaler chạy 1 lần.

## E. Overhead (Table III)

Trong một lần chạy D, mở thêm cửa sổ tmux:

```bash
pid=$(pgrep -f pipeline.autoscaler); while sleep 30; do ps -o %cpu=,rss= -p $pid; done | tee Logs/overhead_autoscaler.txt
kubectl top pod -n istio-system | grep prometheus          # chi phí monitoring
```

Thời gian train lấy từ `train_time_s` trong `metrics.json`. Lưu ý: báo đo trên 10,64 core, còn ở đây train trên GPU T4 của Colab.
