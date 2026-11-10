# Chạy DeepScaler online trên EC2

Hướng dẫn chạy model DeepScaler đã train (`model_states/ds_s2021`) làm autoscaler thật cho Online Boutique, thay cho HPA. Các bước gồm: SSH vào EC2, tải repo, dựng cluster, chạy autoscaler và phát tải.

- Môi trường **phải giống lúc thu dữ liệu**: EC2 **c6i.2xlarge** (8 vCPU, 16 GB), Ubuntu 22.04/24.04, ổ gp3 50 GB, kind + Istio 1.22.3, trace FIFA với `SCALE=1`.
- Model học CPU%, latency và memory của đúng phần cứng này. Đổi loại máy thì phải thu dữ liệu và train lại.
- DeepScaler dự đoán số pod mỗi 30 s, giống chu kỳ trong báo (Sec. V-B: *"performs service instance prediction every 30 seconds"*).

Trước khi bật EC2, nhớ **push lên GitHub** các file mới: `pipeline/collect.py`, `pipeline/autoscaler.py`, `pipeline/uncertainty_report.py`, `pipeline/make_traces.py`, `deploy/04_run_deepscaler.sh` và hướng dẫn này. EC2 lấy code bằng `git clone` / `git pull`.

Thay các chỗ `<...>` bằng giá trị của bạn:

| Biến | Ý nghĩa |
|---|---|
| `<KEY.pem>` | File private key của EC2 (ví dụ `~/.ssh/deepscaler.pem`) |
| `<EC2_IP>` | Public IPv4 của instance |

---

## 0. Trên máy của bạn: chuẩn bị

Security group của EC2 chỉ cần mở **cổng 22 (SSH)**. Prometheus và Grafana sẽ xem qua SSH tunnel ở bước 7.

```bash
chmod 400 <KEY.pem>

# Nén model DeepScaler để gửi lên EC2 (thư mục này không có trên GitHub)
cd ~/Projects/DeepScaler
tar czf /tmp/ds_model.tgz model_states/ds_s2021 data/c20
```

## 1. SSH vào EC2 và cài môi trường (chỉ làm 1 lần)

```bash
ssh -i <KEY.pem> ubuntu@<EC2_IP>
```

Trên EC2:

```bash
git clone -b deepscaler-pipeline https://github.com/maintn24/DeepScaler.git
cd DeepScaler
bash deploy/ec2_bootstrap.sh     # docker, python venv ~/dsenv, locust, pandas...
exit                             # logout để nhóm docker có hiệu lực
```

## 2. Gửi model lên EC2

Chạy trên máy của bạn:

```bash
scp -i <KEY.pem> /tmp/ds_model.tgz ubuntu@<EC2_IP>:~/DeepScaler/
ssh -i <KEY.pem> ubuntu@<EC2_IP>
```

Trên EC2:

```bash
cd ~/DeepScaler && git pull
tar xzf ds_model.tgz
ls model_states/ds_s2021/AdapGLA     # phải có AdapGLA.pkl và best_adj_mx.npy
```

## 3. Dựng cluster + Istio + Prometheus + Online Boutique

```bash
tmux new -s ds                   # chạy trong tmux để mất SSH không bị dừng
cd ~/DeepScaler
deploy/00_kind_cluster.sh
deploy/01_istio_prometheus.sh
deploy/02_boutique.sh            # tạo cả HPA, bước 4 sẽ xoá

# Kiểm tra: cả hai lệnh phải trả về giá trị khác rỗng
curl -s 'http://localhost:30090/api/v1/query?query=up' | head -c 200; echo
kubectl -n boutique get pods
```

Mọi pod phải ở trạng thái `2/2 Running` (app + sidecar Istio). Nên chạy Locust 2–3 phút một lần để Prometheus có số liệu `istio_requests_total`.

## 3b. Đo năng lực pod μ (profiling, khoảng 35 phút, làm 1 lần)

μ_i là số req/s mà một pod của service i chịu được khi CPU của **cả pod (app + sidecar Istio)** đạt 50 % request, cùng mục tiêu với HPA. Từ μ suy ra nhãn nhu cầu `demand = max(1, ⌈req/μ⌉)` (mục tiêu O1). Đây là đáp án độc lập với quyết định của autoscaler, dùng để đo độ bất định ở bước 5.

Không dùng HPA. Cố định số pod, rồi tăng tải theo bậc: 10 → 150 user, mỗi bậc 2 phút.

```bash
cd ~/DeepScaler
kubectl -n boutique delete hpa --all --ignore-not-found
# 2 pod cho 9 service (không đụng redis-cart: 2 bản redis sẽ tách dữ liệu giỏ hàng), frontend 6 pod
kubectl -n boutique scale deploy adservice cartservice checkoutservice currencyservice emailservice \
    paymentservice productcatalogservice recommendationservice shippingservice --replicas=2
kubectl -n boutique scale deploy frontend --replicas=6
kubectl -n boutique rollout status deploy --timeout=300s

python -m pipeline.make_traces ramp --out sendFlow/ramp-10-150.req     # 30 phút
echo "START $(date '+%Y-%m-%d %H:%M:%S')" | tee Logs/profile_time.txt
TRACE=sendFlow/ramp-10-150.req RUN_TIME=1800 locust -f pipeline/locustfile.py --headless \
    --host http://localhost:31000 --csv Logs/profile_locust > Logs/profile_locust.log 2>&1
echo "END   $(date '+%Y-%m-%d %H:%M:%S')" | tee -a Logs/profile_time.txt

python -m pipeline.collect --prom http://localhost:30090 --namespace boutique \
    --start "<START>" --end "<END>" --step 30 --out data/raw/profile.csv
python -m pipeline.uncertainty_report --csv data/raw/profile.csv --mu_basis pod --mu_only --out_dir reports
cat reports/mu.json      # xem r2 (nên > 0.7) và "extrapolated" (true = μ nằm ngoài dải đo)
```

Nếu một service có `extrapolated: true`, nghĩa là tải chưa đẩy pod của nó tới 50 % CPU. Khi đó tăng `--hi` của `make_traces ramp` hoặc giảm số pod của service đó, rồi đo lại.

## 3c. Đo độ trễ và μ theo SLA (bản sửa 10/11/2026)

**Độ trễ.** Từ bản này mỗi lần chạy ghi độ trễ ở hai nơi:
- **Locust, từng request** (`Logs/RUN_latency.csv`, bật bằng `LAT_LOG`). Đây là số đo end-to-end chính xác, dùng cho Table II. `pipeline.client_latency` gộp vào CSV thành các cột `client_p95`, `client_slow<ms>`, `client_n`, `client_cpu`.
- **Sidecar Istio** (Prometheus). `deploy/02_boutique.sh` đặt biên bucket tại 100/150/200/250/300 ms (`LAT_BUCKETS`). Nhờ đó `p95 > SLO` không còn bị nội suy trong bucket (100, 250] mặc định. Kiểm tra bằng `deploy/check_latency_buckets.sh 0.2` sau khi đã có traffic.

`05_online_run.sh` và `03_load_and_collect.sh` đã tự làm cả hai việc. Với các lần chạy cũ, chỉ lấy được giá trị xấp xỉ từ Locust history:
```bash
python -m pipeline.client_latency --csv data/raw/RUN.csv --history Logs/RUN_locust_stats_history.csv
python -m pipeline.compare_runs --lat_source client ...   # cột lat_source: client | client~ | istio | istio~
```
Dấu `~` nghĩa là số xấp xỉ. Không trộn nhiều nguồn trong cùng một bảng.

**μ theo SLA.** μ ở bước 3b là tải tại đó CPU của pod đạt 50 % (điểm làm việc của HPA), thận trọng khoảng 2-3 lần so với SLA. Đo lại theo SLA, mỗi lần một service, mất khoảng 1,5-2 giờ:
```bash
deploy/07_profile_sla.sh            # -> reports/mu_sla.json, rồi dùng MU=reports/mu_sla.json / --mu reports/mu_sla.json
```
Service nào có `censored: true` thì μ của nó chỉ là cận dưới, do tải chưa đủ lớn: tăng `HI` hoặc giảm `OTHER` rồi chạy lại cho service đó (`TARGETS="..."`).

## 4. Chạy DeepScaler

Đưa tất cả service về 1 pod để lần chạy nào cũng bắt đầu từ cùng trạng thái:

```bash
kubectl -n boutique delete hpa --all
kubectl -n boutique scale deploy --all --replicas=1
kubectl -n boutique rollout status deploy --timeout=300s
```

**Bước 4a. Chạy thử không scale (`--dry_run`), khoảng 2 phút.** Mục đích là kiểm tra model nạp được và đọc được Prometheus. Cần có tải để metric có giá trị, nên mở cửa sổ tmux thứ hai (`Ctrl-b c`) và phát tải ngắn:

```bash
cd ~/DeepScaler && RUN_TIME=180 locust -f pipeline/locustfile.py --headless --host http://localhost:31000
```

Quay lại cửa sổ đầu (`Ctrl-b 0`):

```bash
cd ~/DeepScaler
DRY_RUN=1 MODEL_DIR=model_states/ds_s2021 deploy/04_run_deepscaler.sh
```

- Lần đầu chạy script sẽ cài PyTorch bản CPU.
- Mỗi 30 s sẽ in ra một dòng dạng `HH:MM:SS {'ad': 1, 'cart': 1, ..., 'frontend': 2, ...}`.
- Chờ khoảng 10 phút (20 bước × 30 s) thì cửa sổ dữ liệu mới đầy. Trước đó dự đoán chưa có ý nghĩa.
- Thấy chạy ổn thì nhấn `Ctrl-C`.

**Bước 4b. Chạy thật.** Đặt tên cho lần chạy, rồi mở 2 cửa sổ tmux.

```bash
cd ~/DeepScaler
echo online_deepscaler_$(date +%Y%m%d_%H%M) > Logs/CURRENT_RUN    # tên lần chạy, 2 cửa sổ dùng chung
kubectl -n boutique scale deploy --all --replicas=1
```

Cửa sổ 1, autoscaler (mỗi lần chạy ghi ra file log riêng):

```bash
cd ~/DeepScaler && RUN=$(cat Logs/CURRENT_RUN)
LOG=Logs/${RUN}_autoscaler.jsonl MODEL_DIR=model_states/ds_s2021 deploy/04_run_deepscaler.sh \
    2>&1 | tee Logs/${RUN}_console.log
```

Cửa sổ 2, phát tải: một vòng trace FIFA là 903 × 5 s, khoảng **75 phút**.

```bash
cd ~/DeepScaler && RUN=$(cat Logs/CURRENT_RUN)
echo "START $(date '+%Y-%m-%d %H:%M:%S')" | tee Logs/${RUN}_time.txt
RUN_TIME=4500 locust -f pipeline/locustfile.py --headless --host http://localhost:31000 \
    --csv Logs/${RUN}_locust > Logs/${RUN}_locust.log 2>&1
echo "END   $(date '+%Y-%m-%d %H:%M:%S')" | tee -a Logs/${RUN}_time.txt
```

Khi Locust dừng, nhấn `Ctrl-C` ở cửa sổ autoscaler. Mỗi quyết định scale (dự đoán thô `pred`, số pod chọn `pods`, số pod hiện tại `current`) đã được ghi vào `Logs/<RUN>_autoscaler.jsonl`.

## 4c. Chạy với model đã tune trên dữ liệu có burst (`ds_burst_tuned`, 04/11/2026)

Model `model_states/ds_burst_tuned` được train trên `train_burst_20261104_0751`. Tham số được chọn theo tập validation (`experiments/RESULTS_free_tune_20261104.md`): cửa sổ 20 bước, 3 block, 50 epoch × 6 vòng, batch 32, ε 0,1, δ 0,2.

Cách chạy giống bước 4b, chỉ đổi tên lần chạy và `MODEL_DIR`. Mỗi cặp DeepScaler + HPA chạy cùng trace FIFA 75 phút, cùng bắt đầu từ 1 pod mỗi service.

| Lần chạy | Tên `RUN` | Autoscaler |
|---|---|---|
| DeepScaler đã tune | `online_dstuned_<ngày giờ>` | `MODEL_DIR=model_states/ds_burst_tuned deploy/04_run_deepscaler.sh` |
| HPA đối chứng | `online_hpa_<ngày giờ>` | `deploy/02_boutique.sh` (tạo lại HPA), không chạy autoscaler |

Nên chạy 2–3 cặp, xen kẽ thứ tự. So sánh theo Table II và Fig. 10:

```bash
python -m pipeline.compare_runs --slo 0.2 --warmup_min 10 --out reports/table2_dstuned \
    DeepScaler=data/raw/online_dstuned_<1>.csv HPA=data/raw/online_hpa_<1>.csv ...
```

## 4d. Chạy một lần bằng script: `deploy/05_online_run.sh` (khuyên dùng cho Table II)

Script gộp các bước 4b/4c/5/6 cho **một** lần chạy, với bất kỳ autoscaler nào:
1. Dựng lại cluster sạch (`FRESH=1`, mặc định).
2. Đưa mọi service về 1 pod, chờ 2 phút.
3. Bật autoscaler chạy nền.
4. Chạy Locust 75 phút.
5. Tắt autoscaler, rồi `collect` metric và call graph.

```bash
tmux new -s ds                     # hoặc tmux attach -t ds
cd ~/DeepScaler && git pull
deploy/05_online_run.sh slo        # hoặc: mmn | hpa | ds
```

- Lệnh tự dừng sau khoảng 1 giờ 40 phút và in ra tên file CSV cùng số dòng (khoảng 151).
- Trong lúc chạy có thể `Ctrl-b d` rồi tắt SSH.
- Theo dõi bằng `tail -f Logs/$(cat Logs/CURRENT_RUN)_console.log`. Riêng HPA không có log autoscaler, nên xem bằng `kubectl -n boutique get hpa`.
- Chạy liên tiếp nhiều lần: `deploy/05_online_run.sh slo && deploy/05_online_run.sh mmn`. Đợt sau đổi thứ tự.

Chi tiết từng baseline (SLO [48], MMN [33], AWS [47]) và lệnh so sánh: `guide/EXPERIMENTS.md`, mục D.

## 5. Xuất metric và đo độ bất định của lần chạy

Dùng đúng `START` và `END` trong `Logs/<RUN>_time.txt`. Ngoài 5 feature, `collect.py` còn thu thêm:
- `p95`: latency p95 từng service. `frontend_p95` là latency end-to-end mà Locust thấy.
- `err`: tỉ lệ lỗi 5xx.
- `cpupod`: CPU cả pod, gồm sidecar.

```bash
cd ~/DeepScaler && RUN=$(cat Logs/CURRENT_RUN)
python -m pipeline.collect --prom http://localhost:30090 --namespace boutique \
    --start "<START>" --end "<END>" --step 30 \
    --out data/raw/${RUN}.csv --adj_out data/raw/${RUN}_od.npy

python -m pipeline.uncertainty_report --csv data/raw/${RUN}.csv \
    --decisions Logs/${RUN}_autoscaler.jsonl --mu reports/mu.json --slo 0.2 \
    --label ${RUN} --out_dir reports/${RUN}
```

`reports/<RUN>/` gồm:
- `summary.json`, `per_service.csv`
- Hai biểu đồ: `E1_E2_residuals.png` (sai số và các bước thiếu pod của từng service) và `E4_E5_timeline.png` (tải, số service thiếu pod, latency so với SLO)

Ý nghĩa các chỉ số:
- **E0:** tỉ lệ quyết định giống hệt số pod hiện tại.
- **E1:** tỉ lệ thiếu pod của từng service.
- **E2:** biên cần có để phủ 90–95 % nhu cầu, và biên đó thay đổi theo thời gian ra sao.
- **E4:** xác suất vi phạm SLO khi có hoặc không có service thiếu pod.
- **E5:** tỉ lệ thiếu pod lúc tải tăng nhanh so với lúc bình thường.

Thêm `--compare reports/offline_ds_s2021/summary.json` để so offline với online (E3).

## 6. Lần chạy đối chứng HPA (để so sánh, cùng tải)

```bash
kubectl -n boutique scale deploy --all --replicas=1
deploy/02_boutique.sh            # tạo lại HPA (CPU 50 %, 1–10 pod)
RUN=online_hpa_$(date +%Y%m%d_%H%M)
echo "START $(date '+%Y-%m-%d %H:%M:%S')" | tee Logs/${RUN}_time.txt
RUN_TIME=4500 locust -f pipeline/locustfile.py --headless --host http://localhost:31000 \
    --csv Logs/${RUN}_locust > Logs/${RUN}_locust.log 2>&1
echo "END   $(date '+%Y-%m-%d %H:%M:%S')" | tee -a Logs/${RUN}_time.txt
# xuất metric như bước 5, rồi đo bằng cùng nhãn nhu cầu (HPA không có dự đoán -> không cần --decisions):
python -m pipeline.uncertainty_report --csv data/raw/${RUN}.csv --mu reports/mu.json --slo 0.2 \
    --label ${RUN} --out_dir reports/${RUN}
```

Table II của báo có độ lệch chuẩn (±), nên mỗi autoscaler nên chạy **2–3 lần**, xen kẽ HPA và DeepScaler.

## 6b. Kịch bản burst

Kịch bản này giữ nguyên trace FIFA lúc train, nhưng thêm 3 đợt tải ×2. Mỗi đợt dài 5 phút và bắt đầu đột ngột ở phút 15, 40 và 60. Báo ghi nhận burst làm tăng vi phạm SLA ở mọi autoscaler dựa trên dự đoán (Sec. V-B, Fig. 10, phút 30–50). Ở đây ta đo xem DeepScaler thiếu pod đến mức nào trong các đợt đó.

```bash
python -m pipeline.make_traces burst --out sendFlow/fifa-burst.req
```

Lặp lại bước 4b (DeepScaler) và bước 6 (HPA), đổi tên lần chạy thành `online_ds_burst_...` / `online_hpa_burst_...`, và thêm `TRACE=sendFlow/fifa-burst.req` vào trước lệnh `locust`. Sau đó xuất metric và chạy `uncertainty_report` như bước 5.

## 7. Theo dõi từ máy của bạn (tuỳ chọn)

```bash
ssh -i <KEY.pem> -L 30090:localhost:30090 -L 30300:localhost:30300 ubuntu@<EC2_IP>
```

Sau đó mở trên trình duyệt:
- Prometheus: http://localhost:30090
- Grafana: http://localhost:30300

## 8. Lấy kết quả về rồi tắt EC2

Chạy trên máy của bạn:

```bash
cd ~/Projects/DeepScaler
scp -i <KEY.pem> "ubuntu@<EC2_IP>:~/DeepScaler/data/raw/{online_*,profile*}" data/raw/
scp -i <KEY.pem> "ubuntu@<EC2_IP>:~/DeepScaler/Logs/{online_*,profile*}" Logs/
scp -i <KEY.pem> -r "ubuntu@<EC2_IP>:~/DeepScaler/reports" .
```

Dữ liệu Prometheus nằm **trong cluster**. Phải chạy `collect.py` (bước 5) cho **mọi** lần chạy và kiểm tra file đã về đủ, rồi mới **Stop/Terminate** instance trên AWS Console. Tắt máy là mất hết, không thu lại được. EC2 tính tiền theo giờ chạy.

---

## Lỗi hay gặp

| Triệu chứng | Cách xử lý |
|---|---|
| `permission denied ... docker.sock` | Chưa logout/login sau `ec2_bootstrap.sh` |
| `Permission denied` khi chạy `deploy/*.sh` | `chmod +x deploy/*.sh` |
| `KeyError` / NaN ở autoscaler lúc đầu | Chưa có tải hoặc chưa đủ 20 bước dữ liệu: phát tải và chờ khoảng 10 phút |
| `load_state_dict` báo sai kích thước | `MODEL_DIR` sai, hoặc model không phải bản 5 feature / cửa sổ 20 (kiểm tra `data/c20/meta.json`) |
| HPA và DeepScaler cùng đổi số pod | Quên `kubectl -n boutique delete hpa --all` |
| Mất SSH giữa chừng | `ssh` lại rồi `tmux attach -t ds` |

## Ghi chú cho luận văn

- **Model đầu vào:** `ds_s2021` được train trên dữ liệu thu khi HPA điều khiển cluster. Đầu vào là 5 feature `pod, cpu, res, req, mem`, cửa sổ 20 bước × 30 s, dự đoán số pod ở bước kế tiếp (nhãn theo Sec. III của báo).
- **Phần tự thiết kế, báo không mô tả:**
  - Luật làm tròn ra số pod: lấy cận trên, trừ khi phần lẻ < 0,35, theo `predict_scale.py` của tác giả.
  - Giới hạn 1–10 pod.
- **Chưa cài:** cơ chế *trust threshold* (Sec. IV-D) và việc học lại định kỳ (Algorithm 1, bước 5).
- **Chi phí suy luận:** autoscaler, Locust và Prometheus chạy chung máy với cluster, nên có thêm một ít overhead. Báo đo khoảng 0,1 core cho mỗi lần dự đoán (Table III).
