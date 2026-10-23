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
kubectl -n boutique scale deploy --all --replicas=2
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
