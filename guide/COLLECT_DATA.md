# Thu dữ liệu train Online Boutique trên EC2 (có burst, vượt SLA)

Hướng dẫn thu dữ liệu train cho DeepScaler, chạy trên một máy EC2, từ lúc tạo máy đến lúc đưa file về.

Trong lúc thu, **HPA** (CPU 50 %, 1–10 pod) quyết định số pod. Đó là nhãn mà DeepScaler học: số pod ở bước kế tiếp (báo DeepScaler, Sec. III). Chu kỳ lấy mẫu là 30 s, đúng bằng chu kỳ dự đoán trong báo (Sec. V-B).

**Vì sao phải thu lại.** Lần thu cũ (`run2_hpa`, 5,1 giờ) có hai vấn đề:
- Không có bước nào vượt SLA 200 ms: p95 tối đa 117 ms.
- Số pod gần như không đổi: dự đoán "giữ nguyên số pod" chỉ sai 0,039 pod.

Với dữ liệu như vậy, DeepScaler không thể hơn DCRNN, T-GCN hay cách giữ nguyên số pod. Báo cũng nêu DeepScaler bị giới hạn bởi *"the quality and quantity of available historical data"* (Sec. VI).

Trace mới `sendFlow/train-burst-6h.req` (6 giờ) được sinh bằng `pipeline/make_traces.py train`, seed 2026:
- Trace FIFA của tác giả (`sendFlow/random-100max.req`) được lặp lại. Mỗi vòng 75 phút nhân với một mức tải 1,0 / 1,5 / 0,6 / 1,8 / 1,2, để HPA dùng nhiều số pod khác nhau và có cả lúc scale in.
- Có 15 burst đột ngột: mỗi burst dài 1–4 phút, tải nhân ×1,8–3,0, tối đa 220 user. HPA không kịp phản ứng nên sẽ có vi phạm SLA, giống hiện tượng báo mô tả ở Fig. 10 (phút 30–50, Sec. V-B).
- Thời điểm các burst được ghi trong `sendFlow/train-burst-6h.req.bursts.csv`.

Thay các chỗ `<...>` bằng giá trị của bạn: `<KEY.pem>` là file key của EC2, `<EC2_IP>` là Public IPv4 của máy.

---

## 1. Tạo máy EC2

Phải **giống lúc thu dữ liệu trước và lúc chạy online**, vì model và μ phụ thuộc vào phần cứng:

| Mục | Giá trị |
|---|---|
| Loại máy | **c6i.2xlarge** (8 vCPU, 16 GB) |
| Hệ điều hành | Ubuntu 22.04 hoặc 24.04 |
| Ổ đĩa | gp3 50 GB |
| Security group, inbound | chỉ **SSH, cổng 22, Source = My IP** |
| Thời gian | khoảng 7,5 giờ (dựng 30 phút, chạy thử 20 phút, thu 6 giờ, collect) |

Nếu IP của bạn đổi giữa chừng (đổi wifi, router cấp IP mới), SSH sẽ báo `Connection timed out`. Khi đó vào Security group, sửa rule, chọn lại **My IP** rồi lưu. Locust vẫn chạy bình thường trong tmux.

Không cần mở thêm cổng nào khác: Locust, Prometheus và `collect` đều chạy trên EC2 và gọi `localhost`. Muốn xem Prometheus hoặc Grafana từ máy bạn thì dùng SSH tunnel:

```bash
ssh -i <KEY.pem> -L 30090:localhost:30090 -L 30300:localhost:30300 ubuntu@<EC2_IP>
```

## 2. Trên máy của bạn: push code, gửi μ

EC2 lấy code bằng `git pull` từ branch `deepscaler-pipeline`, nên phải push trước.

```bash
cd ~/Projects/DeepScaler
git status                       # push hết các thay đổi cần dùng (pipeline/, deploy/, sendFlow/train-burst-6h.req*)
git push
chmod 400 <KEY.pem>
```

## 3. Cài môi trường (mỗi EC2 mới làm 1 lần)

```bash
ssh -i <KEY.pem> ubuntu@<EC2_IP>
git clone -b deepscaler-pipeline https://github.com/maintn24/DeepScaler.git
cd DeepScaler && bash deploy/ec2_bootstrap.sh      # docker, venv ~/dsenv (locust, pandas, ...); kind/kubectl do 00_kind_cluster.sh cài
exit                                               # logout để nhóm docker có hiệu lực
```

Gửi μ đã đo lên EC2, chạy trên máy bạn (dùng đường dẫn tuyệt đối nếu đang ở thư mục khác):

```bash
ssh -i <KEY.pem> ubuntu@<EC2_IP> "mkdir -p ~/DeepScaler/reports"
scp -i <KEY.pem> ~/Projects/DeepScaler/reports/mu.json ubuntu@<EC2_IP>:~/DeepScaler/reports/
```

`mu.json` chỉ dùng để `check_data` kiểm tra thêm nhãn nhu cầu `max(1, ⌈req/μ⌉)`. Không có file này thì việc thu vẫn chạy bình thường.

## 4. Dựng cluster + Istio + Prometheus + Boutique + HPA

```bash
ssh -i <KEY.pem> ubuntu@<EC2_IP>
tmux new -s ds                   # mọi lệnh dài chạy trong tmux
cd ~/DeepScaler && git pull
deploy/00_kind_cluster.sh
deploy/01_istio_prometheus.sh
deploy/02_boutique.sh            # HPA CPU 50 %, 1..10 pod, frontend đưa về 1 pod

# Kiểm tra
curl -s 'http://localhost:30090/api/v1/query?query=up' | head -c 200; echo
kubectl -n boutique get pods,hpa # mọi pod 2/2 Running, đủ 10 HPA
```

Cách dùng tmux:
- Thoát mà vẫn để lệnh chạy: `Ctrl-b d`.
- Vào lại: `tmux attach -t ds`.
- Mở cửa sổ mới: `Ctrl-b c`. Chuyển cửa sổ: `Ctrl-b 0`, `Ctrl-b 1`.

## 5. Chạy thử 20 phút (burst đầu tiên ở phút 13)

Bước này kiểm tra trace có thực sự làm vượt SLA trước khi tốn 6 giờ.

```bash
cd ~/DeepScaler
RUN=smoke_$(date +%Y%m%d_%H%M); echo "START $(date '+%Y-%m-%d %H:%M:%S')" | tee Logs/${RUN}_time.txt
TRACE=sendFlow/train-burst-6h.req RUN_TIME=1200 locust -f pipeline/locustfile.py --headless \
    --host http://localhost:31000 --csv Logs/${RUN}_locust > Logs/${RUN}_locust.log 2>&1
python -m pipeline.collect --last 20 --step 30 --out data/raw/${RUN}.csv
python -m pipeline.check_data data/raw/${RUN}.csv --bursts sendFlow/train-burst-6h.req.bursts.csv \
    --start "$(sed -n 's/^START *//p' Logs/${RUN}_time.txt)"
```

Lệnh `locust` tự dừng sau 20 phút. Kết quả đạt khi dòng `inside bursts` có vi phạm và có dòng `bursts that caused >= 1 violation: 1/1`.

Chưa đạt thì sinh lại trace, rồi chạy thử lại. Ghi giá trị cuối cùng vào luận văn.
- **Không vượt SLA:** `python -m pipeline.make_traces train --out sendFlow/train-burst-6h.req --cap 260 --factor_rng 2.5,4`
- **Vượt SLA cả ngoài burst** (máy hoặc HPA bão hoà): `python -m pipeline.make_traces train --out sendFlow/train-burst-6h.req --cap 200 --scales 0.8,1.2,0.5,1.5,1.0`

Sau khi chạy thử, đưa cluster về trạng thái ban đầu:

```bash
kubectl -n boutique scale deploy --all --replicas=1
sleep 300                        # chờ HPA ổn định
```

## 6. Thu thật 6 giờ

```bash
cd ~/DeepScaler
RUN=train_burst_$(date +%Y%m%d_%H%M); echo "START $(date '+%Y-%m-%d %H:%M:%S')" | tee Logs/${RUN}_time.txt
TRACE=sendFlow/train-burst-6h.req RUN_TIME=21600 locust -f pipeline/locustfile.py --headless \
    --host http://localhost:31000 --csv Logs/${RUN}_locust > Logs/${RUN}_locust.log 2>&1
echo "END   $(date '+%Y-%m-%d %H:%M:%S')" | tee -a Logs/${RUN}_time.txt
```

Lệnh tự dừng sau 6 giờ, không cần Ctrl-C. Trong lúc chờ có thể `Ctrl-b d` và tắt SSH.

Muốn xem tiến độ, mở cửa sổ tmux khác:

```bash
tail -3 ~/DeepScaler/Logs/train_burst_*_locust.log
kubectl -n boutique get hpa
```

## 7. Collect và kiểm tra

```bash
cd ~/DeepScaler
# nếu mở cửa sổ/phiên mới, biến RUN đã mất -> đặt lại:
RUN=$(ls Logs/train_burst_*_time.txt | tail -1 | xargs basename | sed 's/_time.txt//')
START=$(sed -n 's/^START *//p' Logs/${RUN}_time.txt); END=$(sed -n 's/^END *//p' Logs/${RUN}_time.txt)
python -m pipeline.collect --start "$START" --end "$END" --step 30 \
    --out data/raw/${RUN}.csv --adj_out data/raw/${RUN}_od.npy
python -m pipeline.check_data data/raw/${RUN}.csv --bursts sendFlow/train-burst-6h.req.bursts.csv --start "$START"
ls -lh data/raw/${RUN}*          # phải có .csv, _od.npy, _npas.npy
```

Dữ liệu đạt khi:

| Chỉ số (`check_data`) | Mong muốn | `run2_hpa` cũ |
|---|---|---|
| Tỉ lệ vượt SLA 200 ms | khoảng 5–20 % | 0 % |
| Vượt SLA ngoài burst | < 5 % | – |
| Số burst gây vi phạm | ≥ 2/3 | – |
| MAE của cách giữ nguyên số pod | ≥ 0,1 pod | 0,039 |
| CPU app tối đa | < khoảng 5 core | 1,98 |

## 8. Đưa file về máy rồi mới tắt EC2

Chạy trên máy bạn:

```bash
cd ~/Projects/DeepScaler
scp -i <KEY.pem> "ubuntu@<EC2_IP>:~/DeepScaler/data/raw/train_burst_*" data/raw/
scp -i <KEY.pem> "ubuntu@<EC2_IP>:~/DeepScaler/data/raw/smoke_*" data/raw/
scp -i <KEY.pem> "ubuntu@<EC2_IP>:~/DeepScaler/Logs/train_burst_*" Logs/
ls -lh data/raw/train_burst_*
```

Dữ liệu Prometheus nằm **trong cluster**. Phải kiểm tra file đã về đủ rồi mới Stop hoặc Terminate instance. Tắt máy là mất, không thu lại được.

Muốn giữ cluster để chạy online tiếp (`RUN_ONLINE.md`), chọn **Stop**, đừng chọn Terminate. Khi bật lại, IP công khai sẽ đổi và cluster kind phải dựng lại từ bước 4.

## 9. Bước tiếp theo

Train lại trên Colab bằng `notebooks/DeepScaler_train_colab_kaggle.ipynb`:
- Upload `data/raw/${RUN}.csv` và `_od.npy`, `_npas.npy`.
- Đặt `DS_APP="boutique"`.

So DeepScaler với DCRNN, T-GCN và cách giữ nguyên số pod (Fig. 12; xem `EXPERIMENTS.md`, kịch bản C).

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| `scp: stat local "...": No such file` | Bạn đang ở thư mục khác: dùng đường dẫn tuyệt đối `~/Projects/DeepScaler/...` |
| SSH `Connection timed out` | IP của bạn đổi: sửa inbound rule của Security group, chọn lại My IP |
| `permission denied ... docker.sock` | Chưa logout sau `ec2_bootstrap.sh`: `exit` rồi SSH lại |
| `locust: command not found` | venv chưa bật: `source ~/dsenv/bin/activate` |
| `collect` báo cột rỗng hoặc NaN | Prometheus chưa có số liệu: chạy `curl` ở bước 4 và kiểm tra pod `2/2` |
| Locust báo `CPU usage above 90%` | Máy quá tải: giảm `--cap` của trace |
