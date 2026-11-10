# Checklist chạy lại sau khi sửa cách đo (10/11/2026)

Hai thay đổi:
1. **Đo SLA**: bucket Istio đặt tại SLO, cộng log độ trễ từng request của Locust.
2. **μ theo SLA**: thay cho mục tiêu CPU 50 %.

Không phải lần chạy nào cũng phụ thuộc vào hai thay đổi này. Đánh dấu `[x]` khi xong.

## Không cần chạy lại
- `data/raw/train_burst_20261104_0751.csv`, `run2_hpa*`, `run_20261021_1521`: các cột pod/cpu/res/req/mem không phụ thuộc μ hay bucket (`res` là trung bình sum/count, không đọc từ bucket). Nhãn demand được tính lại từ cột `req`.
- Model DeepScaler gốc `model_states/ds_burst_tuned`, cùng các baseline offline DCRNN/T-GCN/LSTM (`experiments/RESULTS_*`): cả hai đều train trên nhãn pod.
- `data/raw/profile.csv` + `reports/mu.json`: μ theo CPU, giữ lại làm cách cũ, và MMN vẫn dùng file này.

## Phải chạy, theo thứ tự (thời gian ước tính trên EC2 c6i.2xlarge)

### A. Chuẩn bị trên EC2
- [ ] `git pull` nhánh `deepscaler-pipeline`, rồi `source ~/dsenv/bin/activate`
- [ ] Dựng cluster có bucket mới: `deploy/00_kind_cluster.sh && deploy/01_istio_prometheus.sh && deploy/02_boutique.sh`
- [ ] Bơm một ít traffic rồi kiểm tra: `deploy/check_latency_buckets.sh 0.2` phải in ra `OK`

### B. Profiling μ theo SLA (~1,5–2 h). Bắt buộc trước bước C và E
- [ ] `deploy/07_profile_sla.sh` → `runs/v2_exact/profile_sla_*` và `reports/mu_sla.json`
- [ ] Xem kết quả in ra:
  - service có `(lower bound)`: tải chưa đủ lớn, chạy lại riêng service đó với `TARGETS=<svc> HI=250` (hoặc `OTHER=2`)
  - service có `mu_sla=nan`: vi phạm SLO ngay từ tải thấp, tức vấn đề p₀; kiểm tra `client_cpu` (Locust có quá tải không)
- [ ] Chép `reports/mu_sla.json` và `runs/v2_exact/` về máy

### C. Model trên nhãn demand (Colab/Kaggle, ~30 phút). Chạy song song với D
- [ ] `python -m pipeline.build_dataset --raw data/raw/train_burst_20261104_0751.csv --split_by target --features demand,cpu,res,req,mem --mu reports/mu_sla.json --out_dir data/boutique_demand`
- [ ] Train với tham số như `ds_burst_tuned` → `model_states/ds_demand`, rồi chép lên EC2
- [ ] Kiểm tra offline: `python -m pipeline.aci_replay --csv data/raw/train_burst_20261104_0751.csv --mu reports/mu_sla.json --forecaster model --model_dir model_states/ds_demand --alpha 0.05`

### D. Baseline online (không cần μ mới; mỗi lần ~1 h 40 vì `FRESH=1` dựng lại cluster)
- [ ] `deploy/run_batch.sh hpa ds hpa ds` (~6,5 h). Chạy `hpa` trước để có REF_CSV cho `slo`/`mmn`

### E. ACI online (cần `ds_demand` từ C và `mu_sla.json` từ B)
- [ ] `ACI_MODEL_DIR=model_states/ds_demand EPS=0.05 deploy/run_batch.sh aci aci` (~3,5 h)
- [ ] Ablation chỉ có ACI, α cố định: `ACI_MODEL_DIR=model_states/ds_demand PACER=off deploy/run_batch.sh aci` (~1 h 40)

### F. Ưu tiên thấp (nếu còn thời gian)
- [ ] `deploy/run_batch.sh slo mmn slo mmn` (~6,5 h). MMN để mặc định `MU=reports/mu.json` để baseline giống trước
- [ ] Các kịch bản lý tưởng / shift (mục tiêu O4): đổi trace qua `TRACE=` của Locust

### G. Sau mỗi đợt
- [ ] Chép về máy: `scp -r ubuntu@<EC2_IP>:~/DeepScaler/runs/v2_exact runs/`
- [ ] `python -m pipeline.index_runs`, rồi ghi status/used_in/note cho từng lần chạy trong `runs/notes.csv`

### H. Tổng hợp
- [ ] Table II mới: `python -m pipeline.compare_runs --lat_source client --slo 0.2 --warmup_min 10 --out reports/v2_exact/table2 HPA=... DeepScaler=... ACI=...`. Cột `lat_source` phải là `client`, không có `~`
- [ ] p₀/p₁ và đường đi của α, β, Q: đọc từ `runs/v2_exact/online_aci_*/..._autoscaler.jsonl`
- [ ] Cập nhật mục "Kết quả sơ bộ" trong `.tex` với `mu_sla.json`, sau đó hỏi Claude cập nhật số liệu

**Tổng thời gian EC2:** khoảng 13–15 h cho A–E, thêm khoảng 6,5 h nếu làm F. Dùng `run_batch.sh` để chạy qua đêm. Chỉ Stop instance sau khi đã chép `runs/v2_exact` về máy.
