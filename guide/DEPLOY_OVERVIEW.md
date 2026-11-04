# Triển khai DeepScaler (Online Boutique) — từ cluster đến model

Quy trình đầy đủ gồm 2 nửa, chạy ở 2 nơi khác nhau:

| Bước | Chạy ở đâu | Lệnh |
|---|---|---|
| 1. Dựng Kubernetes | Máy/cluster Linux có Docker | `deploy/00_kind_cluster.sh` (1 máy) hoặc `setUp-k8s.md` (kubeadm nhiều máy) |
| 2. Istio + Prometheus + metrics-server | Cluster | `deploy/01_istio_prometheus.sh` |
| 3. Online Boutique + HPA | Cluster | `deploy/02_boutique.sh` |
| 4. Bơm tải FIFA + thu dữ liệu | Máy có kubectl/truy cập NodePort | `HOURS=6 deploy/03_load_and_collect.sh` |
| 5. Train | **Colab / Kaggle** (GPU) | `notebooks/DeepScaler_train_colab_kaggle.ipynb`, `DATA_SOURCE="cluster"` |
| 6. Autoscale bằng DeepScaler | Cluster | `deploy/04_run_deepscaler.sh` |

Colab/Kaggle **không** chạy được Kubernetes/Istio (không có Docker, không có quyền privileged), nên bước 1–4 và 6 phải chạy trên máy thật hoặc VM cloud. Chỉ bước 5 chạy trên Colab/Kaggle.

## Cấu hình tối thiểu
- Thử trên 1 máy bằng kind: ≥ 8 vCPU, 16 GB RAM (Boutique ~11 pod + sidecar Istio + Prometheus, HPA scale tới 10 pod/service). Có thể giảm `--max` trong `02_boutique.sh` nếu máy yếu.
- Bài báo gốc: 8 VM (4×12 core/24 GB, 4×24 core/32 GB).

## Dữ liệu
- `03_load_and_collect.sh` tạo `data/raw/<run>.csv` (cột `time, <svc>_<pod|cpu|res|req>`) và `data/raw/<run>_od.npy` (call graph Istio quan sát được).
- Cùng PromQL với `metrics_fetch.py` gốc; mặc định lấy mẫu 30 s. Một vòng trace FIFA = 903 × 5 s ≈ 75 phút; thu ≥ 6 giờ.
- Lúc thu dữ liệu, HPA (CPU 50 %, 1–10 pod) quyết định số pod → đó là nhãn mà DeepScaler học (dự đoán số pod bước kế tiếp).
- Image trong `benchmarks/boutique/boutique.yaml` là `mengchunyang/*:v0.3.6` (Docker Hub). Nếu bị xoá, thay bằng `gcr.io/google-samples/microservices-demo/<service>:v0.3.6`.

## Không có cluster?
`python -m pipeline.simulate` sinh dữ liệu **mô phỏng** cùng định dạng để chạy thử toàn bộ pipeline train (notebook, `DATA_SOURCE="simulated"`). Kết quả trên dữ liệu mô phỏng chỉ để kiểm tra pipeline, không phải kết quả thực nghiệm.

## Kiểm tra nhanh
```bash
curl 'http://localhost:30090/api/v1/query?query=sum(rate(istio_requests_total{destination_workload_namespace="boutique"}[1m]))'
curl 'http://localhost:30090/api/v1/query?query=count(container_spec_cpu_period{namespace="boutique",container="frontend"})'
```
Cả hai phải trả về giá trị khác rỗng trước khi thu dữ liệu.
