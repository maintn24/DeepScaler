# Hướng dẫn

| File | Nội dung |
|---|---|
| [`COLLECT_DATA.md`](COLLECT_DATA.md) | Thu dữ liệu train Online Boutique trên EC2 (trace có burst, vượt SLA): cấu hình máy, lệnh, kiểm tra, đưa file về |
| [`RUN_ONLINE.md`](RUN_ONLINE.md) | Chạy DeepScaler online thay HPA trên EC2; đo μ (profiling); so với HPA |
| [`EXPERIMENTS.md`](EXPERIMENTS.md) | Các kịch bản tái lập trong báo DeepScaler (Fig. 11, Fig. 12, Table II, burst, Table III), cho Boutique, BookInfo, Train-Ticket |
| `notebooks/DeepScaler_tune_burst_colab.ipynb` | Colab: Fig. 12 với giá trị của code, tune tự do (chọn theo validation), baseline cùng cửa sổ, đóng gói model. Mở: `https://colab.research.google.com/github/maintn24/DeepScaler/blob/deepscaler-pipeline/notebooks/DeepScaler_tune_burst_colab.ipynb` |
| [`DEPLOY_OVERVIEW.md`](DEPLOY_OVERVIEW.md) | Tổng quan pipeline: cluster → thu dữ liệu → train trên Colab → autoscale |

Kết quả tune trên Colab: `experiments/RESULTS_tune_grid_20261102.md`. Hướng dẫn dựng Kubernetes nhiều máy của tác giả: `setUp-k8s.md` (ở thư mục gốc).
