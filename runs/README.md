# runs/: mỗi lần chạy một thư mục

```
runs/
  INDEX.md, INDEX.csv     danh mục tự sinh: python -m pipeline.index_runs
  notes.csv               ghi tay: run, status (ref | excluded | failed | new), used_in, note
  v1_istio-p95/<RUN>/     chạy trước 10/11/2026: độ trễ p95 nội suy trong bucket Istio (100, 250] ms -> chỉ tham khảo
  v2_exact/<RUN>/         chạy từ 10/11/2026: bucket Istio tại SLO + log độ trễ từng request (Locust)
```
Trong mỗi `<RUN>/`, tên file giữ nguyên như cũ: `<RUN>.csv` (metrics + cột client_*), `<RUN>_od.npy`, `<RUN>_npas.npy`, `<RUN>_latency.csv`(+`.cpu.csv`), `<RUN>_locust*`, `<RUN>_autoscaler.jsonl`, `<RUN>_console.log`, `<RUN>_time.txt`.

Tên `<RUN>`: `online_<policy>_<yyyymmdd_HHMM>` (policy: hpa | ds | slo | mmn | aci; cũ: deepscaler, dstuned) hoặc `profile_sla_<service>_<stamp>`.

Ở chỗ khác: dữ liệu train/profiling CPU (không phụ thuộc cách đo SLA) ở `data/raw/` + log ở `Logs/`; kết quả tổng hợp ở `reports/<campaign>/`; model ở `model_states/`.
