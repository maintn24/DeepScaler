# Danh mục các lần chạy (tự sinh bởi `python -m pipeline.index_runs`, đừng sửa tay)

Sửa trạng thái/ghi chú trong `runs/notes.csv` rồi chạy lại lệnh trên.

`latency`: client = log từng request (chính xác); istio = p95 sidecar có biên bucket tại 200 ms; `~` = xấp xỉ (nội suy / Locust history). Campaign `v1_istio-p95` chỉ để tham khảo.

## v1_istio-p95

| run | policy | start | rows | latency | status | used_in | note |
|---|---|---|---|---|---|---|---|
| online_deepscaler_20261023_0149 | deepscaler | 2026-10-23 01:50:29 | 152 | istio~ | ref | reports/v1_istio-p95/table2_20261023 | Table II đầu tiên; model ds_s2021 (trước khi tune) |
| online_dstuned_20261104_1637 | dstuned | 2026-11-04 16:41:46 | 151 | istio~ | ref | reports/v1_istio-p95/table2_full; table2_dstuned; load_analysis | DS tuned (ds_burst_tuned) cặp 1 |
| online_dstuned_20261104_2110 | dstuned | 2026-11-04 21:12:37 | 151 | istio~ | excluded | reports/v1_istio-p95/table2_dstuned | lần chạy bị lỗi; loại khỏi load_analysis và table2_full |
| online_dstuned_20261108_1704 | dstuned | 2026-11-08 17:08:02 | 151 | istio~ | ref | reports/v1_istio-p95/table2_full | DS tuned lần 2 (FRESH) |
| online_hpa_20261023_0506 | hpa | 2026-10-23 05:06:17 | 151 | istio~ | ref | reports/v1_istio-p95/table2_20261023 | HPA đối chứng của table2_20261023 |
| online_hpa_20261104_1802 | hpa | 2026-11-04 18:02:11 | 151 | istio~ | ref | reports/v1_istio-p95/table2_full; table2_dstuned; load_analysis | HPA cặp 1 |
| online_hpa_20261104_1943 | hpa | 2026-11-04 19:43:39 | 151 | istio~ | ref | reports/v1_istio-p95/table2_dstuned | HPA cặp 2 |
| online_hpa_20261108_1548 | hpa | 2026-11-08 15:48:14 | 151 | istio~ | ref | reports/v1_istio-p95/table2_full | HPA lần 2 (FRESH) |
| online_mmn_20261108_1900 | mmn | 2026-11-08 19:01:03 |  |  | failed |  | dừng sau ~5 phút; không có CSV |
| online_mmn_20261108_2037 | mmn | 2026-11-08 20:37:56 | 151 | istio~ | ref | reports/v1_istio-p95/table2_full | MMN lần 1 |
| online_mmn_20261108_2226 | mmn | 2026-11-08 22:26:44 | 151 | istio~ | ref | reports/v1_istio-p95/table2_full | MMN lần 2 |
| online_slo_20261108_1851 | slo | 2026-11-08 18:51:32 |  |  | failed |  | dừng sau ~5 phút; không có CSV |
| online_slo_20261108_1916 | slo | 2026-11-08 19:16:51 | 151 | istio~ | ref | reports/v1_istio-p95/table2_full | SLO lần 1 |
| online_slo_20261108_2346 | slo | 2026-11-08 23:47:15 | 151 | istio~ | ref | reports/v1_istio-p95/table2_full | SLO lần 2 |

## data/raw (dữ liệu train / profiling, không phụ thuộc cách đo SLA)

- `data/raw/profile.csv`
- `data/raw/run2_hpa.csv`
- `data/raw/run2_hpa_5f.csv`
- `data/raw/run_20261021_1521.csv`
- `data/raw/smoke_20261104_0719.csv`
- `data/raw/train_burst_20261104_0751.csv`
