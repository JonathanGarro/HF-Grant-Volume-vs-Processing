# Grant Processing Time vs. Volume 2024-2026

Analysis of grant approval queue timing and volume across 2024, 2025, and 2026 (partial). Compares processing speed and throughput for the PO, PD, GO, Legal, CO, and Prez review queues, both foundation-wide and by program.

## Data inputs

| File | Description |
|------|-------------|
| `approvals_00OUf000004eCTBMA2.csv` | Approval step records (step name, start/completed dates) |
| `requests_00OUf000005GbLiMAK.csv` | Grant request records (stage, program, staff assignments) |

Only grants in Active, Closed, or Awarded stages are included. Business days are calculated excluding weekends and US federal holidays.

## Outputs

All outputs are written to the `outputs/` directory.

### Foundation-wide charts

| File | Description |
|------|-------------|
| `queue_stacked_volume.png` | Stacked bar chart — unique grant approval actions per queue per quarter |
| `option_a_lines_per_queue.png` | Volume bars + individual median-days line per queue (dual y-axis) |
| `option_b_weighted_avg.png` | Volume bars + single volume-weighted average days line |
| `option_c_heatmap.png` | Volume bars (left panel) + median days heatmap by queue (right panel) |
| `volume_vs_avgdays_grouped.png` | Grouped bars — volume (solid) and average days (hatched) side by side per quarter |
| `volume_days_with_table.png` | Grouped bars + summary table (average days) |
| `volume_days_with_table_median.png` | Grouped bars + summary table (median days) |
| `foundation_line_trends_median.png` | Line chart — approval step actions, unique grants, and median processing days |
| `foundation_line_trends_mean.png` | Line chart — approval step actions, unique grants, and mean processing days |
| `queue_quarterly_detail.csv` | Quarterly detail table: volume, median, mean, P75, P90 days per queue |

### Per-program charts

Generated for each active program in `outputs/programs/`. Two versions per program (mean and median):

```
outputs/programs/<Program_Name>_mean.png
outputs/programs/<Program_Name>_median.png
```

Programs covered: Education, Environment, Gender Equity & Governance, Performing Arts, U.S. Democracy, Philanthropy, Racial Justice, Economy and Society, Special Projects, SBAC.

## Setup

```bash
uv sync
```

## Usage

```bash
uv run python approvals_analysis.py
```

## Notes

- 2026 data is a partial year (through Q1/early Q2).
- The PA queue is tracked in the requests data and is not included in the approval step analysis.
- Business day calculations use the `holidays` library with US federal holidays for 2021–2027.
