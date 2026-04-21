"""
grants processing: queue timing & volume analysis
compares 2024, 2025, and 2026 across review queues
focuses on PA, PO, PD, Legal, GO, and Prez queues
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import os
import holidays


# ── output dir ──────────────────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)

# ── load data ────────────────────────────────────────────────────────────────
approvals = pd.read_csv("approvals_00OUf000004eCTBMA2.csv", encoding="latin1")
requests  = pd.read_csv("requests_00OUf000005GbLiMAK.csv",  encoding="latin1")

# ── parse dates ──────────────────────────────────────────────────────────────
approvals["start_dt"] = pd.to_datetime(
    approvals["Step Start Date"], format="%m/%d/%Y, %I:%M %p", errors="coerce"
)
approvals["completed_dt"] = pd.to_datetime(
    approvals["Step Completed Date"], format="%m/%d/%Y, %I:%M %p", errors="coerce"
)

approvals["year"]    = approvals["start_dt"].dt.year
approvals["quarter"] = approvals["start_dt"].dt.quarter
approvals["yq"]      = approvals["year"].astype(str) + " Q" + approvals["quarter"].astype(str)

# ── business day calculation (excludes weekends + US federal holidays) ──────
_us_hols = holidays.US(years=range(2021, 2028))
_hol_dates = np.array(sorted(_us_hols.keys()), dtype="datetime64[D]")

def calc_biz_days(start, end):
    """count business days between two datetimes, excluding weekends and US holidays"""
    if pd.isnull(start) or pd.isnull(end):
        return np.nan
    s = np.datetime64(start.date(), "D")
    e = np.datetime64(end.date(), "D")
    if e <= s:
        return 0.0
    return float(np.busday_count(s, e, holidays=_hol_dates))

approvals["biz_days"] = approvals.apply(
    lambda r: calc_biz_days(r["start_dt"], r["completed_dt"]), axis=1
)

# ── filter to years of interest ───────────────────────────────────────────────
years = [2024, 2025, 2026]
df = approvals[approvals["year"].isin(years)].copy()

# ── normalize queue names ─────────────────────────────────────────────────────
queue_map = {
    "PO Approval":             "PO",
    "PD Approval":             "PD",
    "GO Approval":             "GO",
    "Legal Approval":          "Legal",
    "Legal Approval Post-GO":  "Legal",
    "President Approval":      "Prez",
    "Board Approval":          "Board",
    "CO Approval":             "CO",
    "OE Approval":             "OE",
}
df["queue"] = df["Step: Name"].map(queue_map)

# target queues (PA comes from requests, not approvals — handled below)
target_queues = ["PO", "PD", "GO", "Legal", "Prez", "CO"]

df_queues = df[df["queue"].isin(target_queues)].copy()

# ── filter requests to fully approved grants only ─────────────────────────────
# active and closed for all years; awarded also included (relevant for 2026 in-flight)
valid_stages = ["Active", "Closed", "Awarded"]
req_valid_ids = set(
    requests.loc[requests["Stage"].isin(valid_stages), "Request: Reference Number"].dropna()
)
df_queues = df_queues[df_queues["Record Name"].isin(req_valid_ids)].copy()

# ── join to requests to get program info ──────────────────────────────────────
req_slim = requests[
    ["Request: Reference Number", "Top Level Primary Program", "PA", "PO", "PD", "GO"]
].rename(columns={"Request: Reference Number": "Record Name"})

df_queues = df_queues.merge(req_slim, on="Record Name", how="left")

# ── quarter sort order ────────────────────────────────────────────────────────
quarter_order = [f"{y} Q{q}" for y in years for q in range(1, 5)]
# only keep quarters that actually appear in the data
present_quarters = df_queues["yq"].unique()
quarter_order = [q for q in quarter_order if q in present_quarters]

# ── aggregate: volume and median days per queue per quarter ───────────────────
agg = (
    df_queues.groupby(["yq", "queue"])
    .agg(
        volume=("Record Name", "nunique"),
        median_days=("biz_days", "median"),
        mean_days=("biz_days", "mean"),
        p75_days=("biz_days", lambda x: x.quantile(0.75)),
        p90_days=("biz_days", lambda x: x.quantile(0.90)),
    )
    .reset_index()
)
agg["yq"] = pd.Categorical(agg["yq"], categories=quarter_order, ordered=True)
agg = agg.sort_values(["queue", "yq"])

# ── colors: year-based palette ────────────────────────────────────────────────
year_colors = {
    2024: "#4A7B9D",   # muted steel blue
    2025: "#E8963A",   # warm amber
    2026: "#2D6A4F",   # deep forest green (partial year — noted)
}

queue_colors = {
    "PO":    "#1A254E",
    "PD":    "#778218",
    "GO":    "#E89829",
    "Legal": "#4A0F3E",
    "Prez":  "#3E006C",
    "CO":    "#184319",   # deep forest — Economy & Society color
}

# ════════════════════════════════════════════════════════════════════════════
# figure: stacked bars — unique grant count per queue per quarter
# ════════════════════════════════════════════════════════════════════════════

# pivot: rows = quarters, columns = queues, values = unique grant volume
pivot_stacked = agg[agg["queue"].isin(target_queues)].pivot_table(
    index="yq", columns="queue", values="volume", aggfunc="sum"
).reindex(quarter_order).fillna(0)
for q in target_queues:
    if q not in pivot_stacked.columns:
        pivot_stacked[q] = 0
pivot_stacked = pivot_stacked[target_queues]

fig, ax = plt.subplots(figsize=(16, 7))
fig.patch.set_facecolor("#F7F5F0")
ax.set_facecolor("#FAFAF7")

x = np.arange(len(quarter_order))
bar_width = 0.65
bottoms = np.zeros(len(quarter_order))

for queue in target_queues:
    vals = pivot_stacked[queue].values
    ax.bar(x, vals, bottom=bottoms, width=bar_width,
           color=queue_colors[queue], label=queue,
           zorder=3, edgecolor="white", linewidth=0.4)
    # label segment if tall enough to read
    for xi, (v, b) in enumerate(zip(vals, bottoms)):
        if v >= 20:
            ax.text(xi, b + v / 2, str(int(v)),
                    ha="center", va="center", fontsize=6.5,
                    color="white", fontweight="bold")
    bottoms = bottoms + vals

# year divider lines and labels
max_y = bottoms.max()
for yr in years:
    idxs = [i for i, q in enumerate(quarter_order) if q.startswith(str(yr))]
    if not idxs:
        continue
    mid = np.mean(idxs)
    label = str(yr) + (" *" if yr == 2026 else "")
    ax.text(mid, max_y * 1.06, label,
            ha="center", va="bottom", fontsize=11, fontweight="bold", color="#333")
    if idxs[0] > 0:
        ax.axvline(idxs[0] - 0.5, color="#bbb", linewidth=1.0,
                   linestyle="--", zorder=2)

ax.set_ylim(0, max_y * 1.15)
ax.set_xticks(x)
ax.set_xticklabels(
    [q.replace(" ", "\n") for q in quarter_order],
    fontsize=8.5
)
ax.set_ylabel("Unique Grant Approval Actions", fontsize=10)
ax.set_title(
    "Review Queue Volume by Quarter  |  2024 · 2025 · 2026",
    fontsize=14, fontweight="bold", color="#1C1C1C", pad=14
)
ax.legend(title="Queue", loc="upper left", fontsize=9,
          framealpha=0.92, title_fontsize=9)
ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
ax.spines[["top", "right"]].set_visible(False)
ax.annotate("* 2026 is a partial year",
            xy=(1.0, -0.11), xycoords="axes fraction",
            ha="right", fontsize=8, color="#888", style="italic")

plt.tight_layout()
plt.savefig("outputs/queue_stacked_volume.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("saved: outputs/queue_stacked_volume.png")


# ════════════════════════════════════════════════════════════════════════════
# csv export: full quarterly detail
# ════════════════════════════════════════════════════════════════════════════
export_cols = ["yq", "queue", "volume", "median_days", "mean_days", "p75_days", "p90_days"]
agg[agg["queue"].isin(target_queues)][export_cols].sort_values(["queue", "yq"]).to_csv(
    "outputs/queue_quarterly_detail.csv", index=False, float_format="%.2f"
)
print("saved: outputs/queue_quarterly_detail.csv")


# ════════════════════════════════════════════════════════════════════════════
# console summary
# ════════════════════════════════════════════════════════════════════════════
year_agg = (
    df_queues.groupby(["year", "queue"])
    .agg(
        volume=("Record Name", "nunique"),
        median_days=("biz_days", "median"),
    )
    .reset_index()
)
pivot_days = year_agg[year_agg["queue"].isin(target_queues)].pivot(
    index="queue", columns="year", values="median_days"
).reindex(target_queues)
pivot_vol = year_agg[year_agg["queue"].isin(target_queues)].pivot(
    index="queue", columns="year", values="volume"
).reindex(target_queues)

print()
print("=" * 60)
print("YEAR-OVER-YEAR SUMMARY: MEDIAN DAYS IN QUEUE")
print("=" * 60)
print(pivot_days.round(1).to_string())
print()
print("=" * 60)
print("YEAR-OVER-YEAR SUMMARY: UNIQUE GRANTS THROUGH QUEUE")
print("=" * 60)
print(pivot_vol.round(0).to_string())
print()
print("note: 2026 reflects partial year only")


# ════════════════════════════════════════════════════════════════════════════
# shared pivot data for timing charts
# ════════════════════════════════════════════════════════════════════════════

pivot_median = agg[agg["queue"].isin(target_queues)].pivot_table(
    index="yq", columns="queue", values="median_days"
).reindex(quarter_order)

pivot_mean = agg[agg["queue"].isin(target_queues)].pivot_table(
    index="yq", columns="queue", values="mean_days"
).reindex(quarter_order)

# volume-weighted average days across all queues per quarter
agg_all = agg[agg["queue"].isin(target_queues)].copy()
agg_all["weighted"] = agg_all["median_days"] * agg_all["volume"]
wavg = agg_all.groupby("yq").apply(
    lambda g: g["weighted"].sum() / g["volume"].sum()
).reindex(quarter_order)

x = np.arange(len(quarter_order))

def draw_volume(ax, pivot, quarter_order, queue_colors, target_queues, years, alpha=0.75):
    """draw stacked volume bars; return max total height"""
    bottoms = np.zeros(len(quarter_order))
    for queue in target_queues:
        vals = pivot[queue].values
        ax.bar(x, vals, bottom=bottoms, width=0.65,
               color=queue_colors[queue], alpha=alpha,
               zorder=2, edgecolor="white", linewidth=0.3)
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 25:
                ax.text(xi, b + v / 2, str(int(v)),
                        ha="center", va="center", fontsize=6,
                        color="white", fontweight="bold")
        bottoms = bottoms + vals
    max_y = bottoms.max()
    for yr in years:
        idxs = [i for i, q in enumerate(quarter_order) if q.startswith(str(yr))]
        if idxs and idxs[0] > 0:
            ax.axvline(idxs[0] - 0.5, color="#ccc", linewidth=0.9,
                       linestyle="--", zorder=1)
    ax.set_xticks(x)
    ax.set_xticklabels([q.replace(" ", "\n") for q in quarter_order], fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    return max_y


# ════════════════════════════════════════════════════════════════════════════
# option a: stacked volume + one median-days line per queue (dual y-axis)
# ════════════════════════════════════════════════════════════════════════════
fig_a, ax_a1 = plt.subplots(figsize=(17, 7))
fig_a.patch.set_facecolor("#F7F5F0")
ax_a1.set_facecolor("#FAFAF7")
ax_a2 = ax_a1.twinx()

max_y = draw_volume(ax_a1, pivot_stacked, quarter_order, queue_colors, target_queues, years)
ax_a1.set_ylim(0, max_y * 1.18)
ax_a1.set_ylabel("Unique Grant Approval Actions", fontsize=10, color="#444")

line_styles = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]
for i, queue in enumerate(target_queues):
    vals = pivot_median[queue].values.astype(float)
    mask = ~np.isnan(vals)
    if mask.sum() < 2:
        continue
    ax_a2.plot(x[mask], vals[mask],
               color=queue_colors[queue], linewidth=2.2,
               marker="o", markersize=4.5,
               linestyle=line_styles[i % len(line_styles)],
               zorder=5)

ax_a2.set_ylabel("Median Days in Queue", fontsize=10, color="#555")
ax_a2.spines[["top"]].set_visible(False)
ax_a2.grid(False)

for yr in years:
    idxs = [i for i, q in enumerate(quarter_order) if q.startswith(str(yr))]
    if idxs:
        ax_a1.text(np.mean(idxs), max_y * 1.13,
                   str(yr) + (" *" if yr == 2026 else ""),
                   ha="center", fontsize=10, fontweight="bold", color="#333")

vol_h = [plt.Rectangle((0,0),1,1, color=queue_colors[q], alpha=0.75) for q in target_queues]
line_h = [plt.Line2D([0],[0], color=queue_colors[q], linewidth=2,
          marker="o", markersize=4, linestyle=line_styles[i % len(line_styles)])
          for i, q in enumerate(target_queues)]
ax_a1.legend(vol_h + line_h,
             [f"{q} vol" for q in target_queues] + [f"{q} days" for q in target_queues],
             fontsize=7.5, loc="upper left", ncol=2, framealpha=0.9,
             title="bars = volume  |  lines = median days", title_fontsize=7.5)

ax_a1.set_title("Option A: Volume + Median Days Per Queue  |  2024 · 2025 · 2026",
                fontsize=13, fontweight="bold", color="#1C1C1C", pad=12)
ax_a1.annotate("* 2026 partial year", xy=(1.0, -0.10),
               xycoords="axes fraction", ha="right", fontsize=8, color="#888", style="italic")

plt.tight_layout()
plt.savefig("outputs/option_a_lines_per_queue.png", dpi=150, bbox_inches="tight",
            facecolor=fig_a.get_facecolor())
plt.close()
print("saved: outputs/option_a_lines_per_queue.png")


# ════════════════════════════════════════════════════════════════════════════
# option b: stacked volume + single weighted-avg days line (dual y-axis)
# ════════════════════════════════════════════════════════════════════════════
fig_b, ax_b1 = plt.subplots(figsize=(17, 7))
fig_b.patch.set_facecolor("#F7F5F0")
ax_b1.set_facecolor("#FAFAF7")
ax_b2 = ax_b1.twinx()

max_y = draw_volume(ax_b1, pivot_stacked, quarter_order, queue_colors, target_queues, years)
ax_b1.set_ylim(0, max_y * 1.18)
ax_b1.set_ylabel("Unique Grant Approval Actions", fontsize=10, color="#444")

wavg_vals = wavg.values.astype(float)
mask = ~np.isnan(wavg_vals)
ax_b2.plot(x[mask], wavg_vals[mask],
           color="#C0392B", linewidth=2.5, marker="o", markersize=5, zorder=5)
ax_b2.fill_between(x[mask], wavg_vals[mask], alpha=0.08, color="#C0392B")
ax_b2.set_ylabel("Volume-Weighted Avg Median Days", fontsize=10, color="#C0392B")
ax_b2.tick_params(axis="y", labelcolor="#C0392B")
ax_b2.spines[["top"]].set_visible(False)
ax_b2.grid(False)

for yr in years:
    idxs = [i for i, q in enumerate(quarter_order) if q.startswith(str(yr))]
    if idxs:
        ax_b1.text(np.mean(idxs), max_y * 1.13,
                   str(yr) + (" *" if yr == 2026 else ""),
                   ha="center", fontsize=10, fontweight="bold", color="#333")

vol_h = [plt.Rectangle((0,0),1,1, color=queue_colors[q], alpha=0.75) for q in target_queues]
line_h = plt.Line2D([0],[0], color="#C0392B", linewidth=2.5, marker="o", markersize=5)
ax_b1.legend(vol_h + [line_h],
             target_queues + ["Weighted avg days"],
             fontsize=8.5, loc="upper left", framealpha=0.9,
             title="bars = volume  |  line = avg days", title_fontsize=8)

ax_b1.set_title("Option B: Volume + Weighted Average Days  |  2024 · 2025 · 2026",
                fontsize=13, fontweight="bold", color="#1C1C1C", pad=12)
ax_b1.annotate("* 2026 partial year", xy=(1.0, -0.10),
               xycoords="axes fraction", ha="right", fontsize=8, color="#888", style="italic")

plt.tight_layout()
plt.savefig("outputs/option_b_weighted_avg.png", dpi=150, bbox_inches="tight",
            facecolor=fig_b.get_facecolor())
plt.close()
print("saved: outputs/option_b_weighted_avg.png")


# ════════════════════════════════════════════════════════════════════════════
# option c: side-by-side — stacked volume bars (left) + median days heatmap (right)
# ════════════════════════════════════════════════════════════════════════════
from matplotlib.gridspec import GridSpec as GS

fig_c = plt.figure(figsize=(20, 7))
fig_c.patch.set_facecolor("#F7F5F0")
gs = GS(1, 2, figure=fig_c, width_ratios=[2, 1], wspace=0.06,
        left=0.06, right=0.97, top=0.88, bottom=0.14)

ax_c1 = fig_c.add_subplot(gs[0])
ax_c2 = fig_c.add_subplot(gs[1])
ax_c1.set_facecolor("#FAFAF7")

max_y = draw_volume(ax_c1, pivot_stacked, quarter_order, queue_colors, target_queues, years)
ax_c1.set_ylim(0, max_y * 1.18)
ax_c1.set_ylabel("Unique Grant Approval Actions", fontsize=10)
for yr in years:
    idxs = [i for i, q in enumerate(quarter_order) if q.startswith(str(yr))]
    if idxs:
        ax_c1.text(np.mean(idxs), max_y * 1.13,
                   str(yr) + (" *" if yr == 2026 else ""),
                   ha="center", fontsize=10, fontweight="bold", color="#333")
vol_h = [plt.Rectangle((0,0),1,1, color=queue_colors[q], alpha=0.75) for q in target_queues]
ax_c1.legend(vol_h, target_queues, fontsize=8.5, loc="upper left",
             framealpha=0.9, title="Queue")
ax_c1.set_title("Grant Volume by Quarter", fontsize=11, fontweight="bold", pad=8)
ax_c1.annotate("* 2026 partial year", xy=(1.0, -0.13),
               xycoords="axes fraction", ha="right", fontsize=8, color="#888", style="italic")

# heatmap: queues x quarters
heatmap_data = pivot_median[target_queues].T
im = ax_c2.imshow(heatmap_data.values.astype(float), aspect="auto",
                  cmap="YlOrRd", interpolation="nearest")
ax_c2.set_yticks(range(len(target_queues)))
ax_c2.set_yticklabels(target_queues, fontsize=9, fontweight="bold")
ax_c2.set_xticks(range(len(quarter_order)))
ax_c2.set_xticklabels([q.replace(" ", "\n") for q in quarter_order], fontsize=7)
ax_c2.set_title("Median Days in Queue", fontsize=11, fontweight="bold", pad=8)

vmax = np.nanmax(heatmap_data.values.astype(float))
for row in range(heatmap_data.shape[0]):
    for col in range(heatmap_data.shape[1]):
        val = heatmap_data.values[row, col]
        if not np.isnan(val):
            text_color = "white" if val > vmax * 0.55 else "#1C1C1C"
            ax_c2.text(col, row, f"{val:.1f}",
                       ha="center", va="center", fontsize=7,
                       color=text_color, fontweight="bold")

for yr in years:
    idxs = [i for i, q in enumerate(quarter_order) if q.startswith(str(yr))]
    if idxs and idxs[0] > 0:
        ax_c2.axvline(idxs[0] - 0.5, color="white", linewidth=1.5)

plt.colorbar(im, ax=ax_c2, shrink=0.85, label="median days")
fig_c.suptitle("Option C: Volume Bars + Days Heatmap  |  2024 · 2025 · 2026",
               fontsize=13, fontweight="bold", color="#1C1C1C", y=0.97)

plt.savefig("outputs/option_c_heatmap.png", dpi=150, bbox_inches="tight",
            facecolor=fig_c.get_facecolor())
plt.close()
print("saved: outputs/option_c_heatmap.png")


# ════════════════════════════════════════════════════════════════════════════
# figure: grouped stacked bars — volume (left bar) + avg days per queue (right bar)
# each quarter has two bars side by side, same queue colors
# ════════════════════════════════════════════════════════════════════════════

# avg business days per queue per quarter (weekends + US holidays excluded)
pivot_avgdays = agg[agg["queue"].isin(target_queues)].pivot_table(
    index="yq", columns="queue", values="mean_days"
).reindex(quarter_order).fillna(0)
for q in target_queues:
    if q not in pivot_avgdays.columns:
        pivot_avgdays[q] = 0
pivot_avgdays = pivot_avgdays[target_queues]

# also compute median days pivot (alternative)
pivot_med2 = agg[agg["queue"].isin(target_queues)].pivot_table(
    index="yq", columns="queue", values="median_days"
).reindex(quarter_order).fillna(0)
for q in target_queues:
    if q not in pivot_med2.columns:
        pivot_med2[q] = 0
pivot_med2 = pivot_med2[target_queues]

n_quarters = len(quarter_order)
x = np.arange(n_quarters)
bar_w = 0.38        # width of each bar in the pair
gap   = 0.04        # gap between the two bars in a pair
x_vol  = x - (bar_w / 2) - (gap / 2)   # left bar (volume)
x_days = x + (bar_w / 2) + (gap / 2)   # right bar (avg days)

fig, ax1 = plt.subplots(figsize=(18, 8))
fig.patch.set_facecolor("#F7F5F0")
ax1.set_facecolor("#FAFAF7")
ax2 = ax1.twinx()

# ── left bars: stacked volume ─────────────────────────────────────────────
bottoms_vol = np.zeros(n_quarters)
for queue in target_queues:
    vals = pivot_stacked[queue].values
    ax1.bar(x_vol, vals, bottom=bottoms_vol, width=bar_w,
            color=queue_colors[queue], alpha=0.85,
            zorder=3, edgecolor="white", linewidth=0.35)
    for xi, (v, b) in enumerate(zip(vals, bottoms_vol)):
        if v >= 30:
            ax1.text(x_vol[xi], b + v / 2, str(int(v)),
                     ha="center", va="center", fontsize=5.5,
                     color="white", fontweight="bold")
    bottoms_vol += vals

# ── right bars: stacked avg days per queue ───────────────────────────────
bottoms_days = np.zeros(n_quarters)
for queue in target_queues:
    vals = pivot_avgdays[queue].values
    ax2.bar(x_days, vals, bottom=bottoms_days, width=bar_w,
            color=queue_colors[queue], alpha=0.55,
            zorder=3, edgecolor="white", linewidth=0.35,
            hatch="///")
    for xi, (v, b) in enumerate(zip(vals, bottoms_days)):
        if v >= 1.5:
            ax2.text(x_days[xi], b + v / 2, f"{v:.1f}",
                     ha="center", va="center", fontsize=5.5,
                     color="white", fontweight="bold")
    bottoms_days += vals

# ── axes formatting ───────────────────────────────────────────────────────
max_vol  = bottoms_vol.max()
max_days = bottoms_days.max()

ax1.set_ylim(0, max_vol * 1.20)
ax2.set_ylim(0, max_days * 1.20)

ax1.set_ylabel("Unique Grant Approval Actions  (solid bars)", fontsize=10, color="#333")
ax2.set_ylabel("Avg Business Days in Queue  (hatched bars)", fontsize=10, color="#555")
ax2.spines[["top"]].set_visible(False)
ax1.spines[["top", "right"]].set_visible(False)
ax1.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
ax2.grid(False)

# ── x-axis ticks centered between the two bars ───────────────────────────
ax1.set_xticks(x)
ax1.set_xticklabels([q.replace(" ", "\n") for q in quarter_order], fontsize=8.5)

# ── year dividers and labels ──────────────────────────────────────────────
for yr in years:
    idxs = [i for i, q in enumerate(quarter_order) if q.startswith(str(yr))]
    if not idxs:
        continue
    ax1.text(np.mean(idxs), max_vol * 1.14,
             str(yr) + (" *" if yr == 2026 else ""),
             ha="center", fontsize=11, fontweight="bold", color="#333")
    if idxs[0] > 0:
        ax1.axvline(idxs[0] - 0.5, color="#bbb", linewidth=1.0,
                    linestyle="--", zorder=1)

# ── legend ────────────────────────────────────────────────────────────────
solid_handles  = [plt.Rectangle((0,0),1,1, color=queue_colors[q], alpha=0.85)
                  for q in target_queues]
hatch_handles  = [plt.Rectangle((0,0),1,1, color=queue_colors[q], alpha=0.55,
                  hatch="///", edgecolor="white") for q in target_queues]
from matplotlib.lines import Line2D
spacer = Line2D([0],[0], color="none")

ax1.legend(
    solid_handles + [spacer] + hatch_handles,
    [f"{q}  (volume)" for q in target_queues] + [""] +
    [f"{q}  (avg days)" for q in target_queues],
    fontsize=7.5, loc="upper left", ncol=2, framealpha=0.92,
    title="solid = grant volume     hatched = avg days in queue",
    title_fontsize=8
)

ax1.set_title(
    "Review Queue: Volume vs. Avg Days per Step  |  2024 · 2025 · 2026",
    fontsize=14, fontweight="bold", color="#1C1C1C", pad=14
)
ax1.annotate("* 2026 is a partial year",
             xy=(1.0, -0.11), xycoords="axes fraction",
             ha="right", fontsize=8, color="#888", style="italic")

plt.tight_layout()
plt.savefig("outputs/volume_vs_avgdays_grouped.png", dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close()
print("saved: outputs/volume_vs_avgdays_grouped.png")


# ════════════════════════════════════════════════════════════════════════════
# combined figure: grouped bars + summary table beneath
# ════════════════════════════════════════════════════════════════════════════
from matplotlib.gridspec import GridSpec as GS2

# taller figure with explicit layout to prevent overlap
fig2 = plt.figure(figsize=(22, 16))
fig2.patch.set_facecolor("#F7F5F0")
gs2 = GS2(2, 1, figure=fig2, height_ratios=[1.6, 1.0],
          top=0.95, bottom=0.01, left=0.06, right=0.94, hspace=0.05)

ax_chart = fig2.add_subplot(gs2[0])
ax_table  = fig2.add_subplot(gs2[1])
ax_chart.set_facecolor("#FAFAF7")
ax_table.set_visible(False)

ax2b = ax_chart.twinx()

# ── redraw grouped bars on ax_chart ──────────────────────────────────────
bottoms_vol  = np.zeros(n_quarters)
bottoms_days = np.zeros(n_quarters)

for queue in target_queues:
    vols = pivot_stacked[queue].values
    days = pivot_avgdays[queue].values

    ax_chart.bar(x_vol, vols, bottom=bottoms_vol, width=bar_w,
                 color=queue_colors[queue], alpha=0.85,
                 zorder=3, edgecolor="white", linewidth=0.35)
    for xi, (v, b) in enumerate(zip(vols, bottoms_vol)):
        if v >= 30:
            ax_chart.text(x_vol[xi], b + v / 2, str(int(v)),
                          ha="center", va="center", fontsize=5.5,
                          color="white", fontweight="bold")
    bottoms_vol += vols

    ax2b.bar(x_days, days, bottom=bottoms_days, width=bar_w,
             color=queue_colors[queue], alpha=0.55,
             zorder=3, edgecolor="white", linewidth=0.35, hatch="///")
    for xi, (v, b) in enumerate(zip(days, bottoms_days)):
        if v >= 1.5:
            ax2b.text(x_days[xi], b + v / 2, f"{v:.1f}",
                      ha="center", va="center", fontsize=5.5,
                      color="white", fontweight="bold")
    bottoms_days += days

max_vol2  = bottoms_vol.max()
max_days2 = bottoms_days.max()
ax_chart.set_ylim(0, max_vol2 * 1.20)
ax2b.set_ylim(0, max_days2 * 1.20)

ax_chart.set_ylabel("Unique Grant Approval Actions  (solid bars)", fontsize=10, color="#333")
ax2b.set_ylabel("Avg Business Days in Queue  (hatched bars)", fontsize=10, color="#555")
ax2b.spines[["top"]].set_visible(False)
ax_chart.spines[["top", "right"]].set_visible(False)
ax_chart.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
ax2b.grid(False)

ax_chart.set_xticks(x)
ax_chart.set_xticklabels([q.replace(" ", "\n") for q in quarter_order], fontsize=8.5)

for yr in years:
    idxs = [i for i, q in enumerate(quarter_order) if q.startswith(str(yr))]
    if not idxs:
        continue
    ax_chart.text(np.mean(idxs), max_vol2 * 1.14,
                  str(yr) + (" *" if yr == 2026 else ""),
                  ha="center", fontsize=11, fontweight="bold", color="#333")
    if idxs[0] > 0:
        ax_chart.axvline(idxs[0] - 0.5, color="#bbb", linewidth=1.0,
                         linestyle="--", zorder=1)

solid_h = [plt.Rectangle((0,0),1,1, color=queue_colors[q], alpha=0.85) for q in target_queues]
hatch_h = [plt.Rectangle((0,0),1,1, color=queue_colors[q], alpha=0.55,
            hatch="///") for q in target_queues]
from matplotlib.lines import Line2D as L2D
ax_chart.legend(
    solid_h + [L2D([0],[0], color="none")] + hatch_h,
    [f"{q}  (volume)" for q in target_queues] + [""] +
    [f"{q}  (avg days)" for q in target_queues],
    fontsize=7.5, loc="upper left", ncol=2, framealpha=0.92,
    title="solid = grant volume     hatched = avg days in queue",
    title_fontsize=8
)
ax_chart.set_title(
    "Review Queue: Volume vs. Avg Days per Step  |  2024 · 2025 · 2026",
    fontsize=14, fontweight="bold", color="#1C1C1C", pad=14
)

# ── build table data ──────────────────────────────────────────────────────
# rows: one per queue, two metrics each (Vol / Avg Days)
# columns: each quarter + annual totals for each year

# compute annual totals
annual_vol  = {}
annual_days = {}
for yr in years:
    yr_qs = [q for q in quarter_order if q.startswith(str(yr))]
    for queue in target_queues:
        vol_vals  = pivot_stacked.loc[yr_qs, queue].values
        days_vals = pivot_avgdays.loc[yr_qs, queue].values
        # annual volume = sum across quarters
        annual_vol[(yr, queue)]  = int(vol_vals.sum())
        # annual avg days = mean of quarterly means (simple avg)
        valid = days_vals[days_vals > 0]
        annual_days[(yr, queue)] = round(valid.mean(), 1) if len(valid) else 0.0

# column headers: quarter labels + year total labels
col_headers = quarter_order + [f"{yr} Total" for yr in years]
n_cols = len(col_headers)

# row headers: queue name + metric label
row_labels = []
for queue in target_queues:
    row_labels.append(f"{queue}  Vol")
    row_labels.append(f"{queue}  Avg Days")
n_rows = len(row_labels)

# fill cell values
cell_text = []
for queue in target_queues:
    vol_row  = []
    days_row = []
    for q in quarter_order:
        v = pivot_stacked.loc[q, queue] if q in pivot_stacked.index else 0
        d = pivot_avgdays.loc[q, queue] if q in pivot_avgdays.index else 0
        vol_row.append(str(int(v)) if v > 0 else "—")
        days_row.append(f"{d:.1f}" if d > 0 else "—")
    # annual totals
    for yr in years:
        vol_row.append(str(annual_vol[(yr, queue)]))
        days_row.append(f"{annual_days[(yr, queue)]:.1f}")
    cell_text.append(vol_row)
    cell_text.append(days_row)

# ── draw table using matplotlib table ────────────────────────────────────
ax_table.set_visible(True)
ax_table.axis("off")

# color each row pair to match its queue
# vol rows: medium tint, days rows: light tint — clearly differentiated
vol_tint  = "#E4E8EE"   # cool light grey for vol data cells
days_tint = "#F5F5F0"   # warm near-white for days data cells
vol_total_tint  = "#CDD3DC"
days_total_tint = "#E8E8E2"

row_colors = []
for queue in target_queues:
    base = queue_colors[queue]
    # cellColours covers data cells only (not the row label column)
    # n_cols = len(quarter_order) + len(years)
    vol_row  = [vol_tint] * len(quarter_order) + [vol_total_tint] * len(years)
    days_row = [days_tint] * len(quarter_order) + [days_total_tint] * len(years)
    row_colors.append(vol_row)
    row_colors.append(days_row)

tbl = ax_table.table(
    cellText=cell_text,
    rowLabels=row_labels,
    colLabels=col_headers,
    cellColours=row_colors,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.85)

# style header row and row labels
for (r, c), cell in tbl.get_celld().items():
    cell.set_linewidth(0.3)
    cell.set_edgecolor("#ccc")
    if r == 0:
        # column header
        cell.set_facecolor("#2C2C2C")
        cell.set_text_props(color="white", fontweight="bold", fontsize=7)
    if c == -1:
        # row label — vol rows full color, days rows slightly faded
        queue_idx = r - 1
        queue_name = target_queues[queue_idx // 2] if r >= 1 else None
        is_days_row = (queue_idx % 2 == 1) if r >= 1 else False
        if queue_name:
            import matplotlib.colors as mcolors
            base_rgb = mcolors.to_rgb(queue_colors[queue_name])
            if is_days_row:
                # blend toward white for days label
                faded = tuple(c * 0.55 + 0.45 for c in base_rgb)
                cell.set_facecolor(faded)
                cell.set_text_props(color="white", fontweight="normal",
                                    fontsize=7, style="italic")
            else:
                cell.set_facecolor(queue_colors[queue_name])
                cell.set_text_props(color="white", fontweight="bold", fontsize=7.5)
        cell.set_width(0.08)
    # bold the annual total columns
    if c >= len(quarter_order) and r > 0:
        cell.set_text_props(fontweight="bold")



fig2.text(0.99, 0.01, "* 2026 is a partial year",
          ha="right", fontsize=8, color="#888", style="italic")

plt.savefig("outputs/volume_days_with_table.png", dpi=150, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
plt.close()
print("saved: outputs/volume_days_with_table.png")


# ════════════════════════════════════════════════════════════════════════════
# combined figure: grouped bars + summary table beneath
# ════════════════════════════════════════════════════════════════════════════

# taller figure with explicit layout to prevent overlap
fig2 = plt.figure(figsize=(22, 16))
fig2.patch.set_facecolor("#F7F5F0")
gs2 = GS2(2, 1, figure=fig2, height_ratios=[1.6, 1.0],
          top=0.95, bottom=0.01, left=0.06, right=0.94, hspace=0.05)

ax_chart = fig2.add_subplot(gs2[0])
ax_table  = fig2.add_subplot(gs2[1])
ax_chart.set_facecolor("#FAFAF7")
ax_table.set_visible(False)

ax2b = ax_chart.twinx()

# ── redraw grouped bars on ax_chart ──────────────────────────────────────
bottoms_vol  = np.zeros(n_quarters)
bottoms_days = np.zeros(n_quarters)

for queue in target_queues:
    vols = pivot_stacked[queue].values
    days = pivot_med2[queue].values

    ax_chart.bar(x_vol, vols, bottom=bottoms_vol, width=bar_w,
                 color=queue_colors[queue], alpha=0.85,
                 zorder=3, edgecolor="white", linewidth=0.35)
    for xi, (v, b) in enumerate(zip(vols, bottoms_vol)):
        if v >= 30:
            ax_chart.text(x_vol[xi], b + v / 2, str(int(v)),
                          ha="center", va="center", fontsize=5.5,
                          color="white", fontweight="bold")
    bottoms_vol += vols

    ax2b.bar(x_days, days, bottom=bottoms_days, width=bar_w,
             color=queue_colors[queue], alpha=0.55,
             zorder=3, edgecolor="white", linewidth=0.35, hatch="///")
    for xi, (v, b) in enumerate(zip(days, bottoms_days)):
        if v >= 1.5:
            ax2b.text(x_days[xi], b + v / 2, f"{v:.1f}",
                      ha="center", va="center", fontsize=5.5,
                      color="white", fontweight="bold")
    bottoms_days += days

max_vol2  = bottoms_vol.max()
max_days2 = bottoms_days.max()
ax_chart.set_ylim(0, max_vol2 * 1.20)
ax2b.set_ylim(0, max_days2 * 1.20)

ax_chart.set_ylabel("Unique Grant Approval Actions  (solid bars)", fontsize=10, color="#333")
ax2b.set_ylabel("Median Business Days in Queue  (hatched bars)", fontsize=10, color="#555")
ax2b.spines[["top"]].set_visible(False)
ax_chart.spines[["top", "right"]].set_visible(False)
ax_chart.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
ax2b.grid(False)

ax_chart.set_xticks(x)
ax_chart.set_xticklabels([q.replace(" ", "\n") for q in quarter_order], fontsize=8.5)

for yr in years:
    idxs = [i for i, q in enumerate(quarter_order) if q.startswith(str(yr))]
    if not idxs:
        continue
    ax_chart.text(np.mean(idxs), max_vol2 * 1.14,
                  str(yr) + (" *" if yr == 2026 else ""),
                  ha="center", fontsize=11, fontweight="bold", color="#333")
    if idxs[0] > 0:
        ax_chart.axvline(idxs[0] - 0.5, color="#bbb", linewidth=1.0,
                         linestyle="--", zorder=1)

solid_h = [plt.Rectangle((0,0),1,1, color=queue_colors[q], alpha=0.85) for q in target_queues]
hatch_h = [plt.Rectangle((0,0),1,1, color=queue_colors[q], alpha=0.55,
            hatch="///") for q in target_queues]
ax_chart.legend(
    solid_h + [L2D([0],[0], color="none")] + hatch_h,
    [f"{q}  (volume)" for q in target_queues] + [""] +
    [f"{q}  (avg days)" for q in target_queues],
    fontsize=7.5, loc="upper left", ncol=2, framealpha=0.92,
    title="solid = grant volume     hatched = avg days in queue",
    title_fontsize=8
)
ax_chart.set_title(
    "Review Queue: Volume vs. Median Days per Step  |  2024 · 2025 · 2026",
    fontsize=14, fontweight="bold", color="#1C1C1C", pad=14
)

# ── build table data ──────────────────────────────────────────────────────
# rows: one per queue, two metrics each (Vol / Avg Days)
# columns: each quarter + annual totals for each year

# compute annual totals
annual_vol  = {}
annual_days = {}
for yr in years:
    yr_qs = [q for q in quarter_order if q.startswith(str(yr))]
    for queue in target_queues:
        vol_vals  = pivot_stacked.loc[yr_qs, queue].values
        days_vals = pivot_med2.loc[yr_qs, queue].values
        # annual volume = sum across quarters
        annual_vol[(yr, queue)]  = int(vol_vals.sum())
        # annual median days = median of quarterly medians
        valid = days_vals[days_vals > 0]
        annual_days[(yr, queue)] = round(float(np.median(valid)), 1) if len(valid) else 0.0

# column headers: quarter labels + year total labels
col_headers = quarter_order + [f"{yr} Total" for yr in years]
n_cols = len(col_headers)

# row headers: queue name + metric label
row_labels = []
for queue in target_queues:
    row_labels.append(f"{queue}  Vol")
    row_labels.append(f"{queue}  Avg Days")
n_rows = len(row_labels)

# fill cell values
cell_text = []
for queue in target_queues:
    vol_row  = []
    days_row = []
    for q in quarter_order:
        v = pivot_stacked.loc[q, queue] if q in pivot_stacked.index else 0
        d = pivot_med2.loc[q, queue] if q in pivot_med2.index else 0
        vol_row.append(str(int(v)) if v > 0 else "—")
        days_row.append(f"{d:.1f}" if d > 0 else "—")
    # annual totals
    for yr in years:
        vol_row.append(str(annual_vol[(yr, queue)]))
        days_row.append(f"{annual_days[(yr, queue)]:.1f}")
    cell_text.append(vol_row)
    cell_text.append(days_row)

# ── draw table using matplotlib table ────────────────────────────────────
ax_table.set_visible(True)
ax_table.axis("off")

# color each row pair to match its queue
# vol rows: medium tint, days rows: light tint — clearly differentiated
vol_tint  = "#E4E8EE"   # cool light grey for vol data cells
days_tint = "#F5F5F0"   # warm near-white for days data cells
vol_total_tint  = "#CDD3DC"
days_total_tint = "#E8E8E2"

row_colors = []
for queue in target_queues:
    base = queue_colors[queue]
    # cellColours covers data cells only (not the row label column)
    # n_cols = len(quarter_order) + len(years)
    vol_row  = [vol_tint] * len(quarter_order) + [vol_total_tint] * len(years)
    days_row = [days_tint] * len(quarter_order) + [days_total_tint] * len(years)
    row_colors.append(vol_row)
    row_colors.append(days_row)

tbl = ax_table.table(
    cellText=cell_text,
    rowLabels=row_labels,
    colLabels=col_headers,
    cellColours=row_colors,
    loc="center",
    cellLoc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(8)
tbl.scale(1, 1.85)

# style header row and row labels
for (r, c), cell in tbl.get_celld().items():
    cell.set_linewidth(0.3)
    cell.set_edgecolor("#ccc")
    if r == 0:
        # column header
        cell.set_facecolor("#2C2C2C")
        cell.set_text_props(color="white", fontweight="bold", fontsize=7)
    if c == -1:
        # row label — vol rows full color, days rows slightly faded
        queue_idx = r - 1
        queue_name = target_queues[queue_idx // 2] if r >= 1 else None
        is_days_row = (queue_idx % 2 == 1) if r >= 1 else False
        if queue_name:
            import matplotlib.colors as mcolors
            base_rgb = mcolors.to_rgb(queue_colors[queue_name])
            if is_days_row:
                # blend toward white for days label
                faded = tuple(c * 0.55 + 0.45 for c in base_rgb)
                cell.set_facecolor(faded)
                cell.set_text_props(color="white", fontweight="normal",
                                    fontsize=7, style="italic")
            else:
                cell.set_facecolor(queue_colors[queue_name])
                cell.set_text_props(color="white", fontweight="bold", fontsize=7.5)
        cell.set_width(0.08)
    # bold the annual total columns
    if c >= len(quarter_order) and r > 0:
        cell.set_text_props(fontweight="bold")



fig2.text(0.99, 0.01, "* 2026 is a partial year",
          ha="right", fontsize=8, color="#888", style="italic")

plt.savefig("outputs/volume_days_with_table_median.png", dpi=150, bbox_inches="tight",
            facecolor=fig2.get_facecolor())
plt.close()
print("saved: outputs/volume_days_with_table_median.png")

# ════════════════════════════════════════════════════════════════════════════
# per-program charts: mean and median versions with foundation-wide reference lines
# ════════════════════════════════════════════════════════════════════════════

program_colors = {
    "Education":                "#1A254E",
    "Environment":              "#778218",
    "Gender Equity & Governance": "#E89829",
    "Performing Arts":          "#4A0F3E",
    "U.S. Democracy":           "#3E006C",
    "Philanthropy":             "#214240",
    "Racial Justice":           "#C15811",
    "Economy and Society":      "#184319",
    "Special Projects":         "#C5D0C5",
    "SBAC":                     "#414B3F",
}

# programs with enough data to be meaningful
active_programs = [
    "Gender Equity & Governance", "Environment", "Education",
    "U.S. Democracy", "Performing Arts", "Philanthropy",
    "Special Projects", "Economy and Society", "Racial Justice", "SBAC",
]

os.makedirs("outputs/programs", exist_ok=True)

def make_program_chart(program, metric, pivot_program_days,
                       pivot_program_vol,
                       quarter_order, target_queues, queue_colors,
                       years, program_color):
    """
    draw grouped stacked bars for a single program with foundation-wide
    reference lines overlaid in red.
    metric: 'mean' or 'median'
    """
    n_quarters = len(quarter_order)
    x = np.arange(n_quarters)
    bar_w = 0.38
    gap   = 0.04
    x_vol  = x - (bar_w / 2) - (gap / 2)
    x_days = x + (bar_w / 2) + (gap / 2)

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#F7F5F0")
    gs = GS2(2, 1, figure=fig, height_ratios=[1.6, 1.0],
             top=0.95, bottom=0.01, left=0.06, right=0.94, hspace=0.05)
    ax_chart = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    ax_chart.set_facecolor("#FAFAF7")
    ax_table.set_visible(False)
    ax2 = ax_chart.twinx()

    # ── stacked volume bars (program) ─────────────────────────────────────
    bottoms_vol  = np.zeros(n_quarters)
    bottoms_days = np.zeros(n_quarters)

    for queue in target_queues:
        vols = pivot_program_vol[queue].values
        days = pivot_program_days[queue].values

        ax_chart.bar(x_vol, vols, bottom=bottoms_vol, width=bar_w,
                     color=queue_colors[queue], alpha=0.85,
                     zorder=3, edgecolor="white", linewidth=0.35)
        for xi, (v, b) in enumerate(zip(vols, bottoms_vol)):
            if v >= 15:
                ax_chart.text(x_vol[xi], b + v / 2, str(int(v)),
                              ha="center", va="center", fontsize=5.5,
                              color="white", fontweight="bold")
        bottoms_vol += vols

        ax2.bar(x_days, days, bottom=bottoms_days, width=bar_w,
                color=queue_colors[queue], alpha=0.55,
                zorder=3, edgecolor="white", linewidth=0.35, hatch="///")
        for xi, (v, b) in enumerate(zip(days, bottoms_days)):
            if v >= 0.8:
                ax2.text(x_days[xi], b + v / 2, f"{v:.1f}",
                         ha="center", va="center", fontsize=5.5,
                         color="white", fontweight="bold")
        bottoms_days += days

    max_vol  = max(bottoms_vol.max(), 1)
    max_days = max(bottoms_days.max(), 1)

    # scale axes
    ax_chart.set_ylim(0, max_vol * 1.22)
    ax2.set_ylim(0, max_days * 1.22)

    metric_label = "Avg" if metric == "mean" else "Median"
    ax_chart.set_ylabel("Unique Grant Approval Actions  (solid bars)", fontsize=10, color="#333")
    ax2.set_ylabel(f"{metric_label} Business Days in Queue  (hatched bars)", fontsize=10, color="#555")
    ax2.spines[["top"]].set_visible(False)
    ax_chart.spines[["top", "right"]].set_visible(False)
    ax_chart.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax2.grid(False)

    ax_chart.set_xticks(x)
    ax_chart.set_xticklabels([q.replace(" ", "\n") for q in quarter_order], fontsize=8.5)

    for yr in years:
        idxs = [i for i, q in enumerate(quarter_order) if q.startswith(str(yr))]
        if not idxs:
            continue
        ax_chart.text(np.mean(idxs), max_vol * 1.16,
                      str(yr) + (" *" if yr == 2026 else ""),
                      ha="center", fontsize=11, fontweight="bold", color="#333")
        if idxs[0] > 0:
            ax_chart.axvline(idxs[0] - 0.5, color="#bbb", linewidth=1.0,
                             linestyle="--", zorder=1)

    # legend
    solid_h = [plt.Rectangle((0,0),1,1, color=queue_colors[q], alpha=0.85) for q in target_queues]
    hatch_h = [plt.Rectangle((0,0),1,1, color=queue_colors[q], alpha=0.55, hatch="///")
               for q in target_queues]
    ax_chart.legend(
        solid_h + [L2D([0],[0], color="none")] + hatch_h,
        [f"{q}  (volume)" for q in target_queues] + [""] +
        [f"{q}  ({metric_label.lower()} days)" for q in target_queues],
        fontsize=7.5, loc="upper left", ncol=2, framealpha=0.92,
        title=f"solid = grant volume     hatched = {metric_label.lower()} days in queue",
        title_fontsize=8
    )
    ax_chart.set_title(
        f"{program}  |  Review Queue: Volume vs. {metric_label} Business Days  |  2024 · 2025 · 2026",
        fontsize=13, fontweight="bold", color=program_color, pad=14
    )
    ax_chart.annotate("* 2026 is a partial year",
                      xy=(1.0, -0.09), xycoords="axes fraction",
                      ha="right", fontsize=8, color="#888", style="italic")

    # ── table ─────────────────────────────────────────────────────────────
    annual_vol  = {}
    annual_days = {}
    for yr in years:
        yr_qs = [q for q in quarter_order if q.startswith(str(yr))]
        for queue in target_queues:
            vv = pivot_program_vol.loc[yr_qs, queue].values if all(q in pivot_program_vol.index for q in yr_qs) else np.zeros(len(yr_qs))
            dv = pivot_program_days.loc[yr_qs, queue].values if all(q in pivot_program_days.index for q in yr_qs) else np.zeros(len(yr_qs))
            annual_vol[(yr, queue)]  = int(vv.sum())
            valid = dv[dv > 0]
            if metric == "mean":
                annual_days[(yr, queue)] = round(float(valid.mean()), 1) if len(valid) else 0.0
            else:
                annual_days[(yr, queue)] = round(float(np.median(valid)), 1) if len(valid) else 0.0

    col_headers = quarter_order + [f"{yr} Total" for yr in years]
    n_cols = len(col_headers)
    row_labels = []
    for queue in target_queues:
        row_labels.append(f"{queue}  Vol")
        row_labels.append(f"{queue}  {metric_label} Days")

    cell_text = []
    for queue in target_queues:
        vol_row  = []
        days_row = []
        for q in quarter_order:
            v = pivot_program_vol.loc[q, queue] if q in pivot_program_vol.index else 0
            d = pivot_program_days.loc[q, queue] if q in pivot_program_days.index else 0
            vol_row.append(str(int(v)) if v > 0 else "—")
            days_row.append(f"{d:.1f}" if d > 0 else "—")
        for yr in years:
            vol_row.append(str(annual_vol[(yr, queue)]))
            days_row.append(f"{annual_days[(yr, queue)]:.1f}" if annual_days[(yr, queue)] > 0 else "—")
        cell_text.append(vol_row)
        cell_text.append(days_row)

    ax_table.set_visible(True)
    ax_table.axis("off")

    vol_tint        = "#E4E8EE"
    days_tint       = "#F5F5F0"
    vol_total_tint  = "#CDD3DC"
    days_total_tint = "#E8E8E2"

    row_colors = []
    for queue in target_queues:
        base = queue_colors[queue]
        vol_row_c  = [vol_tint]  * len(quarter_order) + [vol_total_tint]  * len(years)
        days_row_c = [days_tint] * len(quarter_order) + [days_total_tint] * len(years)
        row_colors.append(vol_row_c)
        row_colors.append(days_row_c)

    tbl = ax_table.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_headers,
        cellColours=row_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.85)

    import matplotlib.colors as mcolors
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.3)
        cell.set_edgecolor("#ccc")
        if r == 0:
            cell.set_facecolor("#2C2C2C")
            cell.set_text_props(color="white", fontweight="bold", fontsize=7)
        if c == -1:
            queue_idx = r - 1
            queue_name = target_queues[queue_idx // 2] if r >= 1 else None
            is_days_row = (queue_idx % 2 == 1) if r >= 1 else False
            if queue_name:
                base_rgb = mcolors.to_rgb(queue_colors[queue_name])
                if is_days_row:
                    faded = tuple(ch * 0.55 + 0.45 for ch in base_rgb)
                    cell.set_facecolor(faded)
                    cell.set_text_props(color="white", fontweight="normal",
                                        fontsize=7, style="italic")
                else:
                    cell.set_facecolor(queue_colors[queue_name])
                    cell.set_text_props(color="white", fontweight="bold", fontsize=7.5)
            cell.set_width(0.08)
        if c >= len(quarter_order) and r > 0:
            cell.set_text_props(fontweight="bold")

    fig.text(0.99, 0.01, "* 2026 is a partial year",
             ha="right", fontsize=8, color="#888", style="italic")

    safe_name = program.replace(" ", "_").replace("&", "and").replace(".", "")
    fname = f"outputs/programs/{safe_name}_{metric}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    return fname


# ── precompute foundation-wide pivots (already have these) ───────────────
# pivot_stacked = volume, pivot_med2 = median days, pivot_avgdays = mean days

# ── generate per-program charts ───────────────────────────────────────────
for program in active_programs:
    prog_color = program_colors.get(program, "#444444")

    # filter df_queues to this program
    prog_mask = df_queues["Top Level Primary Program"] == program
    df_prog = df_queues[prog_mask].copy()

    if len(df_prog) == 0:
        print(f"skipping {program} — no data")
        continue

    # build program-level agg
    prog_agg = (
        df_prog.groupby(["yq", "queue"])
        .agg(
            volume=("Record Name", "nunique"),
            mean_days=("biz_days", "mean"),
            median_days=("biz_days", "median"),
        )
        .reset_index()
    )
    prog_agg["yq"] = pd.Categorical(prog_agg["yq"], categories=quarter_order, ordered=True)
    prog_agg = prog_agg.sort_values(["queue", "yq"])

    # pivot volume
    p_vol = prog_agg[prog_agg["queue"].isin(target_queues)].pivot_table(
        index="yq", columns="queue", values="volume", aggfunc="sum"
    ).reindex(quarter_order).fillna(0)
    for q in target_queues:
        if q not in p_vol.columns:
            p_vol[q] = 0
    p_vol = p_vol[target_queues]

    # pivot mean days
    p_mean = prog_agg[prog_agg["queue"].isin(target_queues)].pivot_table(
        index="yq", columns="queue", values="mean_days"
    ).reindex(quarter_order).fillna(0)
    for q in target_queues:
        if q not in p_mean.columns:
            p_mean[q] = 0
    p_mean = p_mean[target_queues]

    # pivot median days
    p_med = prog_agg[prog_agg["queue"].isin(target_queues)].pivot_table(
        index="yq", columns="queue", values="median_days"
    ).reindex(quarter_order).fillna(0)
    for q in target_queues:
        if q not in p_med.columns:
            p_med[q] = 0
    p_med = p_med[target_queues]

    for metric, p_days in [("mean", p_mean), ("median", p_med)]:
        fname = make_program_chart(
            program=program,
            metric=metric,
            pivot_program_days=p_days,
            pivot_program_vol=p_vol,
            quarter_order=quarter_order,
            target_queues=target_queues,
            queue_colors=queue_colors,
            years=years,
            program_color=prog_color,
        )
        print(f"saved: {fname}")

print("all program charts complete")




# ════════════════════════════════════════════════════════════════════════════
# foundation-wide line chart: approval steps + approved grants + timing
# three lines: steps (navy), grants (grey), days (red dashed)
# two versions: median and mean
# ════════════════════════════════════════════════════════════════════════════

# total approval step actions per quarter (sum across all queues)
total_steps = pivot_stacked[target_queues].sum(axis=1).reindex(quarter_order).fillna(0).values.astype(float)

# unique grants approved per quarter — keyed to the quarter of first approval step
first_step_yq = (
    df_queues[df_queues["queue"].isin(target_queues)]
    .sort_values("start_dt")
    .groupby("Record Name")["yq"]
    .first()
    .reset_index()
)
grants_per_quarter = (
    first_step_yq.groupby("yq")["Record Name"]
    .nunique()
    .reindex(quarter_order)
    .fillna(0)
    .values.astype(float)
)

# overall median and mean business days per quarter
overall_median = (
    df_queues[df_queues["queue"].isin(target_queues)]
    .groupby("yq")["biz_days"]
    .median()
    .reindex(quarter_order)
    .values.astype(float)
)
overall_mean = (
    df_queues[df_queues["queue"].isin(target_queues)]
    .groupby("yq")["biz_days"]
    .mean()
    .reindex(quarter_order)
    .values.astype(float)
)

x = np.arange(len(quarter_order))
steps_color  = "#1A254E"   # navy
grants_color = "#888888"   # grey
days_color   = "#C0392B"   # red

for timing_vals, timing_label, fname_suffix in [
    (overall_median, "Median", "median"),
    (overall_mean,   "Mean",   "mean"),
]:
    fig, ax1 = plt.subplots(figsize=(16, 7))
    fig.patch.set_facecolor("#F7F5F0")
    ax1.set_facecolor("#FAFAF7")
    ax2 = ax1.twinx()

    # approval steps line (left axis, navy)
    ax1.plot(x, total_steps, color=steps_color, linewidth=2.5, marker="o",
             markersize=6, zorder=4)
    ax1.fill_between(x, total_steps, alpha=0.05, color=steps_color)
    for xi, v in enumerate(total_steps):
        if v > 0:
            ax1.text(xi, v + total_steps.max() * 0.025, str(int(v)),
                     ha="center", va="bottom", fontsize=7,
                     color=steps_color, fontweight="bold")

    # grants approved line (left axis, grey)
    ax1.plot(x, grants_per_quarter, color=grants_color, linewidth=2.0,
             marker="D", markersize=5, linestyle="-", zorder=3, alpha=0.85)
    for xi, v in enumerate(grants_per_quarter):
        if v > 0:
            ax1.text(xi, v - total_steps.max() * 0.055, str(int(v)),
                     ha="center", va="top", fontsize=7,
                     color=grants_color, fontweight="bold")

    # timing line (right axis, red dashed)
    mask = ~np.isnan(timing_vals)
    ax2.plot(x[mask], timing_vals[mask], color=days_color, linewidth=2.5,
             marker="s", markersize=6, linestyle="--", zorder=4)
    ax2.fill_between(x[mask], timing_vals[mask], alpha=0.06, color=days_color)
    for xi, v in zip(x[mask], timing_vals[mask]):
        ax2.text(xi, v + timing_vals[mask].max() * 0.04, f"{v:.1f}",
                 ha="center", va="bottom", fontsize=7.5,
                 color=days_color, fontweight="bold")

    max_left  = max(total_steps.max(), grants_per_quarter.max())
    max_days  = timing_vals[mask].max()

    # year dividers and labels
    for yr in years:
        idxs = [i for i, q in enumerate(quarter_order) if q.startswith(str(yr))]
        if not idxs:
            continue
        ax1.text(np.mean(idxs), max_left * 1.14,
                 str(yr) + (" *" if yr == 2026 else ""),
                 ha="center", fontsize=11, fontweight="bold", color="#333")
        if idxs[0] > 0:
            ax1.axvline(idxs[0] - 0.5, color="#bbb", linewidth=1.0,
                        linestyle="--", zorder=1)

    ax1.set_ylim(0, max_left * 1.22)
    ax2.set_ylim(0, max_days * 1.8)

    ax1.set_ylabel("Count  (approval steps & grants)", fontsize=11, color="#333")
    ax2.set_ylabel(f"{timing_label} Business Days in Queue", fontsize=11, color=days_color)
    ax1.tick_params(axis="y", labelcolor="#333")
    ax2.tick_params(axis="y", labelcolor=days_color)
    ax1.spines[["top", "right"]].set_visible(False)
    ax2.spines[["top"]].set_visible(False)
    ax1.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)
    ax2.grid(False)

    ax1.set_xticks(x)
    ax1.set_xticklabels([q.replace(" ", "\n") for q in quarter_order], fontsize=8.5)

    steps_h  = plt.Line2D([0],[0], color=steps_color,  linewidth=2.5, marker="o", markersize=6)
    grants_h = plt.Line2D([0],[0], color=grants_color, linewidth=2.0, marker="D",
                          markersize=5, alpha=0.85)
    days_h   = plt.Line2D([0],[0], color=days_color,   linewidth=2.5, marker="s",
                          markersize=6, linestyle="--")
    ax1.legend(
        [steps_h, grants_h, days_h],
        ["Grant approval step actions  (left axis)",
         "Unique grants approved  (left axis)",
         f"{timing_label} business days in queue  (right axis)"],
        fontsize=9, loc="upper left", framealpha=0.92
    )

    ax1.set_title(
        f"Review Queue Trends: Steps, Grants & {timing_label} Processing Time  |  2024 · 2025 · 2026",
        fontsize=14, fontweight="bold", color="#1C1C1C", pad=14
    )
    ax1.annotate("* 2026 is a partial year",
                 xy=(1.0, -0.10), xycoords="axes fraction",
                 ha="right", fontsize=8, color="#888", style="italic")

    plt.tight_layout()
    out = f"outputs/foundation_line_trends_{fname_suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"saved: {out}")


# ════════════════════════════════════════════════════════════════════════════
# processing time only: stacked bars showing mean and median days per queue
# no volume bars — just the hatched days bars centered per quarter
# ════════════════════════════════════════════════════════════════════════════

for metric, pivot_days, metric_label, fname_suffix in [
    ("mean",   pivot_avgdays, "Avg",    "mean"),
    ("median", pivot_med2,    "Median", "median"),
]:
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#F7F5F0")
    gs_t = GS2(2, 1, figure=fig, height_ratios=[1.6, 1.0],
               top=0.95, bottom=0.01, left=0.06, right=0.94, hspace=0.05)
    ax_chart = fig.add_subplot(gs_t[0])
    ax_table  = fig.add_subplot(gs_t[1])
    ax_chart.set_facecolor("#FAFAF7")
    ax_table.set_visible(False)

    x = np.arange(n_quarters)
    bar_width_t = 0.65   # full width since only one bar per quarter

    bottoms = np.zeros(n_quarters)
    for queue in target_queues:
        vals = pivot_days[queue].values
        ax_chart.bar(x, vals, bottom=bottoms, width=bar_width_t,
                     color=queue_colors[queue], alpha=0.70,
                     zorder=3, edgecolor="white", linewidth=0.4, hatch="///")
        for xi, (v, b) in enumerate(zip(vals, bottoms)):
            if v >= 0.6:
                ax_chart.text(xi, b + v / 2, f"{v:.1f}",
                              ha="center", va="center", fontsize=6,
                              color="white", fontweight="bold")
        bottoms += vals

    max_days = bottoms.max()
    ax_chart.set_ylim(0, max_days * 1.22)
    ax_chart.set_ylabel(f"{metric_label} Business Days in Queue (stacked per step)", fontsize=10, color="#333")
    ax_chart.spines[["top", "right"]].set_visible(False)
    ax_chart.grid(axis="y", linestyle="--", alpha=0.3, zorder=0)

    ax_chart.set_xticks(x)
    ax_chart.set_xticklabels([q.replace(" ", "\n") for q in quarter_order], fontsize=8.5)

    for yr in years:
        idxs = [i for i, q in enumerate(quarter_order) if q.startswith(str(yr))]
        if not idxs:
            continue
        ax_chart.text(np.mean(idxs), max_days * 1.14,
                      str(yr) + (" *" if yr == 2026 else ""),
                      ha="center", fontsize=11, fontweight="bold", color="#333")
        if idxs[0] > 0:
            ax_chart.axvline(idxs[0] - 0.5, color="#bbb", linewidth=1.0,
                             linestyle="--", zorder=1)

    hatch_h = [plt.Rectangle((0,0),1,1, color=queue_colors[q], alpha=0.70, hatch="///")
               for q in target_queues]
    ax_chart.legend(hatch_h, target_queues, title="Queue",
                    fontsize=8.5, loc="upper left", framealpha=0.92, title_fontsize=9)
    ax_chart.set_title(
        f"Review Queue: {metric_label} Business Days per Step  |  2024 · 2025 · 2026",
        fontsize=14, fontweight="bold", color="#1C1C1C", pad=14
    )
    ax_chart.annotate("* 2026 is a partial year  |  bars are stacked — total height = sum of per-queue days, not end-to-end grant duration",
                      xy=(1.0, -0.09), xycoords="axes fraction",
                      ha="right", fontsize=7.5, color="#888", style="italic")

    # ── table ─────────────────────────────────────────────────────────────
    annual_days = {}
    for yr in years:
        yr_qs = [q for q in quarter_order if q.startswith(str(yr))]
        for queue in target_queues:
            dv = pivot_days.loc[yr_qs, queue].values
            valid = dv[dv > 0]
            if metric == "mean":
                annual_days[(yr, queue)] = round(float(valid.mean()), 1) if len(valid) else 0.0
            else:
                annual_days[(yr, queue)] = round(float(np.median(valid)), 1) if len(valid) else 0.0

    col_headers = quarter_order + [f"{yr} Total" for yr in years]
    row_labels  = [f"{q}  {metric_label} Days" for q in target_queues]

    cell_text = []
    for queue in target_queues:
        row = []
        for q in quarter_order:
            d = pivot_days.loc[q, queue] if q in pivot_days.index else 0
            row.append(f"{d:.1f}" if d > 0 else "—")
        for yr in years:
            v = annual_days[(yr, queue)]
            row.append(f"{v:.1f}" if v > 0 else "—")
        cell_text.append(row)

    ax_table.set_visible(True)
    ax_table.axis("off")

    days_tint       = "#F5F5F0"
    days_total_tint = "#E8E8E2"
    row_colors = [
        [days_tint] * len(quarter_order) + [days_total_tint] * len(years)
        for _ in target_queues
    ]

    tbl = ax_table.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=col_headers,
        cellColours=row_colors,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.85)

    import matplotlib.colors as mcolors
    for (r, c), cell in tbl.get_celld().items():
        cell.set_linewidth(0.3)
        cell.set_edgecolor("#ccc")
        if r == 0:
            cell.set_facecolor("#2C2C2C")
            cell.set_text_props(color="white", fontweight="bold", fontsize=7)
        if c == -1 and r >= 1:
            queue_name = target_queues[r - 1]
            base_rgb = mcolors.to_rgb(queue_colors[queue_name])
            faded = tuple(ch * 0.55 + 0.45 for ch in base_rgb)
            cell.set_facecolor(faded)
            cell.set_text_props(color="white", fontweight="bold", fontsize=7.5)
            cell.set_width(0.09)
        if c >= len(quarter_order) and r > 0:
            cell.set_text_props(fontweight="bold")

    fig.text(0.99, 0.01, "* 2026 is a partial year",
             ha="right", fontsize=8, color="#888", style="italic")

    out = f"outputs/processing_time_only_{fname_suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"saved: {out}")