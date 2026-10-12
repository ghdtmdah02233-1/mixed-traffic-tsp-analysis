from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================================
# 1. 파일 경로
# ====================================

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

FILE = RESULTS / "tsp_full_results.csv"

df = pd.read_csv(FILE)

# 정상 완료한 실험만 사용
df = df[df["status"] == "completed"].copy()

print("=== TSP 정책 효과 분석 ===")
print("정상 완료 실험:", len(df))

# ====================================
# 2. 동일 조건 BASE / TSP 대응 비교
# ====================================

keys = ["traffic", "av_ratio", "seed"]

metrics = [
    "mean_bus_time",
    "headway_std",
    "mean_horizontal_time",
    "mean_vertical_time"
]

base = df[df["policy"] == "BASE"][
    keys + metrics
].copy()

tsp = df[df["policy"] == "TSP"][
    keys + metrics + ["tsp_count"]
].copy()

paired = base.merge(
    tsp,
    on=keys,
    suffixes=("_base", "_tsp"),
    validate="one_to_one"
)

print("정상 대응 실험 쌍:", len(paired))

# 양수 = 버스 시간 단축
paired["bus_saved_s"] = (
    paired["mean_bus_time_base"]
    - paired["mean_bus_time_tsp"]
)

# 양수 = 배차간격 표준편차 감소
paired["headway_improvement_s"] = (
    paired["headway_std_base"]
    - paired["headway_std_tsp"]
)

# 양수 = 가로 차량 시간 단축
paired["horizontal_saved_s"] = (
    paired["mean_horizontal_time_base"]
    - paired["mean_horizontal_time_tsp"]
)

# 양수 = 세로 차량의 추가 통행시간
paired["vertical_extra_s"] = (
    paired["mean_vertical_time_tsp"]
    - paired["mean_vertical_time_base"]
)

# ====================================
# 3. 차량 전체의 시간 편익
# ====================================

# 각 시나리오의 차량 대수
BUS_COUNT = 5
VERTICAL_COUNT = 60

# 양수 = 전체 차량-초 절약
paired["total_vehicle_seconds_saved"] = (
    BUS_COUNT * paired["bus_saved_s"]
    + paired["traffic"] * paired["horizontal_saved_s"]
    - VERTICAL_COUNT * paired["vertical_extra_s"]
)

# ====================================
# 4. 부트스트랩 신뢰구간
# ====================================

rng = np.random.default_rng(42)

def bootstrap_ci(values, n_boot=10000):

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if len(values) < 2:
        return (np.nan, np.nan)

    sample_indices = rng.integers(
        0, len(values),
        size=(n_boot, len(values))
    )

    boot_means = values[sample_indices].mean(axis=1)

    return np.percentile(
        boot_means, [2.5, 97.5]
    )

# ====================================
# 5. 조건별 정책 효과 요약
# ====================================

metric_names = [
    "bus_saved_s",
    "headway_improvement_s",
    "horizontal_saved_s",
    "vertical_extra_s",
    "total_vehicle_seconds_saved"
]

summary_rows = []

for (traffic, ratio), group in paired.groupby(
    ["traffic", "av_ratio"]
):

    row = {
        "traffic": traffic,
        "av_ratio": ratio,
        "paired_runs": len(group),
        "mean_tsp_count": group["tsp_count"].mean()
    }

    for metric in metric_names:

        values = group[metric].dropna().to_numpy()

        row[f"{metric}_mean"] = values.mean()

        low, high = bootstrap_ci(values)

        row[f"{metric}_ci_low"] = low
        row[f"{metric}_ci_high"] = high

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

# ====================================
# 6. CSV 저장
# ====================================

paired.to_csv(
    RESULTS / "tsp_paired_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

summary.to_csv(
    RESULTS / "tsp_policy_analysis.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== 조건별 평균 정책 효과 ===")

display_cols = [
    "traffic",
    "av_ratio",
    "paired_runs",
    "bus_saved_s_mean",
    "headway_improvement_s_mean",
    "vertical_extra_s_mean",
    "total_vehicle_seconds_saved_mean"
]

print(
    summary[display_cols].round(2).to_string(
        index=False
    )
)

# ====================================
# 7. 정책 효과 그래프
# ====================================

def draw_metric(metric, ylabel, title, filename):

    plt.figure(figsize=(9, 5))

    for traffic, group in summary.groupby("traffic"):

        group = group.sort_values("av_ratio")

        means = group[f"{metric}_mean"].to_numpy()
        lows = group[f"{metric}_ci_low"].to_numpy()
        highs = group[f"{metric}_ci_high"].to_numpy()

        errors = [
            np.maximum(0, means - lows),
            np.maximum(0, highs - means)
        ]

        plt.errorbar(
            group["av_ratio"],
            means,
            yerr=errors,
            marker="o",
            capsize=4,
            label=f"Traffic {traffic}"
        )

    plt.axhline(
        0,
        color="gray",
        linestyle="--"
    )

    plt.xlabel("Autonomous Vehicle Ratio (%)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    plt.savefig(
        RESULTS / filename,
        dpi=300
    )

    plt.close()


draw_metric(
    "bus_saved_s",
    "Bus Travel Time Saved (s)",
    "TSP: Bus Travel Time Reduction",
    "tsp_bus_savings.png"
)

draw_metric(
    "headway_improvement_s",
    "Headway Variability Reduction (s)",
    "TSP: Bus Headway Stability",
    "tsp_headway_improvement.png"
)

draw_metric(
    "vertical_extra_s",
    "Extra Travel Time per Vehicle (s)",
    "TSP: Cross-Traffic Delay",
    "tsp_cross_traffic_delay.png"
)

draw_metric(
    "total_vehicle_seconds_saved",
    "Total Vehicle-Seconds Saved",
    "TSP: Net Vehicle Travel Time Benefit",
    "tsp_net_benefit.png"
)

# ====================================
# 8. 전체 결과 확인
# ====================================

print("\n=== 전체 대응 실험 평균 ===")

for metric in metric_names:
    print(
        metric,
        ":",
        round(paired[metric].mean(), 3)
    )

print("\n=== 저장 완료 ===")
print("tsp_paired_comparison.csv")
print("tsp_policy_analysis.csv")
print("tsp_bus_savings.png")
print("tsp_headway_improvement.png")
print("tsp_cross_traffic_delay.png")
print("tsp_net_benefit.png")
