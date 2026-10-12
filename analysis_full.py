from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ====================================
# 1. 데이터 불러오기
# ====================================

BASE = Path(__file__).resolve().parent
RESULTS = BASE / "results"

df = pd.read_csv(RESULTS / "full_runs.csv")

df = df[df["status"] == "completed"].copy()

print("=== 통합 데이터 분석 시작 ===")
print("분석 실험 수:", len(df))

# ====================================
# 2. 실험 조건별 평균 분석
# ====================================

summary = (
    df.groupby(["traffic", "av_ratio"])
    .agg(
        mean_travel_time=("mean_travel_time", "mean"),
        std_travel_time=("mean_travel_time", "std"),
        mean_headway_std=("headway_std", "mean"),
        mean_wait=("mean_intersection_wait", "mean")
    )
    .reset_index()
)

summary.to_csv(
    RESULTS / "analysis_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

# ====================================
# 3. 동일 시드 대응 비교
# ====================================

comparison = []

rng = np.random.default_rng(42)

for traffic in sorted(df["traffic"].unique()):

    group = df[df["traffic"] == traffic]

    baseline = group[group["av_ratio"] == 0][
        ["seed", "mean_travel_time"]
    ].rename(columns={
        "mean_travel_time": "baseline_time"
    })

    for ratio in [25, 50, 75, 100]:

        target = group[group["av_ratio"] == ratio][
            ["seed", "mean_travel_time"]
        ].rename(columns={
            "mean_travel_time": "target_time"
        })

        paired = baseline.merge(
            target, on="seed"
        )

        differences = (
            paired["baseline_time"]
            - paired["target_time"]
        ).to_numpy()

        if len(differences) < 2:
            continue

        # 부트스트랩 신뢰구간
        bootstrap_means = []

        for _ in range(10000):

            sample = rng.choice(
                differences,
                size=len(differences),
                replace=True
            )

            bootstrap_means.append(
                np.mean(sample)
            )

        ci_low, ci_high = np.percentile(
            bootstrap_means,
            [2.5, 97.5]
        )

        comparison.append({
            "traffic": traffic,
            "av_ratio": ratio,
            "mean_reduction_s": np.mean(differences),
            "ci_low": ci_low,
            "ci_high": ci_high,
            "paired_runs": len(differences)
        })

comparison_df = pd.DataFrame(comparison)

comparison_df.to_csv(
    RESULTS / "statistical_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== AV 0% 대비 운행시간 변화 ===")
print(comparison_df.round(3).to_string(index=False))

# ====================================
# 4. 운행시간 비교 그래프
# ====================================

plt.figure(figsize=(9, 5))

for traffic, group in summary.groupby("traffic"):

    group = group.sort_values("av_ratio")

    plt.plot(
        group["av_ratio"],
        group["mean_travel_time"],
        marker="o",
        label=f"Traffic {traffic}"
    )

plt.xlabel("Autonomous Vehicle Ratio (%)")
plt.ylabel("Mean Bus Travel Time (s)")
plt.title("Bus Travel Time by AV Ratio")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS / "full_travel_time.png",
    dpi=300
)
plt.close()

# ====================================
# 5. 배차 안정성 비교 그래프
# ====================================

plt.figure(figsize=(9, 5))

for traffic, group in summary.groupby("traffic"):

    group = group.sort_values("av_ratio")

    plt.plot(
        group["av_ratio"],
        group["mean_headway_std"],
        marker="o",
        label=f"Traffic {traffic}"
    )

plt.xlabel("Autonomous Vehicle Ratio (%)")
plt.ylabel("Headway Standard Deviation (s)")
plt.title("Bus Headway Variability")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS / "full_headway.png",
    dpi=300
)
plt.close()

# ====================================
# 6. TTC 위험 노출률
# ====================================

risk_rows = []

for (traffic, ratio), group in df.groupby(
    ["traffic", "av_ratio"]
):

    human_steps = group["human_leader_steps"].sum()
    av_steps = group["av_leader_steps"].sum()

    human_risk = group["human_risk_steps"].sum()
    av_risk = group["av_risk_steps"].sum()

    risk_rows.append({
        "traffic": traffic,
        "av_ratio": ratio,
        "human_risk_pct": (
            100 * human_risk / human_steps
            if human_steps > 0 else np.nan
        ),
        "av_risk_pct": (
            100 * av_risk / av_steps
            if av_steps > 0 else np.nan
        )
    })

risk_df = pd.DataFrame(risk_rows)

risk_df.to_csv(
    RESULTS / "full_risk_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

plt.figure(figsize=(9, 5))

for traffic, group in risk_df.groupby("traffic"):

    plt.plot(
        group["av_ratio"],
        group["human_risk_pct"],
        marker="o",
        label=f"Human Leader (Traffic {traffic})"
    )

    plt.plot(
        group["av_ratio"],
        group["av_risk_pct"],
        marker="s",
        linestyle="--",
        label=f"AV Leader (Traffic {traffic})"
    )

plt.xlabel("Autonomous Vehicle Ratio (%)")
plt.ylabel("TTC < 2 s Exposure (%)")
plt.title("TTC Risk Exposure by Leader Type")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS / "full_ttc_risk.png",
    dpi=300
)
plt.close()

# ====================================
# 7. 신뢰구간 비교 그래프
# ====================================

plt.figure(figsize=(9, 5))

for traffic, group in comparison_df.groupby("traffic"):

    group = group.sort_values("av_ratio")

    lower_error = (
        group["mean_reduction_s"] - group["ci_low"]
    )

    upper_error = (
        group["ci_high"] - group["mean_reduction_s"]
    )

    plt.errorbar(
        group["av_ratio"],
        group["mean_reduction_s"],
        yerr=[lower_error, upper_error],
        marker="o",
        capsize=4,
        label=f"Traffic {traffic}"
    )

plt.axhline(0, linestyle="--", color="gray")

plt.xlabel("Autonomous Vehicle Ratio (%)")
plt.ylabel("Travel Time Reduction vs 0% (s)")
plt.title("Paired Bootstrap 95% Confidence Intervals")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(
    RESULTS / "full_confidence_interval.png",
    dpi=300
)
plt.close()

print("\n=== 분석 완료 ===")
print("결과 저장 위치:", RESULTS)
print("그래프 4개 생성 완료!")
