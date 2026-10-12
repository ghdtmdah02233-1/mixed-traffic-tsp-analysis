from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# =========================================
# 1. 경로 및 기본 설정
# =========================================

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

INPUT = RESULTS / "tsp_full_results.csv"

BUS_COUNT = 5
VERTICAL_COUNT = 60

CAR_OCCUPANCY = 1.3
BUS_OCCUPANCIES = [10, 20, 30]

rng = np.random.default_rng(42)

# =========================================
# 2. 기존 실험 결과 불러오기
# =========================================

df = pd.read_csv(INPUT)

# 기존 결과는 status가 "complete"일 수 있음
df = df[
    df["status"].isin(["complete", "completed"])
].copy()

keys = ["traffic", "av_ratio", "seed"]

metrics = [
    "mean_bus_time",
    "mean_horizontal_time",
    "mean_vertical_time"
]

base = df[df["policy"] == "BASE"][
    keys + metrics
].copy()

tsp = df[df["policy"] == "TSP"][
    keys + metrics
].copy()

paired = base.merge(
    tsp,
    on=keys,
    suffixes=("_base", "_tsp"),
    validate="one_to_one"
)

if paired.empty:
    raise ValueError(
        "BASE/TSP 대응 실험이 없습니다. "
        "CSV의 status와 policy를 확인하세요."
    )

print("정상 대응 비교:", len(paired), "쌍")

# =========================================
# 3. 정책 적용 전후 시간 차이
# =========================================

paired["bus_saved"] = (
    paired["mean_bus_time_base"]
    - paired["mean_bus_time_tsp"]
)

paired["horizontal_saved"] = (
    paired["mean_horizontal_time_base"]
    - paired["mean_horizontal_time_tsp"]
)

paired["vertical_extra"] = (
    paired["mean_vertical_time_tsp"]
    - paired["mean_vertical_time_base"]
)

# =========================================
# 4. 부트스트랩 신뢰구간
# =========================================

def bootstrap_ci(values, n_boot=10000):

    values = np.asarray(values, dtype=float)

    indices = rng.integers(
        0,
        len(values),
        size=(n_boot, len(values))
    )

    means = values[indices].mean(axis=1)

    return np.percentile(
        means,
        [2.5, 97.5]
    )

# =========================================
# 5. 승객 기준 순편익 계산
# =========================================

all_rows = []

for occupancy in BUS_OCCUPANCIES:

    temp = paired.copy()

    # 버스 승객의 총 시간 절약
    temp["bus_person_saved"] = (
        BUS_COUNT
        * occupancy
        * temp["bus_saved"]
    )

    # 가로 방향 승용차 승객 시간 절약
    temp["horizontal_person_saved"] = (
        temp["traffic"]
        * CAR_OCCUPANCY
        * temp["horizontal_saved"]
    )

    # 세로 방향 승용차 승객 추가 시간
    temp["vertical_person_extra"] = (
        VERTICAL_COUNT
        * CAR_OCCUPANCY
        * temp["vertical_extra"]
    )

    # 전체 승객의 순편익
    temp["net_person_saved"] = (
        temp["bus_person_saved"]
        + temp["horizontal_person_saved"]
        - temp["vertical_person_extra"]
    )

    for (traffic, av), group in temp.groupby(
        ["traffic", "av_ratio"]
    ):

        values = group[
            "net_person_saved"
        ].to_numpy()

        low, high = bootstrap_ci(values)

        all_rows.append({
            "traffic": traffic,
            "av_ratio": av,
            "bus_occupancy": occupancy,
            "car_occupancy": CAR_OCCUPANCY,
            "paired_runs": len(group),
            "net_person_seconds_mean": values.mean(),
            "ci_low": low,
            "ci_high": high
        })

summary = pd.DataFrame(all_rows)

# =========================================
# 6. CSV 저장
# =========================================

OUTPUT = (
    RESULTS / "tsp_passenger_benefit.csv"
)

summary.to_csv(
    OUTPUT,
    index=False,
    encoding="utf-8-sig"
)

print("\n=== 승객 기준 순편익 ===")

print(
    summary[
        [
            "traffic",
            "av_ratio",
            "bus_occupancy",
            "net_person_seconds_mean",
            "ci_low",
            "ci_high"
        ]
    ].round(2).to_string(index=False)
)

# =========================================
# 7. 그래프 2개 생성
# =========================================

for traffic in sorted(
    summary["traffic"].unique()
):

    plt.figure(figsize=(9, 5))

    group = summary[
        summary["traffic"] == traffic
    ]

    for occupancy in BUS_OCCUPANCIES:

        sub = group[
            group["bus_occupancy"] == occupancy
        ].sort_values("av_ratio")

        means = sub[
            "net_person_seconds_mean"
        ].to_numpy()

        lows = sub["ci_low"].to_numpy()
        highs = sub["ci_high"].to_numpy()

        errors = [
            np.maximum(0, means - lows),
            np.maximum(0, highs - means)
        ]

        plt.errorbar(
            sub["av_ratio"],
            means,
            yerr=errors,
            marker="o",
            capsize=4,
            label=f"Bus: {occupancy} passengers"
        )

    plt.axhline(
        0,
        color="gray",
        linestyle="--"
    )

    plt.xlabel("AV Ratio (%)")
    plt.ylabel("Net Person-Seconds Saved")

    plt.title(
        f"TSP Passenger Benefit - Traffic {traffic}"
    )

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        RESULTS / (
            f"tsp_passenger_traffic_{traffic}.png"
        ),
        dpi=300
    )

    plt.close()

# =========================================
# 8. 손익분기 버스 탑승인원 계산
# =========================================

# 순편익이 0이 되는 최소 평균 버스 탑승인원
# 평균 효과 기준이므로 신뢰구간과 구별해야 함

break_even_rows = []

for (traffic, av), group in paired.groupby(
    ["traffic", "av_ratio"]
):

    mean_bus = group["bus_saved"].mean()

    car_benefit = (
        traffic
        * CAR_OCCUPANCY
        * group["horizontal_saved"].mean()
        - VERTICAL_COUNT
        * CAR_OCCUPANCY
        * group["vertical_extra"].mean()
    )

    if mean_bus > 0:

        break_even = (
            -car_benefit
            / (BUS_COUNT * mean_bus)
        )

        # 탑승인원은 음수가 될 수 없음
        break_even = max(0, break_even)

    else:
        break_even = np.nan

    break_even_rows.append({
        "traffic": traffic,
        "av_ratio": av,
        "break_even_bus_occupancy": break_even
    })

break_even_df = pd.DataFrame(
    break_even_rows
)

break_even_df.to_csv(
    RESULTS / "tsp_break_even_occupancy.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\n=== 손익분기 버스 탑승인원 ===")

print(
    break_even_df.round(2).to_string(
        index=False
    )
)

print("\n분석 완료!")
print("저장 파일:")
print("tsp_passenger_benefit.csv")
print("tsp_break_even_occupancy.csv")
print("tsp_passenger_traffic_100.png")
print("tsp_passenger_traffic_200.png")
