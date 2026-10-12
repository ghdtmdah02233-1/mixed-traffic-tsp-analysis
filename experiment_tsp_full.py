from pathlib import Path
import random
import xml.etree.ElementTree as ET

import pandas as pd
import traci
import sumolib

# =====================================
# 1. 기본 설정
# =====================================

BASE = Path(__file__).resolve().parent
SUMO_DIR = BASE / "sumo"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

SUMO_EXE = (
    r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
)

NETWORK = SUMO_DIR / "mixed_traffic_signal.net.xml"
STOPS = SUMO_DIR / "bus_stops.add.xml"
ROUTES = SUMO_DIR / "tsp_full_test.rou.xml"

TRAFFIC_LEVELS = [100, 200]
AV_RATIOS = [0, 25, 50, 75, 100]

SEEDS = list(range(42, 52))

VERTICAL_TRAFFIC = 60
DEMAND_DURATION = 480
MAX_TIME = 1500

BUS_COUNT = 5
BUS_IDS = [f"bus_{i}" for i in range(BUS_COUNT)]

TLS_ID = "J5"
APPROACH_DISTANCE = 100
DELAY_THRESHOLD = 3
MAX_EXTENSION = 5
BASE_GREEN = 30

EDGE_LENGTH = sumolib.net.readNet(
    str(NETWORK)
).getEdge("E0").getLength()

OUTPUT = RESULTS / "tsp_full_results.csv"
EVENT_OUTPUT = RESULTS / "tsp_full_events.csv"

for path in [Path(SUMO_EXE), NETWORK, STOPS]:
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")


# =====================================
# 2. 차량 경로 생성
# =====================================

def create_routes(traffic, ratio, seed):

    rng = random.Random(seed)
    root = ET.Element("routes")

    ET.SubElement(root, "vType", {
        "id": "human",
        "vClass": "passenger",
        "accel": "2.6",
        "decel": "4.5",
        "sigma": "0.5",
        "tau": "1.2"
    })

    ET.SubElement(root, "vType", {
        "id": "autonomous",
        "vClass": "passenger",
        "accel": "2.0",
        "decel": "4.5",
        "sigma": "0",
        "tau": "1.5"
    })

    ET.SubElement(root, "vType", {
        "id": "bus",
        "vClass": "bus",
        "length": "12",
        "accel": "1.2",
        "decel": "4"
    })

    ET.SubElement(root, "route", {
        "id": "horizontal",
        "edges": "E0 E0.672"
    })

    ET.SubElement(root, "route", {
        "id": "vertical",
        "edges": "E2 E3"
    })

    vehicles = []

    av_count = round(traffic * ratio / 100)
    assignment = list(range(traffic))
    rng.shuffle(assignment)
    av_indices = set(assignment[:av_count])

    # 가로 방향 차량
    for i in range(traffic):

        vtype = (
            "autonomous"
            if i in av_indices
            else "human"
        )

        vehicles.append(ET.Element("vehicle", {
            "id": f"h_{i}",
            "type": vtype,
            "route": "horizontal",
            "depart": f"{i * DEMAND_DURATION / traffic:.2f}"
        }))

    # 세로 방향 차량
    for i in range(VERTICAL_TRAFFIC):

        vehicles.append(ET.Element("vehicle", {
            "id": f"v_{i}",
            "type": "human",
            "route": "vertical",
            "depart": f"{i * DEMAND_DURATION / VERTICAL_TRAFFIC:.2f}"
        }))

    # 버스 5대
    for i in range(BUS_COUNT):

        bus = ET.Element("vehicle", {
            "id": f"bus_{i}",
            "type": "bus",
            "route": "horizontal",
            "depart": str(60 + i * 60),
            "departLane": "0"
        })

        ET.SubElement(bus, "stop", {
            "busStop": "bs_0",
            "duration": "15"
        })

        vehicles.append(bus)

    vehicles.sort(
        key=lambda v: float(v.get("depart"))
    )
    root.extend(vehicles)

    tree = ET.ElementTree(root)
    ET.indent(tree, space="    ")
    tree.write(
        ROUTES,
        encoding="utf-8",
        xml_declaration=True
    )

    return ROUTES


# =====================================
# 3. 정책별 시뮬레이션
# =====================================

def run_policy(traffic, ratio, seed, policy):

    departures = {}
    arrivals = {}
    events = []

    extended_this_green = False

    traci.start([
        SUMO_EXE,
        "-n", str(NETWORK),
        "-r", str(ROUTES),
        "-a", str(STOPS),
        "--step-length", "1",
        "--seed", str(seed),
        "--no-step-log", "true",
        "--time-to-teleport", "-1"
    ])

    try:
        # 신호 프로그램 확인
        logic = traci.trafficlight.getAllProgramLogics(
            TLS_ID
        )[0]

        states = [p.state for p in logic.phases]

        if (
            len(states) != 4
            or states[0] != "GGGr"
            or states[1] != "yyyr"
            or states[2] != "rrrG"
            or states[3] != "rrry"
        ):
            raise RuntimeError(
                f"예상과 다른 신호 프로그램: {states}"
            )

        for step in range(MAX_TIME):

            traci.simulationStep()
            now = traci.simulation.getTime()

            for vid in traci.simulation.getDepartedIDList():
                departures[vid] = now

            for vid in traci.simulation.getArrivedIDList():
                arrivals[vid] = now

            if policy == "TSP":

                phase = traci.trafficlight.getPhase(TLS_ID)

                if phase != 0:
                    extended_this_green = False

                if phase == 0 and not extended_this_green:

                    remaining = (
                        traci.trafficlight.getNextSwitch(TLS_ID)
                        - now
                    )

                    spent = (
                        traci.trafficlight.getSpentDuration(
                            TLS_ID
                        )
                    )

                    if (
                        0 < remaining <= MAX_EXTENSION
                        and spent >= BASE_GREEN - MAX_EXTENSION
                    ):

                        active = set(
                            traci.vehicle.getIDList()
                        )

                        for bid in BUS_IDS:

                            if bid not in active:
                                continue

                            if traci.vehicle.getRoadID(bid) != "E0":
                                continue

                            pos = traci.vehicle.getLanePosition(
                                bid
                            )

                            distance = EDGE_LENGTH - pos

                            loss = traci.vehicle.getTimeLoss(
                                bid
                            )

                            if (
                                0 <= distance <= APPROACH_DISTANCE
                                and loss >= DELAY_THRESHOLD
                            ):

                                extra = min(
                                    MAX_EXTENSION,
                                    max(
                                        0,
                                        BASE_GREEN + MAX_EXTENSION
                                        - (spent + remaining)
                                    )
                                )

                                if extra <= 0:
                                    continue

                                traci.trafficlight.setPhaseDuration(
                                    TLS_ID,
                                    remaining + extra
                                )

                                extended_this_green = True

                                events.append({
                                    "traffic": traffic,
                                    "av_ratio": ratio,
                                    "seed": seed,
                                    "time": now,
                                    "bus_id": bid,
                                    "extension_s": extra
                                })

                                break

            # 전체 차량 운행 완료 여부 확인
            if (
                now >= DEMAND_DURATION
                and traci.simulation.getMinExpectedNumber() == 0
            ):
                break

    finally:
        traci.close()

    # =====================================
    # 4. 차량별 운행시간 계산
    # =====================================

    groups = {
        "bus": [],
        "horizontal": [],
        "vertical": []
    }

    for vid, dep in departures.items():

        if vid not in arrivals:
            continue

        travel = arrivals[vid] - dep

        if vid.startswith("bus_"):
            groups["bus"].append(travel)
        elif vid.startswith("h_"):
            groups["horizontal"].append(travel)
        elif vid.startswith("v_"):
            groups["vertical"].append(travel)

    # 버스 도착 간격
    bus_arrivals = [
        arrivals[bid]
        for bid in BUS_IDS
        if bid in arrivals
    ]

    headway_std = None

    if len(bus_arrivals) == BUS_COUNT:

        import statistics

        bus_arrivals.sort()

        headways = [
            bus_arrivals[i] - bus_arrivals[i - 1]
            for i in range(1, BUS_COUNT)
        ]

        headway_std = statistics.stdev(headways)

    def mean_or_none(values):
        return sum(values) / len(values) if values else None

    result = {
        "traffic": traffic,
        "av_ratio": ratio,
        "seed": seed,
        "policy": policy,
        "completed_buses": len(groups["bus"]),
        "completed_horizontal": len(groups["horizontal"]),
        "completed_vertical": len(groups["vertical"]),
        "mean_bus_time": mean_or_none(groups["bus"]),
        "mean_horizontal_time": mean_or_none(
            groups["horizontal"]
        ),
        "mean_vertical_time": mean_or_none(
            groups["vertical"]
        ),
        "headway_std": headway_std,
        "tsp_count": len(events),
        "total_extension_s": sum(
            e["extension_s"] for e in events
        ),
        "status": (
            "completed"
            if (
                len(groups["bus"]) == BUS_COUNT
                and len(groups["horizontal"]) == traffic
                and len(groups["vertical"]) == VERTICAL_TRAFFIC
            )
            else "incomplete"
        )
    }

    return result, events


# =====================================
# 5. A/B 반복 실험
# =====================================

results = []
all_events = []

total = (
    len(TRAFFIC_LEVELS)
    * len(AV_RATIOS)
    * len(SEEDS)
    * 2
)

count = 0

print("=== TSP 정책 비교 실험 시작 ===")
print("총 실험 횟수:", total)

for traffic in TRAFFIC_LEVELS:
    for ratio in AV_RATIOS:
        for seed in SEEDS:

            # A와 B가 완전히 동일한 경로 파일 사용
            create_routes(traffic, ratio, seed)

            for policy in ["BASE", "TSP"]:

                count += 1

                print(
                    f"[{count}/{total}] "
                    f"Traffic={traffic}, AV={ratio}%, "
                    f"Seed={seed}, Policy={policy}",
                    flush=True
                )

                result, events = run_policy(
                    traffic, ratio, seed, policy
                )

                results.append(result)
                all_events.extend(events)

                # 중간 저장
                pd.DataFrame(results).to_csv(
                    OUTPUT,
                    index=False,
                    encoding="utf-8-sig"
                )

                pd.DataFrame(
                    all_events,
                    columns=[
                        "traffic", "av_ratio", "seed",
                        "time", "bus_id", "extension_s"
                    ]
                ).to_csv(
                    EVENT_OUTPUT,
                    index=False,
                    encoding="utf-8-sig"
                )

                print(
                    "버스 평균:",
                    result["mean_bus_time"],
                    "| TSP 횟수:",
                    result["tsp_count"],
                    "| 상태:",
                    result["status"],
                    flush=True
                )

print("\n=== 전체 정책 실험 완료 ===")

df = pd.DataFrame(results)

summary = (
    df.groupby(
        ["traffic", "av_ratio", "policy"]
    )
    .agg(
        mean_bus_time=("mean_bus_time", "mean"),
        mean_headway_std=("headway_std", "mean"),
        mean_horizontal_time=(
            "mean_horizontal_time", "mean"
        ),
        mean_vertical_time=(
            "mean_vertical_time", "mean"
        ),
        mean_tsp_count=("tsp_count", "mean"),
        valid_runs=(
            "status",
            lambda x: (x == "completed").sum()
        )
    )
    .reset_index()
)

summary.to_csv(
    RESULTS / "tsp_full_summary.csv",
    index=False,
    encoding="utf-8-sig"
)

print(summary.to_string(index=False))
print("결과 저장 완료!")
