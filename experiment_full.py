from pathlib import Path
import random
import xml.etree.ElementTree as ET
import statistics

import pandas as pd
import traci
import sumolib

# ==========================================
# 1. 기본 설정
# ==========================================

BASE = Path(__file__).resolve().parent
SUMO_DIR = BASE / "sumo"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)

SUMO_EXE = (
    r"C:\Program Files (x86)\Eclipse\Sumo\bin\sumo.exe"
)

NETWORK = SUMO_DIR / "mixed_traffic_signal.net.xml"
STOPS = SUMO_DIR / "bus_stops.add.xml"
ROUTES = SUMO_DIR / "full_experiment.rou.xml"

TRAFFIC_LEVELS = [100, 200]
AV_RATIOS = [0, 25, 50, 75, 100]
SEEDS = list(range(42, 52))

VERTICAL_TRAFFIC = 60
DEMAND_DURATION = 480
MAX_TIME = 1200

BUS_COUNT = 5
BUS_INTERVAL = 60
BUS_START = 60

TTC_LIMIT = 2.0
MIN_SPEED = 5 / 3.6
BRAKING_LIMIT = -3.0

BUS_IDS = [
    f"bus_{i}" for i in range(BUS_COUNT)
]

OUTPUT_RUNS = RESULTS / "full_runs.csv"
OUTPUT_BUSES = RESULTS / "full_buses.csv"
OUTPUT_SUMMARY = RESULTS / "full_summary.csv"

# 기존 파일 확인
for path in [Path(SUMO_EXE), NETWORK, STOPS]:
    if not path.exists():
        raise FileNotFoundError(f"파일 없음: {path}")

net = sumolib.net.readNet(str(NETWORK))

if not net.getEdge("E0").getConnections(
    net.getEdge("E0.672")
):
    raise RuntimeError("가로 도로 연결 오류")

if not net.getEdge("E2").getConnections(
    net.getEdge("E3")
):
    raise RuntimeError("세로 도로 연결 오류")

EDGE_LENGTH = net.getEdge("E0").getLength()


# ==========================================
# 2. 차량 경로 자동 생성
# ==========================================

def create_routes(traffic, ratio, seed):

    rng = random.Random(seed)

    root = ET.Element("routes")

    ET.SubElement(root, "vType", {
        "id": "human",
        "vClass": "passenger",
        "accel": "2.6",
        "decel": "4.5",
        "sigma": "0.5",
        "tau": "1.2",
        "color": "1,0.6,0"
    })

    ET.SubElement(root, "vType", {
        "id": "autonomous",
        "vClass": "passenger",
        "accel": "2.0",
        "decel": "4.5",
        "sigma": "0",
        "tau": "1.5",
        "color": "0,0.8,0.4"
    })

    ET.SubElement(root, "vType", {
        "id": "bus",
        "vClass": "bus",
        "length": "12",
        "accel": "1.2",
        "decel": "4",
        "color": "0,0.4,1"
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

    # 같은 seed에서 AV 비율이 증가하면
    # 기존 AV 차량은 그대로 유지되도록 배정
    assignment = list(range(traffic))
    rng.shuffle(assignment)
    av_indices = set(assignment[:av_count])

    # 가로 방향 차량
    for i in range(traffic):

        vehicle_type = (
            "autonomous"
            if i in av_indices
            else "human"
        )

        depart = i * DEMAND_DURATION / traffic

        vehicles.append(ET.Element("vehicle", {
            "id": f"h_{i}",
            "type": vehicle_type,
            "route": "horizontal",
            "depart": f"{depart:.2f}"
        }))

    # 세로 방향 차량
    for i in range(VERTICAL_TRAFFIC):

        depart = i * DEMAND_DURATION / VERTICAL_TRAFFIC

        vehicles.append(ET.Element("vehicle", {
            "id": f"v_{i}",
            "type": "human",
            "route": "vertical",
            "depart": f"{depart:.2f}"
        }))

    # 버스 5대
    for i in range(BUS_COUNT):

        depart = BUS_START + i * BUS_INTERVAL

        bus = ET.Element("vehicle", {
            "id": f"bus_{i}",
            "type": "bus",
            "route": "horizontal",
            "depart": str(depart),
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


# ==========================================
# 3. 개별 실험
# ==========================================

def run_experiment(traffic, ratio, seed):

    route_file = create_routes(traffic, ratio, seed)

    buses = {}

    for i, bid in enumerate(BUS_IDS):
        buses[bid] = {
            "scheduled": BUS_START + i * BUS_INTERVAL,
            "depart": None,
            "arrival": None,
            "stop": 0,
            "intersection_wait": 0,
            "braking": 0,
            "was_braking": False,
            "human_steps": 0,
            "av_steps": 0,
            "human_risk": 0,
            "av_risk": 0,
            "human_events": 0,
            "av_events": 0,
            "risk_active": False,
            "risk_leader": None
        }

    started = False

    try:
        traci.start([
            SUMO_EXE,
            "-n", str(NETWORK),
            "-r", str(route_file),
            "-a", str(STOPS),
            "--step-length", "1",
            "--seed", str(seed),
            "--no-step-log", "true",
            "--time-to-teleport", "-1"
        ])

        started = True

        for step in range(MAX_TIME):

            traci.simulationStep()
            now = traci.simulation.getTime()

            departed = set(
                traci.simulation.getDepartedIDList()
            )
            arrived = set(
                traci.simulation.getArrivedIDList()
            )
            active = set(
                traci.vehicle.getIDList()
            )

            for bid in BUS_IDS:

                b = buses[bid]

                if bid in departed:
                    b["depart"] = now

                if bid in arrived:
                    b["arrival"] = now

                if bid not in active:
                    continue

                speed = traci.vehicle.getSpeed(bid)
                accel = traci.vehicle.getAcceleration(bid)
                road = traci.vehicle.getRoadID(bid)
                position = traci.vehicle.getLanePosition(bid)
                at_stop = traci.vehicle.isAtBusStop(bid)

                # 정류장 정차
                if at_stop:
                    b["stop"] += 1

                # 교차로 앞 정지
                if (
                    road == "E0"
                    and position >= EDGE_LENGTH - 100
                    and speed <= 0.1
                    and not at_stop
                ):
                    b["intersection_wait"] += 1

                # 급제동 사건
                braking = accel < BRAKING_LIMIT

                if braking and not b["was_braking"]:
                    b["braking"] += 1

                b["was_braking"] = braking

                # 버스 앞 차량
                leader = traci.vehicle.getLeader(
                    bid, 200
                )

                current_risk = False
                current_leader = None

                if leader is not None and not at_stop:

                    leader_id, gap = leader
                    leader_type = traci.vehicle.getTypeID(
                        leader_id
                    )

                    if (
                        leader_type in ["human", "autonomous"]
                        and speed >= MIN_SPEED
                    ):
                        front_speed = traci.vehicle.getSpeed(
                            leader_id
                        )

                        relative_speed = speed - front_speed

                        if leader_type == "human":
                            b["human_steps"] += 1
                        else:
                            b["av_steps"] += 1

                        # 유효 TTC
                        if relative_speed > 0 and gap >= 0:
                            ttc = gap / relative_speed
                            current_risk = ttc < TTC_LIMIT

                        if current_risk:

                            current_leader = leader_id

                            if leader_type == "human":
                                b["human_risk"] += 1
                            else:
                                b["av_risk"] += 1

                            # 연속 위험 사건의 시작
                            if (
                                not b["risk_active"]
                                or b["risk_leader"] != leader_id
                            ):
                                if leader_type == "human":
                                    b["human_events"] += 1
                                else:
                                    b["av_events"] += 1

                b["risk_active"] = current_risk
                b["risk_leader"] = current_leader

            # 모든 버스 도착 시 종료
            if all(
                buses[bid]["arrival"] is not None
                for bid in BUS_IDS
            ):
                break

    finally:
        if started:
            traci.close()

    # ======================================
    # 4. 버스별 결과
    # ======================================

    bus_rows = []

    for bid in BUS_IDS:

        b = buses[bid]

        completed = (
            b["depart"] is not None
            and b["arrival"] is not None
        )

        travel_time = (
            b["arrival"] - b["depart"]
            if completed else None
        )

        total_leader_steps = (
            b["human_steps"] + b["av_steps"]
        )

        bus_rows.append({
            "traffic": traffic,
            "av_ratio": ratio,
            "seed": seed,
            "bus_id": bid,
            "scheduled_depart": b["scheduled"],
            "actual_depart": b["depart"],
            "arrival": b["arrival"],
            "travel_time": travel_time,
            "departure_delay": (
                b["depart"] - b["scheduled"]
                if b["depart"] is not None else None
            ),
            "stop_seconds": b["stop"],
            "intersection_wait": b["intersection_wait"],
            "braking_events": b["braking"],
            "human_leader_steps": b["human_steps"],
            "av_leader_steps": b["av_steps"],
            "human_risk_steps": b["human_risk"],
            "av_risk_steps": b["av_risk"],
            "human_risk_events": b["human_events"],
            "av_risk_events": b["av_events"],
            "av_leader_exposure": (
                b["av_steps"] / total_leader_steps
                if total_leader_steps else None
            ),
            "status": (
                "completed" if completed else "incomplete"
            )
        })

    # ======================================
    # 5. 실험별 결과
    # ======================================

    all_completed = all(
        b["arrival"] is not None
        for b in buses.values()
    )

    if all_completed:

        # 실제 도착 순서
        arrivals = sorted(
            b["arrival"] for b in buses.values()
        )

        headways = [
            arrivals[i] - arrivals[i - 1]
            for i in range(1, len(arrivals))
        ]

        headway_std = statistics.stdev(headways)
        headway_error = statistics.mean([
            abs(h - BUS_INTERVAL)
            for h in headways
        ])

        mean_travel = statistics.mean(
            row["travel_time"] for row in bus_rows
        )

    else:
        headway_std = None
        headway_error = None
        mean_travel = None

    human_steps = sum(
        b["human_steps"] for b in buses.values()
    )
    av_steps = sum(
        b["av_steps"] for b in buses.values()
    )

    human_risk = sum(
        b["human_risk"] for b in buses.values()
    )
    av_risk = sum(
        b["av_risk"] for b in buses.values()
    )

    run_row = {
        "traffic": traffic,
        "av_ratio": ratio,
        "seed": seed,
        "completed_buses": sum(
            b["arrival"] is not None
            for b in buses.values()
        ),
        "mean_travel_time": mean_travel,
        "headway_std": headway_std,
        "headway_error": headway_error,
        "mean_intersection_wait": statistics.mean(
            b["intersection_wait"] for b in buses.values()
        ),
        "total_braking_events": sum(
            b["braking"] for b in buses.values()
        ),
        "human_leader_steps": human_steps,
        "av_leader_steps": av_steps,
        "human_risk_steps": human_risk,
        "av_risk_steps": av_risk,
        "human_risk_events": sum(
            b["human_events"] for b in buses.values()
        ),
        "av_risk_events": sum(
            b["av_events"] for b in buses.values()
        ),
        "human_risk_rate": (
            human_risk / human_steps
            if human_steps else None
        ),
        "av_risk_rate": (
            av_risk / av_steps
            if av_steps else None
        ),
        "status": (
            "completed" if all_completed else "incomplete"
        )
    }

    return run_row, bus_rows


# ==========================================
# 6. 전체 100회 실험
# ==========================================

all_runs = []
all_buses = []

total_runs = (
    len(TRAFFIC_LEVELS)
    * len(AV_RATIOS)
    * len(SEEDS)
)

count = 0

print("=== 혼합교통 통합 실험 시작 ===")
print("전체 실험:", total_runs, "회", flush=True)

for traffic in TRAFFIC_LEVELS:
    for ratio in AV_RATIOS:
        for seed in SEEDS:

            count += 1

            print(
                f"[{count}/{total_runs}] "
                f"차량 {traffic}대 / "
                f"AV {ratio}% / Seed {seed}",
                flush=True
            )

            run_row, bus_rows = run_experiment(
                traffic, ratio, seed
            )

            all_runs.append(run_row)
            all_buses.extend(bus_rows)

            # 매 실험마다 중간 저장
            pd.DataFrame(all_runs).to_csv(
                OUTPUT_RUNS,
                index=False,
                encoding="utf-8-sig"
            )

            pd.DataFrame(all_buses).to_csv(
                OUTPUT_BUSES,
                index=False,
                encoding="utf-8-sig"
            )

            print(
                "도착 버스:",
                run_row["completed_buses"],
                "/5 | 평균 운행시간:",
                run_row["mean_travel_time"],
                "| 배차 표준편차:",
                run_row["headway_std"],
                flush=True
            )


# ==========================================
# 7. 통계 요약
# ==========================================

df = pd.DataFrame(all_runs)

# 미완료 시나리오는 통행시간 분석에서 제외
valid = df[df["status"] == "completed"].copy()

summary = (
    valid.groupby(["traffic", "av_ratio"])
    .agg(
        mean_travel_time=("mean_travel_time", "mean"),
        std_travel_time=("mean_travel_time", "std"),
        mean_headway_std=("headway_std", "mean"),
        mean_headway_error=("headway_error", "mean"),
        mean_wait=("mean_intersection_wait", "mean"),
        mean_braking=("total_braking_events", "mean"),
        valid_runs=("seed", "count")
    )
    .reset_index()
)

summary.to_csv(
    OUTPUT_SUMMARY,
    index=False,
    encoding="utf-8-sig"
)

print("\n=== 통합 실험 완료 ===")
print(summary.to_string(index=False))
print("\n결과 저장 완료!")
