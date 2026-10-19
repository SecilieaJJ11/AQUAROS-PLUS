"""AQUAROS PLUS - complete single-file project.
Run:  python3 aquaros_project.py
Needs: python3 (matplotlib optional, for the chart).
"""
import random, csv, os, statistics as st
from dataclasses import dataclass

# ===== aquaros/config.py =====
NORMAL_RANGE = {
    "temperature": (2.0, 30.0),   # deg C
    "salinity":    (30.0, 38.0),  # ppt
    "hydrophone":  (0.0, 60.0),   # dB (simplified)
    "pressure":    (0.0, 10.0),   # bar
}
STATIC_PRIORITY = {"temperature": 2, "salinity": 3, "hydrophone": 8, "pressure": 2}
MAX_PRIORITY = 10.0

# ===== aquaros/scoring.py =====


def is_anomaly(sensor_type: str, value: float) -> bool:
    lo, hi = NORMAL_RANGE[sensor_type]
    return value < lo or value > hi


def static_priority(sensor_type: str) -> float:
    return float(STATIC_PRIORITY[sensor_type])


def smart_priority(sensor_type: str, value: float) -> float:
    """base if in range; otherwise base + boost, boost = min(10-base, 6 + 4*deviation), capped at 10."""
    base = STATIC_PRIORITY[sensor_type]
    lo, hi = NORMAL_RANGE[sensor_type]
    if lo <= value <= hi:
        return float(base)
    span = max(hi - lo, 1e-6)
    deviation = max(lo - value, value - hi) / span
    boost = min(MAX_PRIORITY - base, 6 + 4 * deviation)
    return round(min(MAX_PRIORITY, base + boost), 1)

# ===== aquaros/sensors.py =====


@dataclass
class Message:
    mid: int
    origin: int
    sensor_type: str
    value: float
    created: int            # window index when generated
    priority: float = 0.0   # assigned once at the origin node by the active policy
    anomaly: bool = False
    delivered: int = -1     # window index when it reached the sink


def make_reading(rng: random.Random, p_anomaly: float):
    stype = rng.choice(list(NORMAL_RANGE))
    lo, hi = NORMAL_RANGE[stype]
    span = hi - lo
    if rng.random() < p_anomaly:
        dev = rng.uniform(0.1, 2.0) * span
        value = hi + dev if rng.random() < 0.5 else lo - dev
    else:
        mid, sd = (lo + hi) / 2, span / 8
        value = min(hi, max(lo, rng.gauss(mid, sd)))
    return stype, value

# ===== aquaros/queueing.py =====
class TxQueue:
    def __init__(self, aging: float = 0.0):
        self.aging = aging
        self.items = []

    def push(self, msg):
        self.items.append(msg)

    def __len__(self):
        return len(self.items)

    def pop_batch(self, capacity: int, now: int):
        key = lambda m: (-(m.priority + self.aging * (now - m.created)), m.created, m.mid)
        self.items.sort(key=key)
        batch, self.items = self.items[:capacity], self.items[capacity:]
        return batch

    def requeue(self, msgs):
        self.items.extend(msgs)

# ===== aquaros/routing.py =====
def pick_next_hop(candidates, energy):
    """candidates: list of node ids; energy: dict id -> residual energy. Highest energy wins."""
    return max(candidates, key=lambda n: (energy[n], -n))

# ===== aquaros/network.py =====


def run(policy="static", aging=0.0, seed=0, windows=200, n_sensors=6, n_relays=2,
        gen_per_window=4, capacity=4, relay_capacity=12, loss=0.1, p_anomaly=0.03):
    rng = random.Random(seed)
    sensors = list(range(n_sensors))
    relays = list(range(n_sensors, n_sensors + n_relays))
    queues = {n: TxQueue(aging) for n in sensors + relays}
    energy = {n: 100.0 for n in sensors + relays}
    all_msgs, mid = [], 0
    score = smart_priority if policy == "aquaros" else None

    for w in range(windows):
        for s in sensors:
            for _ in range(gen_per_window):
                stype, val = make_reading(rng, p_anomaly)
                pr = smart_priority(stype, val) if policy == "aquaros" else static_priority(stype)
                m = Message(mid, s, stype, val, w, pr, is_anomaly(stype, val))
                mid += 1
                all_msgs.append(m)
                queues[s].push(m)
        arrivals = []
        for s in sensors:                       # sensors -> relays (SHEEP-inspired hop choice)
            hop = pick_next_hop(relays, energy)
            batch = queues[s].pop_batch(capacity, w)
            lost = []
            for m in batch:
                energy[s] -= 0.05
                if rng.random() < loss:
                    lost.append(m)
                else:
                    energy[hop] -= 0.02
                    arrivals.append((hop, m))
            queues[s].requeue(lost)
        for hop, m in arrivals:
            queues[hop].push(m)
        for r in relays:                        # relays -> sink
            batch = queues[r].pop_batch(relay_capacity, w)
            lost = []
            for m in batch:
                energy[r] -= 0.05
                if rng.random() < loss:
                    lost.append(m)
                else:
                    m.delivered = w
            queues[r].requeue(lost)
    return all_msgs, energy

# ===== aquaros/metrics.py =====


def summarize(msgs, windows, within=3):
    """Latency in windows. Undelivered messages are charged (windows - created), a lower bound."""
    def lat(m):
        return (m.delivered if m.delivered >= 0 else windows) - m.created
    out = {}
    for label, group in (("anomaly", [m for m in msgs if m.anomaly]),
                         ("routine", [m for m in msgs if not m.anomaly])):
        if not group:
            continue
        lats = sorted(lat(m) for m in group)
        out[f"{label}_mean_latency"] = st.mean(lats)
        out[f"{label}_p95_latency"] = lats[int(0.95 * (len(lats) - 1))]
        out[f"{label}_delivered_ratio"] = sum(m.delivered >= 0 for m in group) / len(group)
        out[f"{label}_within_{within}w"] = sum(m.delivered >= 0 and lat(m) <= within for m in group) / len(group)
    return out

# ===== experiment =====
if __name__ == '__main__':
    """Compare four scheduler configurations over many seeds; writes results/results.csv and results/summary.png."""

    CONFIGS = {
        "Existing (static)":              dict(policy="static",  aging=0.0),
        "Static + QLS-inspired aging":    dict(policy="static",  aging=0.1),
        "AQUAROS PLUS":                   dict(policy="aquaros", aging=0.0),
        "AQUAROS PLUS + QLS-inspired":    dict(policy="aquaros", aging=0.1),
    }
    SEEDS, WINDOWS = range(40), 200
    KEYS = ["anomaly_mean_latency", "anomaly_p95_latency", "anomaly_within_3w",
            "anomaly_delivered_ratio", "routine_mean_latency", "routine_delivered_ratio"]

    os.makedirs("results", exist_ok=True)
    rows, table = [], {}
    for name, cfg in CONFIGS.items():
        per_seed = []
        for s in SEEDS:
            msgs, _ = run(seed=s, windows=WINDOWS, **cfg)
            per_seed.append(summarize(msgs, WINDOWS))
        table[name] = {k: st.mean(r[k] for r in per_seed) for k in KEYS}
        table[name].update({k + "_sd": st.pstdev(r[k] for r in per_seed) for k in KEYS})
        rows.append({"config": name, **table[name]})

    with open("results/results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

    print(f"{'config':32s}" + "".join(f"{k[:18]:>20s}" for k in KEYS))
    for name, r in table.items():
        print(f"{name:32s}" + "".join(f"{r[k]:20.3f}" for k in KEYS))

    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    names = list(table)
    cols = ["#7e9bb0", "#4f6c81", "#31d8c4", "#1c7d70"]
    for a, (k, t) in zip(ax, [("anomaly_mean_latency", "Anomaly latency (windows, lower=better)"),
                               ("anomaly_within_3w", "Anomalies delivered within 3 windows"),
                               ("routine_mean_latency", "Routine latency (windows)")]):
        a.bar(range(4), [table[n][k] for n in names], yerr=[table[n][k + "_sd"] for n in names], color=cols, capsize=3)
        a.set_xticks(range(4)); a.set_xticklabels(["Existing", "Static+QLS", "AQUAROS", "AQUAROS+QLS"], rotation=20, fontsize=8)
        a.set_title(t, fontsize=9)
    plt.tight_layout(); plt.savefig("results/summary.png", dpi=130)