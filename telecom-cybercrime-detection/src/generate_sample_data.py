"""
Generate a synthetic telecom / network-traffic dataset that follows the
UNSW-NB15 column layout, so the project runs out-of-the-box.

For your final submission, replace it with the real UNSW-NB15 file
(UNSW_NB15_training-set.csv) - see README.md.

Usage:
    python -m src.generate_sample_data              # 20,000 rows
    python -m src.generate_sample_data --rows 50000
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ATTACK_CATS = ["Generic", "Exploits", "Fuzzers", "DoS", "Reconnaissance",
               "Analysis", "Backdoor", "Shellcode", "Worms"]
PROTOS = ["tcp", "udp", "icmp", "arp", "ospf"]
SERVICES = ["-", "http", "dns", "ftp", "ssh", "smtp", "pop3", "ssl"]
STATES = ["FIN", "CON", "INT", "REQ", "RST", "ACC"]


def generate(rows: int = 20000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    label = (rng.random(rows) < 0.68).astype(int)  # ~68% attacks, like UNSW-NB15
    # Some attacks look like normal traffic and vice-versa (stealthy attacks,
    # noisy normal traffic), so the model cannot reach a fake 100% accuracy.
    flip = rng.random(rows) < 0.10
    atk = (label == 1) ^ flip   # "behaves like an attack" (drives the features)

    def mix(normal, attack, noise=0.35):
        """Draw from two overlapping distributions so the task is not trivial."""
        base = np.where(atk, attack, normal)
        return np.abs(base * rng.lognormal(0, noise, rows))

    def pick(attack_choices, attack_p, normal_choices, normal_p):
        return np.where(atk, rng.choice(attack_choices, rows, p=attack_p),
                        rng.choice(normal_choices, rows, p=normal_p))

    df = pd.DataFrame({
        "id": np.arange(1, rows + 1),
        "dur": mix(1.2, 0.15, 0.9),
        "proto": pick(PROTOS, [.45, .35, .1, .05, .05], PROTOS, [.7, .25, .02, .02, .01]),
        "service": pick(SERVICES, [.5, .15, .15, .05, .05, .04, .03, .03],
                        SERVICES, [.3, .3, .2, .05, .05, .04, .03, .03]),
        "state": pick(STATES, [.3, .1, .4, .1, .05, .05], STATES, [.6, .2, .1, .05, .03, .02]),
        "spkts": mix(10, 6).round(),
        "dpkts": mix(12, 3).round(),
        "sbytes": mix(900, 2200, 0.7).round(),
        "dbytes": mix(2500, 400, 0.8).round(),
        "rate": mix(300, 9000, 0.9),
        "sttl": pick([254, 62, 252], [.6, .3, .1], [31, 62, 254], [.4, .5, .1]),
        "dttl": pick([252, 0, 29], [.5, .3, .2], [29, 252, 60], [.5, .3, .2]),
        "sload": mix(1e5, 6e6, 0.9),
        "dload": mix(3e5, 4e4, 0.9),
        "sloss": mix(1, 3).round(),
        "dloss": mix(1, 1).round(),
        "sinpkt": mix(80, 15, 0.8),
        "dinpkt": mix(60, 20, 0.8),
        "sjit": mix(500, 2500, 0.9),
        "djit": mix(300, 400, 0.9),
        "swin": pick([255, 0], [.85, .15], [255, 0], [.5, .5]),
        "stcpb": rng.integers(0, 4_000_000_000, rows),
        "dtcpb": rng.integers(0, 4_000_000_000, rows),
        "dwin": pick([255, 0], [.6, .4], [255, 0], [.5, .5]),
        "tcprtt": mix(0.05, 0.01, 0.8),
        "synack": mix(0.03, 0.005, 0.8),
        "ackdat": mix(0.02, 0.005, 0.8),
        "smean": mix(90, 230, 0.6).round(),
        "dmean": mix(180, 60, 0.6).round(),
        "trans_depth": rng.integers(0, 3, rows),
        "response_body_len": mix(1500, 80, 1.2).round(),
        "ct_srv_src": mix(5, 25, 0.7).round(),
        "ct_state_ttl": pick([1, 2, 0], [.7, .2, .1], [0, 1, 2], [.55, .2, .25]),
        "ct_dst_ltm": mix(4, 18, 0.7).round(),
        "ct_src_dport_ltm": mix(3, 12, 0.7).round(),
        "ct_dst_sport_ltm": mix(2, 8, 0.7).round(),
        "ct_dst_src_ltm": mix(4, 20, 0.7).round(),
        "is_ftp_login": rng.choice([0, 1], rows, p=[.97, .03]),
        "ct_ftp_cmd": rng.choice([0, 1], rows, p=[.97, .03]),
        "ct_flw_http_mthd": rng.choice([0, 1, 2], rows, p=[.85, .1, .05]),
        "ct_src_ltm": mix(4, 15, 0.7).round(),
        "ct_srv_dst": mix(5, 22, 0.7).round(),
        "is_sm_ips_ports": np.where(atk, rng.choice([0, 1], rows, p=[.9, .1]), 0),
    })
    cats = rng.choice(ATTACK_CATS, rows, p=[.35, .25, .12, .08, .08, .04, .03, .03, .02])
    df["attack_cat"] = np.where(label == 1, cats, "Normal")
    df["label"] = label

    # a little realistic dirt so the preprocessing step has work to do
    for col in ("dur", "sbytes", "rate"):
        df.loc[rng.choice(rows, int(rows * 0.002), replace=False), col] = np.nan
    df = pd.concat([df, df.sample(int(rows * 0.003), random_state=seed)], ignore_index=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--out", default="data/sample_telecom_traffic.csv")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = generate(args.rows)
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows x {df.shape[1]} columns -> {out}")


if __name__ == "__main__":
    main()
