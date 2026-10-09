"""Drive every link cell of rerun item F through Mahimahi (runs inside the VM).

Cells mirror run_link_realism.py in the package: the low_bandwidth and normal
constant links under the same five loss rates, and the three measured LTE uplink
traces each beside a constant link at the same mean capacity, all at loss 0. The
delayed_imagery link is added because item F names that condition. Delay is the
scenario's base_latency_s split evenly over the two directions, so the emulated
round trip equals the model's base latency. The downlink is a 50 Mbps constant
trace so acknowledgements are never the bottleneck.

Payloads are the package byte model: the three summaries, the three feature
vectors and one raw record per modality at the midpoint of its configured range.

Every cell appends to results/emulator/transfers.csv as it runs; a cell whose
completion marker is already present is skipped, so the run resumes after a stop.
"""
import csv
import json
import os
import shutil
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PKG = os.path.join(ROOT, "upstream", "prototype_v2.3")
OUT = os.path.join(ROOT, "results", "emulator")
TRACES_SRC = os.path.join(PKG, "data", "traces")
TRACES = os.path.join(OUT, "traces")
CSV = os.path.join(OUT, "transfers.csv")
HERE = os.path.dirname(os.path.abspath(__file__))
MTU = 1500
PORT = 5201

cfg = json.load(open(os.path.join(PKG, "config.json")))
prov = json.load(open(os.path.join(TRACES_SRC, "provenance.json")))["traces"]
bm = cfg["byte_model"]
SIZES = [bm["summary"][m] for m in ("image", "text", "sensor")] + \
        [bm["feature"][m] for m in ("image", "text", "sensor")] + \
        [(lo + hi) // 2 for lo, hi in (bm["raw_range"][m] for m in ("image", "text", "sensor"))]
SC = {s["name"]: s for s in cfg["scenarios"]}
LOSS = [0.0, 0.01, 0.02, 0.05, 0.10]


def const_trace(bps, seconds=120):
    """Mahimahi trace delivering one MTU packet at a constant rate (looped by mm-link)."""
    name = os.path.join(TRACES, "const_%d.up" % int(bps))
    if not os.path.exists(name):
        pps = bps / (MTU * 8.0)
        n = int(pps * seconds)
        with open(name, "w") as fh:
            for k in range(1, n + 1):
                fh.write("%d\n" % int(k * 1000.0 / pps))
    return name


def cells():
    out = []
    for scn in ("low_bandwidth", "normal"):
        s = SC[scn]
        for p in LOSS:
            out.append(("%s_loss%g" % (scn, p), const_trace(s["bandwidth_bps"]),
                        int(round(s["base_latency_s"] * 500)), p, "const:%d" % s["bandwidth_bps"]))
    s = SC["delayed_imagery"]
    out.append(("delayed_imagery_loss0", const_trace(s["bandwidth_bps"]),
                int(round(s["base_latency_s"] * 500)), 0.0, "const:%d" % s["bandwidth_bps"]))
    d = int(round(SC["low_bandwidth"]["base_latency_s"] * 500))
    for t in ("Verizon-LTE-short.up", "Verizon-LTE-driving.up", "TMobile-LTE-driving.up"):
        bps = prov[t]["mean_mbps"] * 1e6
        out.append(("trace_%s" % t, os.path.join(TRACES_SRC, t), d, 0.0, "trace:%s" % t))
        out.append(("constmatched_%s" % t, const_trace(bps), d, 0.0, "const:%d" % int(bps)))
    return out


def done_cells():
    if not os.path.exists(CSV):
        return set()
    with open(CSV) as fh:
        return {r["cell"] for r in csv.DictReader(fh) if r["payload_bytes"] == "-1"}


def main():
    os.makedirs(TRACES, exist_ok=True)
    for tool in ("mm-link", "mm-delay", "mm-loss"):
        if shutil.which(tool) is None:
            sys.exit("missing %s; install mahimahi first" % tool)
    down = const_trace(50e6)
    server = subprocess.Popen([sys.executable, os.path.join(HERE, "mm_transfer.py"), "server",
                               "--port", str(PORT)])
    time.sleep(0.5)
    skip = done_cells()
    log = open(os.path.join(OUT, "run_log.txt"), "a")
    try:
        for cell, up, delay_ms, loss, link in cells():
            if cell in skip:
                print("skip (complete):", cell); continue
            print("cell:", cell, "delay", delay_ms, "loss", loss, flush=True)
            log.write("%s start %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), cell)); log.flush()
            client = [sys.executable, os.path.join(HERE, "mm_transfer.py"), "client",
                      "--host", "$MAHIMAHI_BASE", "--port", str(PORT),
                      "--sizes", ",".join(map(str, SIZES)), "--cell", cell, "--link", link,
                      "--delay-ms", str(delay_ms), "--loss", str(loss), "--out", CSV]
            inner = "mm-link %s %s -- sh -c '%s'" % (up, down, " ".join(client))
            cmd = "mm-delay %d mm-loss uplink %g %s" % (delay_ms, loss, inner) if loss > 0 \
                else "mm-delay %d %s" % (delay_ms, inner)
            rc = subprocess.call(cmd, shell=True)
            log.write("%s end %s rc=%d\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), cell, rc)); log.flush()
    finally:
        server.terminate()
        log.close()
    env = {"mahimahi": subprocess.run(["dpkg-query", "-W", "-f=${Version}", "mahimahi"],
                                      capture_output=True, text=True).stdout.strip(),
           "kernel": os.uname().release, "python": sys.version.split()[0],
           "sizes": SIZES, "loss_rates": LOSS, "mtu": MTU,
           "delay_rule": "mm-delay = base_latency_s/2 per direction, so RTT = base_latency_s",
           "downlink": "constant 50 Mbps trace"}
    json.dump(env, open(os.path.join(OUT, "environment.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
