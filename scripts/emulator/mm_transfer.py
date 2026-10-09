"""TCP payload transfer through a Mahimahi link shell (rerun item F).

Two roles, both plain Python 3.8 stdlib so they run inside the Ubuntu 20.04 VM.

  server  Listens outside the shell. For each connection it reads an 8-byte
          length prefix, then that many bytes, then writes one byte back. That
          byte is the application-level acknowledgement the client times.
  client  Runs inside `mm-delay ... mm-loss ... mm-link ...`. For every requested
          (payload_bytes, repeat) it opens a fresh TCP connection to the outside
          address, sends the payload and waits for the acknowledgement. It records
          the offset of each transfer from the client's own start, which is within
          a few milliseconds of the shell's start, so the modelled latency can be
          computed at the same position in the capacity trace.

Nothing here reads results; every transfer is appended to the CSV as it finishes.
"""
import argparse
import csv
import os
import socket
import struct
import sys
import time

ACK = b"\x01"


def serve(port):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", port))
    srv.listen(64)
    while True:
        conn, _ = srv.accept()
        try:
            hdr = _read_exact(conn, 8)
            if hdr is None:
                continue
            (n,) = struct.unpack("!Q", hdr)
            if _read_exact(conn, n) is not None:
                conn.sendall(ACK)
        finally:
            conn.close()


def _read_exact(conn, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(min(1 << 16, n - len(buf)))
        if not chunk:
            return None
        buf.extend(chunk)
    return bytes(buf)


def transfer(host, port, n, timeout):
    payload = struct.pack("!Q", n) + os.urandom(n)
    t0 = time.monotonic()
    s = socket.create_connection((host, port), timeout=timeout)
    t_conn = time.monotonic()
    try:
        s.sendall(payload)
        t_sent = time.monotonic()
        ack = s.recv(1)
        t_ack = time.monotonic()
    finally:
        s.close()
    ok = ack == ACK
    return ok, t_conn - t0, t_sent - t0, t_ack - t0


def run_client(args):
    start = time.monotonic()
    sizes = [int(x) for x in args.sizes.split(",")]
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as fh:
        w = csv.writer(fh)
        if new:
            w.writerow(["cell", "link", "delay_ms", "loss", "payload_bytes", "repeat",
                        "offset_s", "ok", "connect_s", "send_done_s", "ack_s"])
        for n in sizes:
            reps = args.repeats_small if n <= args.small_max else args.repeats_large
            for r in range(reps):
                off = time.monotonic() - start
                try:
                    ok, tc, ts, ta = transfer(args.host, args.port, n, args.timeout)
                except Exception as exc:  # timeout or reset: recorded, never hidden
                    ok, tc, ts, ta = False, float("nan"), float("nan"), float("nan")
                    print("  transfer failed:", n, r, exc, file=sys.stderr)
                w.writerow([args.cell, args.link, args.delay_ms, args.loss, n, r,
                            "%.4f" % off, int(ok), "%.6f" % tc, "%.6f" % ts, "%.6f" % ta])
                fh.flush()
        w.writerow([args.cell, args.link, args.delay_ms, args.loss, -1, -1,
                    "%.4f" % (time.monotonic() - start), 1, "", "", ""])  # cell-complete marker
        fh.flush()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="role", required=True)
    s = sub.add_parser("server")
    s.add_argument("--port", type=int, default=5201)
    c = sub.add_parser("client")
    c.add_argument("--host", required=True)
    c.add_argument("--port", type=int, default=5201)
    c.add_argument("--sizes", required=True)
    c.add_argument("--repeats-small", type=int, default=20)
    c.add_argument("--repeats-large", type=int, default=10)
    c.add_argument("--small-max", type=int, default=2048)
    c.add_argument("--timeout", type=float, default=120.0)
    c.add_argument("--cell", required=True)
    c.add_argument("--link", required=True)
    c.add_argument("--delay-ms", type=int, required=True)
    c.add_argument("--loss", type=float, required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    serve(a.port) if a.role == "server" else run_client(a)
