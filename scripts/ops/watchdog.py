#!/usr/bin/env python3
"""Watchdog for the train-plan service. Each stdout line is an alert/event.

    python3 -u scripts/ops/watchdog.py   # e.g. as a Claude Code Monitor

State (log offsets, alert levels) persists beside this file so re-arming the
monitor does not replay old lines. Lives outside /tmp: /tmp is tmpfs here and
a reboot wipes it.
"""
import glob, json, os, re, subprocess, time

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATE = os.path.join(os.path.expanduser("~/.local/state/cyberworld"), "watchdog_state.json")
POLL = 20
HEARTBEAT = 1800          # status line every 30 min
STALL_S = 40 * 60         # no log growth for this long -> stall alert
PAT = re.compile(r"guard\]|STEP_BACK|STOP|Traceback|Error|error:|Killed|OOM|out of memory|"
                 r"device: cpu|Training on device|\[run \]|\[skip|\[plan\]|DRY RUN|nan loss|NaN|"
                 r"FAIL|WARNING|best epoch|resum|Parallel ingest|records \(|Loaded \d+", re.I)
NOISE = re.compile(r"\[INFO\] Arguments:|skipping unreadable capture UCAP172\.31\.69\.25")
TRAIN_PAT = re.compile(r"bita/train\.py|scripts/retrain_|scripts/run_encoder")


def sh(cmd):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:
        return ""


def emit(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def load():
    try:
        with open(STATE) as f:
            return json.load(f)
    except Exception:
        return {"offsets": {}, "mem_level": 0, "gpu": {}, "runs": {}, "last_growth": time.time(),
                "last_hb": 0, "ended": False}


def save(s):
    with open(STATE, "w") as f:
        json.dump(s, f)


def train_pids():
    """Training python processes (not their systemd-run wrapper, not spawn workers)."""
    pids = []
    for line in sh("pgrep -af python").splitlines():
        pid, _, cmd = line.partition(" ")
        if cmd.startswith("systemd-run") or "multiprocessing" in cmd or "watchdog" in cmd:
            continue
        if TRAIN_PAT.search(cmd):
            pids.append((int(pid), cmd))
    return pids


def cg_mem(pid):
    """anon (+kernel) of the pid's cgroup. Not memory.current: that includes
    the PCAP page cache, which the kernel reclaims under the cap."""
    try:
        cg = open(f"/proc/{pid}/cgroup").read().strip().split("::")[-1]
        base = "/sys/fs/cgroup" + cg
        stat = dict(l.split() for l in open(base + "/memory.stat"))
        cur = int(stat["anon"]) + int(stat.get("kernel", 0))
        mx = open(base + "/memory.max").read().strip()
        return cur, (int(mx) if mx != "max" else None)
    except Exception:
        return None, None


def meminfo():
    d = {}
    for l in open("/proc/meminfo"):
        k, v = l.split(":")
        d[k] = int(v.split()[0]) * 1024
    return d


def main():
    s = load()
    os.chdir(REPO)
    while True:
        now = time.time()
        # --- service state
        active = sh("systemctl --user is-active train-plan")
        if active != "active" and not s["ended"]:
            res = sh("systemctl --user show train-plan -p Result -p ExecMainStatus")
            orphans = [p for p, _ in train_pids()]
            emit(f"SERVICE ENDED: state={active} {res.replace(chr(10), ' ')} | orphan train pids={orphans} | "
                 "last plan.log: " + sh("tail -4 plan.log").replace("\n", " || ")[-700:])
            s["ended"] = True
        elif active == "active":
            s["ended"] = False

        # --- new log lines
        logs = ["plan.log"] + sorted(glob.glob("results/training_plan/**/*.log", recursive=True))
        grew = False
        for lf in logs:
            if "/.spill/" in lf:
                continue
            try:
                size = os.path.getsize(lf)
            except OSError:
                continue
            off = s["offsets"].get(lf)
            if off is None:
                off = size if lf == "plan.log" else 0
            if size < off:
                off = 0
            if size > off:
                grew = True
                with open(lf, errors="replace") as f:
                    f.seek(off)
                    chunk = f.read(size - off)
                for line in chunk.splitlines():
                    line = line.strip()
                    if not PAT.search(line) or NOISE.search(line):
                        continue
                    # The same step name recurs once per seed; key by section.
                    if line.startswith(("[plan] compare", "[plan] seeds", "[plan] downstream",
                                        "[plan] summary", "[plan] dry run")):
                        s["section"] = line
                    if line.startswith("[run ]"):
                        key = f"{s.get('section', '')} :: {line}"
                        s["runs"][key] = s["runs"].get(key, 0) + 1
                        if s["runs"][key] > 2:
                            emit(f"LOOP? step started {s['runs'][key]}x: {key}")
                            continue
                    emit(f"[{os.path.basename(lf)}] {line[:400]}")
            s["offsets"][lf] = size
        if grew:
            s["last_growth"] = now

        # --- memory
        pids = train_pids()
        mi = meminfo()
        avail = mi["MemAvailable"] / 2**30
        zram = (mi["SwapTotal"] - mi["SwapFree"]) / 2**30
        worst, memdesc = 0.0, []
        for pid, _cmd in pids:
            cur, mx = cg_mem(pid)
            if cur and mx:
                worst = max(worst, cur / mx)
                memdesc.append(f"pid {pid} anon {cur/2**30:.1f}/{mx/2**30:.0f}G")
        level = 3 if worst > 0.95 else 2 if worst > 0.90 else 1 if worst > 0.80 else 0
        if level > s["mem_level"]:
            emit(f"MEMORY HIGH ({worst:.0%} of cap): {'; '.join(memdesc)} | sys avail {avail:.1f}G zram {zram:.1f}G")
            s["mem_level"] = level
        elif worst < 0.75:
            s["mem_level"] = 0
        if avail < 2.0:
            emit(f"SYSTEM RAM LOW: MemAvailable {avail:.1f}G zram used {zram:.1f}G")

        # --- GPU: does each training pid hold a CUDA context?
        gpu_pids = {int(l) for l in sh("nvidia-smi --query-compute-apps=pid --format=csv,noheader").split()
                    if l.strip().isdigit()}
        gutil = sh("nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader")
        try:
            gmem = int(sh("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits").split()[0])
        except Exception:
            gmem = 0
        glevel = 3 if gmem > 7400 else 2 if gmem > 6500 else 1 if gmem > 5500 else 0
        if glevel > s.get("gpu_level", 0):
            emit(f"GPU MEMORY HIGH: {gmem} MiB of 8188 | per process: "
                 + sh("nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader").replace(chr(10), '; '))
            s["gpu_level"] = glevel
        elif gmem < 5000:
            s["gpu_level"] = 0
        for pid, _cmd in pids:
            on, key = pid in gpu_pids, str(pid)
            prev = s["gpu"].get(key)
            if prev is None:
                s["gpu"][key] = {"on": on, "since": now, "warned": False}
            elif on != prev["on"]:
                emit(f"GPU {'ACQUIRED' if on else 'RELEASED'} by pid {pid} ({gutil})")
                s["gpu"][key] = {"on": on, "since": now, "warned": False}
            elif not on and not prev["warned"] and now - prev["since"] > 90 * 60:
                emit(f"GPU WARNING: pid {pid} {int((now-prev['since'])/60)} min with no CUDA context "
                     f"(still ingesting, or training on CPU?) cpu%={sh(f'ps -o %cpu= -p {pid}')} gpu={gutil}")
                prev["warned"] = True
        live = {str(p) for p, _ in pids}
        s["gpu"] = {k: v for k, v in s["gpu"].items() if k in live}

        # --- stall
        if active == "active" and now - s["last_growth"] > STALL_S:
            cpu = sum(float(sh(f"ps -o %cpu= -p {p}") or 0) for p, _ in pids)
            workers = sh("pgrep -fc multiprocessing.spawn")
            emit(f"STALL? no log growth for {int((now-s['last_growth'])/60)} min; train cpu%={cpu:.0f} "
                 f"ingest workers={workers} gpu={gutil}")
            s["last_growth"] = now

        # --- heartbeat
        if now - s["last_hb"] > HEARTBEAT:
            emit(f"HEARTBEAT svc={active} {'; '.join(memdesc) or 'no train proc'} | gpu {gutil} "
                 f"on_gpu={[p for p, _ in pids if p in gpu_pids]} | ingest workers={sh('pgrep -fc multiprocessing.spawn')} "
                 f"| sys avail {avail:.1f}G zram {zram:.1f}G")
            s["last_hb"] = now

        save(s)
        if s["ended"] and active != "active":
            return
        time.sleep(POLL)


if __name__ == "__main__":
    main()
