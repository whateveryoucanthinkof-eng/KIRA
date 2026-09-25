# CyberWorld Cyber Range, SPAN Tap and Capture Path

**Scope:** the Containerlab enterprise range, its workload generators, the passive SPAN tap and the
sensor-side capture path. The ML architecture is in `docs/ARCHITECTURE.md`. Operational bring-up is
in `docs/LOCAL_SPAN_RUNBOOK.md`.

This material was previously carried inside `docs/ARCHITECTURE.md`. It has been separated because
the problem statement asks for a two-page *architecture* document about the model, not about the
range. Everything below describes the range as it actually is; the retired V3.1 modelling content
that used to sit alongside it has been deleted rather than moved.

The range is optional. The same capture path runs against a real SPAN/mirror NIC by selecting a
site profile with `lab_mode: false` (see `config/sites/`).

---

## 1. Network-first principle

1. A functional enterprise network is created in **Containerlab**.
2. Autonomous workloads run continuously across users and servers.
3. A **passive sensor** taps the network via Linux traffic mirroring (`tc mirred` SPAN).
4. The sensor closes **2.0-second windows** and emits a 5-tuple flow snapshot per window.
5. Everything downstream of that snapshot — host state, inference, forecasting — happens in
   `control_backend`. The sensor builds no feature vector and runs no model.

---

## 2. Topology and segmentation

```
[ External / Attacker Zone: 192.168.100.0/24 ]
                      │
                [ attacker ] (192.168.100.10)
                      │ eth1
                      ▼ eth1 (192.168.100.1)
               [ router-edge ]
                      │ eth2 (172.16.0.1/30)
                      ▼ eth1 (172.16.0.2/30)
             [ router-firewall ]
                      │ eth2 (10.0.0.1/30)
                      ▼ eth1 (10.0.0.2/30)
               [ router-core ]
     ┌────────────────┼────────────────┬────────────────┐
     │ eth2           │ eth3           │ eth4           │ eth5 (SPAN mirror)
     ▼                ▼                ▼                ▼
[ USERS LAN ]    [ SERVERS LAN ]  [ DMZ LAN ]       [ sensor ]
 10.0.1.0/24      10.0.2.0/24      10.0.3.0/24      (passive promiscuous)
 - ws-office      - srv-dns        - dmz-web
 - ws-web         - srv-id
 - ws-file        - srv-app
 - ws-app         - srv-db
                  - srv-file
```

### 2.1 Subnets

| Zone | CIDR | Gateway | Members | Policy |
| :--- | :--- | :--- | :--- | :--- |
| External / Attacker | `192.168.100.0/24` | `192.168.100.1` | `attacker` (`.10`) | Untrusted. Can route only toward DMZ web. |
| Edge–FW transit | `172.16.0.0/30` | `172.16.0.1` | `router-edge`, `router-firewall` | Point-to-point perimeter transit. |
| FW–Core transit | `10.0.0.0/30` | `10.0.0.1` | `router-firewall`, `router-core` | Internal perimeter link. |
| Users LAN | `10.0.1.0/24` | `10.0.1.1` | `ws-office` (`.11`), `ws-web` (`.12`), `ws-file` (`.13`), `ws-app` (`.14`) | Reach Servers LAN and DMZ. No direct DB access. |
| Servers LAN | `10.0.2.0/24` | `10.0.2.1` | `srv-dns` (`.10`), `srv-id` (`.20`), `srv-file` (`.30`), `srv-app` (`.40`), `srv-db` (`.50`) | `srv-db` accepts queries only from `srv-app`. |
| DMZ LAN | `10.0.3.0/24` | `10.0.3.1` | `dmz-web` (`.10`) | Public-facing web service. |
| Passive SPAN | point-to-point | n/a | `router-core:eth5` → `sensor:eth1` | Read-only mirror. |

### 2.2 Filtering and isolation

Enforced on `router-firewall` via `iptables`:

1. **Attacker isolation.** Packets from `192.168.100.0/24` to `10.0.1.0/24` (Users) or `10.0.2.0/24`
   (Servers) are dropped and logged.
2. **DMZ access.** External traffic to `10.0.3.10:80` is accepted.
3. **Database protection.** Direct traffic from `10.0.1.0/24` to `10.0.2.50:5432` is rejected. Only
   `srv-app` (`10.0.2.40`) may open TCP sessions to `srv-db`.
4. **Internal routing.** Users resolve via `srv-dns` (`10.0.2.10:53`), fetch from `srv-file`
   (`10.0.2.30`), use `srv-app` (`10.0.2.40:8000`) and browse `dmz-web` (`10.0.3.10:80`).

---

## 3. Workload generation

Traffic comes from autonomous, heterogeneous client and server processes with temporal jitter:

* **`ws-office`** — periodic DNS lookups (`corp.local`, `web.corp.local`), web browsing on
  `dmz-web`, occasional portal queries to `srv-app`, small document retrievals from `srv-file`, and
  idle periods of 2–8 s simulating human reading time.
* **`ws-web`** — high-frequency HTTP GETs to `dmz-web`, diverse TCP source-port churn.
* **`ws-file`** — bulk uploads and downloads against `srv-file`; high byte volume, large payloads.
* **`ws-app`** — continuous REST queries to `srv-app` (`/api/v1/data`, `/api/v1/query`) and session
  token validation against `srv-id`.
* **`srv-app`** — authenticates against `srv-id` (`10.0.2.20:8080`), queries `srv-db`
  (`10.0.2.50:5432`), returns JSON; produces realistic multi-tier application traffic.
* **`srv-dns`** — authoritative for `corp.local`, `app.corp.local`, `db.corp.local`,
  `web.corp.local`.

External attack campaigns are **operator-driven** (ARM EXTERNAL in the dashboard). The UI does not
inject synthetic attack packets into topology as if they were observed truth.

---

## 4. Passive SPAN tap

The sensor is strictly passive, so observation can never delay or drop enterprise traffic:

* **Mechanism.** Linux `tc mirred` (egress mirror action) on the core router.
* **Mirrored interfaces.** Ingress and egress on `router-core` `eth2`, `eth3`, `eth4` → `eth5`.
* **Sensor tap.** `sensor:eth1` is the far end of `router-core:eth5`, in promiscuous mode
  (`ip link set eth1 promisc on`).
* **Non-interference.** Packets dropped, delayed or processed on the sensor do not affect delivery
  between enterprise nodes.

---

## 5. Capture path (zero-disk hot path)

```
[ sensor:eth1 ]
        │
        ▼
[ Sniffer ] telemetry/capture/sniffer.py
  - non-blocking AF_PACKET socket (scapy fallback)
  - microsecond timestamps
  - dispatches canonical packet dicts in memory
        │
        ├──► [ Flow table ] telemetry/flow/flow_table.py
        │      - active 5-tuple hash table (src_ip, dst_ip, src_port, dst_port, proto)
        │      - bidirectional packet/byte counters, IAT and idle intervals, TCP flags
        │
        └──► [ PCAP engine ] telemetry/packet/pcap_engine.py
               - ~30 packet-level features: TTL statistics, TCP window mean/std,
                 retransmission detection, unanswered-SYN ratio, payload moments,
                 IP fragmentation, internal fanout, vertical/horizontal scan scores
               - NOT WIRED to any model. Computed and currently unused.
        │
        ▼ (every 2.0 s boundary)
[ Window close ] telemetry/state/state_builder.py
  - closes [t, t+2.0s), emits the 5-tuple flow snapshot plus window metadata
  - this snapshot is the sensor's entire contract with the backend
        │
        ▼
[ control_backend ]  host state (TGNE-TA latent + flow attributes) and inference
```

No intermediate CSV or PCAP is written on the hot path. If `--record-pcap` or `--record-state` is
enabled, data is queued to a background worker thread (`queue.Queue`) and written without blocking
capture.

### 5.1 Live and replay modes

* **Live** — continuously processes frames off `sensor:eth1`.
* **Replay** — reads historical PCAP through the same parsing and window-closing code, giving
  deterministic, reproducible runs. `LiveStateBuilder.seek_to()` anchors the window clock to the
  first packet timestamp; without it, historical timestamps never close a window.

---

## 6. Latency instrumentation

Each emitted window records:

* `t_obs` — first packet in the window; `t_close` — window boundary (`t + 2.0 s`).
* `t_state_start` / `t_state_end` — snapshot assembly.
* `t_inf_start` / `t_inf_end` — backend model forward pass.

Derived: pipeline latency `t_state_end − t_close`, inference latency `t_inf_end − t_inf_start`, and
their sum. These are *instrumentation points*, not validated results. The former V3.1 targets
(<5 ms, <15 ms, <20 ms) and the "6.6 s average early-warning lead time" figure belonged to the
retired design and are **not** carried forward as claims. Lead time is defined and measured by
`cyberworld_v4/metrics/earlywarning.py` as `t_onset − t_first_valid_alert`, and no measurement
exists yet.

---

## 7. Attack scenarios

The range supports controlled injection from the isolated `attacker` node without changing capture
or model code:

1. **Reconnaissance** — horizontal ping sweeps and vertical SYN scans against `router-edge` and
   `dmz-web`.
2. **Perimeter exploitation** — HTTP credential stuffing or vulnerability probing on `dmz-web`.
3. **Lateral movement** — pivoting into internal subnets toward SMB (445) and RDP (3389).
4. **Command & control** — periodic beaconing with distinctive inter-arrival jitter.

Note: scenarios 1, 3 and 4 were designed to be picked up through packet-level indicators
(`pcap_unanswered_syn_ratio`, `pcap_unique_dst_ports`, `pcap_new_internal_edges`,
`pcap_internal_fanout`, `pcap_pkt_iat_std`). Those features are computed but **not wired to any
model**, so today only their flow-level shadow — peer counts, port counts, rates, connection
density — reaches the forecaster.
