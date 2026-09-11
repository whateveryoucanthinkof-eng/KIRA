# CyberWorld: Enterprise Cyber Range & Live Streaming Telemetry Architecture

**Document ID:** `docs/ARCHITECTURE.md`  
**System Version:** V3.1-Production  
**Authoritative ML Model:** CyberWorldModel V3.1-PCAP  

---

## 1. Executive Architectural Overview

**CyberWorld** is an integrated cyber-range and temporal world-model inference system designed to proactively forecast cyber intrusions before attacks transition into destructive or irreversible phases.

The system is built around the **Network-First Principle**:
1. A believable, fully functional enterprise network is created in **Containerlab**.
2. Realistic enterprise workloads operate continuously across users and servers.
3. A **passive network sensor** taps into the network non-intrusively via Linux traffic mirroring (`tc mirred` SPAN).
4. An **in-memory streaming telemetry pipeline** transforms live packets into 2.0-second state vectors without writing intermediate CSV or PCAP files to disk.
5. An in-memory causal sliding buffer ($L=15$ steps = 30 seconds) feeds the **V3.1 World Model**.
6. The model evaluates transition hazards, MITRE ATT&CK stages, next-state simulation $\mathbf{\hat{S}}_{t+1}$, and forward risk trajectories up to 10 seconds in advance ($K=5$ horizons).

```
   ┌─────────────────────────────────────────────────────────────┐
   │             Containerlab Enterprise Cyber Range             │
   │  ┌──────────────┐   ┌──────────────┐   ┌─────────────────┐  │
   │  │ Users Subnet │   │Servers Subnet│   │   DMZ Subnet    │  │
   │  │ 10.0.1.0/24  │   │ 10.0.2.0/24  │   │   10.0.3.0/24   │  │
   │  └──────┬───────┘   └──────┬───────┘   └────────┬────────┘  │
   │         │                  │                    │           │
   │         └───────────┬──────┴────────────────────┘           │
   │                     ▼                                       │
   │              [ router-core ] ──(SPAN tc mirred)──┐          │
   │                     │                            │          │
   │              [ router-firewall ]                 │          │
   │                     │                            │          │
   │              [ router-edge ]                     │          │
   │                     │                            │          │
   │              [ attacker ] (Isolated External)    │          │
   └──────────────────────────────────────────────────┼──────────┘
                                                      │
                                                      │ Mirrored Real Packets
                                                      ▼
   ┌─────────────────────────────────────────────────────────────┐
   │             In-Memory Streaming Telemetry Pipeline          │
   │                                                             │
   │  Live Packet Stream (sensor eth1 via AF_PACKET)            │
   │     │                                                       │
   │     ├──► In-Memory Flow Table (5-tuple active flows, rates) │
   │     └──► In-Memory PCAP Engine (28 behavioral features)     │
   │     │                                                       │
   │     ▼                                                       │
   │  Time Window Aggregator (2.0s non-overlapping boundaries)   │
   │     │                                                       │
   │     ▼                                                       │
   │  72-D V3.1 Normalized State Vector S(t)                     │
   │     │                                                       │
   │     ▼                                                       │
   │  15-Step Causal Rolling Sequence Buffer [1, 15, 72]         │
   │     │                                                       │
   │     ▼                                                       │
   │  V3.1 Inference Engine (Dual-Branch Feature Transformer)    │
   │     │                                                       │
   │     ├─► Future Risk Trajectory P(t+2s ... t+10s)            │
   │     ├─► MITRE ATT&CK 7-Stage Classification                 │
   │     ├─► Attack Onset Hazard Forecast                        │
   │     ├─► Precursor Reconnaissance Anomaly Indicator          │
   │     └─► Physical Next-State Simulation S_(t+1)              │
   └─────────────────────────────────────────────────────────────┘
```

---

## 2. Enterprise Network Topology & Segmentation

The cyber range is structured into isolated security zones managed by three routing layers:

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
 10.0.1.0/24      10.0.2.0/24      10.0.3.0/24      (Passive Promiscuous)
 - ws-office      - srv-dns        - dmz-web
 - ws-web         - srv-id
 - ws-file        - srv-app
 - ws-app         - srv-db
                  - srv-file
```

### 2.1 Subnet Architecture

| Zone Name | CIDR Prefix | Gateway | Member Nodes | Description & Security Policy |
| :--- | :--- | :--- | :--- | :--- |
| **External / Attacker** | `192.168.100.0/24` | `192.168.100.1` | `attacker` (`.10`) | Untrusted external network. Attacker can ONLY route toward DMZ Web. |
| **Edge-FW Transit** | `172.16.0.0/30` | `172.16.0.1` | `router-edge`, `router-firewall` | Point-to-point perimeter transit. |
| **FW-Core Transit** | `10.0.0.0/30` | `10.0.0.1` | `router-firewall`, `router-core` | Internal perimeter link. |
| **Users LAN** | `10.0.1.0/24` | `10.0.1.1` | `ws-office` (`.11`), `ws-web` (`.12`), `ws-file` (`.13`), `ws-app` (`.14`) | Workstations for employees. Can reach Servers LAN and DMZ. Cannot reach DB directly. |
| **Servers LAN** | `10.0.2.0/24` | `10.0.2.1` | `srv-dns` (`.10`), `srv-id` (`.20`), `srv-file` (`.30`), `srv-app` (`.40`), `srv-db` (`.50`) | Enterprise services. `srv-db` accepts queries exclusively from `srv-app`. |
| **DMZ LAN** | `10.0.3.0/24` | `10.0.3.1` | `dmz-web` (`.10`) | Public-facing web service accessible from internet and internal LAN. |
| **Passive SPAN** | Point-to-Point | N/A | `router-core` `eth5` $\to$ `sensor` `eth1` | Read-only TAP mirror interface. Core router mirrors all routed packets to sensor. |

### 2.2 Security Filtering & Isolation Rules
Enforced on `router-firewall` via `iptables`:
1. **Attacker Isolation:** Packets originating from `192.168.100.0/24` targeting `10.0.1.0/24` (Users) or `10.0.2.0/24` (Servers) are unconditionally dropped and logged.
2. **DMZ Access:** Packets from external networks to `10.0.3.10:80` (DMZ Web) are accepted.
3. **Database Protection:** Direct traffic from `10.0.1.0/24` to `10.0.2.50:5432` (Database) is rejected. Only `srv-app` (`10.0.2.40`) can establish TCP sessions with `srv-db`.
4. **Internal Routing:** Users can resolve domains via `srv-dns` (`10.0.2.10:53`), fetch documents from `srv-file` (`10.0.2.30`), access applications via `srv-app` (`10.0.2.40:8000`), and browse `dmz-web` (`10.0.3.10:80`).

---

## 3. Workload Generation Profiles

Enterprise traffic is generated by autonomous, heterogeneous client and server processes with realistic temporal jitter:

* **`ws-office` (General Office Worker):**
  * Periodic DNS lookups (`corp.local`, `web.corp.local`).
  * Web browsing on `dmz-web`.
  * Occasional employee portal queries to `srv-app`.
  * Small document retrievals from `srv-file`.
  * Natural idle periods (2–8 seconds) simulating human reading time.
* **`ws-web` (Web-Centric User):**
  * High-frequency HTTP GET requests to `dmz-web`.
  * Diverse TCP source port churn and rapid connections.
* **`ws-file` (Data / Analytics User):**
  * Bulk uploads and downloads against `srv-file`.
  * High byte volume transfers with larger packet payloads.
* **`ws-app` (Application Operator):**
  * Continuous REST API queries to `srv-app` (`/api/v1/data`, `/api/v1/query`).
  * Session token validations against `srv-id`.
* **`srv-app` (Application Backend):**
  * Upon receiving client requests, performs authentication checks against `srv-id` (`10.0.2.20:8080`).
  * Queries `srv-db` (`10.0.2.50:5432`) over TCP with SQL commands.
  * Formats JSON responses to clients, generating realistic multi-tier application traffic.
* **`srv-dns` (Enterprise DNS):**
  * Authoritative DNS daemon answering queries for `corp.local`, `app.corp.local`, `db.corp.local`, `web.corp.local`.

---

## 4. Passive Sensor & Traffic Mirroring (SPAN)

The network sensor operates in **pure passive mode** to guarantee that observation never becomes a bottleneck or introduces inline delay to enterprise operations:

* **Mirroring Mechanism:** Linux `tc mirred` (traffic control action mirred egress mirror) configured on the Core router.
* **Mirrored Interfaces:** Ingress and egress queues on `router-core`'s interfaces (`eth2`, `eth3`, `eth4`) are mirrored to `eth5`.
* **Sensor Tap:** `sensor` interface `eth1` is bound to the other end of `router-core:eth5` in promiscuous mode (`ip link set eth1 promisc on`).
* **Non-Interference Guarantee:** Packets dropped, delayed, or processed on the sensor do not affect packet delivery between enterprise nodes.

---

## 5. Streaming Telemetry Architecture (Zero-Disk Hot Path)

The telemetry subsystem (`telemetry/`) ingests raw packets and computes normalized 72-D network states entirely in memory:

```
[ Sensor Interface: eth1 ]
            │
            ▼
[ Sniffer Engine ] (telemetry/capture/sniffer.py)
  - Non-blocking socket (AF_PACKET / scapy fallback)
  - Microsecond timestamp capture
  - Dispatches canonical packet dicts to memory
            │
            ├───► [ Flow Tracker ] (telemetry/flow/flow_table.py)
            │       - Active 5-tuple hash table: (src_ip, dst_ip, src_port, dst_port, proto)
            │       - Bidirectional packet/byte counters
            │       - Inter-arrival times, idle intervals, TCP flags
            │       - Port delta & entropy trackers
            │
            └───► [ PCAP Engine ] (telemetry/packet/pcap_engine.py)
                    - Reuses mathematical definitions from model/src/pcap_features.py
                    - Unanswered SYN ratio & handshake completion
                    - TCP window mean/std & sequence retransmission detection
                    - RFC-1918 internal graph host-to-host edge expansion & fanout
                    - L4 payload moments & IP fragmentation
            │
            ▼ (Every 2.0s Timer Boundary)
[ State Builder ] (telemetry/state/state_builder.py)
  - Closes window [t, t+2.0s)
  - Fuses 42 flow features + 2 metadata + 28 PCAP features -> 72-D vector
  - Applies logarithmic transform: ln(1 + max(0, x))
  - Applies frozen standard scaling: (x_log - mu) / (sigma + 1e-5)
  - Enqueues into in-memory sequence buffer: CausalSlidingBuffer(maxlen=15)
            │
            ▼ (When buffer has 15 states)
[ V3.1 Model Adapter ] (telemetry/inference/adapter.py)
  - Feeds tensor [1, 15, 72] to CyberWorldModelEXP14B(in_dim=72)
  - Obtains multi-task forecast
  - Measures processing latency (packet -> state -> prediction)
```

### 5.1 Dual-Mode Operation (Live vs Replay)
* **Live Mode:** Continuously processes frames off `sensor:eth1`.
* **Replay Mode:** Reads historical PCAP files through the exact same parsing and state-building logic, allowing deterministic demo reproduction for competition judges.
* **Asynchronous Recording:** If `--record-pcap` or `--record-state` is enabled, data is enqueued to a background worker thread via a queue (`queue.Queue`) and saved to disk without blocking the live hot path.

---

## 6. Latency Instrumentation

Every emitted prediction logs precise execution timing:

* $t_{\text{obs}}$: Timestamp of first packet in window.
* $t_{\text{close}}$: Exact boundary time of window closure ($t + 2.0\text{s}$).
* $t_{\text{state\_start}}$, $t_{\text{state\_end}}$: Feature extraction and scaling duration.
* $t_{\text{inf\_start}}$, $t_{\text{inf\_end}}$: Model forward pass duration.

**Key Latency Metrics:**
* **Pipeline Latency:** $\Delta t_{\text{pipe}} = t_{\text{state\_end}} - t_{\text{close}}$ (Target: $< 5\text{ms}$).
* **Inference Latency:** $\Delta t_{\text{inf}} = t_{\text{inf\_end}} - t_{\text{inf\_start}}$ (Target: $< 15\text{ms}$ on CPU).
* **Total Processing Delay:** $\Delta t_{\text{total}} = \Delta t_{\text{pipe}} + \Delta t_{\text{inf}}$ (Target: $< 20\text{ms}$).
* **Early Warning Lead Time:** $6.6\text{ seconds}$ average advance notice before malicious impact, clearly separated from processing delay.

---

## 7. Future Attack Scenarios (Phase 2 Readiness)

The architecture is prepared for controlled cyber attack injection from the isolated `attacker` node in future phases without altering telemetry or model code:

1. **Reconnaissance:** Horizontal ping sweeps and vertical SYN port scans against `router-edge` and `dmz-web`. (Triggers `pcap_unanswered_syn_ratio` and `pcap_unique_dst_ports`).
2. **Perimeter Exploitation:** HTTP credential stuffing or vulnerability probing on `dmz-web`.
3. **Lateral Movement:** Compromise pivoting into internal subnets targeting SMB (445) and RDP (3389). (Triggers `pcap_new_internal_edges` and `pcap_internal_fanout`).
4. **Command & Control (C2):** Periodic beaconing with distinctive packet inter-arrival jitter. (Triggers `pcap_pkt_iat_std` and `flow_iat_std`).
