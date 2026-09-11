Create the frontend control system for the existing Cyber Network Predictor dashboard. Do not invent backend functionality. Every control below must correspond to the real backend behavior.

BACKEND:
REST:
GET /api/status
GET /api/site
GET /api/topology
POST /api/command/{cmd_name}
POST /api/mitigate

WebSocket:
WS /ws

Events:
prediction, topology_update, ml_reset, system_status, command_started, command_completed, command_output, attack_event.

COMMAND EXECUTION:
Only one command can run at a time. When command_started arrives, show the command as EXECUTING and disable conflicting controls. When command_completed arrives, use its success/exit status to show SUCCESS or FAILED. Show command_output messages in the console.

LAB MODE CONTROLS:

START NETWORK
Call POST /api/command/start_network.
Backend runs ./scripts/deploy.sh to deploy the Containerlab enterprise network.
On success network_online=true.
Only available when network is offline.

HEALTH CHECK
Call POST /api/command/healthcheck.
Backend runs ./scripts/healthcheck.sh.
Checks connectivity/security policy of the deployed network.
Requires network_online=true.

STOP NETWORK
Call POST /api/command/stop_network.
Backend runs ./scripts/destroy.sh.
Destroys the Containerlab network. Also clears workload/attack state and stops telemetry.
Require confirmation.

RESET ENVIRONMENT
Call POST /api/command/reset_environment.
Resets operational state: stops ML/workloads, disarms external monitoring, clears topology, clears isolated hosts, blocked IPs, blocked ports and revoked-credential state, and resets ML history.
This is NOT the same as destroying Containerlab.

TELEMETRY:

START SENSOR
Call POST /api/command/start_telemetry.
Starts passive SPAN/network packet telemetry. In Lab Mode it starts the lab telemetry service; in Local SPAN Mode it captures from the configured network interface.
It resets telemetry/model history and begins streaming flow windows.
This does NOT start ML.
Requires the network/sensor environment to be available.

STOP SENSOR
Call POST /api/command/stop_telemetry.
Stops passive packet capture, stops ML, clears predictions and resets topology.

VERIFY
Call POST /api/command/verify_telemetry.
Lab Mode only.
Runs ./scripts/verify_telemetry.sh to verify packet capture and telemetry state generation.
This is a diagnostic check, not ML inference.

ML WORLD MODEL:

START ML
Call POST /api/command/start_ml.
Requires the sensor/telemetry to already be running.
Activates live ML inference on incoming telemetry windows.
Display ML: LIVE.

STOP ML
Call POST /api/command/stop_ml.
Stops ML inference and clears current prediction/model history.
The telemetry sensor can remain running.
Display ML: STANDBY.

NORMAL WORKLOADS — LAB ONLY:

START WORKLOADS
Call POST /api/command/start_normal_traffic.
Starts normal enterprise background traffic using workloads/user_workload.py.
Traffic profiles represent realistic office browsing/DNS/idle, web-heavy, file-heavy and application-heavy behavior.
This is NORMAL traffic, not attack traffic.

STOP WORKLOADS
Call POST /api/command/stop_normal_traffic.
Stops the background workload processes.

EXTERNAL ATTACK MONITORING:

ARM EXTERNAL
Call POST /api/command/start_attack.
Sets external_attack_armed=true and attack_running=true and broadcasts an attack event.
IMPORTANT: this does NOT launch an attack. It arms the system to observe external/operator-driven attack traffic.
Label it "ARM EXTERNAL" or "ARM EXTERNAL MONITORING", never "LAUNCH ATTACK".

DISARM
Call POST /api/command/stop_attack.
Sets attack_running=false and external_attack_armed=false and broadcasts the stopped state.
Display EXTERNAL ARMED / EXTERNAL IDLE.

DEFENSIVE SOAR:

These actions currently RECORD mitigation state in the backend. They do NOT directly configure a firewall, router, switch or identity provider. Never claim that actual infrastructure enforcement occurred.

ISOLATE HOST
POST /api/mitigate
Payload: {"action":"ISOLATE_HOST","target":"<IP>"}
Records the host as isolated. The telemetry/model pipeline filters flows involving isolated hosts.
Show the target IP.

BLOCK IP
POST /api/mitigate
Payload: {"action":"BLOCK_IP","target":"<IP>"}
Records the IP as blocked. Telemetry filtering excludes matching flows.
Open an IP input modal before submitting.

BLOCK PORT
POST /api/mitigate
Payload: {"action":"BLOCK_PORT","target":"<PORT>"}
Records the port as blocked and telemetry filtering excludes matching destination ports.
Current frontend defaults to port 80, so make the port explicit in the UI instead of hiding it.

REVOKE CREDENTIALS
POST /api/mitigate
Payload: {"action":"REVOKE_CREDENTIALS","target":"compromised_admin"}
Records credential revocation state.
It does NOT actually revoke an account through an identity provider.
Show the affected account explicitly.

CLEAR DEFENSES
POST /api/mitigate
Payload: {"action":"CLEAR_DEFENSES","target":null}
Clears isolated hosts, blocked IPs, blocked ports and revoked credential state and resets model history.
This is NOT the same as RESET ENVIRONMENT.

INVESTIGATION:

NETWORK TOPOLOGY
Make hosts clickable.
Selecting a host opens details showing hostname, IP, role, zone, traffic volume and risk score.
Provide a close button.

EXPLAINABILITY
"VIEW TOP FEATURE DRIVERS" expands the feature-attribution details behind the prediction.
When expanded, change to "HIDE FEATURE DRIVERS".

CONSOLE:

Filters:
ALL
CONTROL
MODEL
ATTACK
TELEMETRY

AUTO-SCROLL:
ON follows new log entries.
PAUSED lets the operator inspect historical logs.

EXPORT:
Export the current console log.

CLEAR:
Clear the visible console only.

CONTROL DEPENDENCIES:

START NETWORK → network online → START SENSOR → telemetry running → START ML.

HEALTH CHECK and VERIFY require the lab network.

START WORKLOADS requires the lab network.

START ML requires telemetry.

STOP SENSOR also stops ML.

STOP NETWORK clears/stops lab-dependent services.

Only one command can execute at a time.

DESIGN:

Create a professional enterprise SOC/network-operations interface.

Use:
white/off-white background, white cards, subtle gray borders, compact spacing, 6–8px radius, dark typography, restrained green/amber/red/blue status colors, monospace for IPs/timestamps/logs.

Avoid:
neon, cyberpunk, glowing borders, glassmorphism, sci-fi HUDs, giant gauges, excessive gradients and decorative AI graphics.

Organize controls into:
LAB / SITE
NETWORK
TELEMETRY
ML WORLD MODEL
WORKLOADS
EXTERNAL MONITORING
DEFENSIVE SOAR

Give primary operational controls the strongest visual hierarchy:
START NETWORK, START SENSOR, START ML.

Make destructive actions visually distinct and require confirmation:
STOP NETWORK, RESET, ISOLATE HOST, BLOCK IP, BLOCK PORT, REVOKE CREDENTIALS.

Every control needs clear states:
AVAILABLE
DISABLED
EXECUTING
SUCCESS
FAILED
RUNNING
STOPPED

Show why a disabled control is unavailable, for example:
"Start ML — telemetry sensor is offline."

Keep the UI modular and frontend-only. Do not create a new backend, database or authentication system. The interface must map directly to the REST and WebSocket contract above and remain easy to connect to the existing React implementation.