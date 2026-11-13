# Safety Guideline for AI Agents — Unitree Go2

This document defines the behaviors and state changes that **AI agents must NOT perform** while operating on the Unitree Go2. It is meant to be attached to any agent prompt, API contract, or control loop. Violations can cause injury to people, damage to hardware, or loss of the robot.

> This is a "do not" list. Agents are free to operate within the safe envelope described below unless these constraints are explicitly overridden by a human operator.

---

## 0. Scope & Definitions

- **Agent** = any software (LLM-driven controller, script, or autonomous module) that issues commands to the Go2.
- **Operator** = a human who holds physical or remote authority and can override agents.
- **Safe envelope** = the set of allowed robot states and command ranges. Everything outside it requires explicit human approval.

---

## 1. Motion & Operation Modes (DO NOT TOUCH)

The Go2's operating modes control how much autonomy and speed it allows. Agents must not change these unless explicitly permitted.

- **Do NOT activate / switch to running or fast trot modes.** These increase speed and dynamic forces beyond what an AI agent should safely command, especially on uneven terrain or near people.
- **Do NOT change gait / locomotion mode** from standing or slow-walk to anything more aggressive (e.g., sprint, jump, dance routines) without human approval.
- **Do NOT enable autonomous "follow me" / social navigation modes** on their own. Follow mode can drive the robot toward people and objects unexpectedly.
- **Do NOT change the height / leg-offset calibration** (e.g., raise the robot to a higher standing height). Altering this affects balance and center of mass.
- **Do NOT command rapid or high-torque joint movements** without a validated, human-approved trajectory.

---

## 2. Safety Systems & Sensors (DO NOT TOUCH)

These are protective features. Disabling them removes the robot's ability to avoid collisions and falls.

- **Do NOT disable object detection / obstacle avoidance.** Keep collision detection active at all times while the robot is moving.
- **Do NOT disable emergency stop (E-stop) logic** or override the collision-response behavior.
- **Do NOT disable tilt / fall-detection** or the robot's auto-balance and self-righting safeguards.
- **Do NOT turn off or bypass IMU-based stabilization.** A disabled IMU can cause the robot to tip over.
- **Do NOT disable low-battery / power shutdown protection**, or force the robot to keep moving past a safe charge threshold.
- **Do NOT override floor / edge detection** (avoid falling down stairs or off ledges).

---

## 3. Limits & Command Constraints (DO NOT EXCEED)

Agents should operate within conservative bounds. Do not exceed these unless a human explicitly approves higher limits:

- **Do NOT exceed the approved maximum speed** for the current environment (default to slow-walk indoors and near people).
- **Do NOT command travel outside the approved area / geofence.** Stay inside the known, human-defined workspace.
- **Do NOT approach obstacles closer than the minimum safe distance** while moving.
- **Do NOT exceed the approved heading / joint-angle limits** or command poses the robot was not calibrated for.
- **Do NOT command sustained high load** (e.g., carrying objects) beyond the robot's rated capacity without approval.

---

## 4. Teleoperation & Control Authority (DO NOT STEAL)

- **Do NOT take control away from the human operator / remote controller.** The operator always retains authority.
- **Do NOT disable the teleop / joystick safety key** or remove the human from the control loop.
- **Do NOT switch control ownership** between agent and operator autonomously. Ownership transfer must be an explicit, approved handshake.
- **Do NOT continue autonomous motion when a human issues a stop.** On any E-stop or stop command, the agent must immediately release all motion commands.

---

## 5. Emergency Behavior (DO NOT VIOLATE)

When safety is triggered, agents must react as follows and must not do the opposite:

- **On collision / obstacle trigger:** Stop, hold pose, and notify the operator. Do *not* keep pushing forward.
- **On fall / tilt detection:** Release to a safe pose; do *not* command aggressive recovery maneuvers without human approval.
- **On E-stop:** Immediately cease all motion commands and wait for the reset by an operator.
- **On loss of sensors (camera / depth):** Do *not* continue moving autonomously. Switch to a stopped, notify state.
- **On communication loss with controller:** Do *not* wander or continue a task. Stop and hold in place until link is restored.

---

## 6. Environmental & Human Safety (DO NOT IGNORE)

- **Do NOT operate without a clear human-supervision plan.** Never run unsupervised in occupied spaces.
- **Do NOT operate near stairs, ledges, water, or steep ramps** without explicit approval and active edge detection.
- **Do NOT run at high speed on slippery, uneven, or cluttered surfaces.** Slow down or stop.
- **Do NOT approach people from behind or at speed.** Keep a safe distance and slow down when humans are present.
- **Do NOT lift, push, or interact with people** — the Go2 is a robotic platform, not designed for human contact payloads.

---

## 7. Software, Code & Configuration (DO NOT MODIFY freely)

- **Do NOT modify safety-critical parameters** (speed limits, obstacle thresholds, tilt thresholds) in configuration files without review.
- **Do NOT disable logging / telemetry.** Keep motion, collision, and E-stop logs intact for audit.
- **Do NOT remove or bypass safety guards** in the control code (try/except swallowing of sensor errors, suppressed warnings).
- **Do NOT deploy untested agent code directly to the robot.** Validate in simulation or a controlled area first.

---

## 8. Agent Design Requirements (MUST INCLUDE)

Every agent operating the Go2 must include these safety features by design:

1. **Explicit stop command** — always accept a remote/human stop and honor it immediately.
2. **Heartbeat / watchdog** — if control link or sensor data stops, the agent stops the robot automatically.
3. **Speed governor** — auto reduce speed when humans or obstacles are detected nearby.
4. **State machine** — always know its current mode (stopped / walking / teleop) and only transition with approval.
5. **Fail-safe default** — if uncertain, the agent should STOP rather than continue.
6. **Audit log** — record every command, state change, and safety event.

---

## Quick Reference Card

| Category | Agents must NOT |
|----------|-----------------|
| Modes | Activate running / fast / jump / dance modes |
| Sensors | Disable object detection or collision avoidance |
| Safety | Disable E-stop, tilt, fall, or IMU protection |
| Limits | Exceed approved speed / area / joint limits |
| Control | Steal authority from the human operator |
| Emergency | Keep moving on collision / fall / E-stop |
| Env. | Run near stairs, people, or uneven terrain unsupervised |
| Code | Modify safety params or disable logging |

---

*Review this guideline periodically and update it whenever new operation modes or sensors are added to the Unitree Go2.*

## Handling of development on the GO2
The go2s Nvidia Jetson should never be touched, work is only permitted within the designated container.
Never change any configurations or touch anything outside the container.
