# Chopstix: Robotic Manipulation with Chopsticks

Teaching a dexterous robotic hand to grasp and lift simple items using chopsticks via machine learning in simulation.

---

## 📌 Topic Overview

This project explores using machine learning to teach a robotic hand how to grasp simple items and lift them using chopsticks. All development and experiments are conducted in physical simulation (e.g., MuJoCo, Isaac Sim, Meta's Project SuperDex), as physical hardware access to high-degree-of-freedom dexterous hands is outside the scope of this 4-month project.

---

## 💡 Motivations

- **Dexterous Manipulation Challenge:** Using chopsticks requires fine coordination, contact stability, and dynamic force adjustment—representing a classic open challenge at the cutting edge of robotic manipulation and dexterous hand learning.
- **Relatability & Fun:** As regular chopstick users, building an autonomous system capable of dynamic chopstick manipulation is both technically compelling and entertaining to demonstrate to peers.

---

## 🔬 Proposed Approach

### 1. Imitation Learning (IL)
- **Methods:** Behavior Cloning (BC), DeepMimic, or DAgger.
- **Role:** Sets an initial kinematic and dynamic prior for how the hand should grip chopsticks and manipulate target objects. Having proper grasp demonstrations provides early guidance to avoid unnatural or unstable grips.
- **Data Collection:** Demonstrations can be gathered via human video, motion capture, or simulated teleoperation.
- **Challenges:** Retargeting human hand kinematics and link lengths to robotic hand morphology, along with data cleaning and contact alignment.

### 2. Reinforcement Learning (RL)
- **Environments:** Physics-based simulators including **MuJoCo**, **Isaac Sim**, and **Meta's Project SuperDex**.
  - *Project SuperDex* is particularly well-suited due to native contact/soft-tissue compliance modeling and a Meta Quest 3 teleoperation pipeline for demonstration collection in simulation.
- **Reward Formulation:**
  - Distance between chopstick tips and target object.
  - Grip force stability and contact persistence.
  - Successful lift height and trajectory smoothness.

### 3. Hybrid Strategies
- Initialize the policy using Imitation Learning to learn the fundamental grasp posture.
- Fine-tune via Reinforcement Learning to optimize force control, disturbance rejection, and grasp stability.
- Use simulated teleoperation to supply additional expert interventions when the policy fails.

### 4. Evaluation Metrics
- **Success Rate:** Percentage of successful pinch-and-lift trials without dropping the item.
- **Grip Stability:** Average normal and shear force variance across trials.

---

## 📅 Milestones & Timeline

| Timeline | Milestone |
| :--- | :--- |
| **End of September** | Baseline tests for LEAP hand in MuJoCo |
| **Mid October** | Chopsticks actuated under PD control (compatible with LEAP/Shadow controller setup) |
| **End of October** | Set up complete simulation and training environment |
| **November** | Policy training (IL / RL exploration) |
| **End of November** | Final demo and evaluation |

---

## 🤖 Hardware & Compute Notes

- **Robotic Hands:**
  - **[Shadow Robot Dexterous Hand](https://shadowrobot.com/dexterous-hand-series/):** Primary proposed hand, chosen for its high degrees of freedom and human-like dexterity.
  - **LEAP Hand:** Used as an initial exploratory platform and baseline comparison.
- **Compute Resources:**
  - If following the RL route, GPU acceleration will utilize the University's NVIDIA Tesla clusters (coordinating with Prof. Sanjeev Bedi for access upon proposal acceptance).

---

## 📁 Repository Structure

```text
├── README.md               # Project proposal, overview, and roadmap
├── .gitignore              # Ignored files (virtual environments, cache)
├── move_shadow_joint.py    # MuJoCo script testing Shadow Hand joint actuation and passive viewer
├── test_shadow.py          # Actuator and joint range inspection for Shadow Hand
├── test_leap.py            # Inspection script for LEAP hand kinematics and actuators
└── move_joint.py           # Joint control scratchpad
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [MuJoCo](https://mujoco.org/)
- `mujoco-menagerie`

### Running the Scripts

Activate your virtual environment and run the test scripts:

```bash
# Test Shadow Hand joint movement with interactive viewer
python move_shadow_joint.py
# or with mjpython on macOS:
mjpython move_shadow_joint.py

# Inspect Shadow Hand actuators
python test_shadow.py

# Inspect LEAP hand model & mappings
python test_leap.py
```
