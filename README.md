# Chopstix: Robotic Manipulation with Chopsticks

Teaching a dexterous robotic hand to be asian. 

### Prerequisites
- Python 3.10+
- [MuJoCo](https://mujoco.org/)
- `mujoco-menagerie`

### Running the Scripts

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
