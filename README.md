# TrackMania Nations Forever AI

A self-learning neural network AI that plays TrackMania Nations Forever.
It watches the screen, processes the image to detect track borders, and
controls the car using the arrow keys and S (brake) — just like a human.

Inspired by / borrows patterns from:
- [TMAI](https://github.com/LouisDeOliveira/TMAI) — LIDAR raycasting and
  DirectInput key simulation patterns
- [TMInterface](https://donadigo.com/tminterface/) — optional game-state API

---

## Controls sent to the game

| Key         | Action       |
|-------------|--------------|
| Arrow Up    | Gas          |
| Arrow Left  | Steer left   |
| Arrow Right | Steer right  |
| S           | Brake        |
| Delete      | Restart race |

## Hotkeys while the AI is running

| Key | Function               |
|-----|------------------------|
| P   | Pause / unpause        |
| T   | Toggle training thread |
| S   | Force-save model       |
| R   | Manually restart race  |
| Q   | Quit                   |

---

## Requirements

- Python 3.9+
- TrackMania Nations Forever (windowed mode recommended)
- See `requirements.txt`

### Optional but strongly recommended: TMInterface

Install [TMInterface](https://donadigo.com/tminterface/) to let the AI read
speed and car physics directly from game memory instead of relying on OCR.

After installing TMInterface:
```
pip install tminterface
```
Then launch the game **through TMInterface** (not directly via Steam).

---

## Setup

```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

---

## Usage

1. Launch TrackMania Nations Forever (windowed mode).
2. Load a track and get to the starting line.
3. Run the AI:
   ```
   python main.py
   ```
4. Switch focus to the TrackMania window.
5. Press **P** to unpause and let the AI drive.

---

## How it works

### Screen capture & LIDAR
Each frame the raw game window is captured. It is processed with edge
detection (grayscale → threshold → Canny → dilate → blur) to produce a clean
binary image of track borders. 16 rays are cast from the bottom-centre
of this image to measure how much road is visible in each direction.

### Neural network
A small CNN stacks 4 consecutive 128×128 grayscale frames as input channels
(for temporal context — the AI can "see" motion). The output is 4 sigmoid
neurons representing the probability of holding each key:
`[gas, steer_left, steer_right, brake]`.

### Reward system
- **+speed / 400** — reward proportional to how fast the car is going
- **+gas × 2** — encourage pressing the accelerator
- **−0.3** per step — small constant cost to encourage finishing quickly
- **−roll penalty** — penalise flipping (TMInterface only)
- **Escalating stuck penalty** — the AI automatically restarts the race when
  it has been nearly stationary for too long
- **No edge penalty** — wall-riding is perfectly fine and sometimes fastest

### Training
Experiences (frame, action, reward) are stored in a replay buffer of 10 000
entries. A background thread samples random batches and trains the network
continuously using advantage-weighted regression.

---

## Project structure

| File                  | Purpose                                      |
|-----------------------|----------------------------------------------|
| `main.py`             | Main loop, hotkeys, display overlay          |
| `screen_capture.py`   | Frame capture, edge detection, LIDAR         |
| `input_controller.py` | DirectInput scan-code keyboard control       |
| `neural_network.py`   | CNN model, replay buffer, training           |
| `reward_system.py`    | Reward calculation (TMInterface + fallback)  |
