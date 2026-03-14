"""
TrackMania Nations Forever AI – main loop
==========================================
Controls (while running):
  p  – pause / unpause
  t  – toggle background training
  s  – force-save model
  r  – manually restart race (also happens automatically when stuck)
  q  – quit

Keys sent to the game (fixed, no configuration needed):
  Arrow Up    – gas
  Arrow Left  – steer left
  Arrow Right – steer right
  S           – brake
"""

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import keyboard
import cv2
import win32gui
import threading
from collections import deque

from screen_capture import ScreenCapture
from input_controller import InputController
from neural_network import OsuNeuralNetwork
from reward_system import RewardSystem

GAME_WINDOW = "TmForever"


def is_window_foreground(window_title: str) -> bool:
    try:
        hwnd = win32gui.GetForegroundWindow()
        return window_title.lower() in win32gui.GetWindowText(hwnd).lower()
    except Exception:
        return False


def continuous_training(neural_network, stop_event, pause_event, save_event,
                        save_interval=500):
    print("Training thread started.")
    epoch_counter = 0
    while not stop_event.is_set():
        if pause_event.is_set():
            time.sleep(0.5)
            continue
        if len(neural_network.memory_buffer) < 32:
            time.sleep(2)
            continue
        batch_size = min(32, len(neural_network.memory_buffer))
        t0 = time.time()
        history = neural_network.train(batch_size=batch_size, epochs=1)
        epoch_counter += 1
        if epoch_counter % save_interval == 0 or save_event.is_set():
            neural_network.save_model()
            print(f"\n[Training] Model saved at epoch {epoch_counter}")
            save_event.clear()
        if epoch_counter % 10 == 0 and history:
            loss = history.history.get('loss', [0])[0]
            print(f"[Training] Epoch {epoch_counter}  loss={loss:.4f}  "
                  f"dt={time.time()-t0:.2f}s  buf={len(neural_network.memory_buffer)}")
        time.sleep(0.05)
    print("Training thread stopped.")


def draw_overlay(display_frame, action, stats, training, game_active, lidar_obs):
    """
    Draw a HUD overlay onto display_frame in-place.
    action: [gas, left, right, brake]
    """
    h, w = display_frame.shape[:2]
    names  = ["GAS",    "LEFT",  "RIGHT", "BRAKE"]
    colors = [(0,255,0),(0,200,255),(0,200,255),(0,0,255)]

    for i, (name, col) in enumerate(zip(names, colors)):
        state = "ON " if action[i] > 0.5 else "OFF"
        on_col = col if action[i] > 0.5 else (80, 80, 80)
        cv2.putText(display_frame, f"{name}: {state}",
                    (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, on_col, 2)

    # Speed
    spd_text = f"Speed: {stats['speed']:.0f} km/h  (best: {stats['best_speed']:.0f})"
    cv2.putText(display_frame, spd_text, (10, h - 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # Reward
    rew_text = f"Avg reward: {stats['avg_reward']:.3f}  ep: {stats['episode_reward']:.1f}"
    cv2.putText(display_frame, rew_text, (10, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 2)

    # Training status
    train_label = "TRAINING: ON" if training else "TRAINING: OFF"
    train_col = (0, 255, 0) if training else (0, 0, 255)
    cv2.putText(display_frame, train_label, (10, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, train_col, 2)

    # Game window status
    gw_label = "TMNF ACTIVE" if game_active else "TMNF NOT ACTIVE"
    gw_col   = (0, 255, 0) if game_active else (0, 0, 255)
    cv2.putText(display_frame, gw_label, (w - 220, h - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, gw_col, 2)

    # TMInterface indicator
    if stats.get('tmi_active'):
        cv2.putText(display_frame, "TMI", (w - 55, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # LIDAR bar chart along the bottom edge
    if lidar_obs is not None and len(lidar_obs) > 0:
        bar_w = w // len(lidar_obs)
        for i, dist in enumerate(lidar_obs):
            bar_h = int(dist * 40)
            x0 = i * bar_w
            cv2.rectangle(display_frame,
                          (x0, h - bar_h - 5), (x0 + bar_w - 2, h - 5),
                          (100, 255, 100), -1)


def main():
    print("=" * 50)
    print("  TrackMania Nations Forever AI")
    print("=" * 50)
    print("Controls: Up=gas  Left/Right=steer  S=brake")
    print()

    # ── Initialise components ────────────────────────────────────────────────
    screen_capture   = ScreenCapture()
    input_controller = InputController()
    reward_system    = RewardSystem(use_tmi=True)
    neural_network   = OsuNeuralNetwork()

    try:
        neural_network.load_model()
        print("Loaded existing model.")
    except Exception:
        print("No saved model found — starting fresh.")

    # Frame stack: 4 grayscale 128×128 frames → shape (128,128,4)
    FRAME_STACK_SIZE = 4
    FRAME_SIZE = (128, 128)
    frame_stack = deque(maxlen=FRAME_STACK_SIZE)
    blank = np.zeros((*FRAME_SIZE, 1), dtype=np.float32)
    for _ in range(FRAME_STACK_SIZE):
        frame_stack.append(blank)

    # Exploration
    epsilon       = 1.0
    epsilon_min   = 0.05
    epsilon_decay = 0.9995

    # Background training thread
    save_interval       = 500
    stop_training_event  = threading.Event()
    pause_training_event = threading.Event()
    save_model_event     = threading.Event()

    training_thread = threading.Thread(
        target=continuous_training,
        args=(neural_network, stop_training_event,
              pause_training_event, save_model_event, save_interval),
        daemon=True,
    )
    training_thread.start()

    # OpenCV debug window
    cv2.namedWindow("TrackMania AI", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TrackMania AI", 480, 360)

    rewards    = []
    fps_values = []
    iteration  = 0
    running    = True
    paused     = True
    training   = True

    print()
    print("PAUSED. Switch to the TrackMania window, then press P to start.")
    print("  P – pause/unpause    T – toggle training")
    print("  S – save model       R – restart race")
    print("  Q – quit")
    print()

    # ── Main loop ────────────────────────────────────────────────────────────
    while running:
        loop_start = time.time()

        # ── Hot-keys ──────────────────────────────────────────────────────────
        if keyboard.is_pressed('p'):
            paused = not paused
            if paused:
                input_controller.release_all()
                print("\n[AI] Paused.")
            else:
                print("[AI] Running.")
            time.sleep(0.3)

        if keyboard.is_pressed('t'):
            training = not training
            if training:
                pause_training_event.clear()
                print("[AI] Training ON")
            else:
                pause_training_event.set()
                print("[AI] Training OFF")
            time.sleep(0.3)

        if keyboard.is_pressed('s'):
            save_model_event.set()
            print("[AI] Saving model...")
            time.sleep(0.3)

        if keyboard.is_pressed('r'):
            print("[AI] Manual race restart.")
            input_controller.restart_race()
            reward_system.reset_episode()
            time.sleep(0.3)

        if keyboard.is_pressed('q'):
            running = False
            continue

        # ── Paused mode ───────────────────────────────────────────────────────
        if paused:
            frame = screen_capture.capture()
            disp  = frame.copy()
            cv2.putText(disp, "PAUSED",
                        (disp.shape[1] // 2 - 80, disp.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)
            cv2.imshow("TrackMania AI", disp)
            cv2.waitKey(1)
            time.sleep(0.1)
            continue

        # ── Active mode ───────────────────────────────────────────────────────
        game_active = is_window_foreground(GAME_WINDOW)
        if not game_active and iteration % 30 == 0:
            print(f"[AI] WARNING: {GAME_WINDOW} window not active!")

        # Capture
        frame             = screen_capture.capture()
        processed_frame   = screen_capture.capture_preprocessed(target_size=FRAME_SIZE)
        lidar_obs         = screen_capture.get_lidar_obs(n_rays=16)

        # Stack frames
        frame_stack.append(processed_frame)
        stacked_frame = np.concatenate(list(frame_stack), axis=-1)  # (128,128,4)

        # Predict action
        action = neural_network.predict(stacked_frame, epsilon=epsilon)

        # Get stats for overlay
        stats = reward_system.get_stats()

        # Draw overlay on a copy of the raw frame
        display_frame = frame.copy()
        draw_overlay(display_frame, action, stats, training, game_active, lidar_obs)
        cv2.imshow("TrackMania AI", display_frame)
        cv2.waitKey(1)

        # Send inputs to game
        if game_active:
            input_controller.execute_action(action)

        # Reward
        reward = reward_system.calculate_reward(frame, action, lidar_obs)
        rewards.append(reward)

        # Store experience
        neural_network.add_to_memory(stacked_frame, action, reward)

        # Auto-restart if stuck
        if reward_system.get_stats()['stuck']:
            print("[AI] Car stuck — restarting race.")
            input_controller.restart_race()
            reward_system.reset_episode()
            # Reset frame stack
            for _ in range(FRAME_STACK_SIZE):
                frame_stack.append(blank)

        # Decay exploration
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        iteration += 1

        # Periodic console stats
        if iteration % 200 == 0:
            s = reward_system.get_stats()
            print(f"\n[AI] iter={iteration}  speed={s['speed']:.0f} km/h  "
                  f"avg_speed={s['avg_speed']:.0f}  best={s['best_speed']:.0f}  "
                  f"eps={epsilon:.3f}  buf={len(neural_network.memory_buffer)}")

        # Periodic reward plot
        if iteration % 500 == 0 and len(rewards) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(rewards[-200:])
            plt.title('Reward – last 200 steps')
            plt.xlabel('Step')
            plt.ylabel('Reward')
            plt.tight_layout()
            plt.savefig('rewards.png')
            plt.close()

        # FPS tracking
        fps = 1.0 / max(time.time() - loop_start, 1e-6)
        fps_values.append(fps)
        if iteration % 10 == 0:
            avg_fps    = float(np.mean(fps_values[-10:]))
            avg_reward = float(np.mean(rewards[-10:])) if rewards else 0.0
            print(f"[AI] iter={iteration:6d}  fps={avg_fps:5.1f}  "
                  f"reward={avg_reward:+.3f}  eps={epsilon:.3f}", end='\r')

    # ── Shutdown ──────────────────────────────────────────────────────────────
    print("\n[AI] Shutting down...")
    input_controller.release_all()
    stop_training_event.set()
    training_thread.join(timeout=5)
    cv2.destroyAllWindows()

    neural_network.save_model()
    reward_system.shutdown()

    s = reward_system.get_stats()
    print("\nFinal stats:")
    print(f"  Total steps  : {s['total_steps']}")
    print(f"  Best speed   : {s['best_speed']:.0f} km/h")
    print(f"  Avg speed    : {s['avg_speed']:.0f} km/h")
    print(f"  Avg reward   : {s['avg_reward']:.4f}")

    plt.figure(figsize=(10, 4))
    plt.plot(rewards)
    plt.title('Full reward history')
    plt.xlabel('Step'); plt.ylabel('Reward')
    plt.tight_layout()
    plt.savefig('rewards_total.png')

    plt.figure(figsize=(10, 4))
    plt.plot(fps_values)
    plt.title('FPS history')
    plt.xlabel('Step'); plt.ylabel('FPS')
    plt.tight_layout()
    plt.savefig('fps_total.png')

    print("Done.")


if __name__ == "__main__":
    main()
