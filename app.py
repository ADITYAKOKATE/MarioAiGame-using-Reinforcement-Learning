import time
import threading
import io
import os
import cv2
from queue import Queue, Empty
from flask import Flask, render_template, Response, jsonify

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from mario_env import MarioEnv

# Make pygame headless so it doesn't open a desktop window
os.environ["SDL_VIDEODRIVER"] = "dummy"

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    return r


# --- Global State ---
game_running = False
frame_queue = Queue(maxsize=30)
log_queue = Queue(maxsize=100)
game_thread = None

def make_env():
    def _init():
        # Force rgb_array to capture frames without needing a window
        env = MarioEnv(render_mode="rgb_array")
        return env
    return _init

def game_loop():
    global game_running
    
    # Initialize Environment
    # Important: Create environment here inside the thread
    env = DummyVecEnv([make_env()])
    env = VecFrameStack(env, n_stack=4)
    
    # Find model
    model = None
    paths_to_try = ["mario_ppo_final", "mario_ppo_interrupted", "train/mario_model_200000_steps"]
    for path in paths_to_try:
        try:
            model = PPO.load(path)
            log_queue.put({"type": "info", "msg": f"Model loaded: {path}"})
            break
        except FileNotFoundError:
            pass
            
    if model is None:
        log_queue.put({"type": "error", "msg": "Could not find trained model."})
        game_running = False
        return

    obs = env.reset()
    done = False
    total_reward = 0
    step = 0
    stuck_counter = 0
    last_x_pos = 0

    log_queue.put({"type": "success", "msg": "Game loop started."})

    try:
        while game_running:
            action, _states = model.predict(obs, deterministic=True)
            
            # --- Smart Heuristics ---
            current_x = env.envs[0].player_pos[0]
            current_y = env.envs[0].player_pos[1]
            
            if abs(current_x - last_x_pos) < 1:
                stuck_counter += 1
            else:
                stuck_counter = 0
            last_x_pos = current_x
            
            heuristic_active = False
            # If stuck
            if stuck_counter > 10:
                action = [2]
                heuristic_active = True
                if stuck_counter > 15: stuck_counter = 0 
                
            # Pit safety
            if (380 < current_x < 400) or (780 < current_x < 800) or (1280 < current_x < 1300):
                 action = [2]
                 heuristic_active = True
                 stuck_counter = 0

            # Step the environment
            obs, reward, done, info = env.step(action)
            total_reward += reward
            
            # Action logging
            explanation = "FORCE JUMP!" if heuristic_active else ""
            act_str = ["Stay", "Right", "Jump"][int(action[0])]
            
            # Only log important events (heuristic jump, or every 50th step as a minor checkpoint)
            if heuristic_active:
                log_msg = f"Step {step:04d} | Used Heuristic: {act_str:5} | Pos: ({current_x:.1f}, {current_y:.1f}) {explanation}"
                log_queue.put({"type": "action", "msg": log_msg})
            elif step > 0 and step % 50 == 0:
                 log_msg = f"Checkpoint {step:04d} | Pos: ({current_x:.1f}, {current_y:.1f})"
                 log_queue.put({"type": "info", "msg": log_msg})
            
            # Extract rgb frame for web streaming
            import pygame
            mario = env.envs[0]
            view_rect = pygame.Rect(int(mario.camera_x), 0, mario.window_width, mario.window_height)
            viewport = mario.surf.subsurface(view_rect)
            view_str = pygame.image.tostring(viewport, "RGB")
            frame = np.frombuffer(view_str, dtype=np.uint8).reshape((mario.window_height, mario.window_width, 3))
            
            if frame is not None:
                # Convert frame (RGB) to BGR for OpenCV encoding
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                # Encode as JPEG
                _, buffer = cv2.imencode('.jpg', frame_bgr)
                # Put in queue (non-blocking if full, drop oldest)
                if frame_queue.full():
                    try: frame_queue.get_nowait()
                    except Empty: pass
                frame_queue.put(buffer.tobytes())
                
            step += 1
            # Add a slight delay for streaming speed (Approx 30fps)
            time.sleep(1/30.0)
            
            if done:
                 # Extract scalar from numpy array if wrapped:
                 final_reward = float(total_reward[0]) if isinstance(total_reward, np.ndarray) else float(total_reward)
                 
                 log_queue.put({"type": "info", "msg": f"Episode finished. Total Reward: {final_reward:.1f}"})
                 if final_reward > 90:
                    log_queue.put({"type": "success", "msg": "🏆 Goal Reached!"})
                    break # Stop when goal is reached!
                    
                 obs = env.reset()
                 total_reward = 0
                 stuck_counter = 0
                 step = 0
                 
    except Exception as e:
        import traceback
        tb_str = traceback.format_exc()
        print(f"TRACEBACK:\n{tb_str}")
        log_queue.put({"type": "error", "msg": f"Error in loop: {str(e)}: {tb_str}"})

        
    finally:
        env.close()
        game_running = False
        log_queue.put({"type": "info", "msg": "Game stopped."})


# --- Flask Routes ---

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/start', methods=['POST'])
def start_game():
    global game_running, game_thread
    if not game_running:
        game_running = True
        # Clear queues
        while not frame_queue.empty(): frame_queue.get()
        while not log_queue.empty(): log_queue.get()
        
        game_thread = threading.Thread(target=game_loop)
        game_thread.daemon = True
        game_thread.start()
        return jsonify({"status": "started"}), 200
    return jsonify({"status": "already running"}), 400

@app.route('/stop', methods=['POST'])
def stop_game():
    global game_running
    game_running = False
    return jsonify({"status": "stopped"}), 200

@app.route('/status')
def get_status():
    global game_running
    return jsonify({"running": game_running})

@app.route('/logs')
def get_logs():
    logs = []
    while not log_queue.empty():
        try:
            logs.append(log_queue.get_nowait())
        except Empty:
            break
    return jsonify(logs)

def generate_video_stream():
    # If game isn't running, return a placeholder black screen
    blank_frame = np.zeros((240, 256, 3), dtype=np.uint8)
    _, buffer = cv2.imencode('.jpg', blank_frame)
    blank_bytes = buffer.tobytes()
    
    while True:
        if game_running:
            try:
                # Get latest frame
                frame_bytes = frame_queue.get(timeout=1.0)
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            except Empty:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')
        else:
             yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + blank_bytes + b'\r\n')
             time.sleep(0.5)

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
