import numpy as np
import math
import os

from config import ObsIndex, ObservationConfig, RewardConfig

class RewardCalculator:
    def __init__(self, reward_weights, environment):
        self.reward = reward_weights
        self.env = environment
        
        # State tracking
        self.start_position = None
        self.start_tile_x = 0
        self.start_tile_y = 0
        self.initialized_spawns = False
        
        self.current_distance = 0.0
        self.distance_total = 0.0
        self.average_distance = 0.0
        self.beat_max_distance_this_run = False
        self.resets = 0

        # FIX: Persistent best distance. Do not reset this on DEAD signal.
        self.min_dist_to_target = float('inf')

        # Movement tracking
        self.just_landed = False
        self.starting_jump = False
        self.leaping = False
        self.leaping_gap = False
        self.leap_start_x = 0.0 

        self.max_distance = 0.0

        # Stagnation tracking
        self.current_tile = (0, 0)
        self.frames_in_tile = 0
        self.stagnation_threshold = 60 

        self.tile_counts = {}
        self.active_spawns = []

        # NEW: Alive reward tracking
        self.frames_alive = 0

        self.reward_totals = {
            "dead": 0.0,
            "terminal": 0.0,
            "stagnation": 0.0,
            "frontier": 0.0,
            "alive": 0.0,
            "distance": 0.0
        }

    def calculate(self, last_state, current_state, signal):
        if self.start_position is None:
            self.start_position = (current_state[ObsIndex.X_POS], current_state[ObsIndex.Y_POS])
            self.start_tile_x = int(current_state[ObsIndex.X_POS] // RewardConfig.TILE_SIZE)
            self.start_tile_y = int(current_state[ObsIndex.Y_POS] // RewardConfig.TILE_SIZE)
            self.current_tile = (self.start_tile_x, self.start_tile_y)

        frame_rewards = {k: 0.0 for k in self.reward_totals.keys()}
        frame_rewards["terminal"] = self._process_terminal_messages(signal)

        # --- Initialize spawns ---
        if not self.initialized_spawns:
            if not self.env.spawns:
                self.env.spawns = self.env.pipe.request_spawns()
            
            if self.env.spawns:
                self.active_spawns = list(self.env.spawns) 
                self.initialized_spawns = True
                print(f"{{RewardCalculator}}: Initialized {len(self.active_spawns)} spawns.")        

        # Count survival time
        self.frames_alive += 1

        # --- Give alive reward every step (while alive) ---
        if signal != "DEAD":
            frame_rewards["alive"] = self.reward.get("ALIVE", 0.01)  # typical range: 0.005 – 0.03

        if signal == "DEAD":
            frame_rewards["dead"] = self.reward.get("DEAD", -5.0)
            frame_rewards["distance"] = self._reward_distance(current_state, True)
            # NOTE: We do NOT reset min_dist_to_target here to prevent death-farming
            self._update_totals(frame_rewards)
            return frame_rewards

        self.just_landed = (current_state[ObsIndex.ON_GROUND] == 1.0 and last_state[ObsIndex.ON_GROUND] == 0.0)
        self.starting_jump = (last_state[ObsIndex.ON_GROUND] == 1.0 and current_state[ObsIndex.ON_GROUND] == 0.0)
        
        if last_state is not None:
            frame_rewards["frontier"] = self._reward_frontier(last_state, current_state)
            frame_rewards["stagnation"] = self._punish_stagnation(current_state)
            frame_rewards["distance"] = self._reward_distance(current_state)

        if self.just_landed:
            self.leaping_gap = False
            self.leaping = False

        self._update_totals(frame_rewards)
        return frame_rewards

    def _reward_frontier(self, last_state, current_state):
        if not self.active_spawns:
            return 0.0

        char_curr = np.array([current_state[ObsIndex.X_POS], current_state[ObsIndex.Y_POS]])
        spawn_array = np.array(self.active_spawns)
        distances_to_all = np.linalg.norm(spawn_array - char_curr, axis=1)
        closest_idx = np.argmin(distances_to_all)
        curr_dist = distances_to_all[closest_idx]
        if curr_dist < RewardConfig.SPAWN_POSITION_THRESHOLD:
            self.active_spawns.pop(closest_idx)
            self.min_dist_to_target = float('inf') 
            self.beat_max_distance_this_run = False
            self.average_distance = 0.0
            self.distance_total = 0.0
            
            return self.reward.get("FRONTIER_COLLECT", 100.0)

        if self.min_dist_to_target == float('inf'):
            self.min_dist_to_target = curr_dist
            return 0.0

        if curr_dist < self.min_dist_to_target:
            progress = self.min_dist_to_target - curr_dist
            self.min_dist_to_target = curr_dist
            return progress * self.reward.get("FRONTIER_PROGRESS_WEIGHT", 0.1)
        else:
            prev_char_pos = np.array([last_state[ObsIndex.X_POS], last_state[ObsIndex.Y_POS]])
            prev_dist = np.linalg.norm(spawn_array[closest_idx] - prev_char_pos)
            
            if curr_dist < prev_dist:
                return self.reward.get("DIRECTION_BUMP", 0.005)
        
        return 0.0

    def _reward_distance(self, current_state, isDead=False):
        if not self.start_position:
            return 0.0

        x, y = current_state[ObsIndex.X_POS], current_state[ObsIndex.Y_POS]
        curr_dist = math.dist(self.start_position, (x, y))

        if isDead:
            if curr_dist < self.average_distance:
                return -self.reward.get("DISTANCE", 0.1)
            else:
                return self.reward.get("DISTANCE", 0.1)

        if curr_dist > self.max_distance:
            improvement = curr_dist - self.max_distance
            self.max_distance = curr_dist
            return improvement * self.reward.get("DISTANCE", 0.1)

        return 0.0 

    def _punish_stagnation(self, current_state): 
        tile_coord = (int(current_state[ObsIndex.X_POS] // RewardConfig.TILE_SIZE), 
                      int(current_state[ObsIndex.Y_POS] // RewardConfig.TILE_SIZE))

        self.tile_counts[tile_coord] = self.tile_counts.get(tile_coord, 0) + 1
        
        if self.tile_counts[tile_coord] > self.stagnation_threshold: 
            return self.reward.get("STAGNATION", -0.01)
        
        return 0.0

    def _update_totals(self, frame_rewards):
        for key in frame_rewards:
            if key in self.reward_totals:
                self.reward_totals[key] += frame_rewards[key]

    def _process_terminal_messages(self, signal) -> float:
        if signal == "LEVEL_COMPLETE":
            return self.reward.get("LEVEL_COMPLETE", 1000.0)
        return 0.0

    def reset(self):
        self.resets += 1
        self.distance_total += self.current_distance
        self.average_distance = self.distance_total / self.resets

        for key in self.reward_totals:
            self.reward_totals[key] = 0.0
        
        # Reset per-episode counters
        self.beat_max_distance_this_run = False
        self.current_distance = 0.0
        self.frames_alive = 0
        self.tile_counts = {}
        self.start_position = None
        self.min_dist_to_target = float('inf')   # ← you can also keep it persistent if desired
