import socket
import time
import os
import numpy as np
import subprocess

from io import TextIOWrapper
from typing import Optional
from config import (
    ServerConfig,
    ObservationConfig,
    PipeManagerConfig,
    LevelConfig,   # <-- NEW: import your level map
)

class PipeManager:
    def __init__(self, host: str, port: int, buffer_size: int = ServerConfig.BUFFER_SIZE, instance_index: int = 0):
        self.host = host
        self.port = port
        self.buffer_size = buffer_size
        self.instance_index = instance_index
        self.socket: Optional[socket.socket] = None
        self.connection: Optional[socket.socket] = None
        self.file_buffer: Optional[TextIOWrapper] = None
        self.process: Optional[subprocess.Popen] = None

        # NEW: curriculum → Celeste room mapping
        self.level_map = LevelConfig.LEVEL_ID_MAP

        self._cleanup_specific_port()
        self.launch_celeste_instance(self.port, instance_index)
        self._connect_to_server()
        self._perform_handshake()

    # ---------------------------------------------------------
    # Port cleanup
    # ---------------------------------------------------------
    def _cleanup_specific_port(self):
        if os.name == 'nt':
            try:
                cmd = f'netstat -ano | findstr LISTENING | findstr :{self.port}'
                output = subprocess.check_output(cmd, shell=True).decode()
                if output:
                    pid = output.strip().split()[-1]
                    print(f"PipeManager [{self.port}]: Cleaning port {self.port} (PID {pid})")
                    subprocess.run(['taskkill', '/F', '/PID', pid, '/T'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    time.sleep(1.0)
            except subprocess.CalledProcessError:
                pass

    # ---------------------------------------------------------
    # Launch Celeste instance
    # ---------------------------------------------------------
    def launch_celeste_instance(self, port, instance_id):
        process_path = PipeManagerConfig.CELESTE_PATH.format(instance_id)
        if not os.path.exists(process_path):
            raise FileNotFoundError(f"Celeste not found at: {process_path}")

        args = [
            process_path,
            "--port", str(port),
            "--graphics", "Vulkan",
            "--windowed",
            "--noborder",
            "--disable-splash"
        ]

        self.process = subprocess.Popen(
            args,
            cwd=os.path.dirname(process_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )

    # ---------------------------------------------------------
    # Connect to Celeste mod server
    # ---------------------------------------------------------
    def _connect_to_server(self):
        print(f"PipeManager [{self.port}]: Connecting...")
        for _ in range(40):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.socket.connect((self.host, self.port))
                self.connection = self.socket
                print(f"--- PipeManager [{self.port}]: Connected! ---")
                return
            except (ConnectionRefusedError, OSError):
                time.sleep(1.5)
        raise TimeoutError(f"Failed to connect on port {self.port}")

    # ---------------------------------------------------------
    # Handshake
    # ---------------------------------------------------------
    def _perform_handshake(self):
        self.socket.settimeout(2.0)
        for _ in range(10):
            try:
                self.socket.sendall(b"START_LEVEL\n")
                data = self.socket.recv(self.buffer_size).decode("utf-8")
                if data:
                    self.file_buffer = self.socket.makefile("r", encoding="utf-8")
                    return
            except socket.timeout:
                pass
            time.sleep(1.0)

    # ---------------------------------------------------------
    # Observation
    # ---------------------------------------------------------
    def receive_observation(self):
        try:
            self.connection.settimeout(10.0)
            line = self.file_buffer.readline()
            if not line:
                return None, "CONNECTION_CLOSED"

            raw_data = line.strip()
            if "READY" in raw_data:
                return self.receive_observation()
            if raw_data == "DEAD":
                return None, "DEAD"

            obs_size = ObservationConfig.TOTAL_FEATURE_COUNT
            numerical_vector = np.fromstring(raw_data, sep=",", dtype=np.float32)

            if len(numerical_vector) < obs_size:
                padded = np.zeros(obs_size, dtype=np.float32)
                padded[:len(numerical_vector)] = numerical_vector[:obs_size]
                return padded, "PARTIAL_TICK"

            return numerical_vector, "TICK"

        except Exception:
            return None, "ERROR"

    # ---------------------------------------------------------
    # Action
    # ---------------------------------------------------------
    def send_action(self, action_vector):
        if self.connection:
            try:
                msg = ",".join(map(str, [int(a) for a in action_vector])) + "\n"
                self.connection.sendall(msg.encode("utf-8"))
            except:
                self.close()

    # ---------------------------------------------------------
    # Spawn request
    # ---------------------------------------------------------
    def request_spawns(self):
        if not self.connection:
            return []
        try:
            self.connection.sendall(b"GET_SPAWNS\n")
            line = self.file_buffer.readline().strip()
            if line.startswith("SPAWNS:"):
                return self._parse_spawn_string(line.replace("SPAWNS:", ""))
        except Exception as e:
            print(f"Error requesting spawns: {e}")
        return []

    def _parse_spawn_string(self, line):
        spawns = []
        for entry in line.split(";"):
            if "," in entry:
                x, y = entry.split(",")
                try:
                    spawns.append((float(x), float(y)))
                except ValueError:
                    pass
        return spawns

    # ---------------------------------------------------------
    # Reset
    # ---------------------------------------------------------
    def send_reset(self):
        if self.connection:
            try:
                self.connection.sendall(b"-1\n")
            except:
                self.close()

    # ---------------------------------------------------------
    # Level loading (NEW)
    # ---------------------------------------------------------
    def send_load_level_by_name(self, room_name: str):
        """Send a LOAD_LEVEL command using a Celeste room name."""
        if self.connection:
            try:
                msg = f"LOAD_LEVEL,{room_name}\n"
                self.connection.sendall(msg.encode("utf-8"))
            except:
                self.close()

    def send_load_level_by_index(self, level_index: int):
        """Send a LOAD_LEVEL command using your curriculum index."""
        if level_index not in self.level_map:
            raise KeyError(f"Level index {level_index} not found in LevelConfig.LEVEL_ID_MAP")

        room_name = self.level_map[level_index]
        self.send_load_level_by_name(room_name)

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------
    def close(self):
        try:
            if self.file_buffer:
                self.file_buffer.close()
            if self.connection:
                self.connection.close()
            if self.process:
                self.process.terminate()
        except:
            pass
