
import socket
import time
import os
import numpy as np
import subprocess

from io import TextIOWrapper
from typing import Optional
from config import ServerConfig, ObservationConfig, PipeManagerConfig

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

        self._cleanup_specific_port()
        self.launch_celeste_instance(self.port, instance_index)
        self._connect_to_server()
        self._perform_handshake()

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

    def launch_celeste_instance(self, port, instance_id):
        process_path = PipeManagerConfig.CELESTE_PATH.format(instance_id)
        if not os.path.exists(process_path):
            raise FileNotFoundError(f"Celeste not found at: {process_path}")

        args = [process_path, "--port", str(port), "--graphics", "Vulkan", 
                "--windowed", "--noborder", "--nolog", "--disable-splash"]

        self.process = subprocess.Popen(args, cwd=os.path.dirname(process_path),
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def _connect_to_server(self):
        print(f"PipeManager [{self.port}]: Connecting...")
        for i in range(40):
            try:
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                # {Crucial: TCP_NODELAY prevents the 8 FPS lag}
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.socket.connect((self.host, self.port))
                self.connection = self.socket 
                print(f"--- PipeManager [{self.port}]: Connected! ---")
                return
            except (ConnectionRefusedError, OSError):
                time.sleep(1.5)
        raise TimeoutError(f"Failed to connect on port {self.port}")

    def _perform_handshake(self):
        self.socket.settimeout(2.0)
        for i in range(10):
            try:
                self.socket.sendall(b"START_LEVEL\n")
                data = self.socket.recv(self.buffer_size).decode('utf-8')
                if data:
                    self.file_buffer = self.socket.makefile("r", encoding="utf-8")
                    return
            except socket.timeout:
                continue
            time.sleep(1.0)

    def receive_observation(self):
        try:
            self.connection.settimeout(10.0) 
            line = self.file_buffer.readline()
            if not line: return None, "CONNECTION_CLOSED"
            
            raw_data = line.strip()
            if "READY" in raw_data: return self.receive_observation()
            if raw_data == "DEAD": return None, "DEAD"
            
            obs_size = ObservationConfig.TOTAL_FEATURE_COUNT
            numerical_vector = np.fromstring(raw_data, sep=',', dtype=np.float32)

            if len(numerical_vector) < obs_size:
                padded = np.zeros(obs_size, dtype=np.float32)
                padded[:len(numerical_vector)] = numerical_vector[:obs_size]
                return padded, "PARTIAL_TICK"

            return numerical_vector, "TICK"
        except Exception as e:
            return None, "ERROR"

    def send_action(self, action_vector):
        if self.connection:
            try:
                message = ",".join(map(str, [int(a) for a in action_vector])) + "\n"
                self.connection.sendall(message.encode("utf-8"))
            except:
                self.close()

    # {RESTORED: Missing method causing Trial 3 crash}
    def request_spawns(self):
        if not self.connection: return []
        try:
            self.connection.sendall(b"GET_SPAWNS\n")
            line = self.file_buffer.readline().strip()
            if line.startswith("SPAWNS:"):
                return self._parse_spawn_string(line.replace("SPAWNS:", ""))
        except Exception as e:
            print(f"Error requesting spawns: {e}")
        return []

    # {RESTORED: Helper for request_spawns}
    def _parse_spawn_string(self, line):
        spawn_points = []
        for entry in line.split(";"):
            if "," in entry:
                coords = entry.split(",")
                try:
                    spawn_points.append((float(coords[0]), float(coords[1])))
                except ValueError: continue
        return spawn_points

    def send_reset(self):
        if self.connection:
            try: self.connection.sendall(b"-1\n")
            except: self.close()

    def close(self):
        try:
            if self.file_buffer: self.file_buffer.close()
            if self.connection: self.connection.close()
            if self.process:
                self.process.terminate()
        except: pass

