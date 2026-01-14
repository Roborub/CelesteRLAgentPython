
class ModelConfig:
    VERSION = 2.5

class OptunaConfig:
    TRIALS = 50
    STARTUP_TRIALS = 5
    WARMUP_STEPS = 10_000
    TIMESTEPS = 1_000_000
    PRUNING_CHECK_RATE = 1_000
    CELESTE_INSTANCE_COUNT = 1

class ObservationConfig:
    GRID_SIZE = 15 
    CATEGORY_COUNT = 8
    STATIC_FEATURE_COUNT = 9
    TOTAL_FEATURE_COUNT = STATIC_FEATURE_COUNT + (GRID_SIZE * GRID_SIZE * CATEGORY_COUNT)

class ServerConfig:
    HOST = "127.0.0.1"
    PORT = 5555
    BUFFER_SIZE = 2048

class PipeManagerConfig:
    CELESTE_PATH = "C:/Program Files (x86)/Steam/steamapps/common/Celeste{}/Celeste.exe"

class ObsIndex:
    X_POS = 0
    Y_POS = 1
    X_POS_REL = 2
    Y_POS_REL = 3
    X_VELOCITY = 4
    Y_VELOCITY = 5
    DASHES = 6
    STAMINA = 7
    ON_GROUND = 8
    CLIMBING = 9
    FACING = 10
    TILE_DATA = 11 

class RewardConfig:
    LEAP_THRESHOLD = 8.0
    TILE_SIZE = 8
    SPAWN_POSITION_THRESHOLD = 16.0

class LevelConfig:
    LEVEL_ID_MAP = {
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "3b"
    }

class EnvironmentConfig:
    Policy = "MultiInputPolicy"
    RunFinalModel = False
    UseCustomWeights = True
    RESUME_LAST_RUN = True 
    FRAME_SKIP = 4
    MAX_EPISODE_STEPS = 15000
    LAYER_NEURONS = 256
    LAYERS = 2
    ENTROPY_COEF = 0.01
    OBS_CLIPPING = 10.0
    LEARNING_RATE = 0.0003
    N_STEPS = 4096
    BATCH_SIZE = 256
    N_EPOCHS = 10
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    VERBOSE = 1
    TENSOR_LOG = "./celeste_tensorboard/"

class RewardWeights:
    LEVEL_COMPLETE = (0.0, 2000.0) 
    DEAD = (-50.0, -1.0)          
    LEAP = (0.0, 1.0)
    GAP_JUMPED = (0.0, 20.0)
    EXPLORATION = (0.0, 5.0)
    STAGNATION = (-5.0, 0)
    DISTANCE = (0.0, 5.0)
    
    FRONTIER_COLLECT = (1.0, 200.0) 
    
    FRONTIER_PROGRESS_WEIGHT = (0.01, 0.2)

CustomRewardWeights = {
    "LEVEL_COMPLETE": 500.0,         
    "DEAD": -5.0,                    
    "ALIVE": 0.0,                    

    "DISTANCE": 0.1,                 
    "FRONTIER_COLLECT": 50.0,        
    "FRONTIER_PROGRESS_WEIGHT": 0.05,

    "STAGNATION": -0.05,             
    "LEAP": 0.0,
    "GAP_JUMPED": 0.0,
    "EXPLORATION": 0.0,
}
