from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
LANES_CONFIG_PATH = BASE_DIR / "data" / "configs" / "lanes.json"
CALIBRATION_CONFIG_PATH = BASE_DIR / "data" / "configs" / "calibration.json"
TEST_VIDEO_PATH = BASE_DIR / "data" / "videos" / "highway_mini.mp4"
MODEL_WEIGHTS_PATH = BASE_DIR / "data" / "weights" / "best.pt"