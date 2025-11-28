import os
from dotenv import load_dotenv

load_dotenv()

RAW_DATA_PATH = os.getenv("RAW_DATA_PATH")
CLEAN_DATA_PATH = os.getenv("CLEAN_DATA_PATH")
RESULTS_DIR = os.getenv("RESULTS_DIR", "results")
FIGURES_DIR = os.getenv("FIGURES_DIR", "figures")
SEED = int(os.getenv("SEED", 42))