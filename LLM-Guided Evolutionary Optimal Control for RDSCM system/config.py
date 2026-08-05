import json
import os
from dotenv import load_dotenv

load_dotenv()   # Load environment variables from local .env file

# ===================== LLM Configuration =====================
LLM_MODEL = "model_name"   # Designation of the target large language model
LLM_TEMPERATURE = 0.0   # Sampling temperature for LLM generation; 0.0 yields deterministic outputs
LLM_MAX_RETRIES = 3   # Maximum retry attempts when failing to invoke the LLM service
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")   # Authentication key for OpenAI-compatible LLM service, loaded from environment variable
BASE_URL = os.getenv("BASE_URL")  # API request endpoint base URL of the LLM service, loaded from environment variable


# ===================== File Path Configuration =====================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))    # Absolute path of the root directory for the current project
INITIAL_PROMPT_FILE = os.path.join(PROJECT_DIR, "initial_prompt_v1.txt")  # File path storing the initial system prompt fed into LLM
CONTROL_V1_FILE = os.path.join(PROJECT_DIR, "control_v1.py")   # File path of the first-version control strategy script
CONTROL_VN_TEMPLATE = os.path.join(PROJECT_DIR, "control_v{}.py")  # Template path for iteratively updated control strategy scripts with version tags
EVOLUTION_LOG_FILE = os.path.join(PROJECT_DIR, "evolution_log.json")   # Storage path for logs recorded during evolutionary optimization iterations
BEST_PARAMS_TEMPLATE = os.path.join(PROJECT_DIR, "best_params_v{}.json")   # Template path for saving optimized optimal parameter sets with version tags
DIAGNOSTIC_TEMPLATE = os.path.join(PROJECT_DIR, "diagnostic_v{}.txt")   # Template path for diagnostic logs of optimization training process with version tags


# ===================== Optimizer Hyperparameter Configuration =====================
OPTUNA_N_TRIALS = 4000    # Total number of sampling trials executed in Optuna hyperparameter tuning
OPTUNA_SEED = 49   # Fixed random seed to guarantee reproducible Optuna optimization results
CMAES_POPSIZE_BASE = 80   # Base population size of the CMA-ES evolutionary optimization algorithm
CMAES_POPSIZE_PER_DIM = 12   # Dimension-dependent coefficient to adjust CMA-ES population size
CMAES_SIGMA0_STAGE1 = 0.2   # Initial search step standard deviation adopted in the first optimization stage of CMA-ES
CMAES_N_STARTUP_TRIALS = 400   # Quantity of random warm-up sampling trials prior to formal CMA-ES iterative optimization
