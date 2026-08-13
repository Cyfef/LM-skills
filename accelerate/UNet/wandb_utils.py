# wandb_utils.py
import os
import wandb
from typing import Dict, Any, Optional

# ============================================================
# 1️⃣ 【CONFIGURATION ZONE】 – All WandB settings are here.
# ============================================================
WANDB_PROJECT = "UNet_accelerate"                     # Project name
WANDB_ENTITY = None                        # Team name (None for personal account)
WANDB_TAGS = ["UNet", "Carvana", "accelerate"]           # Experiment tags
WANDB_NAME = "run_v1"                      # Run name 
WANDB_MODE = os.getenv("WANDB_MODE", "online")  # online / offline / dryrun

# Default hyperparameters (automatically logged to WandB)
DEFAULT_CONFIG = {
    "learning_rate": 1e-5,
    "batch_size": 1,
    "epochs": 5,
    "optimizer": "Adam",
    "model_arch": "UNet",
    "dataset": "Carvana",
}
# ============================================================

# ---------- Internal state (do not modify) ----------
_RUN = None
_ENABLED = True
_CURRENT_CONFIG = DEFAULT_CONFIG.copy()


# ============================================================
# 2️⃣ 【PUBLIC INTERFACE】 – Functions to be called in training scripts
# ============================================================

def wandb_update_config(**kwargs) -> None:
    """
    Dynamically update hyperparameters (e.g., from command line or YAML config).
    Example: wandb_update_config(learning_rate=0.01, batch_size=64)
    """
    _CURRENT_CONFIG.update(kwargs)


def wandb_init() -> None:
    """
    Initialize a WandB run. Takes no arguments – all settings are read
    from the configuration zone above.
    """
    global _RUN, _ENABLED

    # Check if WandB is disabled via environment variable
    if os.getenv("WANDB_DISABLED", "false").lower() == "true":
        _ENABLED = False
        print("⚠️ WandB disabled by env var (WANDB_DISABLED=true).")
        return

    try:
        _RUN = wandb.init(
            project=WANDB_PROJECT,
            entity=WANDB_ENTITY if WANDB_ENTITY else None,
            tags=WANDB_TAGS,
            name=WANDB_NAME,
            mode=WANDB_MODE,
            config=_CURRENT_CONFIG,
        )
        _ENABLED = True
        url = _RUN.get_url() if _RUN else "N/A"
        print(f"✅ WandB initialized. View at: {url}")
    except Exception as e:
        print(f"❌ WandB init failed: {e}. Running without logging.")
        _ENABLED = False


def wandb_log(metrics: Dict[str, Any], step: Optional[int] = None) -> None:
    """
    Log training metrics (e.g., loss, accuracy). Accepts a dict and
    automatically handles step numbering. If WandB is disabled, this
    function becomes a silent no‑op.
    """
    if not _ENABLED or _RUN is None:
        return
    try:
        _RUN.log(metrics, step=step)
    except Exception as e:
        print(f"⚠️ WandB log error: {e}")


def wandb_save_artifact(
    model_path: str,
    artifact_name: Optional[str] = None,
    metadata: Optional[Dict] = None,
) -> None:
    """
    Save a model file as an Artifact (version control). If artifact_name
    is not provided, it defaults to WANDB_NAME + "_model".
    """
    if not _ENABLED or _RUN is None:
        return
    try:
        name = artifact_name or f"{WANDB_NAME}_model"
        artifact = wandb.Artifact(name=name, type="model")
        artifact.add_file(model_path)
        if metadata:
            artifact.metadata = metadata
        _RUN.log_artifact(artifact, aliases=["latest"])
        print(f"✅ Artifact saved: {name}")
    except Exception as e:
        print(f"⚠️ Artifact save error: {e}")


def wandb_finish() -> None:
    """Finish the current run. Call this in a finally block to ensure cleanup."""
    global _RUN, _ENABLED
    if _ENABLED and _RUN is not None:
        try:
            _RUN.finish()
            print("✅ WandB run finished.")
        except Exception:
            pass
    _ENABLED = False
    _RUN = None
