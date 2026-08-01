import os

_run = None
_enabled = True


def _wandb_seems_configured():
    if os.environ.get("WANDB_API_KEY"):
        return True
    netrc_path = os.path.expanduser("~/.netrc")
    if os.path.exists(netrc_path):
        try:
            with open(netrc_path, encoding="utf-8") as f:
                return "api.wandb.ai" in f.read()
        except OSError:
            return False
    return False


def init_run(project, name, config=None):
    global _run, _enabled
    try:
        import wandb
    except ImportError:
        _enabled = False
        return None

    mode = os.environ.get("WANDB_MODE")
    if mode is None:
        mode = "online" if _wandb_seems_configured() else "offline"

    try:
        _run = wandb.init(project=project, name=name, config=config or {}, mode=mode, reinit="finish_previous")
    except Exception as e:
        print(f"  [wandb] init failed ({e}); continuing without logging")
        _enabled = False
        _run = None
    return _run


def log(data, step=None):
    global _enabled
    if not _enabled or _run is None:
        return
    try:
        _run.log(data, step=step)
    except Exception as e:
        print(f"  [wandb] log failed ({e}); disabling further logging for this run")
        _enabled = False


def finish():
    global _run
    if _run is not None:
        try:
            _run.finish()
        except Exception:
            pass
    _run = None
