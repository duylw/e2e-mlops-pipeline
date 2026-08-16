import gc
import pathlib

import _pytest.pathlib
import matplotlib.pyplot as plt
import pytest

# Safely ignore Windows file handle lock permission errors during pytest temp directory cleanup
_orig_rm_rf = _pytest.pathlib.rm_rf


def _safe_rm_rf(path):
    try:
        _orig_rm_rf(path)
    except (PermissionError, OSError):
        pass


_pytest.pathlib.rm_rf = _safe_rm_rf


def pytest_configure(config):
    # Ensure basetemp and its parent directories exist across any OS / CI runner
    basetemp = config.option.basetemp
    if basetemp:
        pathlib.Path(basetemp).mkdir(parents=True, exist_ok=True)


@pytest.fixture(autouse=True)
def cleanup_resources_after_test():
    yield
    plt.close("all")
    try:
        from mlflow.store.db.utils import _EngineRegistry

        _EngineRegistry._registry.clear()
    except Exception:
        pass
    gc.collect()
