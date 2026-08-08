from .algorithm import DLDAlgorithm
from .artifacts import DLDPreCorrectionArtifact, persist_precorrection_atomically
from .config import DLDConfig
from .networks import DLDLabelPredictor, timestep_embedding
from .objective import DLDObjectiveResult, construct_direction, dld_objective, sample_forward_state
from .precorrection import (
    DLDPartitionResult,
    NeighborDistributionResult,
    PARTITION_CLEAN,
    PARTITION_HARD,
    PARTITION_NOISY,
    construct_y0,
    construct_yn,
    kl_ps_to_pw,
    partition_samples,
    weighted_neighbor_distribution,
)
from .sampling import accelerated_timesteps, sample_labels
from .schedules import DirectionalDiffusionSchedule
from .state import DLDPhase, DLDState

# ``algorithms/dld.py`` is the pre-existing synthetic/compatibility
# implementation.  Keep its public primitives available while the canonical
# paper workflow uses the richer package-level artifact above.  Loading the
# old module under a private name avoids changing either workflow's equations
# or checkpoint format during the namespace migration.
import importlib.util as _importlib_util
from pathlib import Path as _Path
import sys as _sys

_legacy_spec = _importlib_util.spec_from_file_location(
    "_lnl_toolbox_legacy_dld", _Path(__file__).resolve().parent.parent / "dld.py"
)
if _legacy_spec is not None and _legacy_spec.loader is not None:
    _legacy_module = _importlib_util.module_from_spec(_legacy_spec)
    _sys.modules[_legacy_spec.name] = _legacy_module
    _legacy_spec.loader.exec_module(_legacy_module)
    DLDPrecorrectionArtifact = _legacy_module.DLDPrecorrectionArtifact
    build_knn_label_distribution = _legacy_module.build_knn_label_distribution
    precorrect_two_views = _legacy_module.precorrect_two_views
    weighted_mse = _legacy_module.weighted_mse

__all__ = [name for name in globals() if not name.startswith("_")]
