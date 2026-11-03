"""Public exports, loaded only when requested."""

from importlib import import_module

_EXPORTS = {'load_checkpoint': ('.checkpoint', 'load_checkpoint'),
 'save_checkpoint': ('.checkpoint', 'save_checkpoint'),
 'run_experiment': ('.experiment', 'run_experiment'),
 'ExperimentService': ('.service', 'ExperimentService'),
 'SweepResult': ('.sweep', 'SweepResult'),
 'run_sweep': ('.sweep', 'run_sweep'),
 'FeatureSnapshot': ('.snapshots', 'FeatureSnapshot'),
 'collect_feature_snapshot': ('.snapshots', 'collect_feature_snapshot'),
 'collect_posterior_snapshot': ('.snapshots', 'collect_posterior_snapshot'),
 'pretrain_noisy_classifier': ('.snapshots', 'pretrain_noisy_classifier'),
 'EarlyStopping': ('.early_stopping', 'EarlyStopping'),
 'ArtifactRef': ('.artifacts', 'ArtifactRef'),
 'ArtifactStore': ('.artifacts', 'ArtifactStore'),
 'PipelineArtifacts': ('.pipeline', 'PipelineArtifacts'),
 'PipelinePhase': ('.pipeline', 'PipelinePhase'),
 'PipelineState': ('.pipeline', 'PipelineState'),
 'StandardNoisyERMPipeline': ('.pipeline', 'StandardNoisyERMPipeline')}
__all__ = ['load_checkpoint', 'save_checkpoint', 'run_experiment', 'ExperimentService', 'SweepResult', 'run_sweep', 'collect_posterior_snapshot', 'FeatureSnapshot', 'collect_feature_snapshot', 'pretrain_noisy_classifier', 'EarlyStopping', 'ArtifactRef', 'ArtifactStore', 'PipelineArtifacts', 'StandardNoisyERMPipeline', 'PipelinePhase', 'PipelineState']

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value

def __dir__():
    return sorted(set(globals()) | set(__all__))
