"""Public exports, loaded only when requested."""

from importlib import import_module

_EXPORTS = {'Algorithm': ('.algorithm', 'Algorithm'),
 'Batch': ('.batch', 'Batch'),
 'Component': ('.component', 'Component'),
 'Stateful': ('.component', 'Stateful'),
 'ExperimentContext': ('.context', 'ExperimentContext'),
 'Evaluator': ('.evaluator', 'Evaluator'),
 'Artifact': ('.result', 'Artifact'),
 'CandidateLabelResult': ('.result', 'CandidateLabelResult'),
 'PseudoLabelResult': ('.result', 'PseudoLabelResult'),
 'SoftTargetResult': ('.result', 'SoftTargetResult'),
 'StepResult': ('.result', 'StepResult'),
 'RunState': ('.state', 'RunState'),
 'ObjectiveConsumer': ('.objectives', 'ObjectiveConsumer'),
 'ObjectiveResult': ('.objectives', 'ObjectiveResult'),
 'ArtifactRef': ('.storage', 'ArtifactRef'),
 'ArtifactSink': ('.storage', 'ArtifactSink'),
 'Checkpoint': ('.storage', 'Checkpoint'),
 'CheckpointStore': ('.storage', 'CheckpointStore'),
 'CandidateSetProvider': ('.targets', 'CandidateSetProvider'),
 'ComplementaryLabelResult': ('.targets', 'ComplementaryLabelResult'),
 'LabelProvider': ('.targets', 'LabelProvider'),
 'PseudoLabelProvider': ('.targets', 'PseudoLabelProvider'),
 'SoftTargetProvider': ('.targets', 'SoftTargetProvider'),
 'TargetInput': ('.targets', 'TargetInput')}
__all__ = ['Algorithm', 'Artifact', 'ArtifactRef', 'ArtifactSink', 'CandidateLabelResult', 'CandidateSetProvider', 'Batch', 'Component', 'Checkpoint', 'CheckpointStore', 'Evaluator', 'ExperimentContext', 'ComplementaryLabelResult', 'LabelProvider', 'ObjectiveConsumer', 'ObjectiveResult', 'PseudoLabelProvider', 'PseudoLabelResult', 'RunState', 'ObjectiveConsumer', 'Stateful', 'SoftTargetProvider', 'SoftTargetResult', 'StepResult', 'TargetInput']

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value

def __dir__():
    return sorted(set(globals()) | set(__all__))
