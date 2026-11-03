"""Public exports, loaded only when requested."""

from importlib import import_module

_EXPORTS = {'BinaryBenchmark': ('.binary_benchmarks', 'BinaryBenchmark'),
 'load_binary_npz': ('.binary_benchmarks', 'load_binary_npz'),
 'CifarData': ('.cifar', 'CifarData'),
 'default_data_root': ('.cifar', 'default_data_root'),
 'load_cifar10': ('.cifar', 'load_cifar10'),
 'load_cifar100': ('.cifar', 'load_cifar100'),
 'summarize_cifar': ('.cifar', 'summarize_cifar'),
 'DataProtocol': ('.contracts', 'DataProtocol'),
 'DataRequirements': ('.contracts', 'DataRequirements'),
 'DataRole': ('.contracts', 'DataRole'),
 'DataSpec': ('.contracts', 'DataSpec'),
 'DatasetAdapter': ('.contracts', 'DatasetAdapter'),
 'DatasetIdentity': ('.contracts', 'DatasetIdentity'),
 'InputSpec': ('.contracts', 'InputSpec'),
 'NoiseDescriptor': ('.contracts', 'NoiseDescriptor'),
 'RawDatasetSplit': ('.contracts', 'RawDatasetSplit'),
 'Sample': ('.contracts', 'Sample'),
 'SampleKey': ('.contracts', 'SampleKey'),
 'UnsupportedDatasetSplitError': ('.contracts', 'UnsupportedDatasetSplitError'),
 'LocalDatasetCatalog': ('.local_catalog', 'LocalDatasetCatalog'),
 'LocalDatasetRecord': ('.local_catalog', 'LocalDatasetRecord'),
 'MentorFeatureDataset': ('.curriculum', 'MentorFeatureDataset'),
 'MentorFeatureRecord': ('.curriculum', 'MentorFeatureRecord'),
 'NeighborGraphArtifact': ('.neighbors', 'NeighborGraphArtifact'),
 'NoisyTargetDataset': ('.noisy_dataset', 'NoisyTargetDataset'),
 'BinaryPreprocessingConfig': ('.preprocessing', 'BinaryPreprocessingConfig'),
 'BinaryPreprocessor': ('.preprocessing', 'BinaryPreprocessor'),
 'DatasetRegistry': ('.registry', 'DatasetRegistry'),
 'SemiSupervisedBatch': ('.semi_supervised', 'SemiSupervisedBatch'),
 'TorchCifarDataset': ('.torch_cifar', 'TorchCifarDataset'),
 'build_cifar_transform': ('.torch_cifar', 'build_cifar_transform'),
 'stratified_split': ('.torch_cifar', 'stratified_split'),
 'IndexedDatasetView': ('.views', 'IndexedDatasetView'),
 'InputViewDataset': ('.views', 'InputViewDataset')}
__all__ = ['BinaryBenchmark', 'BinaryPreprocessingConfig', 'BinaryPreprocessor', 'CifarData', 'DataProtocol', 'DataRequirements', 'DataRole', 'DataSpec', 'DatasetAdapter', 'DatasetIdentity', 'DatasetRegistry', 'IndexedDatasetView', 'InputViewDataset', 'InputSpec', 'MentorFeatureDataset', 'MentorFeatureRecord', 'NeighborGraphArtifact', 'NoisyTargetDataset', 'NoiseDescriptor', 'RawDatasetSplit', 'LocalDatasetCatalog', 'LocalDatasetRecord', 'Sample', 'SampleKey', 'UnsupportedDatasetSplitError', 'SemiSupervisedBatch', 'TorchCifarDataset', 'build_cifar_transform', 'default_data_root', 'load_binary_npz', 'load_cifar10', 'load_cifar100', 'stratified_split', 'summarize_cifar']

def __getattr__(name):
    if name not in _EXPORTS:
        raise AttributeError(name)
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(module, __name__), attribute)
    globals()[name] = value
    return value

def __dir__():
    return sorted(set(globals()) | set(__all__))
