from __future__ import annotations

from collections.abc import Iterable, Sequence

from lnl_toolbox.core import Algorithm, Batch, Evaluator, ExperimentContext, RunState, StepResult


def _record(result: StepResult, state: RunState, evaluators: Sequence[Evaluator]) -> None:
    state.metrics.update(result.metrics)
    for evaluator in evaluators:
        evaluator.update(result, state)


def run_cycles(
    algorithm: Algorithm,
    batches: Iterable[Batch],
    cycles: int,
    context: ExperimentContext,
    *,
    phase: str = "default",
    evaluators: Sequence[Evaluator] = (),
) -> RunState:
    """Execute a task-neutral lifecycle.

    The runner has no knowledge of models, optimizers, labels, gradients, or devices.
    Re-iterable loaders should be passed when more than one cycle is requested.
    """

    state = RunState(phase=phase)
    algorithm.setup(context)
    algorithm.on_run_start(state)
    try:
        for cycle in range(cycles):
            if state.stopped:
                break
            state.cycle = cycle
            algorithm.on_cycle_start(state)
            for batch in batches:
                if state.stopped:
                    break
                _record(algorithm.step(batch, state), state, evaluators)
                state.step += 1
            _record(algorithm.on_cycle_end(state), state, evaluators)
        _record(algorithm.on_run_end(state), state, evaluators)
        for evaluator in evaluators:
            state.metrics.update(evaluator.compute(state))
        return state
    finally:
        algorithm.close()


def run_epochs(
    algorithm: Algorithm,
    batches: Iterable[Batch],
    epochs: int,
    context: ExperimentContext,
) -> RunState:
    """Compatibility name for callers that interpret one cycle as one epoch."""

    return run_cycles(algorithm, batches, epochs, context, phase="train")
