from types import SimpleNamespace

import numpy as np

from pysisyphus.optimizers.Optimizer import Optimizer


class BareOptimizer(Optimizer):
    def get_step(self, *args, **kwargs):
        raise NotImplementedError


class StagedCalculator:
    def __init__(self):
        self.has_staged_state_survey = True
        self.discarded = 0

    def discard_staged_state_survey(self):
        self.has_staged_state_survey = False
        self.discarded += 1


def test_stop_rolls_back_unevaluated_state_endpoint():
    calculator = StagedCalculator()
    geometry = SimpleNamespace(
        calculator=calculator,
        coords=np.array([1.4, -0.2]),
    )
    optimizer = BareOptimizer.__new__(BareOptimizer)
    optimizer.step_controller = object()
    optimizer.geometry = geometry
    optimizer.coords = [np.array([1.0, 0.0])]
    optimizer._state_step_transaction_open = False
    optimizer._state_step_rollback_coords = None
    optimizer._state_step_rollback_cart_coords = None
    optimizer._state_step_tracking_revision = None
    optimizer.log = lambda message: None

    optimizer.rollback_uncommitted_state_step()

    np.testing.assert_allclose(geometry.coords, [1.0, 0.0])
    assert calculator.discarded == 1
    assert not calculator.has_staged_state_survey


def test_legacy_optimizer_without_controller_is_untouched():
    calculator = StagedCalculator()
    geometry = SimpleNamespace(
        calculator=calculator,
        coords=np.array([1.4, -0.2]),
    )
    optimizer = BareOptimizer.__new__(BareOptimizer)
    optimizer.step_controller = None
    optimizer.geometry = geometry
    optimizer.coords = [np.array([1.0, 0.0])]

    optimizer.rollback_uncommitted_state_step()

    np.testing.assert_allclose(geometry.coords, [1.4, -0.2])
    assert calculator.discarded == 0


def test_gradient_failure_restores_preproposal_geometry_after_stage_was_cleared():
    calculator = StagedCalculator()
    # Reproduce a lower-level error path that has already cleared the stage
    # after the optimizer moved to the uncommitted endpoint.
    calculator.has_staged_state_survey = False
    geometry = SimpleNamespace(
        calculator=calculator,
        coords=np.array([1.4, -0.2]),
    )
    optimizer = BareOptimizer.__new__(BareOptimizer)
    optimizer.step_controller = object()
    optimizer.geometry = geometry
    optimizer.coords = [np.array([1.0, 0.0]), np.array([1.4, -0.2])]
    optimizer._state_step_transaction_open = True
    optimizer._state_step_rollback_coords = np.array([1.0, 0.0])
    optimizer._state_step_rollback_cart_coords = np.array([1.0, 0.0])
    optimizer._state_step_tracking_revision = None
    optimizer.log = lambda message: None

    optimizer.rollback_uncommitted_state_step()

    np.testing.assert_allclose(geometry.coords, [1.0, 0.0])
    assert calculator.discarded == 1
    assert not optimizer._state_step_transaction_open


def test_converged_geometry_skips_electronic_trial_survey():
    class ExplodingController:
        def select_step(self, optimizer, step):
            raise AssertionError("a converged geometry must not be surveyed")

    optimizer = BareOptimizer.__new__(BareOptimizer)
    optimizer.step_controller = ExplodingController()
    optimizer.geometry = SimpleNamespace(calculator=SimpleNamespace())
    optimizer.check_convergence = lambda **kwargs: (True, object())
    optimizer.log = lambda message: None
    step = np.array([0.0, 0.0])

    returned = optimizer.apply_step_controller_unless_converged(step)

    assert returned is step
