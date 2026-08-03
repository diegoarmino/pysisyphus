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


class CachedGeometry:
    """Small Geometry double whose coordinate setter invalidates results."""

    def __init__(self, calculator, coords, energy, forces):
        self.calculator = calculator
        self._coords = np.asarray(coords, dtype=float)
        self._energy = float(energy)
        self._forces = np.asarray(forces, dtype=float)

    @property
    def coords(self):
        return self._coords

    @coords.setter
    def coords(self, value):
        self._coords = np.asarray(value, dtype=float)
        self._energy = None
        self._forces = None

    @property
    def cart_coords(self):
        return self._coords

    @cart_coords.setter
    def cart_coords(self, value):
        self.coords = value

    @property
    def has_energy(self):
        return self._energy is not None

    @property
    def has_forces(self):
        return self._forces is not None

    @property
    def energy(self):
        if self._energy is None:
            raise AssertionError("rollback attempted a new energy calculation")
        return self._energy

    @property
    def cart_forces(self):
        if self._forces is None:
            raise AssertionError("rollback attempted a new force calculation")
        return self._forces

    def set_results(self, results):
        self._energy = results.get("energy")
        self._forces = results.get("forces")


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
    optimizer._state_step_rollback_results = None
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
    optimizer._state_step_rollback_results = None
    optimizer._state_step_tracking_revision = None
    optimizer.log = lambda message: None

    optimizer.rollback_uncommitted_state_step()

    np.testing.assert_allclose(geometry.coords, [1.0, 0.0])
    assert calculator.discarded == 1
    assert not optimizer._state_step_transaction_open


def test_rollback_restores_cached_energy_and_forces_without_recalculation():
    calculator = StagedCalculator()
    geometry = CachedGeometry(
        calculator,
        coords=[1.4, -0.2],
        energy=-12.5,
        forces=[0.3, -0.4],
    )
    optimizer = BareOptimizer.__new__(BareOptimizer)
    optimizer.step_controller = object()
    optimizer.geometry = geometry
    optimizer.coords = [np.array([1.0, 0.0])]
    optimizer._state_step_transaction_open = True
    optimizer._state_step_rollback_coords = np.array([1.0, 0.0])
    optimizer._state_step_rollback_cart_coords = np.array([1.0, 0.0])
    optimizer._state_step_rollback_results = {
        "energy": -13.25,
        "forces": np.array([0.1, -0.2]),
    }
    optimizer._state_step_tracking_revision = None
    optimizer.log = lambda message: None

    optimizer.rollback_uncommitted_state_step()

    np.testing.assert_allclose(geometry.coords, [1.0, 0.0])
    assert geometry.energy == -13.25
    np.testing.assert_allclose(geometry.cart_forces, [0.1, -0.2])
    assert optimizer._state_step_rollback_results is None


def test_step_controller_captures_preproposal_cached_results():
    class Controller:
        def select_step(self, optimizer, step):
            selected = SimpleNamespace(
                factor=0.5,
                root=8,
                status=SimpleNamespace(value="ACCEPT"),
            )
            return SimpleNamespace(step=0.5 * step, selected=selected)

    calculator = StagedCalculator()
    geometry = CachedGeometry(
        calculator,
        coords=[1.0, 0.0],
        energy=-13.25,
        forces=[0.1, -0.2],
    )
    optimizer = BareOptimizer.__new__(BareOptimizer)
    optimizer.step_controller = Controller()
    optimizer.geometry = geometry
    optimizer.step_control_results = []
    optimizer._state_step_transaction_open = False
    optimizer._state_step_rollback_coords = None
    optimizer._state_step_rollback_cart_coords = None
    optimizer._state_step_rollback_results = None
    optimizer._state_step_tracking_revision = None
    optimizer.on_step_control = lambda original, controlled: None
    optimizer.log = lambda message: None

    controlled = optimizer.apply_step_controller(np.array([0.2, -0.4]))

    np.testing.assert_allclose(controlled, [0.1, -0.2])
    assert optimizer._state_step_transaction_open
    assert optimizer._state_step_rollback_results["energy"] == -13.25
    np.testing.assert_allclose(
        optimizer._state_step_rollback_results["forces"], [0.1, -0.2]
    )


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
