from types import SimpleNamespace

import numpy as np
import pytest

from pysisyphus.optimizers.step_control import (
    FatalStateSurveyError,
    NoAcceptableStateStep,
    NonTransactionalSurveyError,
    StateAwareStepController,
    TrialStatus,
    make_step_controller,
)


class FakeGeometry:
    def __init__(self, calculator):
        self.calculator = calculator
        self.coords = np.zeros(2)
        self.atoms = ("H", "H")

    def get_temporary_coords(self, coords):
        return np.asarray(coords, dtype=float)


class FakeOptimizer:
    def __init__(self, calculator):
        self.geometry = FakeGeometry(calculator)
        self.energies = [-1.0]
        self.trust_max = 4.0


class FakeCalculator:
    def __init__(self, outcomes):
        self.outcomes = outcomes
        self.tracking_revision = 3
        self.staged = None
        self.surveyed = []
        self.discard_calls = 0
        self.coordinate_tolerance = 1.0e-10

    def survey_state(self, atoms, cart_coords, *, factor):
        self.surveyed.append(factor)
        outcome = self.outcomes[factor]
        decision = SimpleNamespace(
            status=outcome[0],
            selected_root=outcome[1],
            selected_energy_eh=outcome[2],
            best_score=outcome[3],
            margin=outcome[4],
            reason=outcome[5],
        )
        return SimpleNamespace(decision=decision, factor=factor)

    def stage_state_survey(self, survey, *, expected_cart_coords):
        self.staged = survey, expected_cart_coords

    def discard_staged_state_survey(self):
        self.staged = None
        self.discard_calls += 1


def test_larger_step_can_bridge_manifold():
    calc = FakeCalculator(
        {
            0.5: ("ACCEPT", 2, -0.95, 0.90, 0.30, "clean but uphill"),
            1.0: ("MANIFOLD", 2, -1.10, 0.70, 0.01, "mixed pair"),
            1.5: ("ACCEPT", 3, -1.20, 0.88, 0.22, "clean endpoint"),
        }
    )
    optimizer = FakeOptimizer(calc)
    controller = StateAwareStepController(factors=(0.5, 1.0, 1.5))

    result = controller.select_step(optimizer, np.array([1.0, 0.0]))

    assert result.selected.factor == 1.5
    assert result.selected.root == 3
    assert result.evaluations[0].status is TrialStatus.MANIFOLD
    np.testing.assert_allclose(result.step, [1.5, 0.0])
    np.testing.assert_allclose(calc.staged[1], [1.5, 0.0])
    controller.validate_applied_geometry(np.array([1.5, 0.0]))
    with pytest.raises(Exception, match="differs"):
        controller.validate_applied_geometry(np.array([1.4, 0.0]))
    with pytest.raises(Exception, match="tolerance 1e-10"):
        controller.validate_applied_geometry(np.array([1.5 + 1.0e-9, 0.0]))
    assert calc.tracking_revision == 3


def test_identity_guards_precede_energy_ranking():
    calc = FakeCalculator(
        {
            1.0: ("ACCEPT", 2, -1.10, 0.90, 0.20, "clean"),
            1.5: ("ACCEPT", 4, -1.30, 0.50, 0.02, "weak identity"),
        }
    )
    controller = StateAwareStepController(
        factors=(1.0, 1.5), min_score=0.70, min_margin=0.10
    )
    result = controller.select_step(FakeOptimizer(calc), np.array([1.0, 0.0]))
    assert result.selected.factor == 1.0
    assert calc.surveyed == [1.0]


def test_exhaustive_mode_can_rank_all_clean_endpoints_by_energy():
    calc = FakeCalculator(
        {
            1.0: ("ACCEPT", 2, -1.10, 0.90, 0.20, "clean"),
            1.5: ("ACCEPT", 2, -1.30, 0.88, 0.18, "lower endpoint"),
        }
    )
    controller = StateAwareStepController(
        factors=(1.0, 1.5), fallback_only=False
    )

    result = controller.select_step(
        FakeOptimizer(calc), np.array([1.0, 0.0])
    )

    assert calc.surveyed == [1.0, 1.5]
    assert result.selected.factor == 1.5


def test_explicit_bridge_policy_can_exceed_optimizer_trust_radius():
    calc = FakeCalculator(
        {
            1.0: ("MANIFOLD", None, -1.05, 0.75, 0.01, "mixed"),
            1.5: ("ACCEPT", 2, -1.20, 0.90, 0.20, "clean bridge"),
        }
    )
    optimizer = FakeOptimizer(calc)
    optimizer.trust_max = 1.1
    controller = StateAwareStepController(
        factors=(1.0, 1.5),
        respect_trust_max=False,
        max_step_norm=1.6,
    )

    result = controller.select_step(optimizer, np.array([1.0, 0.0]))

    assert result.selected.factor == 1.5


def test_no_acceptable_endpoint_is_a_hard_stop():
    calc = FakeCalculator(
        {
            0.5: ("RETRY", 2, -1.10, 0.40, 0.20, "low overlap"),
            1.0: ("MANIFOLD", 2, -1.20, 0.80, 0.01, "mixed"),
        }
    )
    controller = StateAwareStepController(factors=(0.5, 1.0))
    with pytest.raises(NoAcceptableStateStep, match="No acceptable"):
        controller.select_step(FakeOptimizer(calc), np.array([1.0, 0.0]))
    assert calc.staged is None
    assert calc.discard_calls == 1


def test_detects_survey_that_mutates_committed_revision():
    class MutatingCalculator(FakeCalculator):
        def survey_state(self, atoms, cart_coords, *, factor):
            result = super().survey_state(atoms, cart_coords, factor=factor)
            self.tracking_revision += 1
            return result

    calc = MutatingCalculator(
        {1.0: ("ACCEPT", 2, -1.10, 0.90, 0.20, "clean")}
    )
    controller = StateAwareStepController(factors=(1.0,))
    with pytest.raises(NonTransactionalSurveyError):
        controller.select_step(FakeOptimizer(calc), np.array([1.0, 0.0]))
    assert calc.discard_calls == 1


def test_halt_is_not_hidden_by_another_acceptable_factor():
    calc = FakeCalculator(
        {
            1.0: ("HALT", None, None, 0.0, 0.0, "invalid normalization"),
            1.5: ("ACCEPT", 2, -1.10, 0.90, 0.20, "clean"),
        }
    )
    controller = StateAwareStepController(factors=(1.0, 1.5))
    with pytest.raises(FatalStateSurveyError, match="invalid normalization"):
        controller.select_step(FakeOptimizer(calc), np.array([1.0, 0.0]))
    assert calc.staged is None
    assert calc.discard_calls == 1


def test_mapping_factory_and_restart_history():
    controller = make_step_controller(
        {"type": "state_aware", "factors": [1.0], "require_descent": False}
    )
    calc = FakeCalculator(
        {1.0: ("ACCEPT", 2, -0.90, 0.90, 0.20, "allowed uphill")}
    )
    controller.select_step(FakeOptimizer(calc), np.array([0.5, 0.0]))
    info = controller.get_restart_info()

    restored = StateAwareStepController(factors=(1.0,))
    restored.set_restart_info(info)
    assert restored.history == controller.history
