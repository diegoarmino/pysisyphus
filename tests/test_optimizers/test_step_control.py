from types import SimpleNamespace

import numpy as np
import pytest

from pysisyphus.Geometry import Geometry
from pysisyphus.optimizers.step_control import (
    FatalStateSurveyError,
    NoAcceptableStateStep,
    NonTransactionalSurveyError,
    StateAwareStepController,
    TrialStatus,
    make_step_controller,
    normalize_trial_evaluation,
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


def test_exhaustive_mode_prefers_stronger_identity_over_lower_energy():
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
    assert result.selected.factor == 1.0


def test_root8_regression_prefers_clean_half_step_over_lower_energy_bridge():
    """Reproduce the first state-8 proposal from the Ru optimization.

    The unscaled proposal was ambiguous.  Of the accepted fallbacks, lambda
    0.5 retained root 8 with much stronger similarity and margin, whereas the
    former energy-first policy selected the lower-energy lambda-1.5/root-5
    endpoint.
    """

    calc = FakeCalculator(
        {
            1.0: ("RETRY", 7, -1222.967519938, 0.590254, 0.037487, "ambiguous"),
            0.5: ("ACCEPT", 8, -1222.964315780, 0.935142, 0.646000, "clean"),
            0.75: ("ACCEPT", 8, -1222.964745675, 0.747799, 0.182961, "clean"),
            1.25: ("ACCEPT", 6, -1222.967807311, 0.692288, 0.240620, "clean"),
            1.5: ("ACCEPT", 5, -1222.968159327, 0.719634, 0.209542, "clean"),
        }
    )
    optimizer = FakeOptimizer(calc)
    optimizer.energies = [-1222.960691738]
    controller = StateAwareStepController(
        factors=(0.5, 0.75, 1.0, 1.25, 1.5),
        fallback_only=True,
    )

    result = controller.select_step(optimizer, np.array([1.0, 0.0]))

    assert calc.surveyed == [1.0, 0.5, 0.75, 1.25, 1.5]
    assert result.selected.factor == 0.5
    assert result.selected.root == 8
    assert result.selected.score == pytest.approx(0.935142)
    np.testing.assert_allclose(result.step, [0.5, 0.0])


def test_global_assignment_diagnostics_are_serialized_for_restart_audits():
    assignment = SimpleNamespace(
        row_best_candidate_root=5,
        target_candidate_root=7,
        target_pair_score=0.534678796,
        target_edge_stability_gap=0.009215957,
        row_global_agree=False,
    )
    result = SimpleNamespace(
        decision=SimpleNamespace(
            status="RETRY",
            selected_root=None,
            selected_energy_eh=-1222.972901173,
            best_score=0.654836563,
            margin=0.120157767,
            reason="row/global disagreement",
            global_assignment=assignment,
        )
    )

    evaluation = normalize_trial_evaluation(result, 1.0)
    serialized = evaluation.serializable()

    assert evaluation.row_best_root == 5
    assert evaluation.global_root == 7
    assert evaluation.global_score == pytest.approx(0.534678796)
    assert evaluation.global_assignment_gap == pytest.approx(0.009215957)
    assert evaluation.row_global_agree is False
    assert serialized["global_root"] == 7
    assert serialized["row_global_agree"] is False


def test_root_manifold_diagnostics_are_serialized_for_every_trial():
    manifold = SimpleNamespace(
        status=SimpleNamespace(value="ROTATED"),
        reference_roots=(5, 7),
        candidate_roots=(7, 5),
        dimension=2,
        continuity=SimpleNamespace(
            singular_values=(0.999, 0.947),
            principal_angles_rad=(0.04473, 0.32673),
        ),
        minimum_singular_value=0.947,
        rms_singular_value=0.973,
        maximum_principal_angle_deg=18.72,
        chordal_distance=0.321,
        geodesic_distance=0.327,
        reference_energy_span_ev=0.045,
        candidate_energy_span_ev=0.062,
        dimension_match=True,
        assignment_closed=True,
        reason="preserved two-root manifold",
    )
    result = SimpleNamespace(
        decision=SimpleNamespace(
            status="MANIFOLD",
            selected_root=None,
            best_score=0.71,
            margin=0.01,
            reason="individual roots are ambiguous",
            manifold_report=manifold,
        )
    )

    evaluation = normalize_trial_evaluation(result, 0.75)
    serialized = evaluation.serializable()

    assert evaluation.manifold_status == "ROTATED"
    assert evaluation.manifold_reference_roots == (5, 7)
    assert evaluation.manifold_candidate_roots == (7, 5)
    assert evaluation.manifold_dimension == 2
    assert evaluation.manifold_singular_values == pytest.approx((0.999, 0.947))
    assert evaluation.manifold_principal_angles_deg == pytest.approx(
        tuple(np.degrees((0.04473, 0.32673)))
    )
    assert evaluation.manifold_min_singular_value == pytest.approx(0.947)
    assert evaluation.manifold_max_angle_deg == pytest.approx(18.72)
    assert evaluation.manifold_dimension_match is True
    assert evaluation.manifold_assignment_closed is True
    assert serialized["manifold_reference_roots"] == [5, 7]
    assert serialized["manifold_candidate_roots"] == [7, 5]
    assert serialized["manifold_singular_values"] == pytest.approx([0.999, 0.947])
    assert serialized["manifold_status"] == "ROTATED"
    assert serialized["manifold_reason"] == "preserved two-root manifold"


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


def test_tric_trial_uses_live_internal_coordinate_frame():
    """A second TRIC step must not be transformed in a reinitialized copy.

    TRIC rotations are defined relative to their initialization geometry.  A
    copied Geometry therefore has a different internal-coordinate frame after
    the first step, even when it contains the same Cartesian coordinates.
    """

    atoms = ("O", "H", "H", "O", "H", "H")
    cart_coords = np.array(
        (
            0.00, 0.00, 0.00,
            1.43, 1.10, 0.00,
            -1.43, 1.10, 0.00,
            7.00, 0.30, 0.20,
            8.43, 1.40, 0.20,
            5.57, 1.40, 0.20,
        )
    )
    geometry = Geometry(atoms, cart_coords, coord_type="tric")
    rng = np.random.default_rng(12)

    first_step = rng.normal(size=geometry.coords.size)
    first_step *= 0.08 / np.linalg.norm(first_step)
    geometry.coords = geometry.coords + first_step

    second_step = rng.normal(size=geometry.coords.size)
    second_step *= 0.15 / np.linalg.norm(second_step)
    before_cart = geometry.cart_coords.copy()
    expected_cart = geometry.get_temporary_coords(geometry.coords + second_step)

    calc = FakeCalculator(
        {1.0: ("ACCEPT", 2, -1.10, 0.90, 0.20, "clean")}
    )
    geometry.set_calculator(calc)
    optimizer = SimpleNamespace(
        geometry=geometry,
        energies=[-1.0],
        trust_max=1.0,
    )
    controller = StateAwareStepController(
        factors=(1.0,),
        require_descent=False,
    )

    result = controller.select_step(optimizer, second_step)

    # Surveying is read-only and uses the current TRIC reference frame.
    np.testing.assert_allclose(geometry.cart_coords, before_cart, atol=1.0e-12)
    np.testing.assert_allclose(calc.staged[1], expected_cart, atol=1.0e-12)

    # Applying the selected internal step reaches exactly the surveyed endpoint.
    geometry.coords = geometry.coords + result.step
    np.testing.assert_allclose(geometry.cart_coords, expected_cart, atol=1.0e-12)
    controller.validate_applied_geometry(geometry.cart_coords)


def test_pure_backtransform_rebuild_request_marks_trial_for_retry():
    class NeedNewInternalsException(Exception):
        pass

    calc = FakeCalculator(
        {1.0: ("ACCEPT", 2, -1.10, 0.90, 0.20, "unreachable")}
    )
    optimizer = FakeOptimizer(calc)

    def require_rebuild(coords):
        raise NeedNewInternalsException

    optimizer.geometry.get_temporary_coords = require_rebuild
    controller = StateAwareStepController(factors=(1.0,))

    with pytest.raises(NoAcceptableStateStep) as exc_info:
        controller.select_step(optimizer, np.array([0.1, 0.0]))

    assert exc_info.value.evaluations[0].status is TrialStatus.RETRY
    assert "rebuilding internal coordinates" in str(exc_info.value)
    assert calc.surveyed == []


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


def test_accepted_but_uphill_trials_report_controller_energy_rejection():
    calc = FakeCalculator(
        {
            0.5: ("ACCEPT", 2, -0.97, 0.90, 0.20, "identity passed"),
            1.0: ("ACCEPT", 2, -0.95, 0.91, 0.21, "identity passed"),
        }
    )
    controller = StateAwareStepController(factors=(0.5, 1.0))

    with pytest.raises(NoAcceptableStateStep) as exc_info:
        controller.select_step(FakeOptimizer(calc), np.array([1.0, 0.0]))

    message = str(exc_info.value)
    assert "candidate energy -0.950000000000 Eh" in message
    assert "current energy -1.000000000000 Eh" in message
    assert "5.000000e-02 Eh" in message
    assert set(exc_info.value.controller_rejections) == {0.5, 1.0}
    assert "5.000000e-02 Eh" in exc_info.value.controller_rejections[1.0]
    assert "3.000000e-02 Eh" in exc_info.value.controller_rejections[0.5]
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
