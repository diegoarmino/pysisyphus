"""Transactional step selection for state-aware optimizations.

The controller in this module deliberately knows nothing about a particular
electronic-structure program.  It defines a small, duck-typed contract for a
calculator that can survey electronic states without mutating its committed
state and can stage one selected survey for the subsequent gradient call.

Rejected trial geometries never enter the optimizer history.  In particular,
``MANIFOLD`` trials do not stop evaluation of longer factors; this permits a
bounded bridge across a locally mixed region when a clean, descending endpoint
is found.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import math
from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np


class StepControlError(RuntimeError):
    """Base class for state-aware step-control failures."""


class NoAcceptableStateStep(StepControlError):
    """Raised when none of the surveyed step factors is acceptable."""

    def __init__(
        self,
        evaluations: Sequence["TrialEvaluation"],
        *,
        controller_rejections: Optional[Mapping[float, str]] = None,
    ):
        self.evaluations = tuple(evaluations)
        self.controller_rejections = dict(controller_rejections or {})
        details = "; ".join(
            f"lambda={trial.factor:g}: {trial.status.value} ({trial.reason})"
            + (
                f"; controller rejected: {self.controller_rejections[trial.factor]}"
                if trial.factor in self.controller_rejections
                else ""
            )
            for trial in self.evaluations
        )
        super().__init__(f"No acceptable state-following trial step. {details}")


class NonTransactionalSurveyError(StepControlError):
    """Raised when a supposedly read-only survey changes calculator state."""


class FatalStateSurveyError(StepControlError):
    """Raised when a HALT survey reveals invalid electronic-state data."""


class TrialStatus(str, Enum):
    ACCEPT = "ACCEPT"
    MANIFOLD = "MANIFOLD"
    RETRY = "RETRY"
    HALT = "HALT"


@dataclass(frozen=True)
class TrialEvaluation:
    """Normalized result returned by a transactional state survey."""

    factor: float
    status: TrialStatus
    energy: Optional[float] = None
    score: Optional[float] = None
    margin: Optional[float] = None
    root: Optional[int] = None
    reason: str = ""
    row_best_root: Optional[int] = None
    global_root: Optional[int] = None
    global_score: Optional[float] = None
    global_assignment_gap: Optional[float] = None
    row_global_agree: Optional[bool] = None
    manifold_status: Optional[str] = None
    manifold_reference_roots: tuple[int, ...] = ()
    manifold_candidate_roots: tuple[int, ...] = ()
    manifold_dimension: Optional[int] = None
    manifold_singular_values: tuple[float, ...] = ()
    manifold_principal_angles_deg: tuple[float, ...] = ()
    manifold_min_singular_value: Optional[float] = None
    manifold_rms_singular_value: Optional[float] = None
    manifold_max_angle_deg: Optional[float] = None
    manifold_chordal_distance: Optional[float] = None
    manifold_geodesic_distance: Optional[float] = None
    manifold_reference_energy_span_ev: Optional[float] = None
    manifold_candidate_energy_span_ev: Optional[float] = None
    manifold_dimension_match: Optional[bool] = None
    manifold_assignment_closed: Optional[bool] = None
    manifold_reason: str = ""
    cart_coords: Optional[np.ndarray] = field(default=None, compare=False, repr=False)
    payload: Any = field(default=None, compare=False, repr=False)

    @property
    def acceptable(self) -> bool:
        return self.status is TrialStatus.ACCEPT

    def serializable(self) -> dict:
        return {
            "factor": self.factor,
            "status": self.status.value,
            "energy": self.energy,
            "score": self.score,
            "margin": self.margin,
            "root": self.root,
            "reason": self.reason,
            "row_best_root": self.row_best_root,
            "global_root": self.global_root,
            "global_score": self.global_score,
            "global_assignment_gap": self.global_assignment_gap,
            "row_global_agree": self.row_global_agree,
            "manifold_status": self.manifold_status,
            "manifold_reference_roots": list(self.manifold_reference_roots),
            "manifold_candidate_roots": list(self.manifold_candidate_roots),
            "manifold_dimension": self.manifold_dimension,
            "manifold_singular_values": list(self.manifold_singular_values),
            "manifold_principal_angles_deg": list(self.manifold_principal_angles_deg),
            "manifold_min_singular_value": self.manifold_min_singular_value,
            "manifold_rms_singular_value": self.manifold_rms_singular_value,
            "manifold_max_angle_deg": self.manifold_max_angle_deg,
            "manifold_chordal_distance": self.manifold_chordal_distance,
            "manifold_geodesic_distance": self.manifold_geodesic_distance,
            "manifold_reference_energy_span_ev": self.manifold_reference_energy_span_ev,
            "manifold_candidate_energy_span_ev": self.manifold_candidate_energy_span_ev,
            "manifold_dimension_match": self.manifold_dimension_match,
            "manifold_assignment_closed": self.manifold_assignment_closed,
            "manifold_reason": self.manifold_reason,
        }


@dataclass(frozen=True)
class StepControlResult:
    """Chosen optimizer step and the complete audit trail of its surveys."""

    step: np.ndarray
    selected: TrialEvaluation
    evaluations: tuple[TrialEvaluation, ...]

    def serializable(self) -> dict:
        return {
            "selected_factor": self.selected.factor,
            "selected_root": self.selected.root,
            "selected_energy": self.selected.energy,
            "evaluations": [trial.serializable() for trial in self.evaluations],
        }


def _status_from(value: Any) -> TrialStatus:
    if isinstance(value, TrialStatus):
        return value
    # Interoperate with tdentrack's enum without importing it.
    value = getattr(value, "value", value)
    try:
        return TrialStatus(str(value).upper())
    except ValueError as exc:
        allowed = ", ".join(status.value for status in TrialStatus)
        raise StepControlError(f"Unknown survey status {value!r}; expected one of {allowed}.") from exc


def _field(result: Any, name: str, default: Any = None) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def normalize_trial_evaluation(result: Any, factor: float) -> TrialEvaluation:
    """Convert a calculator-specific survey record into the common contract."""

    decision = _field(result, "decision")
    source = decision if decision is not None else result
    status = _status_from(_field(source, "status"))
    selected_root = _field(source, "selected_root", _field(source, "root"))
    selected_energy = _field(
        source,
        "selected_energy_eh",
        _field(source, "selected_energy", _field(source, "energy")),
    )
    score = _field(source, "score", _field(source, "best_score"))
    margin = _field(source, "margin")
    reason = str(_field(source, "reason", ""))
    assignment = _field(source, "global_assignment")
    row_best_root = _field(assignment, "row_best_candidate_root")
    global_root = _field(assignment, "target_candidate_root")
    global_score = _field(assignment, "target_pair_score")
    global_assignment_gap = _field(assignment, "target_edge_stability_gap")
    row_global_agree = _field(assignment, "row_global_agree")
    manifold = _field(source, "manifold_report")
    manifold_status = _field(manifold, "status")
    manifold_status = _field(manifold_status, "value", manifold_status)
    manifold_reference_roots = tuple(
        int(root) for root in (_field(manifold, "reference_roots", ()) or ())
    )
    manifold_candidate_roots = tuple(
        int(root) for root in (_field(manifold, "candidate_roots", ()) or ())
    )
    continuity = _field(manifold, "continuity")
    raw_singular_values = _field(continuity, "singular_values", ())
    raw_principal_angles = _field(continuity, "principal_angles_rad", ())
    if raw_singular_values is None:
        raw_singular_values = ()
    if raw_principal_angles is None:
        raw_principal_angles = ()
    manifold_singular_values = tuple(
        float(value) for value in raw_singular_values
    )
    manifold_principal_angles_deg = tuple(
        float(np.degrees(value))
        for value in raw_principal_angles
    )

    def optional_float(name: str) -> Optional[float]:
        value = _field(manifold, name)
        return None if value is None else float(value)

    def optional_bool(name: str) -> Optional[bool]:
        value = _field(manifold, name)
        return None if value is None else bool(value)

    return TrialEvaluation(
        factor=float(factor),
        status=status,
        energy=None if selected_energy is None else float(selected_energy),
        score=None if score is None else float(score),
        margin=None if margin is None else float(margin),
        root=None if selected_root is None else int(selected_root),
        reason=reason,
        row_best_root=None if row_best_root is None else int(row_best_root),
        global_root=None if global_root is None else int(global_root),
        global_score=None if global_score is None else float(global_score),
        global_assignment_gap=(
            None
            if global_assignment_gap is None
            else float(global_assignment_gap)
        ),
        row_global_agree=(
            None if row_global_agree is None else bool(row_global_agree)
        ),
        manifold_status=None if manifold_status is None else str(manifold_status),
        manifold_reference_roots=manifold_reference_roots,
        manifold_candidate_roots=manifold_candidate_roots,
        manifold_dimension=(
            None
            if _field(manifold, "dimension") is None
            else int(_field(manifold, "dimension"))
        ),
        manifold_singular_values=manifold_singular_values,
        manifold_principal_angles_deg=manifold_principal_angles_deg,
        manifold_min_singular_value=optional_float("minimum_singular_value"),
        manifold_rms_singular_value=optional_float("rms_singular_value"),
        manifold_max_angle_deg=optional_float("maximum_principal_angle_deg"),
        manifold_chordal_distance=optional_float("chordal_distance"),
        manifold_geodesic_distance=optional_float("geodesic_distance"),
        manifold_reference_energy_span_ev=optional_float(
            "reference_energy_span_ev"
        ),
        manifold_candidate_energy_span_ev=optional_float(
            "candidate_energy_span_ev"
        ),
        manifold_dimension_match=optional_bool("dimension_match"),
        manifold_assignment_closed=optional_bool("assignment_closed"),
        manifold_reason=str(_field(manifold, "reason", "") or ""),
        payload=result,
    )


class StateAwareStepController:
    """Survey scaled proposals and stage one clean electronic-state endpoint.

    A compatible calculator must implement::

        survey_state(atoms, cart_coords, *, factor) -> survey
        stage_state_survey(survey, *, expected_cart_coords) -> None

    ``survey_state`` must be read-only with respect to the committed tracking
    reference, root, GBW and calculation history.  If the calculator exposes a
    ``tracking_revision`` attribute, the controller verifies that it remains
    unchanged throughout every survey.
    """

    def __init__(
        self,
        factors: Iterable[float] = (0.5, 0.75, 1.0, 1.25, 1.5),
        primary_factor: float = 1.0,
        fallback_only: bool = True,
        require_descent: bool = True,
        energy_tolerance: float = 0.0,
        min_score: Optional[float] = None,
        min_margin: Optional[float] = None,
        max_step_norm: Optional[float] = None,
        respect_trust_max: bool = True,
        coordinate_tolerance: float = 1.0e-8,
    ) -> None:
        factors = tuple(float(factor) for factor in factors)
        if not factors or any((not math.isfinite(factor)) or factor <= 0 for factor in factors):
            raise ValueError("Step-control factors must be a non-empty sequence of positive finite values.")
        if len(set(factors)) != len(factors):
            raise ValueError("Step-control factors must be unique.")
        self.factors = factors
        self.primary_factor = float(primary_factor)
        if self.primary_factor not in self.factors:
            raise ValueError("primary_factor must be present in factors.")
        self.fallback_only = bool(fallback_only)
        self.require_descent = bool(require_descent)
        self.energy_tolerance = float(energy_tolerance)
        self.min_score = None if min_score is None else float(min_score)
        self.min_margin = None if min_margin is None else float(min_margin)
        self.max_step_norm = None if max_step_norm is None else float(max_step_norm)
        if self.max_step_norm is not None and self.max_step_norm <= 0:
            raise ValueError("max_step_norm must be positive when supplied.")
        self.respect_trust_max = bool(respect_trust_max)
        self.coordinate_tolerance = float(coordinate_tolerance)
        if self.coordinate_tolerance < 0:
            raise ValueError("coordinate_tolerance cannot be negative.")
        self.history: list[dict] = []
        self.last_result: Optional[StepControlResult] = None
        self._last_coordinate_tolerance = self.coordinate_tolerance

    def _controller_rejection_reason(
        self, trial: TrialEvaluation, current_energy: Optional[float]
    ) -> Optional[str]:
        if not trial.acceptable:
            return "electronic-state decision is not ACCEPT"
        if self.min_score is not None and (trial.score is None or trial.score < self.min_score):
            return (
                f"similarity {trial.score!r} is below min_score "
                f"{self.min_score:.12g}"
            )
        if self.min_margin is not None and (trial.margin is None or trial.margin < self.min_margin):
            return (
                f"assignment margin {trial.margin!r} is below min_margin "
                f"{self.min_margin:.12g}"
            )
        if self.require_descent and current_energy is not None:
            if trial.energy is None:
                return "candidate energy is unavailable for the required descent check"
            limit = current_energy + self.energy_tolerance
            if trial.energy > limit:
                return (
                    f"candidate energy {trial.energy:.12f} Eh exceeds current "
                    f"energy {current_energy:.12f} Eh plus tolerance "
                    f"{self.energy_tolerance:.3e} Eh by "
                    f"{trial.energy - limit:.6e} Eh"
                )
        return None

    def _passes_controller_guards(
        self, trial: TrialEvaluation, current_energy: Optional[float]
    ) -> bool:
        return self._controller_rejection_reason(trial, current_energy) is None

    @staticmethod
    def _discard_uncommitted_surveys(calculator: Any) -> None:
        """Best-effort rollback of every probe created for one proposal.

        A calculator may keep several electronically acceptable alternatives
        pending until one endpoint gradient succeeds.  If proposal selection
        aborts before staging, none of those alternatives may leak into the
        next optimizer cycle.
        """

        discard = getattr(calculator, "discard_staged_state_survey", None)
        if callable(discard):
            discard()

    @staticmethod
    def _ranking_key(trial: TrialEvaluation) -> tuple[float, float, float, float]:
        # ACCEPT only certifies that a trial clears the minimum electronic
        # thresholds.  Several accepted endpoints can still differ greatly in
        # how well they preserve the committed state.  Rank by electronic
        # identity first; use endpoint energy only to break an otherwise exact
        # identity/step-scale tie.  Descent remains an independent controller
        # guard in _controller_rejection_reason().
        score = -math.inf if trial.score is None else trial.score
        margin = -math.inf if trial.margin is None else trial.margin
        energy = math.inf if trial.energy is None else trial.energy
        return (-score, -margin, abs(trial.factor - 1.0), energy)

    def select_step(self, optimizer: Any, step: np.ndarray) -> StepControlResult:
        step = np.asarray(step, dtype=float)
        geometry = optimizer.geometry
        calculator = geometry.calculator
        calculator_tolerance = getattr(calculator, "coordinate_tolerance", None)
        if calculator_tolerance is None:
            self._last_coordinate_tolerance = self.coordinate_tolerance
        else:
            calculator_tolerance = float(calculator_tolerance)
            if not math.isfinite(calculator_tolerance) or calculator_tolerance < 0.0:
                raise StepControlError(
                    "Calculator coordinate_tolerance must be finite and non-negative."
                )
            self._last_coordinate_tolerance = min(
                self.coordinate_tolerance, calculator_tolerance
            )
        survey_func = getattr(calculator, "survey_state", None)
        stage_func = getattr(calculator, "stage_state_survey", None)
        if not callable(survey_func) or not callable(stage_func):
            raise StepControlError(
                "State-aware step control requires calculator.survey_state() and "
                "calculator.stage_state_survey()."
            )

        current_energy = None
        if getattr(optimizer, "energies", None):
            current_energy = float(optimizer.energies[-1])

        evaluations: list[TrialEvaluation] = []
        ordered_factors = (self.primary_factor,) + tuple(
            factor for factor in self.factors if factor != self.primary_factor
        )
        for factor in ordered_factors:
            trial_step = factor * step
            if self.max_step_norm is not None and np.linalg.norm(trial_step) > self.max_step_norm:
                evaluations.append(
                    TrialEvaluation(
                        factor=factor,
                        status=TrialStatus.RETRY,
                        reason=f"scaled step norm exceeds {self.max_step_norm:g}",
                    )
                )
                continue
            trust_max = getattr(optimizer, "trust_max", None)
            if (
                self.respect_trust_max
                and trust_max is not None
                and np.linalg.norm(trial_step) > float(trust_max) + 1e-12
            ):
                evaluations.append(
                    TrialEvaluation(
                        factor=factor,
                        status=TrialStatus.RETRY,
                        reason=f"scaled step norm exceeds optimizer trust_max={float(trust_max):g}",
                    )
                )
                continue

            trial_coords = geometry.coords + trial_step
            try:
                # For internal coordinates this performs the same iterative
                # backtransformation used when the accepted step is applied,
                # but ``pure=True`` leaves the live Geometry untouched.  Do not
                # backtransform through Geometry.copy(): TRIC rotation
                # primitives and DLC bases are initialized relative to the
                # copied geometry, so optimizer coordinates from the live
                # internal basis would be interpreted in a different frame.
                cart_coords = geometry.get_temporary_coords(trial_coords)
            except Exception as exc:
                if exc.__class__.__name__ not in (
                    "NeedNewInternalsException",
                    "RebuiltInternalsException",
                ):
                    self._discard_uncommitted_surveys(calculator)
                    raise
                evaluations.append(
                    TrialEvaluation(
                        factor=factor,
                        status=TrialStatus.RETRY,
                        reason="scaled step requires rebuilding internal coordinates",
                    )
                )
                continue
            revision_before = getattr(calculator, "tracking_revision", None)
            try:
                raw = survey_func(geometry.atoms, cart_coords, factor=factor)
                revision_after = getattr(calculator, "tracking_revision", None)
                if revision_before is not None and revision_after != revision_before:
                    raise NonTransactionalSurveyError(
                        f"survey_state changed tracking_revision from {revision_before!r} to {revision_after!r}."
                    )
                evaluation = normalize_trial_evaluation(raw, factor)
            except BaseException:
                self._discard_uncommitted_surveys(calculator)
                raise
            evaluations.append(
                replace(evaluation, cart_coords=np.array(cart_coords, dtype=float, copy=True))
            )
            if evaluations[-1].status is TrialStatus.HALT:
                break

            # In the common case the optimizer's own proposal already has a
            # unique, descending electronic identity.  Avoid launching every
            # fallback TDDFT job unless the primary endpoint actually needs an
            # alternative.  ``fallback_only=False`` retains exhaustive
            # identity-first ranking for workflows that explicitly want it.
            if (
                self.fallback_only
                and factor == self.primary_factor
                and self._passes_controller_guards(evaluations[-1], current_energy)
            ):
                break

        fatal = [trial for trial in evaluations if trial.status is TrialStatus.HALT]
        if fatal:
            details = "; ".join(
                f"lambda={trial.factor:g}: {trial.reason}" for trial in fatal
            )
            self._discard_uncommitted_surveys(calculator)
            raise FatalStateSurveyError(f"Electronic-state survey requested HALT. {details}")

        acceptable = [
            trial
            for trial in evaluations
            if self._passes_controller_guards(trial, current_energy)
        ]
        if not acceptable:
            self._discard_uncommitted_surveys(calculator)
            controller_rejections = {
                trial.factor: reason
                for trial in evaluations
                if (
                    reason := self._controller_rejection_reason(
                        trial, current_energy
                    )
                )
                is not None
            }
            raise NoAcceptableStateStep(
                evaluations,
                controller_rejections=controller_rejections,
            )

        selected = min(acceptable, key=self._ranking_key)
        try:
            stage_func(
                selected.payload,
                expected_cart_coords=np.array(selected.cart_coords, dtype=float, copy=True),
            )
        except BaseException:
            self._discard_uncommitted_surveys(calculator)
            raise
        result = StepControlResult(
            step=selected.factor * step,
            selected=selected,
            evaluations=tuple(evaluations),
        )
        self.last_result = result
        self.history.append(result.serializable())
        return result

    def validate_applied_geometry(self, actual_cart_coords: np.ndarray) -> None:
        """Ensure the committed Geometry matches the electronically surveyed one."""

        if self.last_result is None:
            return
        expected = self.last_result.selected.cart_coords
        actual = np.asarray(actual_cart_coords, dtype=float)
        if expected is None or actual.shape != expected.shape:
            raise StepControlError("Applied geometry is incompatible with the selected state survey.")
        max_error = float(np.max(np.abs(actual - expected))) if actual.size else 0.0
        tolerance = getattr(
            self, "_last_coordinate_tolerance", self.coordinate_tolerance
        )
        if max_error > tolerance:
            raise StepControlError(
                "Applied geometry differs from the selected state survey by "
                f"{max_error:.3g} bohr (tolerance {tolerance:.3g})."
            )

    def get_restart_info(self) -> dict:
        return {"history": list(self.history)}

    def set_restart_info(self, info: Optional[Mapping[str, Any]]) -> None:
        self.history = list((info or {}).get("history", []))


def make_step_controller(config: Any) -> Optional[StateAwareStepController]:
    """Construct a controller from an optimizer keyword value."""

    if config in (None, False):
        return None
    if isinstance(config, StateAwareStepController):
        return config
    if isinstance(config, Mapping):
        kwargs = dict(config)
        kind = kwargs.pop("type", "state_aware")
        if kind not in ("state_aware", "state-aware"):
            raise ValueError(f"Unknown step-controller type {kind!r}.")
        return StateAwareStepController(**kwargs)
    if callable(getattr(config, "select_step", None)):
        return config
    raise TypeError("step_controller must be None, a mapping, or an object implementing select_step().")
