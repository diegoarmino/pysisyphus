"""Transactional ORCA adapter for TDenTrack state following.

This module is intentionally additive.  It does not use
:class:`~pysisyphus.calculators.OverlapCalculator.OverlapCalculator`'s mutable
root-following history.  Instead, an injected survey runner performs an
isolated all-root calculation and returns immutable electronic data to a
``tdentrack.TrackingSession``.  The session advances only after an ORCA
gradient for the staged root has completed successfully at the exact staged
geometry.

The separation is important for state-aware step control: several geometries
may be surveyed from one committed reference, but rejected probes must not
change the root, GBW, optimizer history, or committed electronic snapshot.

``tdentrack`` is an optional dependency of pysisyphus.  Existing applications
can either pass an already constructed, API-compatible ``tracking_session`` or
let :class:`TDenTrackORCA` construct one from ``initial_snapshot``.  The latter
form gives a clear installation error when tdentrack is unavailable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
import re
import threading
from types import MappingProxyType
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from pysisyphus.calculators.ORCA import ORCA


class TDenTrackAdapterError(RuntimeError):
    """Base class for transactional adapter failures."""


class TDenTrackDependencyError(ImportError):
    """Raised when automatic session construction cannot import tdentrack."""


class SurveyProtocolError(TDenTrackAdapterError):
    """Raised when a survey runner violates the adapter contract."""


class NonTransactionalSurveyError(TDenTrackAdapterError):
    """Raised when an isolated survey changes committed calculator state."""


class StateStagingError(TDenTrackAdapterError):
    """Raised when a survey cannot safely be staged for a gradient."""


class StagedGeometryMismatch(StateStagingError):
    """Raised when a gradient geometry differs from the staged geometry."""


class UnstagedGeometryError(StateStagingError):
    """Raised when a new geometry is evaluated without a staged state survey."""


class GradientProtocolError(TDenTrackAdapterError):
    """Raised when a gradient runner returns incomplete or invalid data."""


def _readonly_coordinates(values: Any, *, name: str = "coordinates") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim == 2:
        if array.shape[1] != 3:
            raise ValueError(f"{name} must be flat or have shape (n_atoms, 3).")
        array = array.reshape(-1)
    elif array.ndim != 1:
        raise ValueError(f"{name} must be flat or have shape (n_atoms, 3).")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains non-finite values.")
    # Publishing a view backed by bytes prevents callers from re-enabling
    # writes on an owning ndarray and changing a staged geometry in place.
    readonly = np.frombuffer(np.ascontiguousarray(array).tobytes(), dtype=float)
    readonly.setflags(write=False)
    return readonly


def _coordinates_match(left: Any, right: Any, tolerance: float) -> bool:
    try:
        left_array = np.asarray(left, dtype=float).reshape(-1)
        right_array = np.asarray(right, dtype=float).reshape(-1)
    except (TypeError, ValueError):
        return False
    return left_array.shape == right_array.shape and bool(
        np.allclose(left_array, right_array, rtol=0.0, atol=tolerance)
    )


def _status_value(decision: Any) -> str:
    status = getattr(decision, "status", None)
    return str(getattr(status, "value", status)).upper()


def _decision_is_accepted(decision: Any) -> bool:
    accepted = getattr(decision, "accepted", None)
    if accepted is not None:
        return bool(accepted)
    return _status_value(decision) == "ACCEPT"


def _mapping_copy(values: Mapping[Any, Any]) -> Mapping[Any, Any]:
    return MappingProxyType(dict(values))


@dataclass(frozen=True)
class SurveyData:
    """Electronic data returned by an isolated all-root survey runner.

    ``candidate`` is normally a tdentrack ``ElectronicSnapshot`` with no
    selected root.  Its ``requested_roots`` must declare the all-root window;
    an incomplete window is passed to tdentrack's selector and normally yields
    ``RETRY``.  ``overlaps`` are signed transition-density overlaps against the
    supplied reference snapshot.  Normalization is deliberately left to
    tdentrack's selection policy.
    """

    candidate: Any
    overlaps: Mapping[int, float]
    reference_norm: float = 1.0
    candidate_norms: Mapping[int, float] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)
    signed_overlap_matrix: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )
    reference_roots: tuple[int, ...] = ()
    reference_gram: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )
    candidate_gram: Optional[np.ndarray] = field(
        default=None, repr=False, compare=False
    )
    subspace_continuity: Any = field(default=None, repr=False, compare=False)
    root_overlap_block: Any = field(default=None, repr=False, compare=False)

    def __post_init__(self) -> None:
        overlaps = {int(root): float(value) for root, value in self.overlaps.items()}
        norms = {int(root): float(value) for root, value in self.candidate_norms.items()}
        if any(not math.isfinite(value) for value in overlaps.values()):
            raise ValueError("Survey overlaps contain non-finite values.")
        if any(not math.isfinite(value) for value in norms.values()):
            raise ValueError("Survey candidate norms contain non-finite values.")
        reference_norm = float(self.reference_norm)
        if not math.isfinite(reference_norm):
            raise ValueError("Survey reference norm is not finite.")
        object.__setattr__(self, "overlaps", _mapping_copy(overlaps))
        object.__setattr__(self, "candidate_norms", _mapping_copy(norms))
        object.__setattr__(self, "metadata", _mapping_copy(self.metadata))
        object.__setattr__(self, "reference_norm", reference_norm)
        reference_roots = tuple(int(root) for root in self.reference_roots)
        object.__setattr__(self, "reference_roots", reference_roots)
        if self.signed_overlap_matrix is not None:
            matrix = np.asarray(self.signed_overlap_matrix, dtype=float)
            expected_shape = (len(reference_roots), len(self.candidate.roots))
            if not reference_roots:
                raise ValueError(
                    "reference_roots are required with signed_overlap_matrix."
                )
            if matrix.shape != expected_shape:
                raise ValueError(
                    f"Signed overlap matrix has shape {matrix.shape}, expected "
                    f"{expected_shape}."
                )
            if not np.all(np.isfinite(matrix)):
                raise ValueError("Signed overlap matrix contains non-finite values.")
            immutable = np.frombuffer(
                np.ascontiguousarray(matrix).tobytes(), dtype=float
            ).reshape(matrix.shape)
            immutable.setflags(write=False)
            object.__setattr__(self, "signed_overlap_matrix", immutable)
        for field_name in ("reference_gram", "candidate_gram"):
            values = getattr(self, field_name)
            if values is None:
                continue
            matrix = np.asarray(values, dtype=float)
            roots_for_matrix = (
                reference_roots
                if field_name == "reference_gram"
                else tuple(int(root) for root in self.candidate.roots)
            )
            expected_shape = (len(roots_for_matrix), len(roots_for_matrix))
            if matrix.shape != expected_shape or not np.all(np.isfinite(matrix)):
                raise ValueError(
                    f"{field_name} must be a finite matrix with shape {expected_shape}."
                )
            immutable = np.frombuffer(
                np.ascontiguousarray(matrix).tobytes(), dtype=float
            ).reshape(matrix.shape)
            immutable.setflags(write=False)
            object.__setattr__(self, field_name, immutable)


@dataclass(frozen=True)
class TDenTrackSurvey:
    """Adapter result understood by ``StateAwareStepController``.

    The controller reads ``decision`` for status, root, energy, score and
    margin, then passes this entire object back to ``stage_state_survey``.
    ``_owner`` prevents a result produced by one calculator from being staged
    on another calculator.
    """

    survey: Any
    decision: Any
    coordinates: np.ndarray = field(repr=False, compare=False)
    factor: float
    tracking_revision: int
    data: SurveyData = field(compare=False)
    _owner: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class _StagedSurvey:
    result: TDenTrackSurvey
    expected_coordinates: np.ndarray = field(repr=False, compare=False)


def _coerce_survey_data(result: Any) -> SurveyData:
    if isinstance(result, SurveyData):
        return result
    if isinstance(result, Mapping):
        try:
            candidate = result["candidate"]
            overlaps = result["overlaps"]
        except KeyError as exc:
            raise SurveyProtocolError(
                "A survey mapping must contain 'candidate' and 'overlaps'."
            ) from exc
        return SurveyData(
            candidate=candidate,
            overlaps=overlaps,
            reference_norm=result.get("reference_norm", 1.0),
            candidate_norms=result.get("candidate_norms", {}),
            metadata=result.get("metadata", {}),
            signed_overlap_matrix=result.get("signed_overlap_matrix"),
            reference_roots=tuple(result.get("reference_roots", ())),
            reference_gram=result.get("reference_gram"),
            candidate_gram=result.get("candidate_gram"),
            subspace_continuity=result.get("subspace_continuity"),
            root_overlap_block=result.get("root_overlap_block"),
        )
    raise SurveyProtocolError(
        "survey_runner must return SurveyData or a mapping containing "
        "'candidate' and 'overlaps'."
    )


def _make_tracking_session(initial_snapshot: Any, selector: Any = None) -> Any:
    try:
        from excited_state_diabatizer.state_tracking import TrackingSession
    except ImportError as exc:
        raise TDenTrackDependencyError(
            "TDenTrackORCA requires the optional 'tdentrack' package when "
            "initial_snapshot is used. Install tdentrack or pass an already "
            "constructed tracking_session."
        ) from exc
    return TrackingSession(initial_snapshot, selector=selector)


class TDenTrackORCA(ORCA):
    """Opt-in ORCA calculator with transactional excited-state tracking.

    Parameters
    ----------
    tracking_session
        A tdentrack ``TrackingSession``.  It must already contain a selected,
        committed initial snapshot.  Alternatively, pass ``initial_snapshot``.
    initial_snapshot, selector
        Convenience arguments used to construct a real tdentrack session.
        This path requires the optional ``tdentrack`` package to be importable.
    survey_runner
        Callable with signature ``runner(atoms, coords, *, reference, factor)``.
        It must run in an isolated work directory/calculator and return
        :class:`SurveyData`.  It must not call calculation methods on this
        parent calculator.  ``coords`` is a read-only flat copy in bohr.
    enable_default_survey
        Explicitly construct the built-in isolated ORCA 6.1.1 BSON/``.cis``
        backend when no ``survey_runner`` is supplied.  This is false by
        default so an ordinary ORCA calculator can never launch extra jobs by
        accident.  Backend options are passed in ``default_survey_options``.
    gradient_runner
        Optional offline/testing hook with signature
        ``runner(atoms, coords, *, root, calculator, prepare_kwargs)``.  When
        absent, the normal ORCA ``EnGrad`` implementation is used.
    coordinate_tolerance
        Absolute tolerance in bohr for staged-coordinate checks.  It defaults
        to ``1e-10`` to permit harmless internal-coordinate backtransform
        roundoff while rejecting a different geometry.
    survey_against
        ``"committed"`` compares each probe with the most recently accepted
        state.  ``"anchor"`` uses ``TrackingSession.anchor`` instead, allowing
        a caller-controlled stable-reference policy.

    Notes
    -----
    Legacy ``OverlapCalculator.track`` is always disabled.  A survey runner is
    required because parsing ORCA 6.1.1 all-root artifacts and constructing
    exact inter-geometry AO overlaps belong to the TDenTrack backend, not to
    the optimizer/calculator transaction layer.
    """

    RESTART_KEY = "tdentrack_transaction"
    RESTART_VERSION = 1

    def __init__(
        self,
        *args: Any,
        tracking_session: Any = None,
        initial_snapshot: Any = None,
        selector: Any = None,
        survey_runner: Optional[Callable[..., Any]] = None,
        enable_default_survey: bool = False,
        default_survey_options: Optional[Mapping[str, Any]] = None,
        gradient_runner: Optional[Callable[..., Mapping[str, Any]]] = None,
        coordinate_tolerance: float = 1.0e-10,
        require_declared_root_window: bool = True,
        survey_against: str = "committed",
        update_anchor_on_commit: bool = True,
        **kwargs: Any,
    ) -> None:
        if tracking_session is not None and initial_snapshot is not None:
            raise TypeError("Pass tracking_session or initial_snapshot, not both.")
        if tracking_session is None:
            if initial_snapshot is None:
                raise TypeError("TDenTrackORCA requires tracking_session or initial_snapshot.")
            tracking_session = _make_tracking_session(initial_snapshot, selector)
        elif selector is not None:
            raise TypeError("selector is only valid when initial_snapshot constructs the session.")
        enable_default_survey = bool(enable_default_survey)
        if survey_runner is not None and enable_default_survey:
            raise TypeError(
                "Pass survey_runner or enable_default_survey=True, not both."
            )
        if survey_runner is None and not enable_default_survey:
            raise TypeError(
                "TDenTrackORCA requires a callable isolated survey_runner, or "
                "enable_default_survey=True for the ORCA 6.1.1 BSON/CIS backend."
            )
        if survey_runner is not None and not callable(survey_runner):
            raise TypeError("survey_runner must be callable when supplied.")
        if gradient_runner is not None and not callable(gradient_runner):
            raise TypeError("gradient_runner must be callable when supplied.")
        coordinate_tolerance = float(coordinate_tolerance)
        if not math.isfinite(coordinate_tolerance) or coordinate_tolerance < 0.0:
            raise ValueError("coordinate_tolerance must be finite and non-negative.")
        if survey_against not in ("committed", "anchor"):
            raise ValueError("survey_against must be either 'committed' or 'anchor'.")
        if kwargs.get("track", False):
            raise ValueError(
                "Legacy ORCA/OverlapCalculator root tracking cannot be combined "
                "with TDenTrackORCA. Remove 'track: true'."
            )
        kwargs["track"] = False

        super().__init__(*args, **kwargs)
        if gradient_runner is None and re.search(
            r"\btgradlist\b", self.blocks, flags=re.IGNORECASE
        ):
            raise ValueError(
                "TDenTrackORCA does not accept TGradList in the selected-root "
                "gradient block; it must produce exactly one auditable ORCA EnGrad."
            )
        if re.search(
            r"\bfollowiroot\s*(?:=\s*)?(?:true|1)\b",
            self.blocks,
            flags=re.IGNORECASE,
        ):
            raise ValueError(
                "TDenTrackORCA requires FollowIRoot false: native ORCA root "
                "following cannot be combined with external transactional selection."
            )

        self.tracking_session = tracking_session
        self.survey_runner = survey_runner
        self.default_survey_backend = None
        self.gradient_runner = gradient_runner
        self.last_orca_engrad_energy = None
        self.coordinate_tolerance = coordinate_tolerance
        self.require_declared_root_window = bool(require_declared_root_window)
        self.survey_against = survey_against
        self.update_anchor_on_commit = bool(update_anchor_on_commit)
        self._tracking_revision_offset = 0
        self._owner_token = object()
        self._staged_state_survey: Optional[_StagedSurvey] = None
        self._transaction_lock = threading.RLock()
        # Retain the configured selector so restart can build a fresh public
        # TrackingSession without writing its private state.
        self._tracking_selector = getattr(tracking_session, "_selector", None)

        committed = self._committed_snapshot()
        committed_root = self._selected_root(committed)
        if self.root is not None and int(self.root) != committed_root:
            raise ValueError(
                f"ORCA root {self.root} differs from committed TDenTrack root {committed_root}."
            )
        self.root = committed_root
        if enable_default_survey:
            try:
                from pysisyphus.calculators.ORCA6StateSurvey import (
                    ORCA6AllRootSurveyBackend,
                )
            except ImportError as exc:
                raise TDenTrackDependencyError(
                    "Could not import the optional ORCA 6.1.1 state-survey backend."
                ) from exc
            backend = ORCA6AllRootSurveyBackend(
                self, **dict(default_survey_options or {})
            )
            self.default_survey_backend = backend
            self.survey_runner = backend

    @property
    def tracking_revision(self) -> int:
        """Monotonic revision of the committed electronic reference."""

        generation = int(getattr(self.tracking_session, "generation"))
        return self._tracking_revision_offset + generation

    @property
    def has_staged_state_survey(self) -> bool:
        return self._staged_state_survey is not None

    def _committed_snapshot(self) -> Any:
        try:
            committed = self.tracking_session.committed
        except AttributeError as exc:
            raise TypeError("tracking_session must expose a committed snapshot.") from exc
        self._selected_root(committed)
        if not hasattr(committed, "coordinates"):
            raise TypeError("The committed snapshot must expose coordinates.")
        return committed

    @staticmethod
    def _selected_root(snapshot: Any) -> int:
        root = getattr(snapshot, "selected_root", None)
        if root is None:
            raise ValueError("The committed TDenTrack snapshot has no selected root.")
        return int(root)

    def _survey_fingerprint(self) -> tuple[Any, ...]:
        """State which an isolated survey is forbidden to modify."""

        committed = self._committed_snapshot()
        anchor = getattr(self.tracking_session, "anchor", committed)
        pending = getattr(self.tracking_session, "pending", ())
        pending_ids = tuple(getattr(item, "survey_id", id(item)) for item in pending)
        gbw = getattr(self, "gbw", None)
        return (
            self.tracking_revision,
            id(committed),
            getattr(committed, "label", None),
            self._selected_root(committed),
            np.asarray(committed.coordinates, dtype=float).reshape(-1).tobytes(),
            id(anchor),
            getattr(anchor, "label", None),
            getattr(anchor, "selected_root", None),
            np.asarray(anchor.coordinates, dtype=float).reshape(-1).tobytes(),
            pending_ids,
            self.root,
            None if gbw is None else str(gbw),
            self.calc_counter,
            id(self._staged_state_survey),
        )

    def _survey_reference(self) -> Any:
        if self.survey_against == "committed":
            return self._committed_snapshot()
        try:
            return self.tracking_session.anchor
        except AttributeError as exc:
            raise TypeError(
                "survey_against='anchor' requires tracking_session.anchor."
            ) from exc

    def _capture_tracking_state(self) -> dict[str, Any]:
        """Capture the mutable session containers guarded around callbacks."""

        session = self.tracking_session
        names = (
            "_committed",
            "_anchor",
            "_history",
            "_pending",
            "_generation",
            "_counter",
        )
        state: dict[str, Any] = {}
        for name in names:
            if hasattr(session, name):
                value = getattr(session, name)
                if isinstance(value, dict):
                    value = value.copy()
                elif isinstance(value, list):
                    value = value.copy()
                state[name] = value
        # API-compatible test/downstream sessions may expose writable public
        # fields instead of tdentrack's private backing attributes.
        if "_committed" not in state and hasattr(session, "committed"):
            state["committed"] = session.committed
        if "_generation" not in state and hasattr(session, "generation"):
            state["generation"] = session.generation
        return state

    def _restore_tracking_state(self, state: Mapping[str, Any]) -> None:
        session = self.tracking_session
        for name, value in state.items():
            if isinstance(value, dict):
                value = value.copy()
            elif isinstance(value, list):
                value = value.copy()
            setattr(session, name, value)

    def _validate_candidate(self, candidate: Any, coordinates: np.ndarray) -> None:
        if not hasattr(candidate, "coordinates") or not hasattr(candidate, "roots"):
            raise SurveyProtocolError(
                "Survey candidate must be an ElectronicSnapshot-compatible object "
                "with coordinates and roots."
            )
        if not _coordinates_match(
            candidate.coordinates, coordinates, self.coordinate_tolerance
        ):
            raise SurveyProtocolError(
                "Survey candidate coordinates do not match the requested trial geometry."
            )
        if getattr(candidate, "selected_root", None) is not None:
            raise SurveyProtocolError(
                "An all-root survey candidate must be unselected (selected_root=None)."
            )
        roots = tuple(int(root) for root in candidate.roots)
        if not roots:
            raise SurveyProtocolError("An all-root survey returned no roots.")
        requested = tuple(int(root) for root in getattr(candidate, "requested_roots", ()))
        if self.require_declared_root_window and not requested:
            raise SurveyProtocolError(
                "An all-root survey must declare candidate.requested_roots so root-window "
                "completeness can be audited."
            )

    def survey_state(
        self,
        atoms: Sequence[str],
        cart_coords: Any,
        *,
        factor: float = 1.0,
    ) -> TDenTrackSurvey:
        """Survey a trial geometry without advancing committed state.

        The runner executes before anything is registered with the tracking
        session.  Runner failures and protocol errors therefore leave even the
        pending-survey collection unchanged.  Non-accepted decisions are
        discarded immediately; accepted alternatives remain pending until one
        is staged and its selected-root gradient succeeds.
        """

        factor = float(factor)
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError("Survey factor must be finite and positive.")
        coordinates = _readonly_coordinates(cart_coords, name="trial coordinates")

        with self._transaction_lock:
            reference = self._survey_reference()
            before = self._survey_fingerprint()
            tracking_state = self._capture_tracking_state()
            parent_state = (
                self.root,
                getattr(self, "gbw", None),
                self.calc_counter,
                self._staged_state_survey,
            )
            try:
                raw_data = self.survey_runner(
                    tuple(atoms),
                    coordinates,
                    reference=reference,
                    factor=factor,
                )
            except BaseException:
                if self._survey_fingerprint() != before:
                    self._restore_tracking_state(tracking_state)
                    (
                        self.root,
                        self.gbw,
                        self.calc_counter,
                        self._staged_state_survey,
                    ) = parent_state
                    raise NonTransactionalSurveyError(
                        "survey_runner failed after modifying parent calculator or "
                        "tracking-session state. Use an isolated child calculator/workdir."
                    )
                raise
            if self._survey_fingerprint() != before:
                self._restore_tracking_state(tracking_state)
                (
                    self.root,
                    self.gbw,
                    self.calc_counter,
                    self._staged_state_survey,
                ) = parent_state
                raise NonTransactionalSurveyError(
                    "survey_runner modified parent calculator or tracking-session state. "
                    "Use an isolated child calculator/workdir."
                )

            data = _coerce_survey_data(raw_data)
            self._validate_candidate(data.candidate, coordinates)
            candidate_roots = set(int(root) for root in data.candidate.roots)
            unknown_scores = set(data.overlaps) - candidate_roots
            if unknown_scores:
                raise SurveyProtocolError(
                    f"Survey overlap scores contain unknown roots {sorted(unknown_scores)}."
                )

            try:
                survey = self.tracking_session.survey(
                    data.candidate,
                    data.overlaps,
                    reference_norm=data.reference_norm,
                    candidate_norms=data.candidate_norms,
                    step_scale=factor,
                    against=self.survey_against,
                    subspace_continuity=data.subspace_continuity,
                    root_overlap_block=data.root_overlap_block,
                )
                decision = self.tracking_session.select(survey.survey_id)
            except BaseException:
                # TrackingSession.survey validates before insertion, but select
                # errors occur after insertion and must be rolled back.
                try:
                    if "survey" in locals():
                        self.tracking_session.discard(survey.survey_id)
                except (KeyError, AttributeError):
                    pass
                raise

            result = TDenTrackSurvey(
                survey=survey,
                decision=decision,
                coordinates=coordinates,
                factor=factor,
                tracking_revision=self.tracking_revision,
                data=data,
                _owner=self._owner_token,
            )
            if not _decision_is_accepted(decision):
                self.tracking_session.discard(survey.survey_id)
            return result

    def stage_state_survey(
        self,
        result: TDenTrackSurvey,
        *,
        expected_cart_coords: Any,
    ) -> None:
        """Stage one accepted survey for the subsequent selected-root gradient."""

        with self._transaction_lock:
            if not isinstance(result, TDenTrackSurvey) or result._owner is not self._owner_token:
                raise StateStagingError("The survey result was not produced by this calculator.")
            if not _decision_is_accepted(result.decision):
                raise StateStagingError(
                    "Only an ACCEPT decision can be staged, got "
                    f"{_status_value(result.decision)!r}."
                )
            if result.tracking_revision != self.tracking_revision:
                raise StateStagingError("Cannot stage a survey from an earlier tracking revision.")
            decision_generation = int(getattr(result.decision, "generation"))
            if decision_generation != int(getattr(self.tracking_session, "generation")):
                raise StateStagingError("Cannot stage a survey from an earlier session generation.")
            selected_root = getattr(result.decision, "selected_root", None)
            if selected_root is None or int(selected_root) not in result.data.candidate.roots:
                raise StateStagingError("Accepted decision has no valid candidate root.")

            expected = _readonly_coordinates(
                expected_cart_coords, name="expected staged coordinates"
            )
            if not _coordinates_match(
                expected, result.coordinates, self.coordinate_tolerance
            ):
                raise StagedGeometryMismatch(
                    "Controller coordinates differ from the surveyed candidate geometry."
                )
            if not _coordinates_match(
                expected, result.data.candidate.coordinates, self.coordinate_tolerance
            ):
                raise StagedGeometryMismatch(
                    "Candidate snapshot coordinates differ from the geometry being staged."
                )

            # Re-evaluation proves that the accepted survey is still pending and
            # that its decision was not fabricated or made stale.
            current = self.tracking_session.select(result.survey.survey_id)
            if current != result.decision:
                raise StateStagingError("The staged decision no longer matches its pending survey.")
            self._staged_state_survey = _StagedSurvey(result, expected)

    def clear_staged_state_survey(self, *, discard_pending: bool = False) -> int:
        """Clear a staged endpoint, optionally discarding all pending probes.

        This is useful after an optimizer-level abort.  It never changes the
        committed electronic snapshot or tracking revision.
        """

        with self._transaction_lock:
            self._staged_state_survey = None
            if discard_pending:
                return int(self.tracking_session.discard())
            return 0

    def discard_staged_state_survey(self) -> int:
        """Optimizer hook: discard the stage and every uncommitted alternative."""

        return self.clear_staged_state_survey(discard_pending=True)

    @staticmethod
    def _validate_gradient_result(result: Any, coordinates: np.ndarray) -> Mapping[str, Any]:
        if not isinstance(result, Mapping):
            raise GradientProtocolError("Gradient runner must return a mapping.")
        if "energy" not in result or "forces" not in result:
            raise GradientProtocolError("Gradient result must contain 'energy' and 'forces'.")
        energy = float(result["energy"])
        forces = np.asarray(result["forces"], dtype=float).reshape(-1)
        if not math.isfinite(energy):
            raise GradientProtocolError("Gradient energy is not finite.")
        if forces.shape != coordinates.shape:
            raise GradientProtocolError(
                f"Gradient forces have shape {forces.shape}, expected {coordinates.shape}."
            )
        if not np.all(np.isfinite(forces)):
            raise GradientProtocolError("Gradient forces contain non-finite values.")
        return result

    def _normalize_native_gradient_energy(
        self,
        result: Mapping[str, Any],
        snapshot: Any,
        requested_root: int,
        *,
        tolerance_eh: float = 5.0e-5,
    ) -> Mapping[str, Any]:
        """Use the audited selected-state energy with ORCA's excited gradient.

        Some ORCA 6.1.1 TDA EnGrad paths write the corrected reference energy
        to ``.engrad``/``FINAL SINGLE POINT ENERGY`` even though the force is
        for the requested excited root. The all-root snapshot contains the
        corresponding corrected state energy. Accept an EnGrad scalar that
        matches either the selected state or ORCA's audited final-energy anchor,
        retain it on the calculator and in ORCA's files for provenance, and
        expose only the selected-state energy to the optimizer.
        """

        requested_root = int(requested_root)
        try:
            selected_energy = float(snapshot.energies_eh[requested_root])
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise GradientProtocolError(
                f"Snapshot has no finite energy for selected root {requested_root}."
            ) from exc
        raw_energy = float(result["energy"])
        if not math.isfinite(selected_energy) or not math.isfinite(raw_energy):
            raise GradientProtocolError(
                "Selected-state and ORCA EnGrad energies must be finite."
            )
        metadata = dict(getattr(snapshot, "metadata", {}))
        allowed = {"selected state": selected_energy}
        final_energy = metadata.get("final_single_point_energy_eh")
        if final_energy is not None:
            final_energy = float(final_energy)
            if not math.isfinite(final_energy):
                raise GradientProtocolError(
                    "Audited FINAL SINGLE POINT ENERGY must be finite."
                )
            allowed["FINAL SINGLE POINT ENERGY anchor"] = final_energy
        errors = {
            label: abs(raw_energy - value) for label, value in allowed.items()
        }
        if min(errors.values()) > float(tolerance_eh):
            raise GradientProtocolError(
                f"ORCA EnGrad energy {raw_energy:.12f} Eh matches neither the "
                "selected-state nor audited final-anchor energy; errors are "
                f"{errors} Eh."
            )
        normalized = dict(result)
        self.last_orca_engrad_energy = raw_energy
        normalized["energy"] = selected_energy
        return normalized

    def _validate_native_gradient_output(self, requested_root: int) -> None:
        """Certify the retained ORCA EnGrad output before state commitment.

        A finite ``.engrad`` file is insufficient evidence of electronic
        identity: a stale or conflicting IRoot directive can yield a perfectly
        valid gradient for the wrong state.  ORCA 6.1.1 echoes its effective
        input and reports the root/multiplicity associated with ``DE(CIS)``;
        both independent records must match the transaction.
        """

        try:
            output_path = Path(self.out)
        except (AttributeError, TypeError) as exc:
            raise GradientProtocolError(
                "The native ORCA gradient produced no retained output to audit."
            ) from exc
        if not output_path.is_file():
            raise GradientProtocolError(
                f"The retained ORCA gradient output does not exist: {output_path}."
            )
        text = output_path.read_text(errors="replace")
        if not self.check_termination(text):
            raise GradientProtocolError(
                f"The retained ORCA gradient output did not terminate normally: {output_path}."
            )

        echoed_lines = re.findall(
            r"^\s*\|\s*\d+>\s?(.*)$", text, flags=re.MULTILINE
        )
        if not echoed_lines:
            raise GradientProtocolError(
                "Could not locate ORCA's echoed input while auditing the gradient root."
            )
        echoed_input = "\n".join(echoed_lines)
        input_roots = {
            int(value)
            for value in re.findall(
                r"\biroot(?!mult)\s*(?:=\s*)?(\d+)",
                echoed_input,
                flags=re.IGNORECASE,
            )
        }
        if input_roots != {int(requested_root)}:
            raise GradientProtocolError(
                "ORCA's echoed gradient input does not contain exactly the "
                f"requested IRoot {requested_root}; found {sorted(input_roots)}."
            )

        input_multiplicities = {
            value.lower()
            for value in re.findall(
                r"\birootmult\s*(?:=\s*)?(singlet|triplet)\b",
                echoed_input,
                flags=re.IGNORECASE,
            )
        }
        expected_label = "triplet" if self.triplets else "singlet"
        if self.triplets and input_multiplicities != {"triplet"}:
            raise GradientProtocolError(
                "The echoed ORCA gradient input does not select IRootMult triplet; "
                f"found {sorted(input_multiplicities)}."
            )
        if not self.triplets and "triplet" in input_multiplicities:
            raise GradientProtocolError(
                "The echoed ORCA gradient input selects a triplet for a singlet transaction."
            )

        reported = re.findall(
            r"DE\([^)]*\)\s*=.*?\(\s*(?:(singlet|triplet)\s+)?root\s+(\d+)\s*\)",
            text,
            flags=re.IGNORECASE,
        )
        if len(reported) != 1:
            raise GradientProtocolError(
                "Expected exactly one ORCA DE(...) root marker for the main EnGrad, "
                f"found {len(reported)}."
            )
        label, root_text = reported[0]
        reported_root = int(root_text)
        # Some ORCA builds omit the spin label from ``DE(CIS)`` even though
        # the echoed IRootMult directive is unambiguous. Only override that
        # independently certified multiplicity when ORCA prints a label.
        reported_label = label.lower() if label else expected_label
        if reported_root != int(requested_root) or reported_label != expected_label:
            raise GradientProtocolError(
                "ORCA reported a gradient for "
                f"{reported_label} root {reported_root}, but the transaction requested "
                f"{expected_label} root {requested_root}."
            )

        state_roots = {
            int(value)
            for value in re.findall(
                r"State of interest\s*\.{3}\s*(\d+)", text, flags=re.IGNORECASE
            )
        }
        if state_roots != {int(requested_root)}:
            raise GradientProtocolError(
                "ORCA's state-of-interest report is inconsistent with the "
                f"requested root {requested_root}; found {sorted(state_roots)}."
            )

    def _run_selected_gradient(
        self,
        atoms: Sequence[str],
        coordinates: np.ndarray,
        root: int,
        prepare_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        if self.gradient_runner is not None:
            return self.gradient_runner(
                tuple(atoms),
                coordinates,
                root=int(root),
                calculator=self,
                prepare_kwargs=dict(prepare_kwargs),
            )
        # Calling super avoids recursion into this transactional override.
        return super().get_forces(atoms, coordinates, **dict(prepare_kwargs))

    def _gradient_at_committed_geometry(
        self,
        atoms: Sequence[str],
        coordinates: np.ndarray,
        prepare_kwargs: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        committed = self._committed_snapshot()
        if not _coordinates_match(
            coordinates, committed.coordinates, self.coordinate_tolerance
        ):
            raise UnstagedGeometryError(
                "A gradient at a new geometry requires an accepted staged state survey."
            )
        root = self._selected_root(committed)
        previous_root = self.root
        self.root = root
        try:
            result = self._run_selected_gradient(atoms, coordinates, root, prepare_kwargs)
            if self.gradient_runner is None:
                self._validate_native_gradient_output(root)
                result = self._normalize_native_gradient_energy(
                    result, committed, root
                )
            return self._validate_gradient_result(result, coordinates)
        except BaseException:
            self.root = previous_root
            raise

    def get_forces(
        self, atoms: Sequence[str], coords: Any, **prepare_kwargs: Any
    ) -> Mapping[str, Any]:
        """Calculate a gradient and atomically commit its staged state.

        With no staged survey, only the already committed geometry is allowed;
        this supports the first optimizer gradient and harmless reevaluations.
        At a new geometry, the selected root is installed only for the duration
        of the gradient call.  Any exception or malformed result restores the
        previous root and leaves the tracking session untouched.
        """

        coordinates = _readonly_coordinates(coords, name="gradient coordinates")
        with self._transaction_lock:
            staged = self._staged_state_survey
            if staged is None:
                return self._gradient_at_committed_geometry(
                    atoms, coordinates, prepare_kwargs
                )

            if not _coordinates_match(
                coordinates, staged.expected_coordinates, self.coordinate_tolerance
            ):
                raise StagedGeometryMismatch(
                    "Gradient coordinates do not match the staged state-survey geometry."
                )
            result = staged.result
            if result.tracking_revision != self.tracking_revision:
                raise StateStagingError("Staged survey became stale before its gradient.")
            root = int(result.decision.selected_root)
            previous_root = self.root
            previous_gbw = getattr(self, "gbw", None)
            survey_gbw = getattr(result.data.candidate, "artifacts", {}).get("gbw")
            if survey_gbw is not None:
                survey_gbw = Path(survey_gbw)
                if not survey_gbw.exists():
                    raise StateStagingError(
                        f"Staged survey GBW does not exist: {survey_gbw}"
                    )
                self.gbw = survey_gbw
            self.root = root
            try:
                gradient = self._run_selected_gradient(
                    atoms, coordinates, root, prepare_kwargs
                )
                if self.gradient_runner is None:
                    self._validate_native_gradient_output(root)
                    gradient = self._normalize_native_gradient_energy(
                        gradient, result.data.candidate, root
                    )
                gradient = self._validate_gradient_result(gradient, coordinates)
                committed = self.tracking_session.commit(
                    result.decision,
                    update_anchor=self.update_anchor_on_commit,
                )
            except BaseException:
                self.root = previous_root
                self.gbw = previous_gbw
                raise

            self.root = self._selected_root(committed)
            # A native ORCA run normally points self.gbw at its new kept GBW.
            # An injected runner may instead return it explicitly; otherwise a
            # geometry-matched survey GBW remains preferable to the old GBW.
            returned_gbw = gradient.get("gbw")
            if returned_gbw is not None:
                self.gbw = Path(returned_gbw)
            elif survey_gbw is None and getattr(self, "gbw", None) == previous_gbw:
                self.gbw = previous_gbw
            self._staged_state_survey = None
            return gradient

    @staticmethod
    def _snapshot_to_restart(snapshot: Any) -> dict[str, Any]:
        return {
            "label": str(snapshot.label),
            "coordinates": np.asarray(snapshot.coordinates, dtype=float).tolist(),
            "roots": [int(root) for root in snapshot.roots],
            "selected_root": int(snapshot.selected_root),
            "requested_roots": [int(root) for root in getattr(snapshot, "requested_roots", ())],
            "energies_eh": {
                int(k): float(v)
                for k, v in getattr(snapshot, "energies_eh", {}).items()
            },
            "excitation_energies_ev": {
                int(k): float(v)
                for k, v in getattr(snapshot, "excitation_energies_ev", {}).items()
            },
            "multiplicities": {
                int(k): int(v) for k, v in getattr(snapshot, "multiplicities", {}).items()
            },
            "spin_squared": {
                int(k): float(v) for k, v in getattr(snapshot, "spin_squared", {}).items()
            },
            "artifacts": {
                str(k): str(v) for k, v in getattr(snapshot, "artifacts", {}).items()
            },
            "metadata": dict(getattr(snapshot, "metadata", {})),
        }

    def _snapshot_from_restart(self, state: Mapping[str, Any]) -> Any:
        snapshot_class = self._committed_snapshot().__class__
        try:
            return snapshot_class(
                label=state["label"],
                coordinates=np.asarray(state["coordinates"], dtype=float),
                roots=tuple(state["roots"]),
                selected_root=state["selected_root"],
                requested_roots=tuple(state.get("requested_roots", ())),
                energies_eh=state.get("energies_eh", {}),
                excitation_energies_ev=state.get("excitation_energies_ev", {}),
                multiplicities=state.get("multiplicities", {}),
                spin_squared=state.get("spin_squared", {}),
                artifacts={k: Path(v) for k, v in state.get("artifacts", {}).items()},
                metadata=state.get("metadata", {}),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise TDenTrackAdapterError(
                "Could not reconstruct the committed ElectronicSnapshot from restart data."
            ) from exc

    def get_restart_info(self) -> dict[str, Any]:
        """Extend calculator restart data with the committed tracking state.

        Pending probes and a staged-but-uncommitted endpoint are deliberately
        omitted.  On restart the optimizer must survey its next proposed
        geometry again.  Adaptive-anchor history cannot yet be restored through
        TrackingSession's public API, so the committed snapshot becomes the new
        anchor while the external monotonic revision is preserved.
        """

        with self._transaction_lock:
            info = super().get_restart_info()
            info[self.RESTART_KEY] = {
                "version": self.RESTART_VERSION,
                "tracking_revision": self.tracking_revision,
                "committed": self._snapshot_to_restart(self._committed_snapshot()),
            }
            return info

    def set_restart_info(self, restart_info: Mapping[str, Any]) -> None:
        """Restore the committed reference and discard all uncommitted probes."""

        with self._transaction_lock:
            base_info = dict(restart_info)
            try:
                state = base_info.pop(self.RESTART_KEY)
            except KeyError as exc:
                raise TDenTrackAdapterError(
                    f"Restart data contain no {self.RESTART_KEY!r} section."
                ) from exc
            if int(state.get("version", -1)) != self.RESTART_VERSION:
                raise TDenTrackAdapterError(
                    f"Unsupported TDenTrackORCA restart version {state.get('version')!r}."
                )
            # Calculator.set_restart_info mutates its input by popping chkfiles;
            # pass our private copy rather than the caller's mapping.
            super().set_restart_info(base_info)
            snapshot = self._snapshot_from_restart(state["committed"])
            session_class = self.tracking_session.__class__
            try:
                session = session_class(snapshot, selector=self._tracking_selector)
            except TypeError:
                # A duck-typed session used by a downstream application may not
                # expose tdentrack's optional selector keyword.
                session = session_class(snapshot)
            self.tracking_session = session
            self._tracking_revision_offset = int(state["tracking_revision"])
            self._staged_state_survey = None
            self.root = self._selected_root(snapshot)


__all__ = [
    "GradientProtocolError",
    "NonTransactionalSurveyError",
    "StagedGeometryMismatch",
    "StateStagingError",
    "SurveyData",
    "SurveyProtocolError",
    "TDenTrackAdapterError",
    "TDenTrackDependencyError",
    "TDenTrackORCA",
    "TDenTrackSurvey",
    "UnstagedGeometryError",
]
