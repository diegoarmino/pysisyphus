from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
import re

import numpy as np
import pytest

from pysisyphus.calculators.ORCA import ORCA
from pysisyphus.calculators.TDenTrackORCA import (
    GradientProtocolError,
    NonTransactionalSurveyError,
    StagedGeometryMismatch,
    StateStagingError,
    SurveyData,
    SurveyProtocolError,
    TDenTrackORCA,
    UnstagedGeometryError,
)


def test_orca_rewrites_equals_iroot_without_duplicate(tmp_path):
    calc = ORCA(
        keywords="hf sto-3g",
        blocks="%tddft nroots=3 iroot=1 triplets=true end",
        root=1,
        check_mem=False,
        wavefunction_dump=False,
        out_dir=tmp_path,
    )
    calc.root = 2

    block = calc.get_block_str()

    assert len(
        re.findall(
            r"\biroot(?!mult)\s*(?:=\s*)?\d+", block, flags=re.IGNORECASE
        )
    ) == 1
    assert re.search(r"\biroot\s+2\b", block, re.IGNORECASE)
    assert re.search(
        r"\birootmult\s+triplet\b", block, re.IGNORECASE
    )


def test_orca_rejects_multiple_iroot_directives(tmp_path):
    with pytest.raises(ValueError, match="multiple IRoot"):
        ORCA(
            keywords="hf sto-3g",
            blocks="%tddft nroots 3 iroot 1 iroot=2 end",
            check_mem=False,
            wavefunction_dump=False,
            out_dir=tmp_path,
        )


def _native_gradient_output(root=2, label="Triplet", input_root=None):
    input_root = root if input_root is None else input_root
    return f"""
INPUT FILE
|  1> %tddft
|  2>   nroots 3
|  3>   iroot={input_root}
|  4>   irootmult triplet
|  5> end
    DE(CIS) = 0.123456 Eh ({label} root {root})
TD-DFT GRADIENT
State of interest                  ... {root}
                 EXCITED STATE GRADIENT DONE
                             ****ORCA TERMINATED NORMALLY****
"""


def test_native_gradient_output_certifies_root_and_multiplicity(
    tmp_path, initial_snapshot
):
    calc = make_calc(
        tmp_path,
        initial_snapshot,
        lambda *args, **kwargs: None,
        successful_gradient([]),
    )
    output = tmp_path / "gradient.out"
    output.write_text(_native_gradient_output())
    calc.out = output

    calc._validate_native_gradient_output(2)
    output.write_text(_native_gradient_output(label=""))
    calc._validate_native_gradient_output(2)


@pytest.mark.parametrize(
    "text, message",
    (
        (_native_gradient_output(root=1), "requested IRoot 2"),
        (_native_gradient_output(root=2, input_root=1), "requested IRoot 2"),
        (_native_gradient_output(root=2, label="Singlet"), "singlet root 2"),
    ),
)
def test_native_gradient_output_rejects_wrong_state(
    tmp_path, initial_snapshot, text, message
):
    calc = make_calc(
        tmp_path,
        initial_snapshot,
        lambda *args, **kwargs: None,
        successful_gradient([]),
    )
    output = tmp_path / "gradient.out"
    output.write_text(text)
    calc.out = output

    with pytest.raises(GradientProtocolError, match=message):
        calc._validate_native_gradient_output(2)


class Status(str, Enum):
    ACCEPT = "ACCEPT"
    RETRY = "RETRY"


@dataclass(frozen=True)
class Snapshot:
    label: str
    coordinates: np.ndarray
    roots: tuple[int, ...]
    selected_root: int | None = None
    requested_roots: tuple[int, ...] = ()
    energies_eh: dict[int, float] = field(default_factory=dict)
    excitation_energies_ev: dict[int, float] = field(default_factory=dict)
    multiplicities: dict[int, int] = field(default_factory=dict)
    spin_squared: dict[int, float] = field(default_factory=dict)
    artifacts: dict[str, Path] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def with_selected_root(self, root: int):
        return replace(self, selected_root=int(root))


@dataclass(frozen=True)
class PendingSurvey:
    survey_id: str
    generation: int
    candidate: Snapshot
    overlaps: dict[int, float]
    step_scale: float
    subspace_continuity: object = None
    root_overlap_block: object = None


@dataclass(frozen=True)
class Decision:
    status: Status
    survey_id: str
    generation: int
    selected_root: int | None
    selected_energy_eh: float | None
    best_score: float
    margin: float
    reason: str = ""

    @property
    def accepted(self):
        return self.status is Status.ACCEPT


class Session:
    """Small API-compatible stand-in; no tdentrack/ORCA installation needed."""

    def __init__(self, initial: Snapshot, selector=None):
        self.committed = initial
        self.generation = 0
        self._selector = selector
        self._pending = {}
        self._counter = 0

    @property
    def pending(self):
        return tuple(self._pending.values())

    def survey(
        self,
        candidate,
        overlaps,
        *,
        reference_norm,
        candidate_norms,
        step_scale,
        against,
        subspace_continuity=None,
        root_overlap_block=None,
    ):
        assert against == "committed"
        self._counter += 1
        survey = PendingSurvey(
            f"survey-{self._counter}",
            self.generation,
            candidate,
            dict(overlaps),
            step_scale,
            subspace_continuity,
            root_overlap_block,
        )
        self._pending[survey.survey_id] = survey
        return survey

    def select(self, survey_id):
        survey = self._pending[survey_id]
        root, score = max(survey.overlaps.items(), key=lambda item: item[1])
        ordered = sorted(survey.overlaps.values(), reverse=True)
        second = ordered[1] if len(ordered) > 1 else 0.0
        status = Status(survey.candidate.metadata.get("status", "ACCEPT"))
        energy = survey.candidate.energies_eh.get(root)
        return Decision(
            status,
            survey_id,
            survey.generation,
            root if status is Status.ACCEPT else None,
            energy,
            score,
            score - second,
            "mock selector",
        )

    def commit(self, decision, *, update_anchor):
        assert update_anchor
        assert decision.accepted
        survey = self._pending[decision.survey_id]
        self.committed = survey.candidate.with_selected_root(decision.selected_root)
        self._pending.clear()
        self.generation += 1
        return self.committed

    def discard(self, survey_id=None):
        if survey_id is None:
            count = len(self._pending)
            self._pending.clear()
            return count
        return self._pending.pop(survey_id)


@pytest.fixture
def initial_coords():
    return np.array([0.0, 0.0, 0.0, 1.4, 0.0, 0.0])


@pytest.fixture
def initial_snapshot(initial_coords):
    return Snapshot(
        "initial",
        initial_coords,
        (1, 2, 3),
        selected_root=1,
        requested_roots=(1, 2, 3),
        energies_eh={1: -10.0, 2: -9.9, 3: -9.8},
        multiplicities={1: 3, 2: 3, 3: 3},
    )


def candidate(coords, *, status="ACCEPT", with_requested_roots=True):
    return Snapshot(
        "trial",
        np.array(coords, copy=True),
        (1, 2, 3),
        requested_roots=(1, 2, 3) if with_requested_roots else (),
        energies_eh={1: -9.8, 2: -10.1, 3: -9.7},
        multiplicities={1: 3, 2: 3, 3: 3},
        metadata={"status": status},
    )


def make_calc(tmp_path, initial_snapshot, survey_runner, gradient_runner):
    return TDenTrackORCA(
        keywords="uhf b3lyp def2-svp tightscf",
        blocks="%tddft nroots 3 triplets true end",
        root=1,
        tracking_session=Session(initial_snapshot),
        survey_runner=survey_runner,
        gradient_runner=gradient_runner,
        wavefunction_dump=False,
        check_mem=False,
        out_dir=tmp_path,
    )


def test_transactional_calculator_requires_explicit_survey_backend(
    tmp_path, initial_snapshot
):
    with pytest.raises(TypeError, match="requires a callable isolated survey_runner"):
        TDenTrackORCA(
            keywords="uhf b3lyp def2-svp",
            blocks="%tddft nroots 3 triplets true end",
            root=1,
            tracking_session=Session(initial_snapshot),
            wavefunction_dump=False,
            check_mem=False,
            out_dir=tmp_path,
        )


def test_transactional_calculator_rejects_legacy_tracking(
    tmp_path, initial_snapshot
):
    with pytest.raises(ValueError, match="Legacy ORCA/OverlapCalculator"):
        TDenTrackORCA(
            keywords="uhf b3lyp def2-svp",
            blocks="%tddft nroots 3 triplets true end",
            root=1,
            track=True,
            tracking_session=Session(initial_snapshot),
            survey_runner=lambda *args, **kwargs: None,
            wavefunction_dump=False,
            check_mem=False,
            out_dir=tmp_path,
        )


def successful_gradient(calls):
    def run(atoms, coords, *, root, calculator, prepare_kwargs):
        calls.append((atoms, np.array(coords), root, dict(prepare_kwargs)))
        return {"energy": -10.1, "forces": np.full_like(coords, 0.01)}

    return run


def test_survey_stage_gradient_then_commit_is_transactional(
    tmp_path, initial_snapshot, initial_coords
):
    trial_coords = initial_coords + 0.1
    references = []

    def survey(atoms, coords, *, reference, factor):
        references.append((reference, factor, np.array(coords)))
        return SurveyData(candidate(coords), {1: 0.2, 2: 0.91, 3: 0.1})

    gradient_calls = []
    calc = make_calc(
        tmp_path, initial_snapshot, survey, successful_gradient(gradient_calls)
    )
    revision = calc.tracking_revision

    result = calc.survey_state(("H", "H"), trial_coords, factor=1.25)

    assert result.decision.accepted
    assert result.decision.selected_root == 2
    assert calc.tracking_revision == revision
    assert calc.tracking_session.committed is initial_snapshot
    assert calc.root == 1
    assert len(calc.tracking_session.pending) == 1
    assert references[0][0] is initial_snapshot

    calc.stage_state_survey(result, expected_cart_coords=trial_coords)
    assert calc.has_staged_state_survey
    with pytest.raises(StagedGeometryMismatch):
        calc.get_forces(("H", "H"), trial_coords + 1.0e-4)
    assert not gradient_calls
    assert calc.tracking_revision == revision

    forces = calc.get_forces(("H", "H"), trial_coords)

    assert forces["energy"] == pytest.approx(-10.1)
    assert gradient_calls[0][2] == 2
    assert calc.root == 2
    assert calc.tracking_session.committed.selected_root == 2
    np.testing.assert_allclose(calc.tracking_session.committed.coordinates, trial_coords)
    assert calc.tracking_revision == revision + 1
    assert not calc.has_staged_state_survey
    assert calc.tracking_session.pending == ()


def test_failed_gradient_restores_root_without_commit(
    tmp_path, initial_snapshot, initial_coords
):
    trial_coords = initial_coords + 0.2

    def survey(atoms, coords, *, reference, factor):
        return SurveyData(candidate(coords), {1: 0.1, 2: 0.9, 3: 0.2})

    def fail(atoms, coords, *, root, calculator, prepare_kwargs):
        assert calculator.root == 2
        raise RuntimeError("mock EnGrad failure")

    calc = make_calc(tmp_path, initial_snapshot, survey, fail)
    result = calc.survey_state(("H", "H"), trial_coords, factor=1.0)
    calc.stage_state_survey(result, expected_cart_coords=trial_coords)

    with pytest.raises(RuntimeError, match="mock EnGrad failure"):
        calc.get_forces(("H", "H"), trial_coords)

    assert calc.root == 1
    assert calc.tracking_revision == 0
    assert calc.tracking_session.committed is initial_snapshot
    assert calc.has_staged_state_survey


def test_invalid_gradient_result_does_not_commit(
    tmp_path, initial_snapshot, initial_coords
):
    trial_coords = initial_coords + 0.2

    def survey(atoms, coords, *, reference, factor):
        return SurveyData(candidate(coords), {1: 0.1, 2: 0.9, 3: 0.2})

    def invalid(atoms, coords, *, root, calculator, prepare_kwargs):
        return {"energy": -10.1, "forces": np.zeros(3)}

    calc = make_calc(tmp_path, initial_snapshot, survey, invalid)
    result = calc.survey_state(("H", "H"), trial_coords)
    calc.stage_state_survey(result, expected_cart_coords=trial_coords)

    with pytest.raises(GradientProtocolError, match="forces have shape"):
        calc.get_forces(("H", "H"), trial_coords)
    assert calc.tracking_revision == 0
    assert calc.root == 1


def test_rejected_survey_is_discarded_and_cannot_be_staged(
    tmp_path, initial_snapshot, initial_coords
):
    def survey(atoms, coords, *, reference, factor):
        return SurveyData(
            candidate(coords, status="RETRY"), {1: 0.4, 2: 0.41, 3: 0.1}
        )

    calc = make_calc(tmp_path, initial_snapshot, survey, successful_gradient([]))
    result = calc.survey_state(("H", "H"), initial_coords + 0.1)

    assert result.decision.status is Status.RETRY
    assert calc.tracking_session.pending == ()
    assert calc.tracking_revision == 0
    with pytest.raises(StateStagingError, match="Only an ACCEPT"):
        calc.stage_state_survey(
            result, expected_cart_coords=initial_coords + 0.1
        )


def test_survey_protocol_errors_leave_session_untouched(
    tmp_path, initial_snapshot, initial_coords
):
    wrong_coords = initial_coords + 0.3

    def survey(atoms, coords, *, reference, factor):
        return SurveyData(
            candidate(wrong_coords, with_requested_roots=False),
            {1: 0.8, 2: 0.1, 3: 0.1},
        )

    calc = make_calc(tmp_path, initial_snapshot, survey, successful_gradient([]))
    with pytest.raises(SurveyProtocolError, match="coordinates do not match"):
        calc.survey_state(("H", "H"), initial_coords + 0.1)
    assert calc.tracking_session.pending == ()
    assert calc.tracking_session.committed is initial_snapshot
    assert calc.tracking_revision == 0


def test_survey_forwards_metric_aware_root_overlap_block(
    tmp_path, initial_snapshot, initial_coords
):
    root_overlap_block = object()

    def survey(atoms, coords, *, reference, factor):
        return SurveyData(
            candidate(coords),
            {1: 0.1, 2: 0.9, 3: 0.2},
            root_overlap_block=root_overlap_block,
        )

    calc = make_calc(tmp_path, initial_snapshot, survey, successful_gradient([]))
    result = calc.survey_state(("H", "H"), initial_coords + 0.1)

    assert result.survey.root_overlap_block is root_overlap_block
    assert result.data.root_overlap_block is root_overlap_block


def test_survey_requires_declared_all_root_window(
    tmp_path, initial_snapshot, initial_coords
):
    trial_coords = initial_coords + 0.1

    def survey(atoms, coords, *, reference, factor):
        return SurveyData(
            candidate(coords, with_requested_roots=False),
            {1: 0.8, 2: 0.1, 3: 0.1},
        )

    calc = make_calc(tmp_path, initial_snapshot, survey, successful_gradient([]))
    with pytest.raises(SurveyProtocolError, match="requested_roots"):
        calc.survey_state(("H", "H"), trial_coords)
    assert calc.tracking_session.pending == ()
    assert calc.tracking_revision == 0


def test_nonisolated_survey_is_detected_and_parent_root_restored(
    tmp_path, initial_snapshot, initial_coords
):
    holder = {}

    def survey(atoms, coords, *, reference, factor):
        holder["calc"].root = 3
        return SurveyData(candidate(coords), {1: 0.8, 2: 0.1, 3: 0.1})

    calc = make_calc(tmp_path, initial_snapshot, survey, successful_gradient([]))
    holder["calc"] = calc
    with pytest.raises(NonTransactionalSurveyError, match="modified parent"):
        calc.survey_state(("H", "H"), initial_coords + 0.1)
    assert calc.root == 1
    assert calc.tracking_revision == 0
    assert calc.tracking_session.pending == ()


def test_nonisolated_survey_restores_tracking_session_mutations(
    tmp_path, initial_snapshot, initial_coords
):
    holder = {}

    def survey(atoms, coords, *, reference, factor):
        session = holder["calc"].tracking_session
        session.committed = candidate(coords).with_selected_root(3)
        session.generation = 9
        session._pending["foreign"] = object()
        return SurveyData(candidate(coords), {1: 0.8, 2: 0.1, 3: 0.1})

    calc = make_calc(tmp_path, initial_snapshot, survey, successful_gradient([]))
    holder["calc"] = calc

    with pytest.raises(NonTransactionalSurveyError, match="modified parent"):
        calc.survey_state(("H", "H"), initial_coords + 0.1)

    assert calc.tracking_session.committed is initial_snapshot
    assert calc.tracking_session.generation == 0
    assert calc.tracking_session.pending == ()


def test_committed_gradient_allowed_but_new_unstaged_geometry_rejected(
    tmp_path, initial_snapshot, initial_coords
):
    def survey(atoms, coords, *, reference, factor):
        return SurveyData(candidate(coords), {1: 0.9, 2: 0.1, 3: 0.0})

    calls = []
    calc = make_calc(tmp_path, initial_snapshot, survey, successful_gradient(calls))
    calc.get_forces(("H", "H"), initial_coords)
    assert calls[0][2] == 1
    assert calc.tracking_revision == 0

    with pytest.raises(UnstagedGeometryError):
        calc.get_forces(("H", "H"), initial_coords + 0.1)
    assert len(calls) == 1


def test_restart_restores_only_committed_snapshot_and_revision(
    tmp_path, initial_snapshot, initial_coords
):
    trial_coords = initial_coords + 0.1

    def survey(atoms, coords, *, reference, factor):
        return SurveyData(candidate(coords), {1: 0.2, 2: 0.9, 3: 0.1})

    calc = make_calc(tmp_path, initial_snapshot, survey, successful_gradient([]))
    result = calc.survey_state(("H", "H"), trial_coords)
    calc.stage_state_survey(result, expected_cart_coords=trial_coords)
    calc.get_forces(("H", "H"), trial_coords)
    restart_info = calc.get_restart_info()

    fresh = make_calc(
        tmp_path / "fresh", initial_snapshot, survey, successful_gradient([])
    )
    fresh.set_restart_info(restart_info)

    assert fresh.tracking_revision == 1
    assert fresh.root == 2
    assert fresh.tracking_session.committed.selected_root == 2
    np.testing.assert_allclose(
        fresh.tracking_session.committed.coordinates, trial_coords
    )
    assert fresh.tracking_session.pending == ()
    assert not fresh.has_staged_state_survey


def test_optimizer_discard_hook_clears_stage_and_all_pending_alternatives(
    tmp_path, initial_snapshot, initial_coords
):
    def survey(atoms, coords, *, reference, factor):
        return SurveyData(candidate(coords), {1: 0.2, 2: 0.9, 3: 0.1})

    calc = make_calc(tmp_path, initial_snapshot, survey, successful_gradient([]))
    first = calc.survey_state(("H", "H"), initial_coords + 0.1, factor=0.5)
    calc.survey_state(("H", "H"), initial_coords + 0.2, factor=1.0)
    calc.stage_state_survey(first, expected_cart_coords=initial_coords + 0.1)

    assert calc.discard_staged_state_survey() == 2
    assert not calc.has_staged_state_survey
    assert calc.tracking_session.pending == ()
    assert calc.tracking_revision == 0
    assert calc.tracking_session.committed is initial_snapshot
