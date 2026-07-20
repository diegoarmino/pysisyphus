from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest

import pysisyphus.calculators.ORCA6StateSurvey as survey_module
from pysisyphus.calculators.ORCA6StateSurvey import (
    CISActiveSpace,
    ChildSurveyResult,
    LoadedORCA6State,
    ORCA6AllRootSurveyBackend,
    ORCA6ArtifactLoader,
    ORCA6SurveyError,
    bootstrap_tdentrack_snapshot,
    exact_cross_overlap,
    signed_tden_overlap_matrix,
    state_self_norms,
)


class SyntheticWavefunction:
    def __init__(self, atoms, coords, cross_key, cross_matrices):
        self.atoms = tuple(atoms)
        self.coords = np.array(coords, dtype=float)
        self.C = np.stack((np.eye(4), np.eye(4)))
        self.S = np.eye(4)
        self.bf_type = "spherical"
        self.has_shells = True
        self.shells = SimpleNamespace(ordering="orca", shells=())
        self.ecp_electrons = np.zeros(len(atoms), dtype=int)
        self.cross_key = cross_key
        self.cross_matrices = cross_matrices

    def S_with(self, other):
        return self.cross_matrices[(self.cross_key, other.cross_key)]


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


def make_loaded_pair():
    ref_coords = np.array([0.0, 0.0, 0.0, 1.4, 0.0, 0.0])
    cur_coords = ref_coords + 0.1
    cross = {
        ("ref", "ref"): np.eye(4),
        ("cur", "cur"): np.eye(4),
        ("ref", "cur"): np.eye(4),
        ("cur", "ref"): np.eye(4),
    }
    wf_ref = SyntheticWavefunction(("H", "H"), ref_coords, "ref", cross)
    wf_cur = SyntheticWavefunction(("H", "H"), cur_coords, "cur", cross)
    active = CISActiveSpace((0, 1), (2, 3), (0, 1), (2, 3))
    transitions = np.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ]
    )
    loaded_ref = LoadedORCA6State(
        wf_ref, (1, 2), active, transitions, transitions
    )
    loaded_cur = LoadedORCA6State(
        wf_cur, (1, 2), active, transitions, transitions
    )
    return loaded_ref, loaded_cur, ref_coords, cur_coords


def touch_artifacts(directory, prefix):
    artifacts = {}
    for kind, suffix in (
        ("cis", ".cis"),
        ("bson", ".bson"),
        ("gbw", ".gbw"),
        ("output", ".out"),
    ):
        path = directory / f"{prefix}{suffix}"
        path.write_bytes(b"fixture")
        artifacts[kind] = path
    return artifacts


def test_default_backend_builds_signed_full_matrix_and_audit(tmp_path):
    loaded_ref, loaded_cur, ref_coords, cur_coords = make_loaded_pair()
    ref_artifacts = touch_artifacts(tmp_path, "reference")
    reference = Snapshot(
        "reference",
        ref_coords,
        (1, 2),
        selected_root=1,
        requested_roots=(1, 2),
        energies_eh={1: -10.0, 2: -9.9},
        excitation_energies_ev={1: 1.0, 2: 2.0},
        multiplicities={1: 3, 2: 3},
        artifacts=ref_artifacts,
    )
    parent = SimpleNamespace(
        triplets=True,
        out_dir=tmp_path,
        do_tddft=True,
        do_ice=False,
        nroots=2,
        tracking_revision=0,
    )

    def loader(snapshot, *, roots, atoms, coordinates):
        return loaded_ref if snapshot.label == "reference" else loaded_cur

    def child_runner(
        parent,
        atoms,
        coordinates,
        *,
        reference,
        requested_roots,
        audit_dir,
        label,
    ):
        artifacts = touch_artifacts(audit_dir, "candidate")
        return ChildSurveyResult(
            label,
            atoms,
            coordinates,
            requested_roots,
            energies_eh={1: -10.1, 2: -9.8},
            excitation_energies_ev={1: 1.1, 2: 2.2},
            multiplicities={1: 3, 2: 3},
            artifacts=artifacts,
            orca_version="6.1.1",
        )

    block_data = {}

    def make_block(**kwargs):
        block_data.update(kwargs)
        return "synthetic-root-overlap-block"

    backend = ORCA6AllRootSurveyBackend(
        parent,
        artifact_loader=loader,
        child_runner=child_runner,
        audit_root=tmp_path / "audit",
        root_overlap_block_factory=make_block,
    )
    result = backend(
        ("H", "H"), cur_coords, reference=reference, factor=1.25
    )

    np.testing.assert_allclose(result["signed_overlap_matrix"], 2.0 * np.eye(2))
    assert result["overlaps"] == {1: 2.0, 2: 0.0}
    assert result["reference_norm"] == pytest.approx(2.0)
    assert result["candidate_norms"] == {1: 2.0, 2: 2.0}
    np.testing.assert_allclose(result["reference_gram"], 2.0 * np.eye(2))
    np.testing.assert_allclose(result["candidate_gram"], 2.0 * np.eye(2))
    assert result["root_overlap_block"] == "synthetic-root-overlap-block"
    np.testing.assert_allclose(block_data["overlaps"], 2.0 * np.eye(2))
    np.testing.assert_allclose(block_data["reference_gram"], 2.0 * np.eye(2))
    np.testing.assert_allclose(block_data["candidate_gram"], 2.0 * np.eye(2))
    assert result["candidate"].requested_roots == (1, 2)
    assert result["candidate"].selected_root is None
    manifest = result["candidate"].artifacts["audit_manifest"]
    assert manifest.is_file()
    assert "signed_overlap_matrix" in manifest.read_text()


def test_signed_contraction_and_self_norms_are_not_absolute():
    loaded_ref, loaded_cur, _, _ = make_loaded_pair()
    candidate_alpha = -np.array(loaded_cur.transition_alpha)
    candidate_beta = -np.array(loaded_cur.transition_beta)
    reversed_state = LoadedORCA6State(
        loaded_cur.wavefunction,
        loaded_cur.roots,
        loaded_cur.active_space,
        candidate_alpha,
        candidate_beta,
    )
    matrix = signed_tden_overlap_matrix(
        loaded_ref, reversed_state, np.eye(4)
    )
    np.testing.assert_allclose(matrix, -2.0 * np.eye(2))
    np.testing.assert_allclose(state_self_norms(loaded_ref), (2.0, 2.0))


def test_exact_cross_overlap_rejects_transpose_mismatch():
    loaded_ref, loaded_cur, _, _ = make_loaded_pair()
    loaded_ref.wavefunction.cross_matrices[("cur", "ref")] = 0.5 * np.eye(4)
    with pytest.raises(ORCA6SurveyError, match="transpose error"):
        exact_cross_overlap(loaded_ref, loaded_cur, atol=1.0e-8)


def test_artifact_loader_uses_bson_coefficients_and_cis_ranges(
    tmp_path, monkeypatch
):
    loaded_ref, _, coords, _ = make_loaded_pair()
    artifacts = touch_artifacts(tmp_path, "state")
    snapshot = Snapshot(
        "state",
        coords,
        (1, 2),
        selected_root=1,
        requested_roots=(1, 2),
        artifacts=artifacts,
    )
    X = np.array(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 1.0]],
        ]
    )
    monkeypatch.setattr(
        survey_module.Wavefunction,
        "from_file",
        lambda path: loaded_ref.wavefunction,
    )
    header = SimpleNamespace(
        layout="orca-standard-vector-records",
        alpha_occ_start=0,
        alpha_occ_end=1,
        alpha_virt_start=2,
        alpha_virt_end=3,
        beta_occ_start=0,
        beta_occ_end=1,
        beta_virt_start=2,
        beta_virt_end=3,
        alpha_nocc=2,
        alpha_nvirt=2,
        beta_nocc=2,
        beta_nvirt=2,
    )
    parser_calls = []

    def cis_parser(path, **kwargs):
        parser_calls.append(kwargs)
        states = {
            global_root: {
                "global_root": global_root,
                "multiplicity": 3,
                "orca_gradient_iroot": local_root,
                "response_component": "x-plus-y",
                "alpha": X[local_root - 1],
                "beta": X[local_root - 1],
            }
            for global_root, local_root in ((3, 1), (4, 2))
        }
        return header, states

    state = ORCA6ArtifactLoader(
        triplets=True, tda=False, cis_parser=cis_parser
    )(
        snapshot, roots=(1, 2), atoms=("H", "H"), coordinates=coords
    )

    assert state.active_space.alpha_occ == (0, 1)
    assert state.active_space.beta_virt == (2, 3)
    assert state.transition_alpha.shape == (2, 2, 2)
    assert np.all(np.isfinite(state.transition_alpha))
    assert state.root_metadata[1]["global_root"] == 3
    assert state.root_metadata[2]["global_root"] == 4
    assert parser_calls == [{"multiplicity": 3, "tda": False}]


def test_artifact_loader_uses_global_roots_for_open_shell_reference(
    tmp_path, monkeypatch
):
    loaded_ref, _, coords, _ = make_loaded_pair()
    artifacts = touch_artifacts(tmp_path, "open_shell")
    snapshot = Snapshot(
        "open_shell",
        coords,
        (1, 2),
        selected_root=1,
        requested_roots=(1, 2),
        multiplicities={1: 3, 2: 5},
        artifacts=artifacts,
    )
    monkeypatch.setattr(
        survey_module.Wavefunction,
        "from_file",
        lambda path: loaded_ref.wavefunction,
    )
    header = SimpleNamespace(
        layout="orca-standard-vector-records",
        alpha_occ_start=0,
        alpha_occ_end=1,
        alpha_virt_start=2,
        alpha_virt_end=3,
        beta_occ_start=0,
        beta_occ_end=1,
        beta_virt_start=2,
        beta_virt_end=3,
        alpha_nocc=2,
        alpha_nvirt=2,
        beta_nocc=2,
        beta_nvirt=2,
    )
    transitions = loaded_ref.transition_alpha
    parser_calls = []

    def cis_parser(path, **kwargs):
        parser_calls.append(kwargs)
        return header, {
            root: {
                "global_root": root,
                "multiplicity": multiplicity,
                "root_within_multiplicity": 1,
                "orca_gradient_iroot": 1,
                "response_component": "tda-x",
                "alpha": transitions[root - 1],
                "beta": transitions[root - 1],
            }
            for root, multiplicity in ((1, 3), (2, 5))
        }

    state = ORCA6ArtifactLoader(
        triplets=False,
        multiplicity=3,
        tda=True,
        cis_parser=cis_parser,
    )(snapshot, roots=(1, 2), atoms=("H", "H"), coordinates=coords)

    assert state.roots == (1, 2)
    assert state.root_metadata[1]["multiplicity"] == 3
    assert state.root_metadata[2]["multiplicity"] == 5
    assert parser_calls == [{"multiplicity": None, "tda": True}]


def test_artifact_loader_rejects_duplicate_multiplicity_local_iroot(
    tmp_path, monkeypatch
):
    loaded_ref, _, coords, _ = make_loaded_pair()
    artifacts = touch_artifacts(tmp_path, "state")
    snapshot = Snapshot("state", coords, (1, 2), artifacts=artifacts)
    monkeypatch.setattr(
        survey_module.Wavefunction,
        "from_file",
        lambda path: loaded_ref.wavefunction,
    )
    header = SimpleNamespace(layout="orca-standard-vector-records")
    states = {
        root: {"multiplicity": 3, "orca_gradient_iroot": 1}
        for root in (3, 4)
    }

    with pytest.raises(ORCA6SurveyError, match="more than one state"):
        ORCA6ArtifactLoader(
            triplets=True,
            cis_parser=lambda *args, **kwargs: (header, states),
        )(snapshot, roots=(1, 2), atoms=("H", "H"), coordinates=coords)


@pytest.mark.parametrize(
    "excited_block",
    (
        "%tddft iroot 2 nroots 4 triplets true end",
        "%tddft iroot=2 nroots = 4 triplets true end",
    ),
)
def test_sanitize_blocks_changes_only_excited_state_block(excited_block):
    blocks = f"%scf maxiter 100 end\n{excited_block}\n%mdci nroots 9 end"
    sanitized = ORCA6AllRootSurveyBackend._sanitize_blocks(blocks, 6)
    assert not re.search(r"\biroot(?!mult)\b", sanitized, re.IGNORECASE)
    assert re.search(r"\bnroots\s+6\b", sanitized, re.IGNORECASE)
    assert "%mdci nroots 9" in sanitized


@pytest.mark.parametrize(
    "blocks, message",
    (
        (
            "%tddft nroots 4 end\n%cis nroots 4 end",
            "exactly one ORCA TDDFT/CIS",
        ),
        ("%tddft nroots 4 nroots=5 end", "Multiple NRoots"),
        ("%tddft iroot 1 iroot=2 nroots 4 end", "Multiple IRoot"),
    ),
)
def test_sanitize_blocks_rejects_ambiguous_excited_state_input(blocks, message):
    with pytest.raises(ORCA6SurveyError, match=message):
        ORCA6AllRootSurveyBackend._sanitize_blocks(blocks, 6)


@pytest.mark.parametrize(
    "blocks, expected",
    (
        ("%tddft tda true end", True),
        ("%tddft TDA = FALSE end", False),
        ("%tddft nroots 4 end", None),
    ),
)
def test_tda_hint_is_explicit_or_left_for_robust_parser(blocks, expected):
    assert survey_module._tda_hint_from_blocks(blocks) is expected


def test_tda_hint_rejects_duplicate_directives():
    with pytest.raises(ORCA6SurveyError, match="Multiple TDA"):
        survey_module._tda_hint_from_blocks("%tddft tda true tda=false end")


def test_bootstrap_builds_complete_multiplicity_local_snapshot(tmp_path):
    loaded_ref, _, coords, _ = make_loaded_pair()
    artifacts = touch_artifacts(tmp_path, "initial")
    artifacts["output"].write_text(
        "Program Version 6.1.1 - RELEASE\n****ORCA TERMINATED NORMALLY****\n"
    )

    class CompletedCalculator:
        do_tddft = True
        do_ice = False
        triplets = True
        mult = 1
        nroots = 2
        cis = artifacts["cis"]
        bson = artifacts["bson"]
        gbw = artifacts["gbw"]
        out = artifacts["output"]
        kept_history = {}

        @staticmethod
        def check_termination(text):
            return "ORCA TERMINATED NORMALLY" in text

        @staticmethod
        def parse_all_energies(*, text, triplets):
            assert triplets
            return np.array([-10.2, -10.0, -9.9])

    def loader(snapshot, *, roots, atoms, coordinates):
        return loaded_ref

    snapshot = bootstrap_tdentrack_snapshot(
        CompletedCalculator(),
        ("H", "H"),
        coords,
        selected_root=2,
        artifact_loader=loader,
        snapshot_factory=Snapshot,
    )

    assert snapshot.roots == snapshot.requested_roots == (1, 2)
    assert snapshot.selected_root == 2
    assert snapshot.multiplicities == {1: 3, 2: 3}
    assert snapshot.energies_eh == {1: -10.0, 2: -9.9}
    assert snapshot.metadata["root_numbering"] == "multiplicity-local-iroot"
    assert set(("cis", "bson", "gbw", "output")) <= set(snapshot.artifacts)


def test_bootstrap_open_shell_uses_global_roots_and_cis_multiplicities(tmp_path):
    loaded_ref, _, coords, _ = make_loaded_pair()
    loaded_ref = LoadedORCA6State(
        loaded_ref.wavefunction,
        loaded_ref.roots,
        loaded_ref.active_space,
        loaded_ref.transition_alpha,
        loaded_ref.transition_beta,
        root_metadata={
            1: {"global_root": 1, "multiplicity": 3},
            2: {"global_root": 2, "multiplicity": 5},
        },
    )
    artifacts = touch_artifacts(tmp_path, "open_shell_initial")
    artifacts["output"].write_text(
        "Program Version 6.1.1 - RELEASE\n****ORCA TERMINATED NORMALLY****\n"
    )

    class CompletedCalculator:
        do_tddft = True
        do_ice = False
        triplets = False
        mult = 3
        nroots = 2
        cis = artifacts["cis"]
        bson = artifacts["bson"]
        gbw = artifacts["gbw"]
        out = artifacts["output"]
        kept_history = {}

        @staticmethod
        def check_termination(text):
            return "ORCA TERMINATED NORMALLY" in text

        @staticmethod
        def parse_all_energies(*, text, triplets):
            assert not triplets
            return np.array([-10.2, -10.0, -9.9])

    snapshot = bootstrap_tdentrack_snapshot(
        CompletedCalculator(),
        ("H", "H"),
        coords,
        selected_root=1,
        artifact_loader=lambda *args, **kwargs: loaded_ref,
        snapshot_factory=Snapshot,
    )

    assert snapshot.roots == (1, 2)
    assert snapshot.multiplicities == {1: 3, 2: 5}
    assert snapshot.metadata["root_numbering"] == "global-state-iroot"
