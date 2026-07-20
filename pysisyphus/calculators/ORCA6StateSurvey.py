"""Isolated ORCA 6.1.1 all-root surveys for transactional optimization.

The backend in this module is deliberately independent of pysisyphus'
historical ``OverlapCalculator`` state.  A child :class:`ORCA` calculator runs
an energy-only TDDFT calculation in a unique retained directory.  Reference
and candidate BSON wavefunctions provide exact analytic inter-geometry AO
overlaps, while the corresponding ``.cis`` files provide the full signed
``X + Y`` transition-density amplitudes.

The child runner and artifact loader are injectable.  This both makes the
transaction scientifically auditable and permits offline tests with synthetic
wavefunctions; no test needs an ORCA executable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import struct
from types import MappingProxyType, SimpleNamespace
from typing import Any, Callable, Mapping, Optional, Sequence
import uuid

import numpy as np

from pysisyphus.calculators.ORCA import ORCA
from pysisyphus.constants import AU2EV
from pysisyphus.wavefunction import Wavefunction


class ORCA6SurveyError(RuntimeError):
    """Raised when an all-root ORCA survey cannot be trusted."""


MULTIPLICITY_LOCAL_ROOTS = "multiplicity-local-iroot"
GLOBAL_STATE_ROOTS = "global-state-iroot"
ROOT_NUMBERING_MODES = (MULTIPLICITY_LOCAL_ROOTS, GLOBAL_STATE_ROOTS)

_FINAL_ENERGY_RE = re.compile(
    r"FINAL SINGLE POINT ENERGY\s+"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?)"
)


def _root_numbering_for(calculator: Any) -> str:
    """Return the ORCA root convention used by a calculator.

    Restricted-reference spin-adapted triplets use an IRoot ordinal within the
    triplet block.  Unrestricted open-shell calculations instead optimize the
    global ``STATE N`` ordinal, even when the converged states have different
    approximate multiplicities.
    """

    return (
        MULTIPLICITY_LOCAL_ROOTS
        if bool(getattr(calculator, "triplets", False))
        else GLOBAL_STATE_ROOTS
    )


def _multiplicities_from_loaded(
    state: "LoadedORCA6State", *, default: int
) -> dict[int, int]:
    default = int(default)
    if default <= 0:
        raise ORCA6SurveyError("The fallback multiplicity must be positive.")
    multiplicities = {}
    for root in state.roots:
        value = state.root_metadata.get(root, {}).get("multiplicity", default)
        value = default if value is None else int(value)
        if value <= 0:
            raise ORCA6SurveyError(
                f"CIS root {root} has invalid multiplicity {value}."
            )
        multiplicities[root] = value
    return multiplicities


def _align_state_energies_to_final(
    all_energies: Any,
    output_text: str,
    *,
    anchor_root: int,
) -> tuple[np.ndarray, float]:
    """Add ORCA's state-independent correction to every TDDFT root energy.

    ``parse_all_energies`` constructs state totals from the printed ``E(SCF)``
    and excitation energies.  ORCA applies state-independent contributions such
    as D3(BJ) later, and the EnGrad ``.engrad`` energy contains them.  The final
    energy therefore anchors either the selected gradient root or root zero for
    an energy-only all-root survey.  Applying one common shift preserves every
    excitation energy while keeping surveys, gradients, and optimizer descent
    checks on the same potential-energy scale.
    """

    energies = np.asarray(all_energies, dtype=float).copy()
    if energies.ndim != 1 or not np.all(np.isfinite(energies)):
        raise ORCA6SurveyError("ORCA state energies must be a finite vector.")
    anchor_root = int(anchor_root)
    if anchor_root < 0 or anchor_root >= energies.size:
        raise ORCA6SurveyError(
            f"Energy anchor root {anchor_root} is outside 0..{energies.size - 1}."
        )
    matches = _FINAL_ENERGY_RE.findall(output_text)
    if not matches:
        raise ORCA6SurveyError(
            "ORCA output lacks FINAL SINGLE POINT ENERGY; state energies "
            "cannot be aligned with EnGrad energies."
        )
    final_energy = float(matches[-1].replace("D", "E").replace("d", "e"))
    if not math.isfinite(final_energy):
        raise ORCA6SurveyError("ORCA final single-point energy is non-finite.")
    correction = final_energy - float(energies[anchor_root])
    energies += correction
    return energies, correction


def _readonly_array(values: Any, *, ndim: Optional[int] = None) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if ndim is not None and array.ndim != ndim:
        raise ORCA6SurveyError(
            f"Expected a {ndim}-dimensional array, got shape {array.shape}."
        )
    if not np.all(np.isfinite(array)):
        raise ORCA6SurveyError("Electronic-state data contain NaN or infinity.")
    result = np.frombuffer(
        np.ascontiguousarray(array).tobytes(), dtype=float
    ).reshape(array.shape)
    result.setflags(write=False)
    return result


def _atom_labels(atoms: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(atom).strip().lower() for atom in atoms)


def _path_mapping(values: Mapping[str, Any]) -> Mapping[str, Path]:
    return MappingProxyType({str(key): Path(value).resolve() for key, value in values.items()})


@dataclass(frozen=True)
class CISActiveSpace:
    """Zero-based inclusive ORCA MO ranges for both spin channels."""

    alpha_occ: tuple[int, int]
    alpha_virt: tuple[int, int]
    beta_occ: tuple[int, int]
    beta_virt: tuple[int, int]

    @staticmethod
    def _slice(bounds: tuple[int, int]) -> slice:
        start, end = bounds
        return slice(start, end + 1)

    def occ_slice(self, spin: int) -> slice:
        return self._slice(self.alpha_occ if spin == 0 else self.beta_occ)

    def virt_slice(self, spin: int) -> slice:
        return self._slice(self.alpha_virt if spin == 0 else self.beta_virt)

    def as_jsonable(self) -> dict[str, list[int]]:
        return {
            "alpha_occ": list(self.alpha_occ),
            "alpha_virt": list(self.alpha_virt),
            "beta_occ": list(self.beta_occ),
            "beta_virt": list(self.beta_virt),
        }


@dataclass(frozen=True)
class LoadedORCA6State:
    """BSON orbitals and signed transition amplitudes for one geometry."""

    wavefunction: Any
    roots: tuple[int, ...]
    active_space: CISActiveSpace
    transition_alpha: np.ndarray = field(repr=False, compare=False)
    transition_beta: np.ndarray = field(repr=False, compare=False)
    root_metadata: Mapping[int, Mapping[str, Any]] = field(
        default_factory=dict, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        roots = tuple(int(root) for root in self.roots)
        alpha = _readonly_array(self.transition_alpha, ndim=3)
        beta = _readonly_array(self.transition_beta, ndim=3)
        if alpha.shape[0] != len(roots) or beta.shape[0] != len(roots):
            raise ORCA6SurveyError(
                "The number of parsed CIS vectors does not match the declared roots."
            )
        metadata = {
            int(root): MappingProxyType(dict(values))
            for root, values in self.root_metadata.items()
        }
        if set(metadata) - set(roots):
            raise ORCA6SurveyError(
                "CIS root metadata contain roots absent from the loaded state."
            )
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "transition_alpha", alpha)
        object.__setattr__(self, "transition_beta", beta)
        object.__setattr__(self, "root_metadata", MappingProxyType(metadata))


@dataclass(frozen=True)
class ChildSurveyResult:
    """Retained result of an isolated child ORCA calculation."""

    label: str
    atoms: tuple[str, ...]
    coordinates: np.ndarray = field(repr=False, compare=False)
    roots: tuple[int, ...]
    energies_eh: Mapping[int, float]
    excitation_energies_ev: Mapping[int, float]
    multiplicities: Mapping[int, int]
    artifacts: Mapping[str, Path]
    orca_version: Optional[str] = None
    energy_correction_eh: float = 0.0

    def __post_init__(self) -> None:
        if not self.label:
            raise ValueError("A child survey requires a label.")
        roots = tuple(int(root) for root in self.roots)
        if not roots or len(set(roots)) != len(roots):
            raise ValueError("Child survey roots must be non-empty and unique.")
        coordinates = _readonly_array(self.coordinates)
        energies = {int(root): float(value) for root, value in self.energies_eh.items()}
        excitations = {
            int(root): float(value)
            for root, value in self.excitation_energies_ev.items()
        }
        multiplicities = {
            int(root): int(value) for root, value in self.multiplicities.items()
        }
        for name, values in (
            ("energies", energies),
            ("excitation energies", excitations),
        ):
            if set(values) != set(roots) or not all(
                math.isfinite(value) for value in values.values()
            ):
                raise ValueError(f"Child survey {name} are incomplete or non-finite.")
        if set(multiplicities) != set(roots):
            raise ValueError("Child survey multiplicities are incomplete.")
        energy_correction = float(self.energy_correction_eh)
        if not math.isfinite(energy_correction):
            raise ValueError("Child survey energy correction must be finite.")
        object.__setattr__(self, "atoms", tuple(self.atoms))
        object.__setattr__(self, "coordinates", coordinates)
        object.__setattr__(self, "roots", roots)
        object.__setattr__(self, "energies_eh", MappingProxyType(energies))
        object.__setattr__(
            self, "excitation_energies_ev", MappingProxyType(excitations)
        )
        object.__setattr__(self, "multiplicities", MappingProxyType(multiplicities))
        object.__setattr__(self, "artifacts", _path_mapping(self.artifacts))
        object.__setattr__(self, "energy_correction_eh", energy_correction)


def read_cis_active_space(path: Path) -> CISActiveSpace:
    """Read the standard ORCA ``.cis`` active ranges used by parse_orca_cis."""

    path = Path(path)
    try:
        with path.open("rb") as handle:
            raw = handle.read(9 * 4)
        if len(raw) != 9 * 4:
            raise ORCA6SurveyError(f"CIS header is truncated: {path}")
        _, *ranges = struct.unpack("9i", raw)
    except OSError as exc:
        raise ORCA6SurveyError(f"Could not read CIS header {path}: {exc}") from exc
    alpha = tuple(int(value) for value in ranges[:4])
    beta = tuple(int(value) for value in ranges[4:])
    if all(value == -1 for value in beta):
        beta = alpha
    elif any(value < 0 for value in beta):
        raise ORCA6SurveyError(
            f"Partially defined beta active-space ranges in {path}: {beta}."
        )
    active = CISActiveSpace(
        alpha_occ=(alpha[0], alpha[1]),
        alpha_virt=(alpha[2], alpha[3]),
        beta_occ=(beta[0], beta[1]),
        beta_virt=(beta[2], beta[3]),
    )
    for name, bounds in active.as_jsonable().items():
        start, end = bounds
        if start < 0 or end < start:
            raise ORCA6SurveyError(
                f"Invalid zero-based {name} range {start}..{end} in {path}."
            )
    return active


def _tdentrack_parse_cis_amplitudes(*args: Any, **kwargs: Any) -> Any:
    try:
        from excited_state_diabatizer.cis_io import parse_cis_amplitudes
    except ImportError as exc:
        raise ORCA6SurveyError(
            "The ORCA 6.1.1 artifact loader requires tdentrack's robust CIS "
            "parser. Install a compatible tdentrack version."
        ) from exc
    return parse_cis_amplitudes(*args, **kwargs)


def _tda_hint_from_blocks(blocks: str) -> Optional[bool]:
    """Return an explicit ORCA TDA setting, or ``None`` for parser inference."""

    matches = re.findall(
        r"\btda\s*(?:=\s*)?(true|false|1|0)\b",
        str(blocks),
        flags=re.IGNORECASE,
    )
    if len(matches) > 1:
        raise ORCA6SurveyError(
            "Multiple TDA directives make the TDDFT/CIS input ambiguous."
        )
    if not matches:
        return None
    return matches[0].lower() in ("true", "1")


class ORCA6ArtifactLoader:
    """Load one retained BSON/``.cis`` pair without reading raw GBW bytes."""

    def __init__(
        self,
        *,
        triplets: bool,
        multiplicity: Optional[int] = None,
        root_numbering: Optional[str] = None,
        tda: Optional[bool] = None,
        cis_parser: Optional[Callable[..., Any]] = None,
        energy_tolerance_eh: float = 5.0e-6,
    ) -> None:
        self.triplets = bool(triplets)
        self.multiplicity = int(
            3 if self.triplets else (1 if multiplicity is None else multiplicity)
        )
        if self.multiplicity <= 0:
            raise ValueError("multiplicity must be a positive integer.")
        if root_numbering is None:
            root_numbering = (
                MULTIPLICITY_LOCAL_ROOTS if self.triplets else GLOBAL_STATE_ROOTS
            )
        if root_numbering not in ROOT_NUMBERING_MODES:
            raise ValueError(
                f"root_numbering must be one of {ROOT_NUMBERING_MODES}, got "
                f"{root_numbering!r}."
            )
        self.root_numbering = root_numbering
        self.tda = None if tda is None else bool(tda)
        self.cis_parser = (
            _tdentrack_parse_cis_amplitudes if cis_parser is None else cis_parser
        )
        self.energy_tolerance_eh = float(energy_tolerance_eh)
        if (
            not math.isfinite(self.energy_tolerance_eh)
            or self.energy_tolerance_eh < 0.0
        ):
            raise ValueError("energy_tolerance_eh must be finite and non-negative.")

    def __call__(
        self,
        snapshot: Any,
        *,
        roots: Sequence[int],
        atoms: Sequence[str],
        coordinates: np.ndarray,
    ) -> LoadedORCA6State:
        artifacts = getattr(snapshot, "artifacts", {})
        try:
            cis_path = Path(artifacts["cis"])
            bson_path = Path(artifacts["bson"])
        except KeyError as exc:
            raise ORCA6SurveyError(
                f"Snapshot {getattr(snapshot, 'label', '<unknown>')!r} must retain "
                "both 'cis' and 'bson' artifacts."
            ) from exc
        for kind, path in (("cis", cis_path), ("bson", bson_path)):
            if not path.is_file():
                raise ORCA6SurveyError(
                    f"Retained {kind} artifact does not exist: {path}"
                )
        try:
            wavefunction = Wavefunction.from_file(bson_path)
        except Exception as exc:
            raise ORCA6SurveyError(
                f"Could not parse ORCA 6 BSON wavefunction {bson_path}: {exc}"
            ) from exc
        try:
            header, parsed = self.cis_parser(
                cis_path,
                multiplicity=(
                    self.multiplicity
                    if self.root_numbering == MULTIPLICITY_LOCAL_ROOTS
                    else None
                ),
                tda=self.tda,
            )
        except Exception as exc:
            raise ORCA6SurveyError(
                f"Could not parse ORCA 6 CIS amplitudes {cis_path}: {exc}"
            ) from exc

        roots = tuple(int(root) for root in roots)
        by_root: dict[int, Mapping[str, Any]] = {}
        for global_root, state in parsed.items():
            encoded_multiplicity = state.get("multiplicity")
            if (
                self.root_numbering == MULTIPLICITY_LOCAL_ROOTS
                and encoded_multiplicity is not None
                and int(encoded_multiplicity) != self.multiplicity
            ):
                raise ORCA6SurveyError(
                    f"CIS global root {global_root} has multiplicity "
                    f"{encoded_multiplicity}, expected {self.multiplicity}."
                )
            if self.root_numbering == GLOBAL_STATE_ROOTS:
                root = int(global_root)
            else:
                root = state.get("orca_gradient_iroot")
                if root is None:
                    if getattr(header, "layout", "").startswith("orca-legacy-"):
                        root = global_root
                    else:
                        raise ORCA6SurveyError(
                            f"CIS global root {global_root} lacks multiplicity-local "
                            "ORCA gradient IRoot metadata."
                        )
                root = int(root)
            if root in by_root:
                raise ORCA6SurveyError(
                    f"CIS data map more than one state to {self.root_numbering} "
                    f"root {root}."
                )
            by_root[root] = state
        if tuple(sorted(by_root)) != roots:
            raise ORCA6SurveyError(
                f"CIS {self.root_numbering} roots do not match the requested "
                f"window {roots}; available roots are {tuple(sorted(by_root))}."
            )
        active = CISActiveSpace(
            alpha_occ=(header.alpha_occ_start, header.alpha_occ_end),
            alpha_virt=(header.alpha_virt_start, header.alpha_virt_end),
            beta_occ=(header.beta_occ_start, header.beta_occ_end),
            beta_virt=(header.beta_virt_start, header.beta_virt_end),
        )
        expected_alpha = (header.alpha_nocc, header.alpha_nvirt)
        expected_beta = (header.beta_nocc, header.beta_nvirt)
        transitions_alpha = []
        transitions_beta = []
        root_metadata: dict[int, dict[str, Any]] = {}
        snapshot_multiplicities = getattr(snapshot, "multiplicities", {})
        snapshot_excitations = getattr(snapshot, "excitation_energies_ev", {})
        for root in roots:
            state = by_root[root]
            alpha = np.asarray(state["alpha"], dtype=float)
            beta = np.asarray(state["beta"], dtype=float)
            if alpha.shape != expected_alpha or beta.shape != expected_beta:
                raise ORCA6SurveyError(
                    f"CIS local root {root} has alpha/beta shapes "
                    f"{alpha.shape}/{beta.shape}; expected "
                    f"{expected_alpha}/{expected_beta}."
                )
            if state.get("response_component") not in (None, "tda-x", "x-plus-y"):
                raise ORCA6SurveyError(
                    f"CIS local root {root} has unsupported response component "
                    f"{state.get('response_component')!r}."
                )
            snapshot_multiplicity = snapshot_multiplicities.get(root)
            state_multiplicity = state.get("multiplicity")
            if (
                snapshot_multiplicity is not None
                and state_multiplicity is not None
                and int(snapshot_multiplicity) != int(state_multiplicity)
            ):
                raise ORCA6SurveyError(
                    f"Snapshot root {root} has multiplicity "
                    f"{snapshot_multiplicity}, but CIS encodes "
                    f"{state_multiplicity}."
                )
            if (
                self.root_numbering == MULTIPLICITY_LOCAL_ROOTS
                and snapshot_multiplicity is not None
                and int(snapshot_multiplicity) != self.multiplicity
            ):
                raise ORCA6SurveyError(
                    f"Snapshot local root {root} has multiplicity "
                    f"{snapshot_multiplicity}, expected {self.multiplicity}."
                )
            cis_excitation = state.get("excitation_energy_eh")
            snapshot_excitation = snapshot_excitations.get(root)
            if cis_excitation is not None and snapshot_excitation is not None:
                error = abs(
                    float(cis_excitation) - float(snapshot_excitation) / AU2EV
                )
                if error > self.energy_tolerance_eh:
                    raise ORCA6SurveyError(
                        f"CIS/output excitation-energy mismatch for "
                        f"{self.root_numbering} "
                        f"{root}: {error:.3e} Eh exceeds "
                        f"{self.energy_tolerance_eh:.3e} Eh."
                    )
            transitions_alpha.append(alpha)
            transitions_beta.append(beta)
            root_metadata[root] = {
                key: state[key]
                for key in (
                    "global_root",
                    "multiplicity",
                    "root_within_multiplicity",
                    "orca_gradient_iroot",
                    "orca_root_index",
                    "vector_record_index",
                    "tda",
                    "response_component",
                    "restricted",
                )
                if key in state
            }
        Ta = np.stack(transitions_alpha)
        Tb = np.stack(transitions_beta)
        return LoadedORCA6State(
            wavefunction,
            roots,
            active,
            Ta,
            Tb,
            root_metadata=root_metadata,
        )


def _validate_wavefunction(
    state: LoadedORCA6State,
    atoms: Sequence[str],
    coordinates: np.ndarray,
    *,
    atol: float,
) -> None:
    wf = state.wavefunction
    prefix = "Invalid ORCA BSON wavefunction: "
    if not getattr(wf, "has_shells", False):
        raise ORCA6SurveyError(prefix + "AO shell data are absent.")
    if _atom_labels(wf.atoms) != _atom_labels(atoms):
        raise ORCA6SurveyError(prefix + "atom identities/order changed.")
    wf_coords = np.asarray(wf.coords, dtype=float).reshape(-1)
    coordinates = np.asarray(coordinates, dtype=float).reshape(-1)
    if wf_coords.shape != coordinates.shape or not np.allclose(
        wf_coords, coordinates, atol=atol, rtol=0.0
    ):
        raise ORCA6SurveyError(prefix + "coordinates do not match the survey geometry.")
    C = np.asarray(wf.C, dtype=float)
    if C.ndim != 3 or C.shape[0] != 2 or not np.all(np.isfinite(C)):
        raise ORCA6SurveyError(prefix + f"invalid MO coefficient shape {C.shape}.")
    S = np.asarray(wf.S, dtype=float)
    if S.shape != (C.shape[1], C.shape[1]) or not np.all(np.isfinite(S)):
        raise ORCA6SurveyError(prefix + "invalid same-geometry AO overlap dimensions.")
    symmetry_error = float(np.max(np.abs(S - S.T)))
    if symmetry_error > atol:
        raise ORCA6SurveyError(
            prefix + f"AO self-overlap is not symmetric (error {symmetry_error:.3e})."
        )
    for spin, coeffs in (("alpha", C[0]), ("beta", C[1])):
        metric = coeffs.T @ S @ coeffs
        error = float(np.max(np.abs(metric - np.eye(coeffs.shape[1]))))
        if error > atol:
            raise ORCA6SurveyError(
                prefix
                + f"{spin} MOs are not orthonormal (error {error:.3e}, "
                f"tolerance {atol:.3e})."
            )

    for spin, transition in enumerate(
        (state.transition_alpha, state.transition_beta)
    ):
        occ = state.active_space.occ_slice(spin)
        virt = state.active_space.virt_slice(spin)
        if occ.stop > C[spin].shape[1] or virt.stop > C[spin].shape[1]:
            raise ORCA6SurveyError(prefix + "CIS active range exceeds the MO space.")
        if transition.shape[1:] != (
            occ.stop - occ.start,
            virt.stop - virt.start,
        ):
            raise ORCA6SurveyError(
                prefix + "CIS amplitude dimensions differ from active MO ranges."
            )


def exact_cross_overlap(
    reference: LoadedORCA6State,
    candidate: LoadedORCA6State,
    *,
    atol: float,
) -> np.ndarray:
    """Return validated ``<AO(reference)|AO(candidate)>``."""

    wf_ref = reference.wavefunction
    wf_cur = candidate.wavefunction
    prefix = "Cannot construct exact ORCA cross-geometry AO overlap: "
    if _atom_labels(wf_ref.atoms) != _atom_labels(wf_cur.atoms):
        raise ORCA6SurveyError(prefix + "atom identities/order changed.")
    if wf_ref.bf_type != wf_cur.bf_type:
        raise ORCA6SurveyError(prefix + "basis-function representations differ.")
    if wf_ref.shells.ordering != wf_cur.shells.ordering:
        raise ORCA6SurveyError(prefix + "AO orderings differ.")
    shells_ref = tuple(wf_ref.shells.shells)
    shells_cur = tuple(wf_cur.shells.shells)
    if len(shells_ref) != len(shells_cur):
        raise ORCA6SurveyError(prefix + "the number of basis shells changed.")
    for index, (shell_ref, shell_cur) in enumerate(zip(shells_ref, shells_cur)):
        discrete_ref = (shell_ref.center_ind, shell_ref.atomic_num, shell_ref.L)
        discrete_cur = (shell_cur.center_ind, shell_cur.atomic_num, shell_cur.L)
        exps_ref = np.asarray(shell_ref.exps)
        exps_cur = np.asarray(shell_cur.exps)
        coeffs_ref = np.asarray(shell_ref.coeffs_org)
        coeffs_cur = np.asarray(shell_cur.coeffs_org)
        arrays_match = (
            exps_ref.shape == exps_cur.shape
            and coeffs_ref.shape == coeffs_cur.shape
            and np.allclose(exps_ref, exps_cur, atol=atol, rtol=1.0e-12)
            and np.allclose(coeffs_ref, coeffs_cur, atol=atol, rtol=1.0e-12)
        )
        if discrete_ref != discrete_cur or not arrays_match:
            raise ORCA6SurveyError(prefix + f"basis shell {index} changed.")
    if not np.array_equal(wf_ref.ecp_electrons, wf_cur.ecp_electrons):
        raise ORCA6SurveyError(prefix + "ECP electron counts changed.")
    try:
        forward = np.asarray(wf_ref.S_with(wf_cur), dtype=float)
        reverse = np.asarray(wf_cur.S_with(wf_ref), dtype=float)
    except Exception as exc:
        raise ORCA6SurveyError(prefix + f"analytic integral evaluation failed: {exc}") from exc
    expected = (wf_ref.C.shape[1], wf_cur.C.shape[1])
    if forward.shape != expected or reverse.shape != expected[::-1]:
        raise ORCA6SurveyError(
            prefix
            + f"AO dimensions are inconsistent: {forward.shape}, {reverse.shape}, "
            f"expected {expected}."
        )
    if not np.all(np.isfinite(forward)) or not np.all(np.isfinite(reverse)):
        raise ORCA6SurveyError(prefix + "cross-overlap contains NaN or infinity.")
    transpose_error = float(np.max(np.abs(forward - reverse.T)))
    if transpose_error > atol:
        raise ORCA6SurveyError(
            prefix
            + f"forward/reverse transpose error {transpose_error:.3e} exceeds "
            f"{atol:.3e}."
        )
    return forward


def signed_tden_overlap_matrix(
    reference: LoadedORCA6State,
    candidate: LoadedORCA6State,
    S_ao: np.ndarray,
) -> np.ndarray:
    """Contract full signed alpha+beta ``X+Y`` transition densities."""

    matrix = np.zeros((len(reference.roots), len(candidate.roots)), dtype=float)
    for spin, (T_ref, T_cur) in enumerate(
        (
            (reference.transition_alpha, candidate.transition_alpha),
            (reference.transition_beta, candidate.transition_beta),
        )
    ):
        C_ref = np.asarray(reference.wavefunction.C[spin], dtype=float)
        C_cur = np.asarray(candidate.wavefunction.C[spin], dtype=float)
        S_mo = C_ref.T @ np.asarray(S_ao, dtype=float) @ C_cur
        occ_ref = reference.active_space.occ_slice(spin)
        occ_cur = candidate.active_space.occ_slice(spin)
        virt_ref = reference.active_space.virt_slice(spin)
        virt_cur = candidate.active_space.virt_slice(spin)
        S_occ = S_mo[occ_ref, occ_cur]
        S_virt = S_mo[virt_ref, virt_cur]
        expected_occ = (T_ref.shape[1], T_cur.shape[1])
        expected_virt = (T_ref.shape[2], T_cur.shape[2])
        if S_occ.shape != expected_occ or S_virt.shape != expected_virt:
            raise ORCA6SurveyError(
                f"CIS/MO cross-overlap dimensions disagree for spin {spin}: "
                f"S_occ={S_occ.shape}/{expected_occ}, "
                f"S_virt={S_virt.shape}/{expected_virt}."
            )
        matrix += np.einsum(
            "ria,ij,ab,sjb->rs",
            T_ref,
            S_occ,
            S_virt,
            T_cur,
            optimize=True,
        )
    if not np.all(np.isfinite(matrix)):
        raise ORCA6SurveyError("Signed transition-density overlap contains NaN/Inf.")
    return matrix


def state_self_norms(state: LoadedORCA6State) -> np.ndarray:
    norms = np.diag(state_self_overlap_matrix(state)).copy()
    threshold = np.finfo(float).eps * 100.0
    if not np.all(np.isfinite(norms)) or np.any(norms <= threshold):
        raise ORCA6SurveyError(
            "Transition-density self-overlap norms must be finite and positive."
        )
    return norms


def state_self_overlap_matrix(
    state: LoadedORCA6State, *, symmetry_tolerance: float = 1.0e-6
) -> np.ndarray:
    """Return and validate the full within-geometry TDen Gram matrix."""

    gram = signed_tden_overlap_matrix(state, state, state.wavefunction.S)
    transpose_error = float(np.max(np.abs(gram - gram.T)))
    if transpose_error > symmetry_tolerance:
        raise ORCA6SurveyError(
            "Transition-density self-overlap Gram matrix is not symmetric "
            f"(error {transpose_error:.3e}, tolerance {symmetry_tolerance:.3e})."
        )
    gram = 0.5 * (gram + gram.T)
    if not np.all(np.isfinite(gram)):
        raise ORCA6SurveyError("Transition-density Gram matrix contains NaN/Inf.")
    return gram


def _tdentrack_root_overlap_block_factory(**kwargs: Any) -> Any:
    try:
        from excited_state_diabatizer.state_tracking import RootOverlapBlock
    except ImportError as exc:
        raise ORCA6SurveyError(
            "The default ORCA 6.1.1 backend requires tdentrack's "
            "RootOverlapBlock. Install a compatible tdentrack version or inject "
            "root_overlap_block_factory for offline testing."
        ) from exc
    return RootOverlapBlock(**kwargs)


def _tdentrack_snapshot_class() -> Any:
    try:
        from excited_state_diabatizer.state_tracking import ElectronicSnapshot
    except ImportError as exc:
        raise ORCA6SurveyError(
            "Bootstrapping an optimization requires tdentrack.ElectronicSnapshot. "
            "Install tdentrack before calling bootstrap_tdentrack_snapshot()."
        ) from exc
    return ElectronicSnapshot


def _completed_calculator_artifacts(calculator: Any) -> dict[str, Path]:
    artifacts: dict[str, Path] = {}
    for kind in ("cis", "bson", "gbw", "out"):
        value = getattr(calculator, kind, None)
        if value is not None:
            artifacts["output" if kind == "out" else kind] = Path(value).resolve()
    history = getattr(calculator, "kept_history", {})
    if history:
        latest = history[max(history)]
        for path in latest.get("inps", ()):
            if str(path).endswith("orca.inp"):
                artifacts["input"] = Path(path).resolve()
                break
    return artifacts


def bootstrap_tdentrack_snapshot(
    calculator: Any,
    atoms: Sequence[str],
    coordinates: np.ndarray,
    *,
    selected_root: int,
    requested_roots: Optional[Sequence[int]] = None,
    label: str = "initial",
    expected_orca_version: Optional[str] = "6.1.1",
    validation_tolerance: float = 1.0e-6,
    tda: Optional[bool] = None,
    artifact_loader: Optional[Callable[..., LoadedORCA6State]] = None,
    snapshot_factory: Optional[Callable[..., Any]] = None,
) -> Any:
    """Build the initial committed snapshot from a completed all-root ORCA job.

    Restricted-reference spin-adapted triplets use multiplicity-local IRoot
    ordinals.  Unrestricted open-shell calculations use global ``STATE N``
    ordinals because states of different approximate multiplicity may be
    interleaved in one root window.

    ``snapshot_factory`` and ``artifact_loader`` are injection points for
    offline tests.  Production calls omit both and receive a real tdentrack
    ``ElectronicSnapshot`` parsed from BSON and ``.cis`` artifacts.
    """

    if not getattr(calculator, "do_tddft", False):
        raise ORCA6SurveyError("Bootstrap calculator has no TDDFT/CIS block.")
    if getattr(calculator, "do_ice", False):
        raise ORCA6SurveyError("ICE-CI cannot bootstrap the ORCA6 TDDFT backend.")
    if requested_roots is None:
        nroots = getattr(calculator, "nroots", None)
        if nroots is None:
            raise ORCA6SurveyError("Bootstrap requires requested_roots or calculator.nroots.")
        requested_roots = tuple(range(1, int(nroots) + 1))
    roots = tuple(int(root) for root in requested_roots)
    expected_roots = tuple(range(1, len(roots) + 1))
    if roots != expected_roots:
        raise ORCA6SurveyError(
            f"Bootstrap roots must be contiguous one-based ordinals "
            f"{expected_roots}, got {roots}."
        )
    selected_root = int(selected_root)
    if selected_root not in roots:
        raise ORCA6SurveyError(
            f"Selected root {selected_root} is outside the complete window {roots}."
        )
    coordinates = np.asarray(coordinates, dtype=float).reshape(-1)
    validation_tolerance = float(validation_tolerance)
    if not math.isfinite(validation_tolerance) or validation_tolerance <= 0.0:
        raise ValueError("validation_tolerance must be finite and positive.")
    if coordinates.shape != (3 * len(atoms),) or not np.all(np.isfinite(coordinates)):
        raise ORCA6SurveyError("Bootstrap coordinates are invalid for the supplied atoms.")
    artifacts = _completed_calculator_artifacts(calculator)
    for required in ("cis", "bson", "gbw", "output"):
        path = artifacts.get(required)
        if path is None or not path.is_file():
            raise ORCA6SurveyError(
                f"Completed ORCA job lacks retained {required} artifact."
            )
    output_text = artifacts["output"].read_text(errors="replace")
    try:
        terminated = bool(calculator.check_termination(output_text))
    except Exception as exc:
        raise ORCA6SurveyError(f"Could not validate ORCA termination: {exc}") from exc
    if not terminated:
        raise ORCA6SurveyError("Bootstrap ORCA output did not terminate normally.")
    version_match = re.search(
        r"Program Version\s+([0-9]+(?:\.[0-9]+)+)", output_text
    )
    version = None if version_match is None else version_match.group(1)
    if expected_orca_version is not None and version != expected_orca_version:
        raise ORCA6SurveyError(
            f"Expected ORCA {expected_orca_version}, got {version!r}."
        )
    try:
        all_energies = np.asarray(
            calculator.parse_all_energies(
                text=output_text,
                triplets=bool(calculator.triplets),
            ),
            dtype=float,
        )
    except Exception as exc:
        raise ORCA6SurveyError(f"Could not parse bootstrap root energies: {exc}") from exc
    expected_shape = (len(roots) + 1,)
    if all_energies.shape != expected_shape or not np.all(np.isfinite(all_energies)):
        raise ORCA6SurveyError(
            f"Bootstrap root energies have shape {all_energies.shape}, expected {expected_shape}."
        )
    all_energies, energy_correction = _align_state_energies_to_final(
        all_energies,
        output_text,
        anchor_root=selected_root,
    )
    ground = float(all_energies[0])
    energies = {root: float(all_energies[root]) for root in roots}
    excitations = {
        root: (energy - ground) * AU2EV for root, energy in energies.items()
    }
    root_numbering = _root_numbering_for(calculator)
    default_multiplicity = 3 if calculator.triplets else int(calculator.mult)

    probe = SimpleNamespace(label=label, artifacts=artifacts)
    loader = (
        ORCA6ArtifactLoader(
            triplets=calculator.triplets,
            multiplicity=default_multiplicity,
            root_numbering=root_numbering,
            tda=(
                _tda_hint_from_blocks(getattr(calculator, "blocks", ""))
                if tda is None
                else tda
            ),
        )
        if artifact_loader is None
        else artifact_loader
    )
    loaded = loader(
        probe,
        roots=roots,
        atoms=atoms,
        coordinates=coordinates,
    )
    _validate_wavefunction(
        loaded, atoms, coordinates, atol=validation_tolerance
    )
    # Parsing the complete CIS window is part of LoadedORCA6State validation.
    if loaded.roots != roots:
        raise ORCA6SurveyError(
            f"Bootstrap CIS roots {loaded.roots} do not match requested roots {roots}."
        )
    multiplicities = _multiplicities_from_loaded(
        loaded, default=default_multiplicity
    )
    factory = _tdentrack_snapshot_class() if snapshot_factory is None else snapshot_factory
    return factory(
        label=label,
        coordinates=coordinates,
        roots=roots,
        selected_root=selected_root,
        requested_roots=roots,
        energies_eh=energies,
        excitation_energies_ev=excitations,
        multiplicities=multiplicities,
        artifacts=artifacts,
        metadata={
            "orca_version": version,
            "root_numbering": root_numbering,
            "state_survey_backend": "orca-6.1.1-bson-cis",
            "energy_anchor_root": selected_root,
            "state_independent_energy_correction_eh": energy_correction,
        },
    )


class ORCA6AllRootSurveyBackend:
    """Callable default backend used by ``TDenTrackORCA`` when enabled."""

    def __init__(
        self,
        parent: Any,
        *,
        requested_roots: Optional[Sequence[int]] = None,
        artifact_loader: Optional[Callable[..., LoadedORCA6State]] = None,
        child_runner: Optional[Callable[..., ChildSurveyResult]] = None,
        audit_root: Optional[Path] = None,
        validation_tolerance: float = 1.0e-6,
        expected_orca_version: Optional[str] = "6.1.1",
        child_prepare_kwargs: Optional[Mapping[str, Any]] = None,
        root_overlap_block_factory: Optional[Callable[..., Any]] = None,
        tda: Optional[bool] = None,
    ) -> None:
        self.parent = parent
        self.root_numbering = _root_numbering_for(parent)
        self.requested_roots = (
            None if requested_roots is None else tuple(int(root) for root in requested_roots)
        )
        self.validation_tolerance = float(validation_tolerance)
        if not math.isfinite(self.validation_tolerance) or self.validation_tolerance <= 0.0:
            raise ValueError("validation_tolerance must be finite and positive.")
        self.expected_orca_version = expected_orca_version
        self.child_prepare_kwargs = dict(child_prepare_kwargs or {})
        self.root_overlap_block_factory = (
            _tdentrack_root_overlap_block_factory
            if root_overlap_block_factory is None
            else root_overlap_block_factory
        )
        self.artifact_loader = (
            ORCA6ArtifactLoader(
                triplets=parent.triplets,
                multiplicity=(3 if parent.triplets else int(parent.mult)),
                root_numbering=self.root_numbering,
                tda=(
                    _tda_hint_from_blocks(parent.blocks)
                    if tda is None
                    else tda
                ),
            )
            if artifact_loader is None
            else artifact_loader
        )
        self.child_runner = self._run_child if child_runner is None else child_runner
        self.audit_root = Path(
            audit_root
            if audit_root is not None
            else Path(parent.out_dir) / "tdentrack_surveys"
        ).resolve()
        self.audit_root.mkdir(parents=True, exist_ok=True)
        self._counter = 0
        if not getattr(parent, "do_tddft", False):
            raise ValueError("The default ORCA6 survey backend requires a %tddft/%cis block.")
        if getattr(parent, "do_ice", False):
            raise ValueError("The ORCA6 TDDFT survey backend does not support ICE-CI.")

    def _roots_for(self, reference: Any) -> tuple[int, ...]:
        roots = self.requested_roots
        if roots is None:
            roots = tuple(int(root) for root in getattr(reference, "requested_roots", ()))
        if not roots:
            nroots = getattr(self.parent, "nroots", None)
            if nroots is not None:
                roots = tuple(range(1, int(nroots) + 1))
        if not roots:
            raise ORCA6SurveyError(
                "No all-root window is configured. Set requested_roots or ORCA nroots."
            )
        expected = tuple(range(1, len(roots) + 1))
        if tuple(roots) != expected:
            raise ORCA6SurveyError(
                f"ORCA all-root windows must be contiguous one-based roots {expected}, "
                f"got {tuple(roots)}."
            )
        return tuple(roots)

    @staticmethod
    def _require_reference_artifacts(reference: Any) -> None:
        artifacts = getattr(reference, "artifacts", {})
        for kind in ("cis", "bson"):
            try:
                path = Path(artifacts[kind])
            except KeyError as exc:
                raise ORCA6SurveyError(
                    f"Committed reference {reference.label!r} lacks retained {kind!r} data. "
                    "The default backend requires persisted cis+bson artifacts, "
                    "including after restart."
                ) from exc
            if not path.is_file():
                raise ORCA6SurveyError(
                    f"Committed reference {kind} artifact does not exist: {path}"
                )

    def __call__(
        self,
        atoms: Sequence[str],
        coordinates: np.ndarray,
        *,
        reference: Any,
        factor: float,
    ) -> Mapping[str, Any]:
        roots = self._roots_for(reference)
        self._require_reference_artifacts(reference)
        if set(roots) - set(int(root) for root in reference.roots):
            raise ORCA6SurveyError(
                "Committed reference snapshot does not contain the complete requested root window."
            )
        self._counter += 1
        unique = uuid.uuid4().hex[:10]
        revision = getattr(self.parent, "tracking_revision", 0)
        label = f"survey-r{revision:04d}-{self._counter:04d}-{unique}"
        audit_dir = self.audit_root / label
        audit_dir.mkdir(parents=False, exist_ok=False)

        reference_state = self.artifact_loader(
            reference,
            roots=tuple(int(root) for root in reference.roots),
            atoms=atoms,
            coordinates=np.asarray(reference.coordinates, dtype=float),
        )
        _validate_wavefunction(
            reference_state,
            atoms,
            np.asarray(reference.coordinates, dtype=float),
            atol=self.validation_tolerance,
        )
        child = self.child_runner(
            self.parent,
            tuple(atoms),
            np.asarray(coordinates, dtype=float).reshape(-1).copy(),
            reference=reference,
            requested_roots=roots,
            audit_dir=audit_dir,
            label=label,
        )
        if not isinstance(child, ChildSurveyResult):
            raise ORCA6SurveyError("child_runner must return ChildSurveyResult.")
        if child.roots != roots:
            raise ORCA6SurveyError(
                f"Child returned roots {child.roots}; requested complete window is {roots}."
            )
        if _atom_labels(child.atoms) != _atom_labels(atoms):
            raise ORCA6SurveyError("Child survey atom identities/order changed.")
        if not np.allclose(
            child.coordinates,
            np.asarray(coordinates, dtype=float).reshape(-1),
            atol=self.validation_tolerance,
            rtol=0.0,
        ):
            raise ORCA6SurveyError("Child survey coordinates changed.")
        if (
            self.expected_orca_version is not None
            and child.orca_version != self.expected_orca_version
        ):
            raise ORCA6SurveyError(
                f"Expected ORCA {self.expected_orca_version}, got {child.orca_version!r}."
            )

        metadata = {
            "state_survey_backend": "orca-6.1.1-bson-cis",
            "survey_factor": float(factor),
            "orca_version": child.orca_version,
            "audit_directory": str(audit_dir),
            "root_numbering": self.root_numbering,
            "energy_anchor_root": 0,
            "state_independent_energy_correction_eh": child.energy_correction_eh,
        }
        manifest_path = audit_dir / "state_survey.json"
        candidate_artifacts = dict(child.artifacts)
        candidate_artifacts["audit_manifest"] = manifest_path
        candidate = reference.__class__(
            label=child.label,
            coordinates=child.coordinates,
            roots=child.roots,
            selected_root=None,
            requested_roots=roots,
            energies_eh=child.energies_eh,
            excitation_energies_ev=child.excitation_energies_ev,
            multiplicities=child.multiplicities,
            artifacts=candidate_artifacts,
            metadata=metadata,
        )
        candidate_state = self.artifact_loader(
            candidate,
            roots=roots,
            atoms=atoms,
            coordinates=child.coordinates,
        )
        _validate_wavefunction(
            candidate_state,
            atoms,
            child.coordinates,
            atol=self.validation_tolerance,
        )
        S_cross = exact_cross_overlap(
            reference_state, candidate_state, atol=self.validation_tolerance
        )
        signed = signed_tden_overlap_matrix(reference_state, candidate_state, S_cross)
        reference_gram = state_self_overlap_matrix(
            reference_state, symmetry_tolerance=self.validation_tolerance
        )
        candidate_gram = state_self_overlap_matrix(
            candidate_state, symmetry_tolerance=self.validation_tolerance
        )
        reference_norm_values = np.diag(reference_gram).copy()
        candidate_norm_values = np.diag(candidate_gram).copy()
        threshold = np.finfo(float).eps * 100.0
        if (
            np.any(reference_norm_values <= threshold)
            or np.any(candidate_norm_values <= threshold)
        ):
            raise ORCA6SurveyError(
                "Transition-density self-overlap norms must be positive."
            )
        reference_norms = dict(zip(reference_state.roots, reference_norm_values))
        candidate_norms = dict(zip(candidate_state.roots, candidate_norm_values))
        try:
            root_overlap_block = self.root_overlap_block_factory(
                reference_roots=reference_state.roots,
                candidate_roots=candidate_state.roots,
                overlaps=signed,
                reference_gram=reference_gram,
                candidate_gram=candidate_gram,
            )
        except Exception as exc:
            if isinstance(exc, ORCA6SurveyError):
                raise
            raise ORCA6SurveyError(
                f"Root-overlap block construction failed: {exc}"
            ) from exc
        try:
            row = reference_state.roots.index(int(reference.selected_root))
        except ValueError as exc:
            raise ORCA6SurveyError(
                "Selected reference root is absent from the loaded CIS vectors."
            ) from exc
        overlaps = {
            root: float(signed[row, column])
            for column, root in enumerate(candidate_state.roots)
        }

        manifest = {
            "format": "pysisyphus-tdentrack-orca6-survey-v1",
            "label": child.label,
            "reference_label": reference.label,
            "reference_root": int(reference.selected_root),
            "factor": float(factor),
            "orca_version": child.orca_version,
            "root_numbering": self.root_numbering,
            "energy_anchor_root": 0,
            "state_independent_energy_correction_eh": child.energy_correction_eh,
            "atoms": list(atoms),
            "coordinates_bohr": np.asarray(coordinates, dtype=float).reshape(-1).tolist(),
            "reference_roots": list(reference_state.roots),
            "candidate_roots": list(candidate_state.roots),
            "signed_overlap_matrix": signed.tolist(),
            "reference_self_norms": {str(k): float(v) for k, v in reference_norms.items()},
            "candidate_self_norms": {str(k): float(v) for k, v in candidate_norms.items()},
            "reference_gram": reference_gram.tolist(),
            "candidate_gram": candidate_gram.tolist(),
            "reference_active_space": reference_state.active_space.as_jsonable(),
            "candidate_active_space": candidate_state.active_space.as_jsonable(),
            "reference_cis_root_metadata": {
                str(root): dict(values)
                for root, values in reference_state.root_metadata.items()
            },
            "candidate_cis_root_metadata": {
                str(root): dict(values)
                for root, values in candidate_state.root_metadata.items()
            },
            "reference_artifacts": {
                str(k): str(v) for k, v in reference.artifacts.items()
            },
            "candidate_artifacts": {
                str(k): str(v) for k, v in candidate.artifacts.items()
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

        return {
            "candidate": candidate,
            "overlaps": overlaps,
            "reference_norm": float(reference_norms[int(reference.selected_root)]),
            "candidate_norms": candidate_norms,
            "signed_overlap_matrix": signed,
            "reference_roots": reference_state.roots,
            "reference_gram": reference_gram,
            "candidate_gram": candidate_gram,
            "root_overlap_block": root_overlap_block,
            "metadata": {
                "audit_manifest": str(manifest_path),
                "cross_overlap_shape": tuple(S_cross.shape),
            },
        }

    @staticmethod
    def _sanitize_blocks(blocks: str, nroots: int) -> str:
        matches = list(re.finditer(
            r"%(?:tddft|cis)\b.*?\bend\b",
            blocks,
            flags=re.IGNORECASE | re.DOTALL,
        ))
        if len(matches) != 1:
            raise ORCA6SurveyError(
                "Expected exactly one ORCA TDDFT/CIS input block for an all-root "
                f"survey, found {len(matches)}."
            )
        match = matches[0]
        excited_block = match.group(0)
        iroot_pattern = r"\biroot(?!mult)\s*(?:=\s*)?\d+"
        iroot_matches = re.findall(iroot_pattern, excited_block, flags=re.IGNORECASE)
        if len(iroot_matches) > 1:
            raise ORCA6SurveyError(
                "Multiple IRoot directives make the TDDFT/CIS block ambiguous."
            )
        excited_block = re.sub(
            iroot_pattern,
            "",
            excited_block,
            flags=re.IGNORECASE,
        )
        nroots_pattern = r"\bnroots\s*(?:=\s*)?\d+"
        nroots_matches = re.findall(
            nroots_pattern, excited_block, flags=re.IGNORECASE
        )
        if len(nroots_matches) > 1:
            raise ORCA6SurveyError(
                "Multiple NRoots directives make the TDDFT/CIS block ambiguous."
            )
        if nroots_matches:
            excited_block = re.sub(
                nroots_pattern,
                f"nroots {nroots}",
                excited_block,
                count=1,
                flags=re.IGNORECASE,
            )
        else:
            excited_block = re.sub(
                r"%(tddft|cis)\b",
                rf"%\1 nroots {nroots}",
                excited_block,
                count=1,
                flags=re.IGNORECASE,
            )
        return blocks[: match.start()] + excited_block + blocks[match.end() :]

    def _run_child(
        self,
        parent: Any,
        atoms: tuple[str, ...],
        coordinates: np.ndarray,
        *,
        reference: Any,
        requested_roots: tuple[int, ...],
        audit_dir: Path,
        label: str,
    ) -> ChildSurveyResult:
        """Run a real, isolated, energy-only ORCA child calculation."""

        blocks = self._sanitize_blocks(parent.blocks, len(requested_roots))
        reference_gbw = getattr(reference, "artifacts", {}).get("gbw")
        if reference_gbw is not None and not Path(reference_gbw).is_file():
            raise ORCA6SurveyError(
                f"Reference GBW initial guess does not exist: {reference_gbw}"
            )
        utility_overrides = getattr(parent, "orca_utility_overrides", {})
        child = ORCA(
            keywords=parent.keywords,
            blocks=blocks,
            gbw=reference_gbw,
            do_stable=parent.do_stable,
            wavefunction_dump=True,
            orca_2json=utility_overrides.get("orca_2json"),
            orca_2mkl=utility_overrides.get("orca_2mkl"),
            root=None,
            nroots=len(requested_roots),
            track=False,
            exact_cross_overlap=False,
            charge=parent.charge,
            mult=parent.mult,
            pal=parent.pal,
            mem=parent.mem,
            check_mem=False,
            retry_calc=parent.retry_calc,
            base_name="all_roots",
            keep_kind="all",
            clean_after=True,
            out_dir=audit_dir,
        )
        # Configuration lookup can otherwise select a different ORCA installation.
        child.base_cmd = parent.base_cmd
        if child.do_stable:
            child.get_stable_wavefunction(atoms, coordinates)
        inp = child.prepare_input(
            atoms,
            coordinates,
            calc_type="",
            **self.child_prepare_kwargs,
        )
        try:
            results = child.run(inp, calc="all_energies")
        except Exception as exc:
            raise ORCA6SurveyError(
                f"Isolated ORCA all-root calculation failed in {audit_dir}: {exc}"
            ) from exc
        if not isinstance(results, Mapping) or "all_energies" not in results:
            raise ORCA6SurveyError(
                f"ORCA all-root calculation returned no energies; audit {audit_dir}."
            )
        artifacts = {}
        for kind in ("cis", "bson", "gbw", "out"):
            path = getattr(child, kind, None)
            if path is not None:
                artifacts["output" if kind == "out" else kind] = Path(path)
        # A preceding stability calculation consumes one or more child cycles;
        # the retained all-root input is therefore the latest history entry,
        # not necessarily cycle zero.
        kept = child.kept_history.get(max(child.kept_history), {}) if child.kept_history else {}
        for path in kept.get("inps", ()):
            if str(path).endswith("orca.inp"):
                artifacts["input"] = Path(path)
                break
        for required in ("cis", "bson", "gbw", "output"):
            if required not in artifacts or not artifacts[required].is_file():
                raise ORCA6SurveyError(
                    f"ORCA child did not retain required {required} artifact in {audit_dir}."
                )
        output_text = artifacts["output"].read_text(errors="replace")
        if not child.check_termination(output_text):
            raise ORCA6SurveyError(
                f"ORCA child output lacks normal termination; audit {audit_dir}."
            )
        match = re.search(r"Program Version\s+([0-9]+(?:\.[0-9]+)+)", output_text)
        version = None if match is None else match.group(1)
        all_energies = np.asarray(results["all_energies"], dtype=float)
        if all_energies.shape != (len(requested_roots) + 1,) or not np.all(
            np.isfinite(all_energies)
        ):
            raise ORCA6SurveyError(
                f"ORCA child energies have shape {all_energies.shape}; expected "
                f"{(len(requested_roots) + 1,)}."
            )
        all_energies, energy_correction = _align_state_energies_to_final(
            all_energies,
            output_text,
            anchor_root=0,
        )
        ground = float(all_energies[0])
        energies = {
            root: float(all_energies[root]) for root in requested_roots
        }
        excitations = {
            root: (energy - ground) * AU2EV for root, energy in energies.items()
        }
        default_multiplicity = 3 if parent.triplets else int(parent.mult)
        probe = SimpleNamespace(
            label=label,
            artifacts=artifacts,
            multiplicities={},
            excitation_energies_ev=excitations,
        )
        loaded = self.artifact_loader(
            probe,
            roots=requested_roots,
            atoms=atoms,
            coordinates=coordinates,
        )
        _validate_wavefunction(
            loaded,
            atoms,
            coordinates,
            atol=self.validation_tolerance,
        )
        multiplicities = _multiplicities_from_loaded(
            loaded, default=default_multiplicity
        )
        return ChildSurveyResult(
            label=label,
            atoms=atoms,
            coordinates=coordinates,
            roots=requested_roots,
            energies_eh=energies,
            excitation_energies_ev=excitations,
            multiplicities=multiplicities,
            artifacts=artifacts,
            orca_version=version,
            energy_correction_eh=energy_correction,
        )


__all__ = [
    "CISActiveSpace",
    "ChildSurveyResult",
    "LoadedORCA6State",
    "ORCA6AllRootSurveyBackend",
    "ORCA6ArtifactLoader",
    "ORCA6SurveyError",
    "exact_cross_overlap",
    "read_cis_active_space",
    "signed_tden_overlap_matrix",
    "bootstrap_tdentrack_snapshot",
    "state_self_overlap_matrix",
    "state_self_norms",
]
