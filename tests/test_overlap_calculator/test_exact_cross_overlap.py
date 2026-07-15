import importlib
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import numpy as np
import pytest

from pysisyphus.calculators.ORCA import parse_orca_cis
from pysisyphus.calculators.OverlapCalculator import (
    ExactCrossOverlapError,
    OverlapCalculator,
    get_tden_overlaps,
    normalize_tden_overlap_matrix,
)
from pysisyphus.wavefunction import Wavefunction


BENZENE_DIR = Path(__file__).parent / "benzene"


def load_cycle(cycle):
    base = BENZENE_DIR / f"calculator_000.{cycle:03d}.orca"
    wavefunction = Wavefunction.from_orca_json(f"{base}.json")
    Xa, Ya, Xb, Yb = parse_orca_cis(
        f"{base}.cis", restricted_same_ab=True
    )
    all_energies = np.arange(Xa.shape[0] + 1, dtype=float)
    overlap_data = (
        wavefunction.C[0],
        Xa,
        Ya,
        wavefunction.C[1],
        Xb,
        Yb,
        all_energies,
    )
    return wavefunction, overlap_data


def make_exact_calculator(tmp_path, normalize_tden=False):
    return OverlapCalculator(
        root=1,
        nroots=4,
        track=True,
        ovlp_type="tden",
        exact_cross_overlap=True,
        normalize_tden=normalize_tden,
        out_dir=tmp_path,
    )


def store_benzene_pair(calc):
    wavefunctions = list()
    for cycle in (0, 1):
        wavefunction, overlap_data = load_cycle(cycle)
        calc.store_overlap_data(
            wavefunction.atoms,
            wavefunction.coords,
            overlap_data=overlap_data,
            wavefunction=wavefunction,
        )
        wavefunctions.append(wavefunction)
    return wavefunctions


def test_exact_cross_overlap_benzene(tmp_path):
    calc = make_exact_calculator(tmp_path)
    wf_ref, wf_cur = store_benzene_pair(calc)

    expected = wf_ref.S_with(wf_cur)
    actual = calc.get_exact_cross_overlap()
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(actual, wf_cur.S_with(wf_ref).T, atol=1e-12)

    # Verify that track_root actually uses the exact matrix, rather than merely
    # exposing it through get_exact_cross_overlap().
    expected_tden = calc.get_tden_overlaps(S_AO=expected)
    calc.track_root()
    np.testing.assert_allclose(calc.overlap_matrices[-1], expected_tden, atol=1e-12)


def test_exact_cross_overlap_is_opt_in(tmp_path):
    calc = OverlapCalculator(out_dir=tmp_path)
    wf, overlap_data = load_cycle(0)
    calc.store_overlap_data(wf.atoms, wf.coords, overlap_data=overlap_data)
    assert calc.overlap_wavefunctions == []


def test_exact_cross_overlap_missing_snapshot_fails(tmp_path):
    calc = make_exact_calculator(tmp_path)
    wf, overlap_data = load_cycle(0)
    with pytest.raises(ExactCrossOverlapError, match="does not provide"):
        calc.store_overlap_data(wf.atoms, wf.coords, overlap_data=overlap_data)


def test_exact_cross_overlap_atom_order_validation(tmp_path):
    calc = make_exact_calculator(tmp_path)
    wf, overlap_data = load_cycle(0)
    wrong_atoms = list(wf.atoms)
    wrong_atoms[0] = "N"
    with pytest.raises(ExactCrossOverlapError, match="atom identities/order"):
        calc.store_overlap_data(
            wrong_atoms,
            wf.coords,
            overlap_data=overlap_data,
            wavefunction=wf,
        )


def test_normalized_tden_overlap_benzene(tmp_path):
    calc = make_exact_calculator(tmp_path, normalize_tden=True)
    wf_ref, wf_cur = store_benzene_pair(calc)
    S_cross = calc.get_exact_cross_overlap()
    actual = calc.get_tden_overlaps(S_AO=S_cross)

    (Xa_ref, Ya_ref, Xb_ref, Yb_ref) = calc.get_ci_coeffs_for(0)
    (Xa_cur, Ya_cur, Xb_cur, Yb_cur) = calc.get_ci_coeffs_for(1)
    raw = get_tden_overlaps(
        wf_ref.C[0],
        wf_ref.C[1],
        Xa_ref,
        Ya_ref,
        Xb_ref,
        Yb_ref,
        wf_cur.C[0],
        wf_cur.C[1],
        Xa_cur,
        Ya_cur,
        Xb_cur,
        Yb_cur,
        S_cross,
    )
    ref_self = get_tden_overlaps(
        wf_ref.C[0],
        wf_ref.C[1],
        Xa_ref,
        Ya_ref,
        Xb_ref,
        Yb_ref,
        wf_ref.C[0],
        wf_ref.C[1],
        Xa_ref,
        Ya_ref,
        Xb_ref,
        Yb_ref,
        wf_ref.S,
    )
    cur_self = get_tden_overlaps(
        wf_cur.C[0],
        wf_cur.C[1],
        Xa_cur,
        Ya_cur,
        Xb_cur,
        Yb_cur,
        wf_cur.C[0],
        wf_cur.C[1],
        Xa_cur,
        Ya_cur,
        Xb_cur,
        Yb_cur,
        wf_cur.S,
    )
    expected = normalize_tden_overlap_matrix(raw, ref_self, cur_self)
    np.testing.assert_allclose(actual, expected, atol=1e-12)

    # Same-geometry transition densities are normalized by norm_ci_coeffs.
    for self_overlap in (ref_self, cur_self):
        np.testing.assert_allclose(np.diag(self_overlap), 1.0, atol=1e-10)


def test_normalized_tden_rejects_zero_norm():
    overlaps = np.eye(2)
    with pytest.raises(ValueError, match="non-positive or vanishing"):
        normalize_tden_overlap_matrix(overlaps, np.diag((1.0, 0.0)), np.eye(2))


def test_orca_prepare_overlap_data_does_not_double_slice_triplets(monkeypatch):
    orca_module = importlib.import_module("pysisyphus.calculators.ORCA")
    nroots = 3
    coeffs = tuple(np.full((nroots, 2, 2), i, dtype=float) for i in range(4))
    call_kwargs = {}

    def fake_parse_orca_cis(*args, **kwargs):
        call_kwargs.update(kwargs)
        return coeffs

    monkeypatch.setattr(orca_module, "parse_orca_cis", fake_parse_orca_cis)
    monkeypatch.setattr(
        orca_module,
        "parse_orca_gbw",
        lambda *args: SimpleNamespace(Ca=np.eye(4), Cb=np.eye(4)),
    )

    calc = object.__new__(orca_module.ORCA)
    calc.exact_cross_overlap = False
    calc.triplets = True
    calc.nroots = nroots
    calc.cis = "dummy.cis"
    calc.gbw = "dummy.gbw"
    calc.parse_all_energies = lambda: np.arange(nroots + 1, dtype=float)

    _, Xa, Ya, _, Xb, Yb, _ = calc.prepare_overlap_data(path=None)
    assert call_kwargs["triplets_only"] is True
    for actual, expected in zip((Xa, Ya, Xb, Yb), coeffs):
        np.testing.assert_array_equal(actual, expected)


def test_orca6_exact_overlap_data_never_parses_raw_gbw(monkeypatch):
    orca_module = importlib.import_module("pysisyphus.calculators.ORCA")
    nroots = 2
    coeffs = tuple(np.full((nroots, 2, 2), i, dtype=float) for i in range(4))
    wavefunction = SimpleNamespace(C=np.stack((np.eye(4), np.eye(4))))

    monkeypatch.setattr(orca_module, "parse_orca_cis", lambda *args, **kwargs: coeffs)

    def fail_parse_gbw(*args, **kwargs):
        raise AssertionError("the ORCA 6 exact path must not parse raw GBW bytes")

    monkeypatch.setattr(orca_module, "parse_orca_gbw", fail_parse_gbw)
    calc = object.__new__(orca_module.ORCA)
    calc.exact_cross_overlap = True
    calc.triplets = True
    calc.nroots = nroots
    calc.cis = "orca6.cis"
    calc.gbw = "orca6.gbw"
    calc._load_overlap_wavefunction = lambda: wavefunction
    calc.parse_all_energies = lambda: np.arange(nroots + 1, dtype=float)

    Ca, _Xa, _Ya, Cb, _Xb, _Yb, _energies = calc.prepare_overlap_data(path=None)
    np.testing.assert_array_equal(Ca, wavefunction.C[0])
    np.testing.assert_array_equal(Cb, wavefunction.C[1])
    assert calc._prepared_overlap_wavefunction is wavefunction
    assert calc.prepare_overlap_wavefunction() is wavefunction
    assert not hasattr(calc, "_prepared_overlap_wavefunction")


def test_orca_utility_command_uses_configured_installation(tmp_path):
    orca_module = importlib.import_module("pysisyphus.calculators.ORCA")
    bin_dir = tmp_path / "orca-6.1.1"
    bin_dir.mkdir()
    executables = {}
    for name in ("orca", "orca_2json", "orca_2mkl"):
        executable = bin_dir / name
        executable.write_text("#!/bin/sh\n")
        executable.chmod(0o755)
        executables[name] = executable.resolve()

    calc = object.__new__(orca_module.ORCA)
    calc.base_cmd = str(executables["orca"])
    calc.orca_utility_overrides = {"orca_2json": None, "orca_2mkl": None}
    calc.log = lambda *args, **kwargs: None

    command = calc.get_orca_utility_command("orca_2json", "orca.gbw", "-bson")
    assert command == [str(executables["orca_2json"]), "orca.gbw", "-bson"]


def test_orca_utility_explicit_override(tmp_path):
    orca_module = importlib.import_module("pysisyphus.calculators.ORCA")
    override = tmp_path / "custom_2json"
    override.write_text("#!/bin/sh\n")
    override.chmod(0o755)

    calc = object.__new__(orca_module.ORCA)
    calc.base_cmd = "/missing/orca"
    calc.orca_utility_overrides = {
        "orca_2json": str(override),
        "orca_2mkl": None,
    }
    calc.log = lambda *args, **kwargs: None
    assert calc.resolve_orca_utility("orca_2json") == str(override.resolve())


def test_orca_run_after_uses_orca6_gbw_argument(tmp_path):
    orca_module = importlib.import_module("pysisyphus.calculators.ORCA")
    calls = []
    calc = object.__new__(orca_module.ORCA)
    calc.cdds = False
    calc.wavefunction_dump = True
    calc.exact_cross_overlap = False
    calc.get_orca_utility_command = lambda utility, *args: [utility, *args]

    def fake_popen(command, cwd=None):
        calls.append((command, cwd))
        return SimpleNamespace(returncode=0)

    calc.popen = fake_popen
    calc.run_after(tmp_path)
    assert calls == [(["orca_2json", "orca.gbw", "-bson"], tmp_path)]
    config = json.loads((tmp_path / "orca.json.conf").read_text())
    assert config["MOCoefficients"] is True
    assert config["Basisset"] is True
    assert config["JSONFormats"] == ["bson"]


def test_orca6_basis_key_is_supported():
    from pysisyphus.io.orca import wavefunction_from_json_dict

    json_path = BENZENE_DIR / "calculator_000.000.orca.json"
    data = json.loads(json_path.read_text())
    for atom in data["Molecule"]["Atoms"]:
        atom["Basis"] = atom.pop("BasisFunctions")
    wavefunction = wavefunction_from_json_dict(data)
    assert wavefunction.S.shape == wavefunction.C[0].shape


def test_orca6_bson_null_does_not_end_document():
    from pysisyphus.io import bson

    # BSON document {"optional": null, "after": 7}. The integer following the
    # null verifies that Null is distinct from the end-of-document marker.
    null_element = b"\x0aoptional\x00"
    int_element = b"\x10after\x00" + struct.pack("<i", 7)
    size = 4 + len(null_element) + len(int_element) + 1
    document = struct.pack("<i", size) + null_element + int_element + b"\x00"
    assert bson.loads(document) == {"optional": None, "after": 7}
