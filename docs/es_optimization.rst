Excited State Optimization
**************************

pysisyphus offers excited state (ES) tracking, to follow diadiabatic states
along an optimization. ES tracking is enabled by putting `track: True` in the
`calc` section of your YAML input. In ES optimizations pysisyphus may use
additional programs (wfoverlap, Multiwfn, jmol) if requested. Please see the
:ref:`installation instructions<pysisrc-label>` for information on how to set
them up.

**Please consider reading the relevant sections (2 and 3; 4 discusses examples) of
the** `pysisyphus paper <https://onlinelibrary.wiley.com/doi/full/10.1002/qua.26390>`_.
Additionally the user should think about the relevance between equilibrium/non-equilibrium
solvation when calculating ES-gradients with implicit solvation. Serveral programs,
e.g., Gaussian use equilibrium solvation when doing ES-optimization. It is in the
responsibility of the user to add the relevant keywords when using pysisyphus, e.g.,
:code:`EqSolv` for Gaussian.

YAML Example
------------

A bare-bone input for the S\ :sub:`1` optimization of the 1H-amino-keto
tautomer of cytosin at the TD-DFT/PBE0/def2-SVP level of theory using ORCA is
shown below. The full example is found
`here <https://github.com/eljost/pysisyphus/tree/master/examples/opt/06_orca_cytosin_s1_opt>`_.

.. code:: yaml

    opt:
    calc:
     type: orca
     keywords: pbe0 def2-svp rijcosx def2/J def2-svp/C
     # Calculate 2 ES by TD-DFT, follow the first one
     blocks: "%tddft nroots 2 iroot 1 tda false end"
     charge: 0
     mult: 1
     pal: 4
     mem: 2000
     # ES-tracking related keywords follow from here
     # Enable ES-tracking, this is important.
     track: True
     # Track ES by transition density overlaps
     ovlp_type: tden
    geom:
     type: redund
     fn: cytosin.xyz

Additional keywords are possible in the `calc` section. The default values are shown
below.

.. code:: yaml

    calc:
     # Controls calculation of charge-density-differences cubes and rendering
     # Cubes are calcualted by Multiwfn, rendering is handled by jmol.
     #
     # Possible values are: (None, calc, render).
     #  None: No cube calculation/rendering
     #  calc: Cube calculation by Multiwfn.
     #  render: Same as 'calc' and cubes are then rendered by jmol.
     cdds: None
     # Overlap type. Using 'wf' requires the external wfoverlap binary. The remaining
     # options are implemented directly in pysisyphus.
     #
     # Possible values are (wf, tden, nto, nto_org). 
     #  wf: Wavefunction overlaps using the external wfoverlap program.
     #  tden: Transition density matrix overlaps.
     #  nto: Natural transition orbital overlaps.
     #  nto_org: Natural transition orbital overlaps as described by García.
     #  top: Transition orbital projection
     ovlp_type: wf
     # Controls the reference cycle that is used in the overlap calculation. The default
     # 'adapt' is recommended.
     #
     # Possible values are (first, previous, adapt)
     #  first: Keep first calculation as reference cycle. Reliable when only minor
     #         geometrical changes are expected.
     #  previous: Use previous cycle as reference.
     #  adapat: Use adaptive algorithm. Please see the pysisyphus paper for a discussion
     #          at the end of section 3.
     ovlp_with: adapt
     # Thresholds controlling the update of the reference cycle.
     # The first number specifies the minimum overlap that must be exceeded, for an update
     # of the reference cycle. Assuming a value of 0.5 (50 %), the reference cycle update
     # is skipped, if the overlaps between the current states and the reference state don't
     # exceed 50 %.
     # The last two numbers define an interval for the ratio between the second highest
     # overlap, and the highest overlap. If the ratio is small, e.g., below 0.3, then both
     # states are sufficiently different, and no reference cycle update is needed. If the ratio
     # is bigger (> 0.6), then the states are quite similar, and an update is currently not
     # advised.
     # Possible values [three positive floats between 0. and 1.]
     adapt_args: [0.5, 0.3, 0.6]
     # Explicitly calculate the AO-overlap matrix in a double molecule calculation. Only
     # supported by Turbomole and Gaussian calculators. If False, the approximate AO
     # overlap matrix is reconstructed from inverting the MO-coefficient matrix.
     #
     # Possible values: (True, False)
     double_mol: False
     # Use analytic cross-geometry AO integrals from two retained wavefunction
     # snapshots. This is currently implemented for ORCA JSON/BSON exports. Unlike
     # the default reconstruction, it evaluates <AO(ref)|AO(cur)> with the basis
     # functions centered at their actual geometries. It is mutually exclusive with
     # double_mol and deliberately remains opt-in for backwards compatibility.
     # Invalid/missing wavefunction exports, changed atom ordering or basis sets, and
     # inconsistent MO/AO ordering cause the calculation to stop.
     # ORCA's orca_2json utility is resolved next to the configured ORCA executable
     # to avoid mixing installations. It can be overridden explicitly with, e.g.,
     # orca_2json: /opt/orca_6_1_1/orca_2json (and similarly for orca_2mkl).
     #
     # Possible values: (True, False)
     exact_cross_overlap: False
     # Absolute tolerance used for basis, transpose-symmetry, coordinate and MO
     # orthonormality checks in the exact backend.
     exact_cross_overlap_atol: 1.0e-6
     # Absolute CI-coefficients below this threshold are ignored in the overlap calculation.
     #
     # Possible values: positive float
     conf_thresh: 0.0001
     #
     # nto/natural transition orbital specific
     #
     # Number of NTOs to consider in the overlap calculation. Only relevant for 'nto'
     # and 'nto_org' ovlp_types.
     #
     # Possible values: positive integer
     use_ntos: 4
     # Dynamically decide on number of NTOs according to their participation ratio. Only
     # relevant for 'nto_org'
     #
     # Possible values: boolean
     pr_nto: False
     # 
     # wfoverlap/wavefunction overlaps specific
     # 
     # Number of core orbitals to neglect in a wfoverlap calculation. Only relevant
     # for the 'wf' ovlp_type. Must be >= 0.
     #
     # Possible values: positive integer
     ncore: 0
     #
     # tden/transition density matrix specific
     #
     # Controls which set of MO coefficients (at current cycle, or the reference cycle)
     # is used to recover the AO overlap matrix.
     #
     # Possible values: (ref, cur)
     mos_ref: cur
     # Controls whether the set of MO coefficents that was NOT used for recovering the AO
     # overlap matrix is re-normalized, using the recovered AO overlap matrix. If set to
     # True and mos_ref = cur, then the MO coefficients at the reference cycle will be re-
     # normalized, and vice versa.
     #
     # Possible values: (True, False)
     mos_renorm: True
     # Normalize every transition-density overlap by the self-overlap norms of its
     # two states. This option changes numerical overlap values and is therefore
     # disabled by default for backwards compatibility. It is only used when
     # ovlp_type is tden.
     #
     # Possible values: (True, False)
     normalize_tden: False

By brief reasoning it would seem that :code:`mos_ref: ref` and :code:`mos_renorm: True` are
more sensible choices, which is possibly true. Right now the present defaults are kept for
legacy reasons, and I'll update them after testing out the alternatives.

Please also see :ref:`Example - Excited State Tracking <Plotting ES optimizations>`
for possible visualizations when optimizing ES.

Transactional ORCA 6.1.1 optimization with TDenTrack
-----------------------------------------------------

``TDenTrackORCA`` is an opt-in Python API for state-aware RFO steps.  Every
trial geometry is evaluated by an isolated, energy-only ORCA child calculation.
The child retains ``.cis``, BSON, GBW, input, and output artifacts in a unique
audit directory.  Exact cross-geometry AO integrals are evaluated from the two
BSON shell sets; the full signed transition-density overlap matrix and both
within-geometry Gram matrices are passed to TDenTrack for root and manifold
analysis.  They are carried as a complete metric-aware root-overlap block;
TDenTrack first detects a near-degenerate manifold from energies and scalar
scores, then takes the corresponding sub-block for principal-angle analysis.
The committed root changes only after a selected-root ``EnGrad``
calculation succeeds at the staged geometry.

The initial all-root calculation closes the bootstrap loop as follows::

   from excited_state_diabatizer.state_tracking import TrackingSession
   from pysisyphus.Geometry import Geometry
   from pysisyphus.calculators.ORCA import ORCA
   from pysisyphus.calculators.ORCA6StateSurvey import (
       bootstrap_tdentrack_snapshot,
   )
   from pysisyphus.calculators.TDenTrackORCA import TDenTrackORCA
   from pysisyphus.optimizers.RFOptimizer import RFOptimizer

   roots = tuple(range(1, 7))
   td_block = "%tddft nroots 6 triplets true tda false end"

   # Run once at the starting geometry. Do not put IRoot in this all-root job.
   seed = ORCA(
       keywords="uhf b3lyp def2-svp tightscf",
       blocks=td_block,
       root=None,
       nroots=len(roots),
       wavefunction_dump=True,
       keep_kind="all",
       out_dir="seed",
   )
   seed.get_all_energies(atoms, initial_cart_coords)
   initial = bootstrap_tdentrack_snapshot(
       seed,
       atoms,
       initial_cart_coords,
       selected_root=2,
       requested_roots=roots,
   )
   session = TrackingSession(initial)

   calc = TDenTrackORCA(
       keywords=seed.keywords,
       blocks=td_block,
       gbw=initial.artifacts["gbw"],
       root=initial.selected_root,
       nroots=len(roots),
       tracking_session=session,
       enable_default_survey=True,
       default_survey_options={
           "requested_roots": roots,
           "expected_orca_version": "6.1.1",
       },
       out_dir="optimization",
   )

   geom = Geometry(atoms, initial_cart_coords, coord_type="redund")
   geom.set_calculator(calc)
   opt = RFOptimizer(
       geom,
       step_controller={
           "type": "state_aware",
           # The factor-1 proposal is always surveyed first. These are used
           # only when that endpoint is mixed, weak, incomplete, or uphill.
           "factors": (0.5, 0.75, 1.0, 1.25, 1.5),
           "primary_factor": 1.0,
           "fallback_only": True,
           "require_descent": True,
       },
       out_dir="optimization",
   )
   opt.run()

Attach ``calc`` to the geometry and enable the optimizer's ``step_controller``
with the desired short/base/long factors.  This calculator is intentionally not
registered as a YAML calculator: a live ``TrackingSession`` and, for advanced
use, Python callback objects are required.

The normal RFO proposal remains inside its current trust radius. Factors above
one may bridge a narrow mixed region up to ``trust_max``. Going beyond that
global maximum is deliberately a separate opt-in: set
``respect_trust_max=False`` *and* provide a finite ``max_step_norm`` in the
step-controller mapping. Accepted fallback endpoints are ranked by state
energy, then overlap score and margin. The controller never commits an
intermediate geometry along the scaled proposal.

Post-step reparametrization and optimizer-specific implicit energy probes are
disabled while transactional control is active. This includes RFO line search,
GDIIS/GEDIIS, L-BFGS line search/regularization, and SQNM's pre-application and
line search. Their unsurveyed geometries would otherwise precede the
transaction boundary. A converged current geometry is detected before any new
all-root survey is launched.

Root numbering follows the reference type.  Closed-shell spin-adapted triplets
use **multiplicity-local ordinals** ``1..N``; root ``k`` maps to triplet
``IRoot k`` and ``IRootMult triplet``.  Unrestricted open-shell references use
the global printed ``STATE N`` ordinal because roots of different approximate
multiplicity can be interleaved in one TDDFT window.  Their per-root
multiplicities are retained from the CIS records so TDenTrack can reject an
incompatible spin manifold without renumbering the ORCA gradient root.  The
backend uses TDenTrack's ORCA 6 CIS parser for both conventions and for both
TDA and non-TDA ``X+Y`` amplitudes.  If a two-record file is exactly ambiguous
between a degenerate TDA pair and one non-TDA ``X+Y/X-Y`` pair, the input should
state ``tda true`` or ``tda false`` explicitly; otherwise the survey fails
closed.

For a closed-shell spin-adapted triplet gradient the adapter inserts and audits
both ``IRoot k`` and ``IRootMult triplet``. This matters in ORCA 6.1.1:
``Triplets true`` requests triplet roots but does not by itself make ``IRoot``
select the triplet block.  An unrestricted open-shell gradient instead uses
the global ``IRoot k`` without ``IRootMult``.
The retained input echo, ``DE(CIS)`` root marker, state-of-interest report, and
normal-termination marker must all agree before the electronic snapshot is
committed. ``FollowIRoot true`` and ``TGradList`` are rejected because native
root following or a multi-gradient job would make that authorization
ambiguous.

ORCA's printed TDDFT state table is formed from ``E(SCF)`` plus excitation
energies, whereas the final ``.engrad`` energy can also contain
state-independent contributions applied later, such as D3(BJ).  ORCA 6.1.1
does not use one invariant energy anchor for every TDA/TDDFT EnGrad path: the
``FINAL SINGLE POINT ENERGY``/``.engrad`` scalar can represent either the
selected excited state or the reference state even though the reported force
is for the requested excited root.  Bootstrap therefore parses the printed
numeric dispersion correction, applies that one common correction to the
whole root window, and audits whether the final scalar is anchored at root zero
or the requested root.  Energy-only excited-state jobs are checked against the
entire parsed state window because ORCA may make the first excited state,
rather than root zero, its final scalar.  The anchor identifies the scalar's
provenance; it does not choose the state followed by the optimizer.

Before a native gradient is returned, its raw scalar must match either the
audited final anchor or the selected-state energy at that geometry.  The raw
value remains in the retained ORCA files and is exposed as the calculator's
``last_orca_engrad_energy`` attribute, while the optimizer result contains only
the selected-state energy from the complete, same-geometry root survey and its
forces.  Excitation energies are unchanged, and optimizer descent checks and
fallback-step ranking remain on one corrected excited-state energy scale.  The
applied correction, final scalar, and anchor root are recorded in snapshot and
survey metadata.

When implicit solvent is active, ``CPCMEQ`` must be stated explicitly in the
``%tddft``/``%cis`` block. ORCA defaults a vertical energy-only calculation to
non-equilibrium LR-CPCM, but switches to equilibrium LR-CPCM when an analytic
gradient is requested. Since this backend deliberately alternates energy-only
all-root surveys and selected-root gradients, relying on those job-type
defaults would compare different excited-state surfaces. The backend therefore
fails before launching ORCA if CPCM/SMD is active and ``CPCMEQ`` is absent. A
relaxed excited-state optimization normally uses ``CPCMEQ true``; an explicitly
frozen-solvent workflow may choose ``false`` as long as every job uses it.

The committed reference must retain readable ``cis`` and ``bson`` artifact
paths; these paths are serialized in calculator restart data.  The built-in
backend currently assumes fixed atoms/order, basis, ECPs, multiplicity, and root
window.  Point charges or other per-call preparation arguments must be supplied
through ``default_survey_options["child_prepare_kwargs"]`` so the all-root
survey and gradient Hamiltonians remain identical.

Optimization of Conical Intersections
-------------------------------------

pysisyphus implements the `projected gradient method using
an updated branching plane`_, as developed
by Maede, Ohno and Morokuma. Currently, CI-optimization is not enabled for YAML input.
An illustrative example is found in *tests/test_conic_intersect*.

.. _projected gradient method using an updated branching plane: https://pubs.acs.org/doi/pdf/10.1021/ct1000268
