"""
banks.py -- the synthetic papers, grouped by intended pattern for the scenario generator.

The A/B/C/D paper sets are built from scenarios.SESSIONS, grouped by label. Those papers are
already cluster-validated (the lifecycle fixture groups them correctly), so they arrive
pre-calibrated; streaming them just re-scores from the intended pattern band and cycles the
bank for variation. Direction and |delta| band are inferred from the papers' own (user - judge)
deltas, so a streamed intended pattern keeps the same sign and magnitude as the hand-authored
fixture.

Read from SESSIONS rather than from the _SESSION_N literals, so that adding a session to the
fixture feeds the banks automatically. It used to name the literals one by one and stopped at
_SESSION_3, so a fourth session was added to the lifecycle fixture and silently reached none of
the streamed scenarios.

U1/U2 are single-paper unicorns the scenarios place directly, not papers_emitted, so they are not
banks here.

New paper sets the long-horizon behaviors need (a weak-signal probe, and one-off clutter)
are authored below as they come online, in the same shape.
"""

from collections import defaultdict

from . import scenarios as SC
from . import scenario_gen as GEN


def _stub(paper):
    return GEN.SyntheticPaper(paper["title"], paper["abstract"], paper["journal"],
                              paper["rationale"], paper["note"] or "")


def _direction_band(papers):
    deltas = [p["user_score"] - p["judge_score"] for p in papers]
    mags = [abs(d) for d in deltas]
    direction = "under" if sum(deltas) > 0 else "over"
    return direction, (round(min(mags), 2), round(max(mags), 2))


def _paper_sets_from_fixture():
    by_label = defaultdict(list)
    for session in SC.SESSIONS:
        for paper in session["papers"]:
            by_label[paper["intended"]].append(paper)
    out = {}
    for label, papers in by_label.items():
        if label in ("U1", "U2"):                    # single-paper unicorns, never streamed
            continue
        direction, band = _direction_band(papers)
        out[label] = GEN.IntendedPatternPapers(label, direction, band,
                                               papers=[_stub(p) for p in papers])
    return out


# A (clinical EEG, over), B (computational theory, under), C (invertebrate, under),
# D (disease-model, over) -- inferred from the fixture rows.
PAPERS_BY_INTENDED_PATTERN = _paper_sets_from_fixture()


# --- new paper sets for the long-horizon behaviors -------------------------------------------

# W -- the weak-signal PROBE: structural connectomics / wiring diagrams with no functional or
# behavioral readout. Under-scored (the profile wants circuit->behavior function, which these
# lack) but only MILDLY preferred by the user -> a small delta, the knob the accumulation sweep
# turns. Deliberately vertebrate, to stay clear of the invertebrate intended pattern C. The four
# rationales
# are meaning-equivalent ("structure without function") so the intended pattern coheres by
# meaning.
_CONNECTOME_STUBS = [
    GEN.SyntheticPaper(
        "A synaptic-resolution wiring diagram of the mouse retina",
        "Serial electron microscopy reconstructs every synapse across the inner plexiform layer, "
        "yielding a complete connectivity matrix of bipolar, amacrine and ganglion cells, with no "
        "physiological recording.",
        "Nature",
        "A dense structural reconstruction with no functional recording or behavioral test."),
    GEN.SyntheticPaper(
        "Connectomic reconstruction of a cortical column",
        "Automated segmentation of a teravoxel EM volume maps the synaptic wiring of layer 2/3 "
        "pyramidal cells and interneurons, quantifying connection probabilities by cell type.",
        "Cell",
        "An anatomical wiring map; the profile values circuit-behavior function, which is absent here."),
    GEN.SyntheticPaper(
        "The complete connectome of the zebrafish spinal locomotor network",
        "Electron-microscopic reconstruction resolves the full set of synaptic connections among "
        "motor neurons and interneurons of a spinal segment, with no activity measured.",
        "Neuron",
        "A structural connectome without any physiology or behavior."),
    GEN.SyntheticPaper(
        "Synaptic wiring of the mouse hippocampal CA3 recurrent network",
        "Volumetric EM reconstruction quantifies recurrent excitatory connectivity among CA3 "
        "pyramidal cells at synaptic resolution, a static anatomical map.",
        "Nature Neuroscience",
        "Connectivity mapping alone; no functional or behavioral readout, which the profile emphasizes."),
]


def connectome_paper_set(delta_band=(0.16, 0.22)):
    """The weak-signal probe, CONNECTOME. `delta_band` is the accumulation sweep knob."""
    return GEN.IntendedPatternPapers("CONNECTOME", "under", delta_band, papers=list(_CONNECTOME_STUBS))


# ONE_OFF -- the one-off clutter stream: unrelated single flags, each a DIFFERENT topic with its
# own rationale, so they never cohere into an intended pattern.
#
# Keep every rationale here clear of the language the real paper sets use, especially
# CONNECTOME's.
# Two of these used to read "Descriptive anatomy without a functional or behavioral link" and
# "Developmental wiring molecules" -- which is CONNECTOME's complaint almost word for word, so
# the cluster step correctly grouped them with connectomics and the ground truth then scored that
# correct behavior as contamination. A clutter paper that plausibly belongs to a real intended
# pattern is not
# clutter; it makes the gate a coin flip instead of a verdict.
#
# They grow the unattached pool (the robustness
# stressor). Small sub-threshold under-deltas: mild single corrections, not strong signal. Each
# may
# legitimately become its own single-paper pattern; the only defect would be several of them
# spuriously grouped together, which no scenario currently tests.
_ONE_OFF_STUBS = [
    GEN.SyntheticPaper(
        "Astrocyte calcium waves shape cortical slice excitability",
        "Two-photon imaging of astrocytic calcium in acute cortical slices reveals slow waves that "
        "modulate local excitability.",
        "Glia",
        "Glial physiology, which the profile explicitly does not follow."),
    GEN.SyntheticPaper(
        "A compact three-photon microscope for deep-tissue imaging",
        "An optical design achieving deeper penetration at lower average power for in vivo "
        "three-photon microscopy.",
        "Optics Express",
        "An instrumentation/methods paper with no circuit or behavioral question."),
    GEN.SyntheticPaper(
        "Neurovascular coupling dynamics in mouse visual cortex",
        "Simultaneous imaging of blood flow and neural activity quantifies the hemodynamic response "
        "lag across cortical layers.",
        "Journal of Cerebral Blood Flow and Metabolism",
        "Vascular coupling, not the neural computation the profile centers on."),
    GEN.SyntheticPaper(
        "Thermal tolerance limits of a reef fish under acute warming",
        "Critical thermal maxima are measured across acclimation temperatures to estimate the "
        "species' warming safety margin.",
        "Journal of Experimental Biology",
        "Comparative physiology and thermal ecology, with no nervous system question."),
    GEN.SyntheticPaper(
        "Microglial activation markers after traumatic brain injury",
        "Histological quantification of microglial morphology tracks the inflammatory time course "
        "after controlled cortical impact.",
        "Journal of Neuroinflammation",
        "Neuroinflammation pathology, outside the profile's interests."),
    GEN.SyntheticPaper(
        "Kinetics of synaptic vesicle endocytosis at a central synapse",
        "Membrane capacitance measurements resolve fast and slow modes of vesicle retrieval after "
        "stimulation at the calyx of Held.",
        "Journal of Physiology",
        "Presynaptic membrane cell biology, not a circuit or behavioral question."),
    GEN.SyntheticPaper(
        "Pharmacokinetics of intranasal oxytocin in rhesus macaques",
        "Plasma and cerebrospinal fluid concentrations are tracked over six hours after intranasal "
        "dosing to establish a delivery profile.",
        "Journal of Pharmacology and Experimental Therapeutics",
        "Drug delivery and pharmacokinetics, with no circuit or behavioral question at issue."),
    GEN.SyntheticPaper(
        "Transcriptional rhythms of clock genes in the suprachiasmatic nucleus",
        "Bioluminescent reporters track circadian oscillations of Per and Cry transcription in SCN "
        "explants under constant conditions.",
        "Journal of Biological Rhythms",
        "Chronobiology of gene transcription, off the profile's circuit-computation focus."),
]


def one_off_paper_set(delta_band=(0.04, 0.11)):
    """The ONE_OFF clutter intended pattern: topically diverse single flags used to grow the unattached pool.
    Deltas kept clearly SUB-threshold (<0.15) so each renders in the 'roughly agreed / context'
    bucket -- a mild one-off correction, not a strong under-scoring."""
    return GEN.IntendedPatternPapers("ONE_OFF", "under", delta_band, papers=list(_ONE_OFF_STUBS))
