"""
paper_pools.py -- pools of synthetic papers that a scenario draws from, one pool per intended
pattern.

A scenario that runs many sessions needs a supply of papers to emit each session. That supply is
a POOL: the papers for one intended pattern, plus the direction and delta size its flags should
carry. scenario_gen cycles a pool to emit flags session after session.

Compare scenarios.py, which is the other place synthetic papers live: that file holds one
hand-authored STORY (specific papers in a specific order, with the human's decisions and the
expectations between them). This file holds reusable supply.

The A/B/C/D pools are derived from scenarios.SESSIONS, grouped by label, so they cost nothing to
maintain: those papers are already validated by the lifecycle fixture grouping them correctly.
Streaming one just re-scores from its band and cycles the pool for variation. Direction and
|delta| band are inferred from the papers' own (user - judge) deltas, so a streamed pattern keeps
the same sign and magnitude as the hand-authored fixture.

Derived from SESSIONS rather than from the _SESSION_N literals, so adding a session to the
fixture feeds the pools automatically. It used to name the literals one by one and stopped at
_SESSION_3, so a fourth session was added and silently reached none of the streamed scenarios.

U1/U2 are single-paper unicorns that scenarios place directly rather than draw, so they get no
pool. Pools a scenario needs but the lifecycle fixture does not supply (the weak-signal probe)
are hand-authored below.
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


def _pools_from_fixture():
    by_label = defaultdict(list)
    for session in SC.SESSIONS:
        for paper in session["papers"]:
            by_label[paper["intended"]].append(paper)
    out = {}
    for label, papers in by_label.items():
        if label in ("U1", "U2"):                    # single-paper unicorns, never streamed
            continue
        direction, band = _direction_band(papers)
        out[label] = GEN.IntendedPatternPool(label, direction, band,
                                             papers=[_stub(p) for p in papers])
    return out


# A (non-invasive human, over), B (formal theory, under), C (invertebrate, under),
# D (disease-model, over) -- inferred from the fixture rows.
POOLS_BY_INTENDED_PATTERN = _pools_from_fixture()


# --- pools the lifecycle fixture does not supply ---------------------------------------------

# W -- the weak-signal PROBE: structural connectomics / wiring diagrams with no functional or
# behavioral readout. Under-scored (the profile wants circuit->behavior function, which these
# lack) but only MILDLY preferred by the user -> a small delta, the knob the accumulation sweep
# turns. Deliberately vertebrate, to stay clear of the invertebrate intended pattern C. The four
# rationales
# are meaning-equivalent ("structure without function") so the intended pattern coheres by
# meaning.
# NOTES ADDED 2026-08-07, and the absence was a real fixture bug rather than an oversight.
# This pool ran twelve sessions with ZERO notes on any flag, because the scenario was built
# while everyone believed a flag was essentially a number and notes were rare. Both are false:
# every one of the user's live January flags carries a note, and the SMALL-delta ones -- which
# is what this pool emits -- are the MOST likely to carry a substantial one, since a flag whose
# score barely moves usually exists to carry the sentence. So the flagship long-horizon
# scenario was running on the one flag shape the user does not produce.
#
# They are kept SHORT and same-meaning on purpose. The point of this pool is still a WEAK
# signal, and the weakness has to stay in the recurrence -- one flag a session, gathering
# slowly -- rather than being cancelled by handing the model four emphatic paragraphs.
_CONNECTOME_STUBS = [
    GEN.SyntheticPaper(
        "A synaptic-resolution wiring diagram of the mouse retina",
        "Serial electron microscopy reconstructs every synapse across the inner plexiform layer, "
        "yielding a complete connectivity matrix of bipolar, amacrine and ganglion cells, with no "
        "physiological recording.",
        "Nature",
        "A dense structural reconstruction with no functional recording or behavioral test.",
        "Wiring alone does not tell me what the circuit does."),
    GEN.SyntheticPaper(
        "Connectomic reconstruction of a cortical column",
        "Automated segmentation of a teravoxel EM volume maps the synaptic wiring of layer 2/3 "
        "pyramidal cells and interneurons, quantifying connection probabilities by cell type.",
        "Cell",
        "An anatomical wiring map; the profile values circuit-behavior function, which is absent here.",
        "Another map with nothing recorded. I want function."),
    GEN.SyntheticPaper(
        "The complete connectome of the zebrafish spinal locomotor network",
        "Electron-microscopic reconstruction resolves the full set of synaptic connections among "
        "motor neurons and interneurons of a spinal segment, with no activity measured.",
        "Neuron",
        "A structural connectome without any physiology or behavior.",
        "Structure without physiology is only half a result for me."),
    GEN.SyntheticPaper(
        "Synaptic wiring of the mouse hippocampal CA3 recurrent network",
        "Volumetric EM reconstruction quantifies recurrent excitatory connectivity among CA3 "
        "pyramidal cells at synaptic resolution, a static anatomical map.",
        "Nature Neuroscience",
        "Connectivity mapping alone; no functional or behavioral readout, which the profile emphasizes.",
        "Anatomy again, no behavior. Mild but it keeps happening."),
]


# Meaning-equal restatements of the four stubs' complaint, rotated across emissions by _emit so a
# long run does not hand the cluster step the same rationale verbatim over and over.
#
# The COUNT is load-bearing. _emit picks paper `seq % 4` and rationale `seq % len(rationales)`, so
# if those two periods share a factor the same paper always draws the same rationale and rotating
# buys nothing. Four papers and THREE rationales (the paper's own plus these two) have lowest
# common multiple 12, so across a 12-session run every paper-rationale pairing appears exactly
# once. Keep that coprime relationship in mind before adding a third template.
_CONNECTOME_RATIONALES = [
    "Anatomy without function: the wiring is mapped in detail, but nothing is recorded and nothing "
    "is perturbed, and the profile asks for a circuit mechanism tied to behavior.",
    "A static structural dataset. Impressive in scale, but with no activity measurement and no "
    "behavioral readout there is nothing linking the connectivity to what the circuit does.",
]


# NOT NEURAL AT ALL, and the bluntness IS the design -- rule 1, easy cases only.
#
# Three earlier drafts violated it and each took a run to disprove: circatidal rhythms in crabs
# (collides with pool C, invertebrate neuroethology, which this fixture declares the user WANTS),
# the same in fish (no collision, but tidal oscillators driving behavior are genuinely
# interesting -- the user's reaction to the word was "instantly interested"), and thermoregulation
# circuits (optogenetics plus behavior plus causal manipulation, i.e. a paper he would want to
# read). On the fish version the model never once produced the declared direction across two
# runs: `under`, then `judge-not-applying` (a value since deleted).
#
# The lesson: for `over`, pick something laughably off-scope. It is synthetic data -- there is no
# reason to hunt for a close call. Cancer cell biology has no neurons, no circuit, no behavior
# and no organism the profile cares about. Nothing to weigh. The judge over-scores it on generic
# quality signals (glam venue, clean knockouts), which is a real failure mode and exactly why
# the flag is a note-carrier: the score is only mildly wrong and the NOTE says what is going on.
#
# NB on markers: the DESTINATION word ("disinterest") is load-bearing because it appears ONLY in
# the note. A topic word would be near-vacuous -- the model reaches for "cancer" from the titles
# alone. The idiosyncratic-phrase check was dropped after flipping green/red across two runs;
# whether a model quotes a colourful phrase is genuinely variable, which is rule 1 again.
_CANCER_STUBS = [
    GEN.SyntheticPaper(
        "A MYC-driven transcriptional switch controls proliferation in colorectal tumour lines",
        "CRISPR knockout of a MYC cofactor in three colorectal cancer cell lines collapses a "
        "proliferative gene programme, and re-expression restores it; xenograft growth is "
        "reduced by two thirds.",
        "Nature",
        "A causal, mechanistically clean result in a top-tier journal, with knockout and rescue "
        "both demonstrated.",
        "This is cancer biology. Not neuroscience at all. Disinterest list."),
    GEN.SyntheticPaper(
        "KRAS-G12C inhibitor resistance arises through adaptive RTK feedback signalling",
        "Time-course phosphoproteomics in treated lung adenocarcinoma cells identifies a "
        "feedback wave restoring pathway flux within 48 hours; combination treatment prevents it.",
        "Cell",
        "Careful causal pharmacology with a strong mechanism and an obvious translational "
        "payoff, in an excellent venue.",
        "Oncology drug resistance. Nothing to do with me. Disinterest list."),
    GEN.SyntheticPaper(
        "An enhancer hijacking event activates a proto-oncogene in pancreatic carcinoma",
        "Hi-C and ATAC-seq in patient-derived organoids map a structural rearrangement that "
        "places a distal enhancer next to the oncogene; CRISPRi of the enhancer abolishes "
        "expression.",
        "Science",
        "Gene-regulatory mechanism established causally in patient-derived material, a strong "
        "result in a leading journal.",
        "Tumour genomics. Wrong field entirely. Put it on the disinterest list."),
    GEN.SyntheticPaper(
        "p53 restoration triggers senescence rather than apoptosis in hepatocellular carcinoma",
        "Inducible p53 re-expression in a mouse liver tumour model produces a senescent "
        "phenotype with a characteristic secretory profile, and clearance depends on innate "
        "immune recruitment.",
        "Nature",
        "A classic tumour-suppressor question answered with a clean inducible system in vivo.",
        "Cancer again. I do not want any of this. Disinterest list."),
]


# Meaning-equal restatements of the judge's WRONG reason, rotated across emissions so a repeat
# is not verbatim. All say the same thing: it is good science in a good journal, judged on
# generic quality rather than on whether the user cares about the subject.
_CANCER_RATIONALES = [
    "Rigorous causal mechanism in a leading journal, with knockout and rescue. The quality of "
    "the work is not in question.",
    "A well-controlled molecular study with a clear mechanistic claim, published in a venue the "
    "user rates highly.",
]


def cancer_pool(delta_band=(0.08, 0.14)):
    """The NOTE-CARRIER probe, CANCER -- see the block above for why the topic is blunt.

    The band is deliberately BELOW every other pool here (the next smallest is connectome at
    0.16) and inside the user's real 0.07-0.15 region. `over` because the judge scores these too
    high; the user wants them gone and says so in words rather than by moving the number far,
    which is what a note-carrier flag IS."""
    return GEN.IntendedPatternPool("CANCER", "over", delta_band,
                                   papers=list(_CANCER_STUBS),
                                   rationale_templates=list(_CANCER_RATIONALES))


def connectome_pool(delta_band=(0.16, 0.22)):
    """The weak-signal probe, CONNECTOME. `delta_band` is the accumulation sweep knob.

    Only four papers, and the long-horizon scenarios run twelve sessions or more, so _emit cycles
    them: without rotation each paper would be emitted three times with an identical title,
    abstract and rationale, differing only in pmid and a little score jitter. Handing the cluster
    step literal duplicates makes coalescing easier than it would be in reality, which would make
    a long-horizon green mean less than it looks."""
    return GEN.IntendedPatternPool("CONNECTOME", "under", delta_band,
                                     papers=list(_CONNECTOME_STUBS),
                                     rationale_templates=list(_CONNECTOME_RATIONALES))
