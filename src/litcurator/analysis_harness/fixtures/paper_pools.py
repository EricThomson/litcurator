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


# TIDAL -- the NOTE-CARRIER probe. Every other pool here encodes its signal in the delta: a big
# number means the judge was badly wrong. This one encodes it in the NOTE, with a deliberately
# SMALL delta, because that is the archetype the fixture was missing entirely and the live data
# is full of. About 30% of the user's real flags sit at |delta| <= 0.15, and they are small
# precisely BECAUSE the score is not what is being corrected -- the flag exists to carry a
# sentence ("put this on my disinterest list", "the mismatch reasoning is bad"). Before this
# pool the fixture's smallest |delta| anywhere was 0.17, so nothing tested a flag whose evidence
# lives in the prose.
#
# THE NOTES ARE THE FIXTURE HERE, and they are built to be graded discretely so ONE run settles
# it (never three runs and an average -- that would mean the case is not easy enough). Each note
# plants three markers that fail at different depths:
#   "circatidal"        the topic word. Survives almost any faithful handling; if this is gone
#                       the user's language never left the cluster step at all.
#   "disinterest list"  the DESTINATION. The user is not describing a taste, they are naming
#                       where the fix goes. This is the actionable half.
#   "tide-table biology" the user's own idiosyncratic phrase. Nothing paraphrases to this, so it
#                       is the sharpest test of whether their VOICE arrives or gets restated.
# All three are unusual enough to have no natural synonym, which is what makes a substring test
# honest rather than a proxy for prose quality.
# DELIBERATELY VERTEBRATE (fish), and that is a fixture-design constraint, not a detail. The
# first draft of this pool used crabs, isopods and amphipods -- which collides head-on with pool
# C, invertebrate neuroethology. Two pools that can be confused for each other produce a red
# that is about the FIXTURE rather than the machinery, which is the exact failure
# pool_calibration exists to catch. Marine fish keep the tidal setting while staying clear of
# C (invertebrates), A (non-invasive human), B (formal theory) and D (disease models).
_TIDAL_STUBS = [
    GEN.SyntheticPaper(
        "Circatidal rhythms persist in isolated killifish gill explants",
        "Explanted gill tissue from the mangrove killifish maintains a ~12.4 h transcriptional "
        "cycle for six days in constant conditions, establishing a peripheral circatidal "
        "oscillator that runs without any input from the nervous system.",
        "Nature",
        "A clean rhythmic-timing result with a well-controlled free-running design, in a "
        "vertebrate the profile has no stated objection to.",
        "Tide-table biology, not neuroscience -- there is no circuit here at all. Put circatidal "
        "rhythm work on my disinterest list."),
    GEN.SyntheticPaper(
        "A molecular clock for circatidal timing in the reef goby",
        "Transcriptomic profiling across the tidal cycle identifies an oscillating gene module "
        "whose period is uncoupled from the circadian clock, with knockdown abolishing the "
        "12.4 h rhythm in gill epithelium.",
        "Cell",
        "Chronobiology with a clear molecular mechanism and a convincing knockdown, in a "
        "non-standard vertebrate model.",
        "Again tide-table biology. Peripheral tissue, no behavior, no circuit. This belongs on "
        "the disinterest list with the rest of the circatidal work."),
    GEN.SyntheticPaper(
        "Entrainment of circatidal swimming by hydrostatic pressure cycles in juvenile plaice",
        "Fish held under cyclic pressure adopt a 12.4 h swimming rhythm that free-runs for four "
        "cycles, identifying pressure as a sufficient zeitgeber for the circatidal oscillator.",
        "Current Biology",
        "A behavioral entrainment study showing an environmental signal driving behavior, which "
        "the profile states an interest in.",
        "Entrainment studies are still tide-table biology to me. Circatidal work goes on the "
        "disinterest list even when there is behavior attached."),
    GEN.SyntheticPaper(
        "Two independent oscillators time circatidal and circadian behavior in Atlantic salmon smolts",
        "Behavioral recording under constant conditions separates a 12.4 h and a 24 h component "
        "with distinct temperature compensation, arguing for two anatomically separate clocks.",
        "Neuron",
        "Dissociating two oscillators is a systems-level organizational claim, and the profile "
        "favours organizational results of this kind.",
        "Still tide-table biology. Add circatidal rhythms to the disinterest list; I do not want "
        "these no matter how clean the dissociation is."),
]


# Meaning-equal restatements of the judge's wrong reason, rotated so the repeats a short run
# produces are not verbatim. Four papers over six emissions means papers 0 and 1 come round
# twice, and handing the cluster step an identical title, abstract AND rationale would make the
# grouping easier than reality -- which matters even here, where the question is note survival
# rather than coalescing, because a trivially clean cluster is a friendlier place for a quote to
# survive than a messy one. Same reasoning as _CONNECTOME_RATIONALES; see the arithmetic note
# there before changing the count.
_TIDAL_RATIONALES = [
    "Rhythmic timing in a marine invertebrate, with a clean oscillator result. The profile's "
    "interest in invertebrate preparations and in how environmental signals drive behavior "
    "makes this worth surfacing.",
    "A well-controlled chronobiology study in a tractable non-model organism. Nothing here "
    "conflicts with the stated disinterests, and the organism is one the profile favours.",
]


def tidal_pool(delta_band=(0.08, 0.14)):
    """The NOTE-CARRIER probe, TIDAL -- see the block above for why the notes are the fixture.

    The band is deliberately BELOW every other pool in this file (the next smallest is
    connectome at 0.16) and inside the user's real 0.07-0.15 region. `over` because the judge
    is scoring these too high; the user wants them gone, and says so in words rather than by
    moving the number far."""
    return GEN.IntendedPatternPool("TIDAL", "over", delta_band,
                                   papers=list(_TIDAL_STUBS),
                                   rationale_templates=list(_TIDAL_RATIONALES))


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
