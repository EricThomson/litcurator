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

# E -- the PROMPT-BLAMED pool, and the only fixture here whose correct `blame` cannot be reached
# from the profile. Review articles, UNDER-scored. scenarios.JUDGE_PROMPT caps secondary
# literature below 0.55; the judge OBEYS that cap and lands them in the 0.25-0.54 band, and the
# user puts them at 0.80 because they are how he keeps up with fields next to his own.
#
# THE WRITTEN RULE IS WRONG, NOT IGNORED, and that is the whole design. The first version of this
# pool had the judge VIOLATING a stated rule, which invited consolidate to reason "the scoring
# procedure already covers this" and DISCARD the candidate -- it did exactly that in 2 of 3 runs
# on 2026-09-19, dropping five flags on the floor and taking pool-calibration red with it. That
# was not the model misbehaving; "already covered" was a defensible reading, and a case that needs
# an argument is a coin flip wearing a verdict. Here nothing is ignored, so there is nothing to
# call covered: the cap is simply set at the wrong value, and the only place to change it is the
# scoring procedure. Tests should be EASY -- not trivial, but unarguable.
#
# THE NOTES NEVER SAY WHERE THE RULE LIVES. If one said "the scoring prompt caps these", the model
# could answer blame='prompt' from the note alone and the check would pass just as well with the
# judge prompt withheld -- which is the configuration this pool exists to prove we have left.
# They state the complaint only, so reaching 'prompt' REQUIRES reading the cap.
#
# Graded against pool D (disease models, OVER-scored) rather than a second new pool: opposite
# directions cannot be fused by a generalization that happens to be true, and D is already
# calibrated and note-complete. FORMAT is the single shared property here, so the topics are
# deliberately spread across the profile's own interests and stay clear of B (formal theory).
_REVIEW_STUBS = [
    GEN.SyntheticPaper(
        "Coding of odour identity in the olfactory system: a review",
        "A review synthesising two decades of work on how odour identity is represented across "
        "receptor neurons, bulb and cortex, drawing together results from several laboratories.",
        "Annual Review of Neuroscience",
        "A secondary-literature summary rather than a study reporting new results.",
        "Reviews are how I keep up with the parts of the field I do not work in. Scored much too low."),
    GEN.SyntheticPaper(
        "Thalamocortical loops in perception: current understanding",
        "A survey of evidence on thalamic contributions to cortical sensory processing, "
        "reconciling competing accounts from anatomy, physiology and behavior.",
        "Trends in Neurosciences",
        "A synthesis of published findings; no new data are reported here.",
        "This kind of synthesis is worth more to me than most single papers. Far too low."),
    GEN.SyntheticPaper(
        "Cerebellar contributions to timing and prediction: a survey",
        "A review covering cerebellar involvement in predictive timing across motor and "
        "non-motor tasks, organised around the experimental findings from each preparation.",
        "Nature Reviews Neuroscience",
        "Secondary literature summarising results reported elsewhere.",
        "A good review saves me a month of reading. This should be near the top, not the bottom."),
    GEN.SyntheticPaper(
        "Hippocampal replay: what twenty years of recordings have shown",
        "A review of replay phenomena across sleep and wake, comparing findings across species "
        "and recording techniques and identifying unresolved discrepancies.",
        "Neuron",
        "A review article rather than primary research.",
        "Scored low again. I actively seek these out and the scores never reflect that."),
    GEN.SyntheticPaper(
        "Dendritic computation: a review of the evidence",
        "A survey of recorded dendritic nonlinearities across cell types, and what each "
        "measurement implies for how a single neuron integrates its inputs.",
        "Nature Reviews Neuroscience",
        "A summary of work published elsewhere, not a new experimental result.",
        "Another review buried. They are consistently rated far below where I would put them."),
]

# Under-scored by 0.30-0.45. The band matters here: `_scores` places an `under` judge in
# [0.25, 0.55], so every generated judge score sits at or below the 0.55 cap the synthetic judge
# prompt states, which is what makes "the judge obeyed a rule that is set wrong" true by
# construction rather than by luck.
POOLS_BY_INTENDED_PATTERN["E"] = GEN.IntendedPatternPool(
    "E", "under", (0.30, 0.45), papers=_REVIEW_STUBS)

# TOOLS and PLANTS -- the two pools of the note_decides_blame scenario, where every note SAYS
# which document the fix belongs in. They are the only pools whose notes do that, by design: in
# every other pool the note is silent on the matter so that blame has to be reached from the
# documents, and here the note is the whole point.
#
# TOOLS: new recording and imaging instruments, UNDER-scored. The scenario's own judge prompt
# caps methods papers below 0.55 (behaviors._METHODS_CAP), the judge obeys it, and the user
# wants them high -- the same shape as pool E, so the written rule would send this to the
# prompt. The notes say PROFILE, in four different ways. That is the override the scenario
# exists to test: on 2026-10-20 the live consolidate step blamed the prompt six times out of six
# for a pattern whose note said "we absolutely have to in profile", because it had found a
# judge-prompt rule it preferred to the user's words.
#
# The rationales describe the paper's NATURE ("a tool, not a finding"), never the cap, exactly
# as pool E's describe "a summary of others' results" without naming its cap. Pool-calibration
# puts TOOLS and E in one session, and two under-scored pools whose rationales both said
# "capped by a rule" would be one fusion away from a fixture red.
_TOOL_STUBS = [
    GEN.SyntheticPaper(
        "A wireless 1024-channel recording system for freely moving rats",
        "A head-mounted system digitises and transmits 1024 channels of extracellular signal for "
        "eight hours on one battery, validated by recording hippocampal units during open-field "
        "foraging; no new biological finding is claimed.",
        "Nature Methods",
        "The primary contribution is a recording system, not a finding about the brain.",
        "I follow new recording tools closely; they decide what experiments are possible next. "
        "This is a taste of mine and it belongs in my profile."),
    GEN.SyntheticPaper(
        "A head-mounted two-photon microscope for imaging during unrestrained behavior",
        "A 2.4 g miniaturised two-photon microscope images several hundred neurons at cellular "
        "resolution while mice run, groom and rear, with motion artefact corrected online; the "
        "paper characterises the instrument rather than reporting a result about the circuit.",
        "Nature Neuroscience",
        "The paper characterises an instrument rather than reporting a result about any circuit.",
        "Imaging during free behavior is exactly the kind of method I want to hear about the day "
        "it comes out. Add it to my profile, under what I follow."),
    GEN.SyntheticPaper(
        "Flexible high-density probes record the same neurons for over a year",
        "Polymer probes with 512 sites per shank track single units across fourteen months of "
        "chronic implantation in mice, with yield and stability quantified against rigid silicon "
        "probes.",
        "Neuron",
        "A new recording technology; no biological question is answered.",
        "Recording the same cells for a year changes what questions can be asked. This goes in "
        "my profile, not into a scoring rule."),
    GEN.SyntheticPaper(
        "Sub-millisecond closed-loop optogenetic control from spike detection to light delivery",
        "A field-programmable hardware pipeline detects spikes and triggers patterned "
        "illumination within 0.4 ms, benchmarked on cerebellar recordings; the work is an "
        "engineering specification with no new circuit finding.",
        "Journal of Neural Engineering",
        "An engineering specification with no new circuit finding.",
        "Closed-loop tools are something I want to see every time one appears. Profile, please: "
        "this is about what I like, not about how to score."),
]

# Same band as pool E, for the same reason: `_scores` places an `under` judge in [0.25, 0.55],
# so every judge score sits at or below the 0.55 cap and "the judge obeyed the rule" is true by
# construction.
POOLS_BY_INTENDED_PATTERN["TOOLS"] = GEN.IntendedPatternPool(
    "TOOLS", "under", (0.30, 0.45), papers=_TOOL_STUBS)

# PLANTS: plant physiology with borrowed neuroscience vocabulary (electrical signalling, sensory
# cues, movement), OVER-scored because the judge matched the words. Neither document mentions
# plants, so the written rule would call this a profile gap; the notes say JUDGE PROMPT, in four
# different ways, and they say WHY in the user's own terms: he will not add "no plants" to his
# profile, the judge should check for a nervous system before matching. Here the note and the
# principle agree (the fix is topic-agnostic reasoning), so this pool is the discrimination
# partner that stops "always say profile" from passing, not a second override; the control
# script in sandbox/blame_fixture/ makes that explicit.
#
# Laughably off-scope on purpose (rule 1, easy cases): no neurons, no animal, nothing a
# neuroscience profile could be read as wanting. Far from CANCER as well, whose notes say the
# opposite thing ("disinterest list" = profile).
_PLANT_STUBS = [
    GEN.SyntheticPaper(
        "Electrical signalling coordinates rapid leaf folding in Mimosa pudica",
        "Surface recordings show action-potential-like depolarisations propagating along the "
        "petiole ahead of leaf folding; blocking ion channels pharmacologically abolishes the "
        "movement, establishing the signal as necessary.",
        "Current Biology",
        "A sensory stimulus drives a rapid movement through a propagating electrical signal that "
        "is shown to be necessary: a causal sensory-to-behavior pathway, which the profile asks for.",
        "Plants have no neurons. The judge saw 'sensory', 'electrical signalling' and 'movement' "
        "and matched on the words. That is a reasoning failure; fix it in the judge prompt. I am "
        "not adding 'no plants' to my profile."),
    GEN.SyntheticPaper(
        "Calcium waves encode wound location in Arabidopsis leaves",
        "Genetically encoded indicators reveal calcium waves that travel from a wound site to "
        "distant leaves within minutes and scale with damage; glutamate-receptor-like channel "
        "mutants lack the wave.",
        "Science",
        "A spatially coded signal propagating from a sensory event, with a channel knockout as "
        "the causal test, in a leading journal.",
        "This is plant physiology. Nothing here is a nervous system, however neuroscience-flavoured "
        "the vocabulary is. The judge prompt needs to say that borrowed words are not a topic "
        "match; that is where this fix goes."),
    GEN.SyntheticPaper(
        "Root tips integrate gravity and moisture cues to steer growth direction",
        "Time-lapse imaging and microfluidic gradients show roots weighting two directional cues, "
        "with auxin transport mutants failing to integrate them.",
        "Nature Plants",
        "Multisensory integration steering a directed movement, tested with mutants: it reads as "
        "sensory decision-making.",
        "Roots growing toward water is not behavior in my sense of the word. The judge reasoned by "
        "analogy from the profile instead of asking whether there is a nervous system at all. A "
        "judge-prompt problem, not a gap in my profile."),
    GEN.SyntheticPaper(
        "Venus flytrap counts touches to trigger digestion",
        "Mechanical stimulation of trigger hairs elicits action potentials; two within thirty "
        "seconds close the trap and five induce digestive enzymes, with hormone signalling "
        "measured at each count.",
        "Current Biology",
        "Mechanosensory stimuli produce counted electrical events that gate a motor response, a "
        "clean stimulus-to-response mechanism.",
        "Charming, and not neuroscience: there is no neuron anywhere in it. Make the judge check "
        "for a nervous system before it matches my interests. That belongs in the judge prompt."),
]

# Over-scored by 0.25-0.40: `_scores` puts the judge in [0.50, 0.85], above the surfacing line,
# and the user well below it.
POOLS_BY_INTENDED_PATTERN["PLANTS"] = GEN.IntendedPatternPool(
    "PLANTS", "over", (0.25, 0.40), papers=_PLANT_STUBS)

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
