"""
scenarios.py -- the synthetic test fixture for the profile analysis machinery, and its spec.

Read this top to bottom before touching the harness. It defines, in plain language, what
"correct" behavior even is -- both within one session and across sessions -- which is
written down nowhere else. The expectations near the bottom are executable, so this spec
cannot quietly drift out of date the way a prose doc would.

------------------------------------------------------------------------------
HOW LITCURATOR WORKS (the whole vocabulary this file uses)
------------------------------------------------------------------------------

PROFILE   Your written description of what you want to read. Plain prose you author.

JUDGE     An LLM that reads a paper plus your profile and scores it 0 to 1: how much you
          would care about it. You read the ones near the top.

FLAG      When the judge gets one wrong, you record your own score for that paper. A flag
          is ONE paper: the paper, the judge's score, your score, and the gap between them
          (the delta). A single correction, and a real row in the database.

PATTERN   A recurring gap that a GROUP of flags share, e.g. "the judge keeps under-scoring
          invertebrate work". The analysis machinery reads a batch of flags and groups
          them into patterns; you then decide what to do with each. A pattern is either:
            OPEN    -- surfaced, still awaiting your decision.
            CLOSED  -- you have decided. Either INCORPORATED (you edited it into your
                       profile) or REJECTED (not a real gap). A closed pattern drops off
                       your to-do list but stays on record forever, so the machinery never
                       re-suggests it.

That is the whole vocabulary: profile, judge, flag, pattern (open or closed). 

------------------------------------------------------------------------------
WHAT THIS HARNESS DOES
------------------------------------------------------------------------------
Building a test harness here is harder than for the judge. 

Here, a SET of flags becomes a (different) SET
of patterns PLUS changes to permanent memory, and the output of review session 3 depends on what 
happened in sessions 1 and 2. So this is a multi-round scenario, with scripted human decisions between rounds.

To make it work, we are using synthetic data. Synthetic articles, a synthetic profile,
and we are synthesizing flags ahead of time. Each synthetic flag is
labeled with the intended PATTERN it belongs to -- the correct grouping, the "right" answer.

After each round we compare the PRODUCED patterns (what the machinery actually made) with
the INTENDED patterns, purely from the provenance graph (which flags ended up attached to which
pattern), never from the model's wording. That keeps every metric objective and hard to
game: you cannot tune a prompt to merely sound right, because the score is about which
flags landed where.

    intended pattern  = a gap we wrote into the synthetic fixture (ground truth)
    produced pattern  = a pattern the machinery generated during profile analysis (what we grade)

------------------------------------------------------------------------------
WHAT GOOD LOOKS LIKE
------------------------------------------------------------------------------

Within one round, given a few clearly distinct intended patterns:
  COVERAGE        every intended pattern shows up in at least one produced pattern
  FRAGMENTATION   about one produced pattern per intended one, not several
  PURITY          a produced pattern's flags come mostly from a single intended pattern
  PRIORITY        only a couple of patterns marked act_now -- the rest can wait
  N=1 THAT COUNTS the two unicorns below: a single paper that IS worth a pattern

Across sessions 
  PRESERVATION    a pattern you carried forward is still there next round, not duplicated
  MERGE           new flags of an already-open pattern attach to it (its flag count grows)
                  instead of creating a second pattern for the same gap
  RECURRENCE      a CLOSED pattern that keeps coming back is caught as a recurrence on
                  that SAME pattern -- never duplicated as a new one, never silently reopened.
                  It logs a `recurred` mark; the pattern stays closed. The recurrence is the
                  point: a decision that will not stick is a signal.
                    rejected + recurs      -> maybe the rejection was wrong; reconsider it.
                    incorporated + recurs  -> the profile edit did not take (you stated the
                                              interest, the judge still ignores it).
  ADDITION        a genuinely new gap produces exactly one new pattern
  TERMINATION     the open pile grows ONLY by genuinely-new gaps, so it converges instead
                  of ratcheting up every session

------------------------------------------------------------------------------
THE UNICORNS (the sharpest test here)
------------------------------------------------------------------------------

A pattern does NOT need multiple papers. One paper can be enough when it names a clean,
bounded category and you left a note saying so ("add this to my disinterest list"). Sitting
on it is EXPENSIVE, not cheap: the category may not reappear for months, so the signal is
just lost.

The two unicorns (U1, U2) are single papers that must each earn a pattern. A single-paper
pattern is fine; the only real mistakes are the opposite ones -- fragmenting a real
multi-paper intended pattern into singles, or fusing unrelated one-offs into one pattern. What makes a
unicorn worth a pattern is that it names a clean, bounded category (usually with a note).

------------------------------------------------------------------------------
THE ARC
------------------------------------------------------------------------------

Session 1  Gaps A, B, C appear for the first time, plus one unicorn.
         Scripted human afterward: incorporate A, reject B, carry C.
Session 2  B comes back (still rejected), C comes back (carried), a new gap D arrives, plus a
         second unicorn. A is absent -- simulating an edit that actually worked.
         Scripted human afterward: carry D.
Session 3  A comes back (the incorporated gap returning = the edit did NOT take), plus more D.
         The open pile must stop growing.
         Scripted human afterward: incorporate C -- carried since session 1, finally acted on.
         That carried-to-incorporated step is a decision CHANGING on a pattern that already
         has a history, which nothing else here covers.
Session 4  Nothing new at all: B returns a SECOND time (still rejected) and C returns for the
         first time since being incorporated. No new gap, no unicorn, nothing to discover.
         This session exists only to ask the questions that need a long history:
           - does a second return land on the SAME closed pattern, so the count accumulates
             into "you rejected this and it has now come back twice"?
           - with every gap already tracked and nothing new to find, does the open pile stop
             growing, or does the machinery invent work to do?
         A session with no new information is the sharpest test of memory there is.
"""

# The profile every session is scored against. Deliberately SILENT on all the intended gaps,
# so each reads as a genuine gap rather than "the judge is ignoring text already here".
PROFILE = """
I follow systems and circuit neuroscience: how neural circuits compute, and how that
computation drives behavior. I care most about work that ties a circuit mechanism to a
behavioral or perceptual function, especially in sensory systems. I prefer causal and
mechanistic studies over purely descriptive ones.
""".strip()

# The intended patterns: the gaps we wrote into the fixture, each the CORRECT grouping the
# machinery should recover. A-D span several papers; U1/U2 are single-paper unicorns.
# (Details in the docstring above.)
INTENDED_PATTERNS = {
    "A": "non-invasive human work is over-scored (judge scores too high)",
    "B": "computational and theoretical work is under-scored (judge too low)",
    "C": "invertebrate neuroethology is under-scored (judge too low)",
    "D": "translational disease-model work is over-scored (judge too high)",
    "U1": "UNICORN: one paper, a clean NAMED disinterest -- must become a pattern",
    "U2": "UNICORN: one paper, a clean NAMED interest -- must become a pattern",
}

# Each paper is a tuple. The first field, `intended`, is the ground-truth label the harness
# grades against; it is never shown to the machinery.
# (intended, title, abstract, journal, judge_score, user_score, judge_rationale, your_note)
_SESSION_1 = [
    # Intended pattern A: NON-INVASIVE HUMAN work. Every paper here shares exactly ONE off-taste
    # property -- it measures humans from the outside, with no access to circuits -- and differs
    # on everything else,
    # especially the METHOD (fMRI, structural MRI, a behavioral battery, PET, MEG). That is
    # deliberate. These papers used to be uniformly scalp EEG, which meant they shared three
    # properties at once (clinical framing, EEG method, correlational design) and each one was
    # a defensible grouping, so which pattern surfaced was a coin flip. A intended pattern only
    # has ONE
    # right answer when its papers have ONE thing in common.
    ("A", "Task fMRI of working memory load in healthy adults",
     "Forty healthy volunteers performed an n-back task during BOLD imaging. Prefrontal "
     "activation scaled with memory load across the group.",
     "Human Brain Mapping", 0.62, 0.15,
     "A neural measure recorded during a working memory task, which matches the stated "
     "interest in circuits supporting behavior.", "no circuit access, not my thing"),
    ("A", "Steady-state visual evoked potentials track motion coherence",
     "Sixty observers viewed moving dot fields during EEG; the evoked response amplitude "
     "scaled with motion coherence and with reported confidence.",
     "Journal of Vision", 0.58, 0.12,
     "A neural response scaling with a perceptual variable the profile names.", ""),
    ("A", "Cortical thickness correlates with vocabulary size in healthy adults",
     "Structural MRI in 120 volunteers showed temporal cortical thickness covaried with "
     "standardized vocabulary scores.",
     "Cerebral Cortex", 0.55, 0.10,
     "A brain-wide anatomical measure tracked against a cognitive outcome.", ""),
    ("A", "Resting-state fMRI connectivity predicts vigilance performance",
     "Functional connectivity measured at rest in 200 volunteers predicted lapse rate on a "
     "subsequent 40-minute vigilance task.",
     "NeuroImage", 0.60, 0.08,
     "A neural measurement linked to a behavioral outcome.", "correlational only, no"),
    ("A", "Magnetoencephalographic alpha power during sustained attention",
     "MEG in 45 healthy volunteers showed posterior alpha power fluctuating with attentional "
     "state during a continuous performance task.",
     "NeuroImage", 0.63, 0.18,
     "Links a spontaneous neural measure to a cognitive function.", ""),

    ("B", "A normative theory of attractor dynamics in working memory circuits",
     "We derive a mathematical account of working memory capacity from attractor "
     "dynamics in recurrent networks, predicting load-dependent oscillations and "
     "explaining capacity limits without synaptic fatigue.",
     "PLOS Computational Biology", 0.30, 0.78,
     "Purely theoretical with no empirical measurement or causal manipulation.",
     "theory like this is exactly what I want"),
    ("B", "An efficient-coding bound on early sensory representations",
     "Using rate-distortion theory we derive the information-maximizing filter for a noisy "
     "sensory channel under a metabolic constraint, and show the optimum shifts with input "
     "statistics.",
     "Neural Computation", 0.26, 0.75,
     "An analytical result rather than an experimental circuit study.", ""),
    # Deliberately NOT about decision-making. This slot briefly held a normative account of
    # decision thresholds, which collided with the dual-nature paper (a drift-diffusion model of
    # cuttlefish decisions): with decision theory inside intended pattern B, B and C blurred
    # through that shared topic and the dual-nature gate could no longer tell them apart. Keep
    # this one clear of anything the dual paper touches.
    ("B", "Conditions for phase locking in networks of coupled oscillators",
     "We derive the coupling strength required for stable phase locking among weakly coupled "
     "oscillators and characterize the transition to incoherence.",
     "Journal of Mathematical Neuroscience", 0.38, 0.82,
     "A theoretical framework rather than a causal experiment.", ""),
    ("B", "Mean-field theory of balanced excitatory-inhibitory cortical networks",
     "Mean-field equations for large spiking networks show balanced states arise "
     "generically, with fluctuation corrections predicting observed firing irregularity.",
     "Journal of Neuroscience", 0.28, 0.72,
     "Physics-style analysis without behavioral linkage.", ""),
    ("B", "Optimal control accounts of motor sequence chunking",
     "Optimal control policies for sequence acquisition reproduce human learning curves "
     "across four experiments and predict how chunk boundaries emerge from cost.",
     "Psychological Review", 0.32, 0.70,
     "Modeling work; the profile emphasizes mechanistic circuit studies.", ""),

    ("C", "Descending interneurons controlling escape in the octopus arm",
     "Recording and ablation in Octopus bimaculoides identify a small descending "
     "population that gates arm withdrawal, showing local circuits can execute escape "
     "without central command.",
     "Current Biology", 0.35, 0.80,
     "An invertebrate preparation, further from the mammalian circuits emphasized.",
     "invertebrate neuroethology is core interest"),
    ("C", "Planarian phototaxis reveals a two-photoreceptor decision circuit",
     "Behavioral and ablation experiments in planaria show two photoreceptor classes "
     "drive a threshold comparison that determines turning direction.",
     "eLife", 0.32, 0.76,
     "A simple invertebrate system with limited circuit resolution.", ""),
    # The rationale here used to end "in a specialist journal", which was a second shared
    # property doing no work: the intended pattern is about the PREPARATION, and naming the
    # venue invited a venue-themed grouping that spans other intended patterns too (several of
    # B's papers also sit in off-mainstream journals). One shared property per intended pattern.
    ("C", "Mechanosensory circuit for prey capture in the jumping spider",
     "Leg mechanoreceptors feed a small interneuron population that triggers the strike, "
     "with latency tuned to prey distance.",
     "Journal of Experimental Biology", 0.30, 0.74,
     "Invertebrate sensory work, distant from the mammalian circuits emphasized.", ""),
    ("C", "Wind-guided navigation in the fly central complex",
     "Two-photon imaging during tethered flight shows central-complex neurons encode "
     "wind direction and combine it with visual heading to steer.",
     "Nature Neuroscience", 0.45, 0.85,
     "Sensory integration for behavior, though in an insect model.", ""),

    # One paper, but a clean NAMED disinterest with an explicit note -> must become a
    # pattern. Deliberately ORTHOGONAL to every other intended gap (not EEG, not theory,
    # not invertebrate, not a disease model), so that being absorbed into a neighbouring
    # pattern cannot be mistaken for correct behavior. And the judge SHOULD like it --
    # rodent circuits, top journal, on-profile methods -- which makes the disagreement stark.
    ("U1", "Cortical slow-wave dynamics during NREM sleep in freely moving mice",
     "Chronic silicon-probe recordings across cortical layers in freely moving mice "
     "reveal that slow-wave propagation direction reverses across the night, and that "
     "closed-loop optogenetic disruption of the reversal impairs next-day performance.",
     "Nature Neuroscience", 0.80, 0.05,
     "Rodent cortical circuits with closed-loop causal manipulation and a behavioral "
     "readout -- squarely the kind of mechanistic circuit work the profile describes.",
     "sleep and circadian work is a hard no for me -- add it to my active disinterest "
     "list, I never want these regardless of how good the circuit work is"),
]

_SESSION_2 = [
    ("B", "Predictive coding as a unifying account of cortical hierarchy",
     "We review and formalize predictive coding across sensory and prefrontal cortex, "
     "proposing a hierarchical generative model that unifies attention and learning.",
     "Neuron", 0.34, 0.76,
     "Synthesis and theory rather than a causal circuit experiment.", ""),
    ("B", "Statistical-mechanics treatment of criticality in cortical networks",
     "A renormalization-group analysis identifies the conditions under which a recurrent "
     "network sits near a critical point, and what that predicts for avalanche exponents.",
     "Neural Computation", 0.27, 0.71,
     "A theoretical result with no measurement.", ""),
    # Pure formal theory, like every other intended-pattern-B paper. This slot used to hold a
    # population-recording ANALYSIS paper ("Latent population dynamics predict choice"), which
    # is a defensibly different gap from normative theory -- so when this gap returned, the
    # machinery could reasonably form a new grouping instead of matching the rejected theory
    # pattern, and whether the recurrence got logged became a coin flip. Same lesson as intended
    # pattern A:
    # one shared property per intended pattern.
    ("B", "Analytical capacity limits of associative memory in recurrent networks",
     "We derive closed-form bounds on the number of retrievable patterns in a recurrent "
     "network as a function of connectivity sparseness, recovering known scaling laws as a "
     "special case.",
     "Physical Review E", 0.36, 0.74,
     "A mathematical result with no empirical measurement.", ""),

    ("C", "A nociceptive escape circuit in Drosophila larvae",
     "Optogenetic dissection identifies the interneurons converting nociceptive input "
     "into the stereotyped rolling escape, with a gating step that sets threshold.",
     "Cell Reports", 0.33, 0.78,
     "An insect preparation rather than a mammalian circuit.", ""),
    # Plainly neuroethology, with no computational framing. This slot used to read "Cuttlefish
    # camouflage as a visual decision PROBLEM", describing a discrete classification over
    # substrate statistics -- which is intended pattern B's language, in intended pattern C's
    # papers. B and C then bled into each other under a growing pool, and the same cuttlefish
    # sat next to the dual-nature paper besides. Keep C's papers about circuits and behavior.
    ("C", "Chromatophore motor control during cuttlefish camouflage",
     "Recording from the chromatophore lobe during background matching identifies the motor "
     "units driving skin pattern changes.",
     "Current Biology", 0.36, 0.80,
     "Invertebrate behavior with limited neural recording.", ""),
    ("C", "Chemotaxis circuit dynamics in C. elegans",
     "Whole-brain imaging during chemotaxis reveals a low-dimensional state sequence "
     "that maps onto the animal's turning decisions.",
     "eLife", 0.34, 0.75,
     "A very small nervous system, distant from the profile's emphasis.", ""),

    # One paper again, opposite sign: a clean NAMED interest with an explicit note.
    #
    # REWRITTEN 2026-08-26, and the old version is worth recording because it broke Rule 1 in a
    # way that took a paid run to see. It was corollary discharge in a WEAKLY ELECTRIC FISH, in
    # a specialist journal, with the judge's rationale calling it "an unusual model" -- and
    # intended pattern C is non-mammalian model organisms penalised for distance from mammalian
    # circuits. The two shared four properties: non-mammalian organism, sensory circuit tied to
    # behavior, specialist journal, under-scored. On 2026-08-26 the model folded it straight
    # into C, and C's own description ("model organisms treated as liabilities due to distance
    # from mammalian circuits") covers an electric fish without strain. Defending the old
    # version needed an argument -- fish are vertebrates, so it is not invertebrate
    # neuroethology -- and a case that needs an argument is a coin flip wearing a verdict.
    # Pool C had already eaten one fixture this way (see the TIDAL note in paper_pools).
    #
    # WHY THIS ONE IS CLEAN. It is mammalian (not C), empirical and causal (not B), not human
    # (not A), not a disease model (not D), not sleep (not U1). And the REASON it is
    # under-scored is the venue, an axis no other pool in this fixture touches -- so it is not
    # a fourth entry in the crowded under-scored-interest category, which is what made the old
    # one confusable in the first place. U1 was always easy because it is the only named
    # disinterest here; a unicorn needs an empty neighbourhood, not just a distinct label.
    ("U2", "Optogenetic silencing of barrel cortex abolishes a learned whisker discrimination",
     "Mice trained on a two-alternative whisker discrimination task lose performance when "
     "layer 4 of barrel cortex is silenced on single trials, and recover within one session "
     "when silencing stops, tying the cortical column causally to the perceptual decision.",
     "Somatosensory Research Letters", 0.42, 0.90,
     "A competent circuit study, but published in a minor specialist journal with limited reach.",
     "this is exactly my core interest -- causal circuit manipulation tied to a perceptual "
     "decision. the venue should not drag it down."),

    ("D", "Deep brain stimulation restores gait in a parkinsonian primate model",
     "Stimulation of the pedunculopontine nucleus in MPTP-treated primates restored "
     "locomotor initiation, with effect size tracking stimulation frequency.",
     "Brain", 0.66, 0.22,
     "A causal circuit manipulation linked to a motor behavior.",
     "not interested in disease-model work"),
    ("D", "Amyloid-beta oligomers impair hippocampal place coding in a mouse model",
     "Tetrode recordings in an Alzheimer's model show place-field instability preceding "
     "plaque deposition, linking soluble oligomers to spatial coding failure.",
     "Neurobiology of Disease", 0.64, 0.20,
     "Hippocampal place coding is directly named in the profile.", ""),
    ("D", "Tau pathology disrupts entorhinal grid cell periodicity",
     "Grid cells in a tauopathy model lose hexagonal periodicity before cell loss, with "
     "deficits tracking behavioral errors in a spatial task.",
     "Nature Medicine", 0.68, 0.25,
     "Grid coding and spatial behavior, central to the stated interests.", ""),
    ("D", "Gene therapy rescues photoreceptor function in a retinal degeneration model",
     "AAV delivery restored photoreceptor responses and visually guided behavior in a "
     "mouse model of retinitis pigmentosa.",
     "Molecular Therapy", 0.60, 0.18,
     "A sensory system with a behavioral readout.", ""),
]

_SESSION_3 = [
    # The same gap returning. Again all patient-cohort framed, again each a different method
    # (actigraphy, a randomized trial, diffusion imaging), so the only thread back to the
    # session-1 papers is the clinical framing itself.
    ("A", "Event-related potentials during natural scene categorization",
     "EEG in 30 volunteers categorizing photographs showed an early negativity whose "
     "latency varied with scene complexity.",
     "Journal of Vision", 0.60, 0.15,
     "A neural response tied to a perceptual outcome.", "still not interested"),
    ("A", "Scalp EEG oscillations during mental rotation",
     "Two hundred volunteers performed a mental-rotation task during EEG; parietal "
     "oscillatory power scaled with rotation angle.",
     "Psychophysiology", 0.57, 0.10,
     "A neural signal that scales with task demand.", ""),
    ("A", "Prefrontal oxygenation during dual-task walking",
     "Functional near-infrared spectroscopy in 80 volunteers showed prefrontal "
     "oxygenation rising when a cognitive task was added to treadmill walking.",
     "Neurophotonics", 0.55, 0.08,
     "A brain measure related to a motor behavioral outcome.", ""),

    ("D", "Alpha-synuclein spreading alters basal ganglia output in a rat model",
     "Progressive synuclein pathology shifted firing patterns in the substantia nigra "
     "and correlated with the emergence of motor deficits.",
     "Neurobiology of Disease", 0.63, 0.20,
     "Basal ganglia circuit function linked to motor behavior.", ""),
    ("D", "Striatal circuit dysfunction in a Huntington's disease mouse model",
     "Two-photon imaging revealed loss of striatal ensemble sparsity preceding overt "
     "motor symptoms in R6/2 mice.",
     "Journal of Neuroscience", 0.61, 0.19,
     "Circuit-level imaging tied to behavioral onset.", ""),
    ("D", "Antisense therapy restores motor function in an ALS model",
     "Antisense oligonucleotide treatment preserved motor neuron counts and grip "
     "strength in SOD1 mice.",
     "Nature", 0.70, 0.28,
     "A strong causal intervention with a behavioral outcome, in a top journal.", ""),
]


_SESSION_4 = [
    # Nothing new. Three B papers and three C papers, both gaps already closed, both echoing the
    # judge rationales their earlier papers drew -- the cluster step groups by what the rationales
    # MEAN, so meaning-equivalent rationales are what make a gap recognizably the same gap.
    #
    # B: pure formal theory, one shared property (a mathematical result with no measurement) and
    # scattered on everything else -- the subfield, the formalism, the journal.
    ("B", "Exact firing-rate solutions for adapting integrate-and-fire populations",
     "We solve the population density equation for integrate-and-fire neurons with "
     "spike-frequency adaptation, giving closed-form transfer functions and the adaptation "
     "timescale at which the population response becomes non-monotonic.",
     "Journal of Computational Neuroscience", 0.29, 0.73,
     "An analytical derivation with no empirical measurement or causal manipulation.",
     "formal theory again -- I keep having to correct this one"),
    ("B", "A variational principle for metabolically efficient spiking codes",
     "Minimizing spike cost subject to a fixed information rate yields an optimal firing "
     "threshold, and we characterize how the optimum shifts with input signal-to-noise.",
     "Neural Computation", 0.31, 0.77,
     "A theoretical optimality argument rather than an experimental circuit study.", ""),
    ("B", "Bifurcation structure of ring attractor networks with heterogeneous coupling",
     "Continuation analysis shows how heterogeneity in recurrent coupling deforms the "
     "attractor manifold and identifies the coupling variance at which the bump destabilizes.",
     "SIAM Journal on Applied Dynamical Systems", 0.25, 0.70,
     "A dynamical-systems analysis with no behavioral or physiological data.", ""),

    # C: invertebrate neuroethology, one shared property (a small identified circuit driving a
    # natural behavior in an invertebrate) and scattered on phylum, sense, and journal. Kept
    # deliberately free of any computational or modeling framing, which is intended pattern B's
    # territory -- the one time a C paper was written as a "decision problem" the two gaps bled
    # into each other and the recurrence result became a coin flip.
    ("C", "Antennal lobe projection neurons gate the upwind surge in the hawkmoth",
     "Intracellular recording and targeted ablation identify the projection neurons whose "
     "activity is required for the odor-triggered upwind surge, with surge probability "
     "tracking their firing.",
     "Journal of Comparative Physiology A", 0.34, 0.79,
     "An insect preparation rather than a mammalian circuit.",
     "invertebrate circuit-to-behavior work, still core for me"),
    ("C", "A single command interneuron triggers backward swimming in the leech",
     "Stimulation of one identified interneuron is sufficient to elicit the full backward "
     "swim motor program, and its ablation abolishes the behavior.",
     "Journal of Neurophysiology", 0.31, 0.76,
     "An invertebrate preparation, further from the mammalian circuits emphasized.", ""),
    ("C", "Statocyst input drives postural righting in the sea slug Clione",
     "Unilateral statocyst removal biases the righting response, and recording from the "
     "identified righting interneurons shows they integrate gravity signals from both sides.",
     "Proceedings of the Royal Society B", 0.36, 0.82,
     "A simple invertebrate system with limited circuit resolution.", ""),
]


def _papers(rows, offset):
    """Turn the terse tuples into flag-ready dicts with stable synthetic pmids. The
    `intended` field is the ground-truth label the harness grades against."""
    out = []
    for i, (intended, title, abstract, journal, judge, yours, rationale, note) in enumerate(rows):
        out.append({
            "intended": intended, "pmid": f"SYN{offset + i:04d}", "title": title,
            "abstract": abstract, "journal": journal, "judge_score": judge,
            "user_score": yours, "rationale": rationale, "note": note,
        })
    return out


# Each round: the papers flagged that round, the scripted human decisions taken AFTER the
# round is recorded (incorporate / reject / carry), and what must be true once it is
# recorded. The `expect` keys are read by analysis_harness/grading.py -- rename in lockstep.
SESSIONS = [
    {
        "name": "R1 -- gaps A, B, C appear for the first time",
        "papers": _papers(_SESSION_1, 1000),
        "then": [("incorporate", "A"), ("reject", "B"), ("carry", "C")],
        "expect": {
            # U1 is a unicorn: ONE paper, but a named disinterest with an explicit note,
            # so it must earn a pattern exactly like the multi-paper gaps.
            "covers": ["A", "B", "C", "U1"],
            # ONE pattern per planted gap. This is a floor-of-competence gate, so the paper sets
            # are built to be trivially unambiguous -- each one shares exactly ONE property and
            # scatters on everything else -- and a split therefore means something is wrong.
            # It used to allow 2, which quietly let intended pattern B fragment into "theory in
            # general"
            # plus "population coding" and then made the recurrence test a coin flip on which
            # half the human happened to reject.
            "max_produced_per_intended": 1,
            "min_purity": 0.6,
            "new_patterns": {"min": 4, "max": 7},
            # NO act_now cap. There used to be one (<= 3) and it could never pass: this round
            # plants FOUR genuinely act-worthy gaps, so any cap below four asks the model to
            # defer one of four equally real findings at random. Testing priority discipline
            # needs a gap that is real but deliberately MARGINAL -- thin evidence, small delta,
            # something that should honestly be deferred. Plant one and the check becomes worth
            # having; until then it only manufactured a permanent red.
        },
    },
    {
        "name": "R2 -- rejected gap returns, carried gap grows, a new gap arrives",
        "papers": _papers(_SESSION_2, 2000),
        "then": [("carry", "D")],
        "expect": {
            "covers": ["D", "U2"],
            # D is a new multi-paper gap; U2 is a new single-paper unicorn. Both must
            # surface -- and U2 must do so even now that the memory shown to the model is
            # populated with existing patterns.
            "new_pattern_for": ["D", "U2"],
            "no_new_pattern_for": ["B"],          # rejected -> caught as recurrence, not a new pattern
            "recurrence_logged_for": ["B"],       # instead: a recurred mark on B's closed pattern
            "merged_into_existing": ["C"],        # carried pattern gains flags, not a duplicate
            "stay_closed": ["A", "B"],
            # NO cap on the absolute size of the open pile. There used to be one and it was the
            # wrong shape: it measured "did the model find more angles than I guessed", which is
            # not a defect. One set of papers genuinely supports several patterns -- a second one
            # making a DIFFERENT diagnosis is a finding, not bloat -- so the honest count varies
            # run to run and any number sits a hair from the actual output.
            # What we actually care about is the TREADMILL: does the pile keep climbing so
            # curation never converges? That is measured directly, and only where it means
            # something, by max_open_pattern_growth in the next session.
        },
    },
    {
        "name": "R3 -- the INCORPORATED gap comes back (the edit did not take)",
        "papers": _papers(_SESSION_3, 3000),
        # C has been open and carried since session 1; the human finally decides it. This is the
        # only place a pattern's decision CHANGES -- carried, then incorporated -- rather than
        # being made once and left alone.
        "then": [("incorporate", "C")],
        "expect": {
            "no_new_pattern_for": ["A", "D"],
            "recurrence_logged_for": ["A"],       # the failed-edit signal
            "merged_into_existing": ["D"],
            "stay_closed": ["A", "B"],
            # Convergence: round 3 re-presents gaps already tracked, so the pile must not keep
            # climbing. One new pattern is allowed because a second pattern over the same
            # flags with the opposite direction can be a genuine second finding; a steady climb
            # session after session is the treadmill we are watching for.
            "max_open_pattern_growth": 1,
        },
    },
    {
        "name": "R4 -- nothing new: B returns a second time, C returns after being incorporated",
        "papers": _papers(_SESSION_4, 4000),
        "then": [],
        "expect": {
            "no_new_pattern_for": ["B", "C"],
            # B's SECOND return, and C's first since the human incorporated it. Both must land
            # on the pattern that already holds their history rather than starting a fresh one:
            # a return only means anything if it accumulates somewhere.
            "recurrence_logged_for": ["B", "C"],
            # Four sessions on, every closed decision is still closed -- including A, closed
            # back in session 1 and returned since.
            "stay_closed": ["A", "B", "C"],
            "max_open_pattern_growth": 1,
        },
    },
]


# Graded once, after the whole arc has run. These are the questions a single session cannot
# ask -- they are about the SHAPE of the run, not its end state.
TERMINAL_EXPECT = {
    # B was rejected in session 1 and came back in sessions 2 and 4. Both returns must land on
    # the same rejected pattern, because "you rejected this and it has come back twice" is the
    # signal that the rejection was wrong, and it only exists if the returns accumulate in one
    # place. The check takes the MAX over B's patterns, so two siblings with one return each
    # fail -- that is fragmentation wearing a passing score.
    "recurrence_accumulates": [{"label": "B", "min_recurrences": 2}],
    # Sessions 3 and 4 contain no new gaps whatsoever, and the human closes C between them, so
    # the honest trajectory over that tail is flat or downward. The cap is 0 rather than
    # something roomier because the per-session caps already allow 1 each: at a cumulative cap
    # of 2 this would pass on exactly the +1-per-session ratchet it exists to catch.
    "open_pile_settles": [{"over_last_sessions": 2, "max_growth": 0}],
    # No taste pattern may carry the OPPOSITE sign to the flags that built it. Cheap, and it
    # guards four other checks rather than only itself -- fragmentation, min_purity,
    # no_new_pattern_for and the chimera carve-out all branch on direction, so an inverted one
    # makes them compare the wrong things while still reporting green. The four labels here take
    # their sign from the papers above: A and D are over-scored, B and C under-scored.
    "direction_not_inverted": [{"label": "A", "taste": "over"},
                               {"label": "B", "taste": "under"},
                               {"label": "C", "taste": "under"},
                               {"label": "D", "taste": "over"}],
}
