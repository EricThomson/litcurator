"""
Domain-specific configuration for litcurator.

To adapt this tool for a different field, update JOURNALS, NEURO_KEYWORD, and the
DOMAIN_FILTER_PROMPT_* prompts in this file. The rest of the pipeline (retrieve,
storage, domain filter, judge) is domain-agnostic.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path.home() / ".litcurator"
LITCURATOR_DB = DATA_DIR / "litcurator.db"
PROFILE_DIR = DATA_DIR / "profile"
USER_PROFILE_PATH = PROFILE_DIR / "user_profile.md"   # the user's interests; the only taste input to the judge
PROMPT_DIR = DATA_DIR / "prompt"
JUDGE_PROMPT_PATH = PROMPT_DIR / "judge_prompt.md"    # the judge's scoring procedure; the OTHER biconvex knob, edited in the prompt workbench

# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------
# Articles per batch before the labeler shows a break screen.
RELEVANCE_BATCH_SIZE = 25
CURATION_BATCH_SIZE = 10

# When a paper is skipped, it is re-inserted at a uniformly random position in a
# rolling window that sits MIN..MAX articles ahead of the current spot -- far
# enough that it does not recur right away, near enough that it comes back during
# this session rather than being flung to the end. Near the end of the pool both
# gaps shrink to fit whatever is left (the window stays as wide as possible), so
# skipped papers spread across the tail instead of piling up at the very end.
# (Skip the last handful and there is nowhere left to put them -- that one is on
# you.)
CURATION_SKIP_MIN_GAP = 25
CURATION_SKIP_MAX_GAP = 75

# Isolated DBs for --ui_test mode (fresh each launch, never the production DB).
UI_TEST_RELEVANCE_DB = DATA_DIR / "ui_test_relevance.db"
UI_TEST_CURATION_DB = DATA_DIR / "ui_test_curation.db"

# Pre-selected labeling queue written by `litcurator prepare_labeling`.
# The relevance labeler uses this if present; delete it to revert to the full pool.
LABELING_QUEUE_FILE = DATA_DIR / "labeling_queue.json"

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

DOMAIN_THRESHOLD = 0.5          # stage 1: domain-filter score at-or-above this advances to the judge
SCORE_THRESHOLD = 0.5           # stage 2: judge score at-or-above this surfaces in the reading list

# ---------------------------------------------------------------------------
# Locked test set
# ---------------------------------------------------------------------------
# November 2025 is the held-out test set: it must be judged exactly once, at the
# final evaluation, with the profile frozen. The pipeline refuses to score any
# window overlapping this range unless explicitly unlocked (final_test=True /
# CLI --final-test), so a stray run can never spend it by accident.

LOCKED_TEST_START = "2025-11-01"
LOCKED_TEST_END = "2025-11-30"

# The locked test set is ALSO sealed as a frozen explicit pmid list (written once
# by `litcurator seal_test_set`). This is the convention-proof guard: development
# label queries subtract this set by set difference, so the held-out papers are
# unreachable by construction even from code that bypasses the date-window guard,
# and even for true-November papers whose pub_date_iso is mis-bucketed by the
# epub-ahead-of-print artifact. The date window above stays as belt-and-suspenders.
LOCKED_TEST_PMIDS_FILE = DATA_DIR / "locked_test_set.json"

# ---------------------------------------------------------------------------
# Judge harness
# ---------------------------------------------------------------------------
# The judge's UNIT TESTS: a small fixture of OBVIOUS papers with loose bands (reject
# <= 0.25, keep >= 0.75, dead middle), each tagged with the boundary it covers. Scored
# under the active profile + prompt, so it guards BOTH knobs (prompt AND profile) at
# once. The fast floor-of-competence gate, distinct from the slow benchmark. See
# litcurator judge_harness.
JUDGE_HARNESS_CASES_FILE = DATA_DIR / "judge_harness_cases.json"

# Each `litcurator judge_harness` run writes a timestamped report here (stamped with
# the prompt + profile it tested), so you accumulate a history to compare across edits.
JUDGE_HARNESS_RUNS_DIR = DATA_DIR / "judge_harness_runs"

# ---------------------------------------------------------------------------
# Journals
# ---------------------------------------------------------------------------
# Journals to search each retrieval run.
# neuro_keyword: string added to query to filter for neuroscience content, or None
# for field-specific journals that need no filtering. General journals use a broad
# keyword to avoid missing papers with thin PubMed metadata.

NEURO_KEYWORD = "(neur* OR brain)"

JOURNALS = [
    # General journals -- broad neuro keyword required
    {"journal": "Science",                                                              "neuro_keyword": NEURO_KEYWORD},
    {"journal": "Nature",                                                               "neuro_keyword": NEURO_KEYWORD},
    {"journal": "Cell",                                                                 "neuro_keyword": NEURO_KEYWORD},
    {"journal": "Curr Biol",                                                            "neuro_keyword": NEURO_KEYWORD},
    {"journal": "Nature communications",                                                "neuro_keyword": NEURO_KEYWORD},
    {"journal": "Proceedings of the National Academy of Sciences of the United States of America", "neuro_keyword": NEURO_KEYWORD},
    {"journal": "eLife",                                                                "neuro_keyword": NEURO_KEYWORD},
    # Neuroscience-specific journals -- no keyword needed
    {"journal": "Nature neuroscience",                                                  "neuro_keyword": None},
    {"journal": "Neuron",                                                               "neuro_keyword": None},
    {"journal": "Nature reviews. Neuroscience",                                         "neuro_keyword": None},
    {"journal": "Trends Neurosci",                                                      "neuro_keyword": None},
    {"journal": "The Journal of neuroscience : the official journal of the Society for Neuroscience", "neuro_keyword": None},
    {"journal": "J Neurophysiol",                                                       "neuro_keyword": None},
    {"journal": "Cerebral cortex (New York, N.Y. : 1991)",                              "neuro_keyword": None},
    {"journal": "Curr Opin Neurobiol",                                                  "neuro_keyword": None},
    {"journal": "Journal of computational neuroscience",                                "neuro_keyword": None},
    {"journal": "Annual review of neuroscience",                                        "neuro_keyword": None},
    {"journal": "Annual review of psychology",                                          "neuro_keyword": NEURO_KEYWORD},
    {"journal": "Neural computation",                                                   "neuro_keyword": None},
]

# ---------------------------------------------------------------------------
# Journal quality
# ---------------------------------------------------------------------------
USER_JOURNAL_RATINGS = {
    "Nature":                                                                              +0.10,
    "Science":                                                                             +0.10,
    "Science (New York, N.Y.)":                                                            +0.10,
    "Cell":                                                                                +0.10,
    "Neuron":                                                                              +0.10,
    "Nature neuroscience":                                                                 +0.10,
    "Nature reviews. Neuroscience":                                                        +0.10,
    "Annual review of neuroscience":                                                       +0.10,
    "Curr Biol":                                                                           +0.05,
    "Trends Neurosci":                                                                     +0.05,
    "Curr Opin Neurobiol":                                                                 +0.05,
    "The Journal of neuroscience : the official journal of the Society for Neuroscience":   0.00,
    "J Neurophysiol":                                                                       0.00,
    "Neural computation":                                                                   0.00,
    "Journal of computational neuroscience":                                                0.00,
    "Proceedings of the National Academy of Sciences of the United States of America":      0.00,
    "Annual review of psychology":                                                          0.00,
    "Cerebral cortex (New York, N.Y. : 1991)":                                              0.00,
    "eLife":                                                                               -0.05,
    "Nature communications":                                                               -0.10,
}

# ---------------------------------------------------------------------------
# Stage 1: Domain filter prompts
# ---------------------------------------------------------------------------

DOMAIN_FILTER_PROMPT_TITLE = """
You are a neuroscience literature filter. Your job is to assess whether a paper's primary focus is at the level of neural systems and behavior, based on the title (do your best).

Score each title from 0.0 to 1.0:
- 1.0 = primary focus is clearly at the level of circuits, systems, behavior, or cognition (including computational work targeting these levels)
- 0.0 = primary focus is clearly molecular, cellular, or subcellular (ion channels, receptors, gene expression, etc.)
- 0.5 = mixed or ambiguous -- systems/behavioral/cognitive elements present but not the primary focus

Ask yourself: what is the primary system of interest in this paper? If it is the immune system, cardiovascular system, metabolism, reproduction, or other non-neural organ systems -- score low, even if neural pathways are mentioned. If the primary system of interest is the nervous system itself, or behavior, score accordingly.

Note that neuroanatomy at the level of identified brain regions or circuits -- cell type composition, synaptic inventories, connectomes, mesoscale connectivity -- is systems neuroscience. The cellular/molecular exclusion applies to single-cell molecular biology (ion channels, receptors, gene expression), not to neural circuit architecture.

Animal behavior studies (predation, reproduction, communication, navigation, learning, social behavior, and the like) are systems neuroscience by default. Behavior is the output of neural systems; the title does not need to name a neural mechanism explicitly.

Research using organoids, ex vivo preparations, or in vitro neural circuit models should score higher, as these are systems-level experimental platforms even when molecular methods are used, but especially if electrophysiological or optical physiological (such as calcium imaging) methods are used.

Computational methods, foundation models, analysis tools, electrophysiology techniques, or optical imaging techniques developed specifically for systems and behavioral neuro should score high, even if the primary contribution is methodological rather than a direct experimental finding about circuits or behavior. This includes tools for in vivo neural recording such as genetically encoded voltage indicators and calcium sensors.

Theoretical, conceptual, and review articles about systems neuroscience topics should score high even without empirical data. Big-picture thinking about neural systems, evolution of nervous systems, and philosophical perspectives on neuroscience are valuable.

Synaptic plasticity -- the strengthening and weakening of synaptic connections, including LTP and LTD -- is one of the cellular substrates of learning and memory and a core systems neuroscience topic. Papers on synaptic plasticity should therefore score higher even if molecular terminology is in the title.

More broadly, molecular terminology in a title does not preclude systems-level content, but the title itself must contain explicit systems-level signals -- such as connectivity, function, behavior, or circuit -- to score above 0.5. Do not infer systems relevance from what a cellular subject ultimately does; this signal must be present in the title.

AI and machine learning papers that address neuroscience questions -- such as learning rules, memory consolidation, neural coding, or computational models of brain function -- should score high. This includes theoretical ML work that informs our understanding of how biological neural systems work.

When in doubt, score higher rather than lower. This is meant to be a quick screen, not a final filter. The goal is to not miss out on articles, so false positives are better than false negatives for this stage. But also bear in mind that mere mention of a behavioral assay does not make a paper systems neuroscience if the primary story is molecular or cellular.

Return a JSON object with two fields: "score" (a number between 0.0 and 1.0) and "reasoning" (one sentence explaining your score).
""".strip()

DOMAIN_FILTER_PROMPT_ABSTRACT = """
You are a neuroscience literature filter. Your job is to assess whether a
paper's primary focus is at the level of neural systems and behavior.

Score each abstract from 0.0 to 1.0:
- 1.0 = primary focus is clearly circuits, systems, behavior, or cognition
        (including computational work targeting these levels)
- 0.0 = primary focus is clearly molecular, cellular, or subcellular
        (ion channels, receptors, gene expression, etc.)
- 0.5 = mixed or ambiguous

Ask: what is the primary system of interest? If it is the immune system,
cardiovascular system, metabolism, reproduction, or other non-neural organ
systems -- score low, even if neural pathways are mentioned. If the paper primarily focuses on the
the nervous system itself, or behavior, score accordingly.

Computational methods, electrophysiology, and optical imaging tools developed
specifically for systems and behavioral neuroscience should score high, even
if the contribution is methodological. This includes in vivo recording tools
such as genetically encoded voltage indicators and calcium sensors.

Theoretical, conceptual, and review articles on systems neuroscience topics
should score high even without empirical data.

Synaptic plasticity -- the strengthening and weakening of synaptic connections,
including LTP and LTD -- is a core systems neuroscience topic and should score
higher even when the framing is mechanistic.

AI and machine learning papers that address neuroscience questions 
-- neural coding, computational models of brain function, memory, plasticity -- should score high. This includes theoretical ML work that
informs our understanding of biological neural systems.

When in doubt, score higher. False positives are better than false negatives
at this stage.

Return a JSON object with two fields: "score" (a number between 0.0 and 1.0) and "reasoning" (one sentence explaining your score).
""".strip()
