"""
Domain-specific configuration for litcurator.

To adapt this tool for a different field, update JOURNALS and the prompts in
this file. The rest of the pipeline (retrieve.py, rank.py, db.py) is domain-agnostic.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

DATA_DIR = Path.home() / ".litcurator"
GROUND_TRUTH_DB = DATA_DIR / "ground_truth.db"   # legacy, kept as backup
LITCURATOR_DB = DATA_DIR / "litcurator.db"
UI_TEST_RELEVANCE_DB = DATA_DIR / "ui_test_relevance.db"
UI_TEST_CURATION_DB = DATA_DIR / "ui_test_curation.db"
UI_TEST_BATCH_SIZE = 20
PROFILE_DIR = DATA_DIR / "profile"
PROFILE_PATH = PROFILE_DIR / "profile.md"
PROFILE_QA_LOG = PROFILE_DIR / "qa_log.md"
USER_FEEDBACK_DIR = DATA_DIR / "user_feedback"

# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

RELEVANCE_BATCH_SIZE = 50
CURATION_BATCH_SIZE = 15
CURATION_THRESHOLD = 1          # human label >= this is "above the noise" (0-5 scale)
LLM_SCORE_THRESHOLD = 0.5       # LLM score >= this is predicted above the noise (0.0-1.0 scale, tuned on val)
PROFILE_UPDATE_MAX_ITERATIONS = 2  # max react loop iterations in profile builder

# ---------------------------------------------------------------------------
# Journals
# ---------------------------------------------------------------------------

"""
Journals to search each retrieval run.
neuro_keyword: string added to query to filter for neuroscience content, or None for
field-specific journals that need no filtering. General journals use a broad keyword
to avoid missing papers with thin PubMed metadata.
"""
NEURO_KEYWORD = "(neur* OR brain)"

JOURNALS = [
    # General journals — broad neuro keyword required
    {"journal": "Science",                                                              "neuro_keyword": NEURO_KEYWORD},
    {"journal": "Nature",                                                               "neuro_keyword": NEURO_KEYWORD},
    {"journal": "Cell",                                                                 "neuro_keyword": NEURO_KEYWORD},
    {"journal": "Curr Biol",                                                            "neuro_keyword": NEURO_KEYWORD},
    {"journal": "Nature communications",                                                "neuro_keyword": NEURO_KEYWORD},
    {"journal": "Proceedings of the National Academy of Sciences of the United States of America", "neuro_keyword": NEURO_KEYWORD},
    {"journal": "eLife",                                                                "neuro_keyword": NEURO_KEYWORD},
    # Neuroscience-specific journals — no keyword needed
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
# Journal score adjustments
# ---------------------------------------------------------------------------

# Per-journal score adjustments applied before thresholding.
# Positive values boost journals where missing a paper is costly.
# Negative values penalize journals that pass too many off-topic titles.
JOURNAL_SCORE_ADJUSTMENTS = {
    "Nature":                          +0.10,
    "Science":                         +0.10,
    "Science (New York, N.Y.)":        +0.10,
    "Cell":                            +0.10,
    "Neuron":                          +0.10,
    "Nature neuroscience":             +0.10,
    "Nature reviews. Neuroscience":    +0.10,
    "Curr Biol":                       +0.05,
    "Nature communications":           -0.10,
    "eLife":                           -0.05,
}

# ---------------------------------------------------------------------------
# Stage 1: Domain filter prompts
# ---------------------------------------------------------------------------

DOMAIN_FILTER_PROMPT_TITLE = """
You are a neuroscience literature filter. Your job is to assess whether a paper's primary focus is at the level of neural systems and behavior, based on the title (do your best).

Score each title from 0.0 to 1.0:
- 1.0 = primary focus is clearly at the level of circuits, systems, behavior, or cognition (including computational work targeting these levels)
- 0.0 = primary focus is clearly molecular, cellular, or subcellular (ion channels, receptors, gene expression, etc.)
- 0.5 = mixed or ambiguous — systems/behavioral/cognitive elements present but not the primary focus

Ask yourself: what is the primary system of interest in this paper? If it is the immune system, cardiovascular system, metabolism, reproduction, or other non-neural organ systems — score low, even if neural pathways are mentioned. If the primary system of interest is the nervous system itself, or behavior, score accordingly.

Research using organoids, ex vivo preparations, or in vitro neural circuit models should score higher, as these are systems-level experimental platforms even when molecular methods are used, but especially if electrophysiological or optical physiological (such as calcium imaging) methods are used.

Computational methods, foundation models, analysis tools, electrophysiology techniques, or optical imaging techniques developed specifically for systems and behavioral neuro should score high, even if the primary contribution is methodological rather than a direct experimental finding about circuits or behavior. This includes tools for in vivo neural recording such as genetically encoded voltage indicators and calcium sensors.

Theoretical, conceptual, and review articles about systems neuroscience topics should score high even without empirical data. Big-picture thinking about neural systems, evolution of nervous systems, and philosophical perspectives on neuroscience are valuable.

Synaptic plasticity — the strengthening and weakening of synaptic connections, including LTP and LTD — is one of the cellular substrates of learning and memory and a core systems neuroscience topic. Papers on synaptic plasticity should therefore score higher even if molecular terminology is in the title.

More broadly, molecular terminology in a title does not preclude systems-level content, but the title itself must contain explicit systems-level signals — such as connectivity, function, behavior, or circuit — to score above 0.5. Do not infer systems relevance from what a cellular subject ultimately does; this signal must be present in the title.

AI and machine learning papers that address neuroscience questions (neuroAI)— such as learning rules, memory consolidation, neural coding, or computational models of brain function — should score high. This includes theoretical ML work that informs our understanding of how biological neural systems work.

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
systems — score low, even if neural pathways are mentioned. If it is the
nervous system itself, or behavior, score accordingly.

Computational methods, electrophysiology, and optical imaging tools developed
specifically for systems and behavioral neuroscience should score high, even
if the contribution is methodological. This includes in vivo recording tools
such as genetically encoded voltage indicators and calcium sensors.

Theoretical, conceptual, and review articles on systems neuroscience topics
should score high even without empirical data.

Synaptic plasticity — the strengthening and weakening of synaptic connections,
including LTP and LTD — is a core systems neuroscience topic and should score
higher even when the framing is mechanistic.

AI and machine learning papers that address neuroscience questions (NeuroAI)
— learning rules, memory consolidation, neural coding, computational models
of brain function — should score high. This includes theoretical ML work that
informs our understanding of biological neural systems.

When in doubt, score higher. False positives are better than false negatives
at this stage.

Return a JSON object with two fields: "score" (a number between 0.0 and 1.0) and "reasoning" (one sentence explaining your score).
""".strip()

DOMAIN_FILTER_PROMPT = DOMAIN_FILTER_PROMPT_TITLE

# ---------------------------------------------------------------------------
# Stage 2: Curation scoring prompt
# ---------------------------------------------------------------------------

CURATION_PROMPT = """
You are a personalized literature curator. Score each neuroscience article against the user's interest profile provided below.

Read the profile carefully. Score based on whether the paper's PRIMARY CONTRIBUTION matches what the user cares about — not whether it mentions topics or key words from the profile. We aren't doing keyword matches here. 

The title has already been screened. Base your score primarily on the abstract, which tells you what the paper actually does.

Treat the topical interests in the profile as illustrative of the user's taste — not as an exclusive and exhaustive list. A strong systems neuroscience paper in an unlisted area should still score well on its own merits. Do not penalize a paper simply for being outside the explicitly listed topics. The list should be used to boost, but not strongly penalize. 

Methodology matters: if a paper uses methods that the user explicitly is not interested in (e.g., EEG), generic topical relevance should not bring the score above threshold unless there are multiple interesting factors pulling it that way along multiple dimensions. 

When in doubt, score higher rather than lower. Missing a good paper is worse than including a borderline one.

Assign a relevance score from 0.0 to 1.0:
  0.0 — Primary contribution is clearly outside the user's interests
  0.5 — Borderline: some relevant elements but not a clear fit (this will be considered above threshold)
  1.0 — Perfect fit: exactly the kind of work this user cares about

Use the full range continuously (include at least two decimal places in the score -- 0.75). Score each article independently.

Also provide:
  rationale (one sentence): what is the paper primarily about, and why does or doesn't it fit the profile?

Return ONLY a JSON array of objects — one object per article, in the same order as input. Your entire response must be valid JSON: a list starting with [ and ending with ], with no explanation or preamble outside the brackets.

Example output for 2 articles:
[
  {"score": 0.85, "rationale": "Primarily reports in vivo calcium imaging of leech ganglion during local bending, directly matching the user's interest in optical physiology of intact circuits and neuroethology."},
  {"score": 0.12, "rationale": "Primarily characterizes receptor trafficking kinetics in dissociated neurons — a molecular/cellular study outside the user's stated interests."}
]
""".strip()
