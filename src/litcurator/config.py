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
GROUND_TRUTH_DB = DATA_DIR / "ground_truth.db"
PROFILE_DIR = DATA_DIR / "profile"
PROFILE_PATH = PROFILE_DIR / "profile.md"
PROFILE_QA_LOG = PROFILE_DIR / "qa_log.md"

# ---------------------------------------------------------------------------
# Labeling
# ---------------------------------------------------------------------------

RELEVANCE_BATCH_SIZE = 25
CURATION_BATCH_SIZE = 10

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
# High-impact journal score bump
# ---------------------------------------------------------------------------

"""
Articles from these journals get a +0.1 score bump before threshold is applied.
Rationale: missing a paper from these venues is a higher cost than a false positive.
"""
HIGH_IMPACT_JOURNALS = {
    "Nature",
    "Science",
    "Science (New York, N.Y.)",
    "Cell",
    "Neuron",
    "Nature neuroscience",
    "Nature reviews. Neuroscience",
}
HIGH_IMPACT_BUMP = 0.1

# ---------------------------------------------------------------------------
# Journal score adjustments
# ---------------------------------------------------------------------------

# Per-journal score adjustments applied before thresholding.
# Negative values penalize journals that pass too many off-topic titles.
JOURNAL_SCORE_ADJUSTMENTS = {
    "Nature communications": -0.1,
    "eLife": -0.05,
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
