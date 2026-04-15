"""
Domain-specific configuration for litcurator.

To adapt this tool for a different field, update JOURNALS and the prompts in
this file. The rest of the pipeline (retrieve.py, rank.py, db.py) is domain-agnostic.
"""

# ---------------------------------------------------------------------------
# Journals
# ---------------------------------------------------------------------------

"""
Journals to search each retrieval run.
neuro_keyword=True means we add 'neuroscience' to the query (general journals).
neuro_keyword=False means the journal is field-specific (no keyword needed).
"""
JOURNALS = [
    # General journals — neuroscience keyword required
    {"journal": "Science",                                                              "neuro_keyword": True},
    {"journal": "Nature",                                                               "neuro_keyword": True},
    {"journal": "Cell",                                                                 "neuro_keyword": True},
    {"journal": "Curr Biol",                                                            "neuro_keyword": True},
    {"journal": "Nature communications",                                                "neuro_keyword": True},
    {"journal": "Proceedings of the National Academy of Sciences of the United States of America", "neuro_keyword": True},
    {"journal": "eLife",                                                                "neuro_keyword": True},
    # Neuroscience-specific journals — no keyword needed
    {"journal": "Nature neuroscience",                                                  "neuro_keyword": False},
    {"journal": "Neuron",                                                               "neuro_keyword": False},
    {"journal": "Nature reviews. Neuroscience",                                         "neuro_keyword": False},
    {"journal": "Trends Neurosci",                                                      "neuro_keyword": False},
    {"journal": "The Journal of neuroscience : the official journal of the Society for Neuroscience", "neuro_keyword": False},
    {"journal": "J Neurophysiol",                                                       "neuro_keyword": False},
    {"journal": "Cerebral cortex (New York, N.Y. : 1991)",                              "neuro_keyword": False},
    {"journal": "Curr Opin Neurobiol",                                                  "neuro_keyword": False},
    {"journal": "Journal of computational neuroscience",                                "neuro_keyword": False},
    {"journal": "Annual review of neuroscience",                                        "neuro_keyword": False},
    {"journal": "Annual review of psychology",                                          "neuro_keyword": True},
    {"journal": "Neural computation",                                                   "neuro_keyword": False},
]

# ---------------------------------------------------------------------------
# Stage 1: Domain filter prompt
# ---------------------------------------------------------------------------

DOMAIN_FILTER_PROMPT = """
You are a neuroscience literature filter. Your job is to assess whether a paper's primary focus is at the level of neural systems and behavior.

Score each abstract from 0.0 to 1.0:
- 1.0 = primary focus is clearly at the level of circuits, systems, behavior, or cognition (including computational work targeting these levels)
- 0.0 = primary focus is clearly molecular, cellular, or subcellular (ion channels, receptors, gene expression, etc.)
- 0.5 = mixed or ambiguous — systems/behavioral/cognitive elements present but not the primary focus

When in doubt, score higher rather than lower. A behavioral assay alone does not make a paper systems neuroscience if the primary story is molecular or cellular. The goal is to not miss out on articles, so false positives are better than false negatives.

Return a JSON object with two fields: "score" (a number between 0.0 and 1.0) and "reasoning" (one sentence explaining your score).
""".strip()
