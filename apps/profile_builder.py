"""
Profile builder -- iteratively updates profile.md based on user feedback flags.

Step 1: Select date periods to include.
Step 2: LLM identifies preference clusters from flags. User confirms/edits.
Step 3: Clusters are distilled to pure principles (article refs stripped).
Step 4: ReAct loop -- LLM proposes profile edits, validates by re-scoring flagged
        articles, iterates up to MAX_ITERATIONS. User reviews diff and accepts.

Run:
    streamlit run apps/profile_builder.py
"""

import difflib
import os
from datetime import date
import anthropic
from dotenv import load_dotenv
import streamlit as st

from litcurator import db
from litcurator.config import (
    LITCURATOR_DB, PROFILE_PATH, CURATION_PROMPT, JOURNAL_SCORE_ADJUSTMENTS,
    PROFILE_UPDATE_MAX_ITERATIONS,
)
from litcurator.evaluate import score_curation_batch, CURATION_MODEL, MODEL_COSTS

load_dotenv()

MAX_ITERATIONS = PROFILE_UPDATE_MAX_ITERATIONS

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

CLUSTER_PROMPT = """
You are analyzing user feedback on a neuroscience literature curation system to identify the preference signals the feedback reveals.

The system scores articles 0-1 against a user interest profile. Articles scoring >= 0.5 are surfaced to the user; articles below 0.5 are filtered out. The user has flagged articles as incorrectly scored, with notes explaining what was wrong.

Your task: identify up to {max_clusters} distinct categories of preference the user is revealing, rank-ordered by importance. Prioritize categories that: (1) affect articles on the wrong side of the 0.5 threshold, (2) recur across multiple articles, (3) involve large score errors. Skip rare or minor patterns that are on the correct side of threshold.

For each category:
- Number and name it concisely (2-5 words)
- Describe the underlying preference principle (1-2 sentences)
- List the flagged articles that belong to it (by number)
- Note the direction: should scores be higher or lower for articles in this category?

Important:
- Represent specific exclusions accurately. "Not interested in eyeblink conditioning unless exceptional" is a valid named exclusion -- do not over-generalize it into a broader principle that may not apply to other paradigms.
- When multiple flags point to the same conceptual gap, group them.
- If uncertain about an interpretation, say so -- the user will review and correct before any edits are made.

Write in plain text. This is a draft for user review, not final profile changes.
""".strip()


DISTILL_PROMPT = """
Restate each preference category below as a general principle and direction only.
Remove all references to specific articles, article titles, article numbers, scores, journals, and article-specific notes.
The output should read as a clean set of general preference guidelines with no trace of the specific papers that motivated them.
For each category output: numbered name, principle description (1-2 sentences), direction.
Return only the distilled categories, nothing else.
""".strip()


EDIT_PROMPT = """
You are making targeted edits to a personalized neuroscience literature curation profile based on confirmed preference categories.

The system scores articles 0-1 against this profile. Articles scoring >= 0.5 are surfaced to the user; articles below 0.5 are filtered out. Flags represent cases where the score was wrong — either a paper scored too high and shouldn't have been surfaced, or too low and should have been.

The existing profile is the source of truth for style, format, and length. Your job is to make the minimum necessary changes — not to rewrite or restructure.

Rules:
- Preserve the existing prose style. If the profile uses casual first-person paragraphs with no headers, your output must too. Do not add headers, bullet points, bold text, or new sections unless they already exist in the profile.
- Keep approximately the same length. Add only what the categories require; do not expand existing points.
- Make targeted edits: insert a sentence or two where needed, adjust existing sentences, or add a short paragraph. Leave everything else word for word.
- Do not escalate or strengthen preferences beyond what the categories state. "Low interest" stays "low interest". Hedged language in the user's notes ("may be", "tends to", "often") must stay hedged -- do not convert soft preferences into absolute rules.
- The profile describes a person's taste, not a scoring rubric. Preferences are tendencies with exceptions, not bright-line rules. Do not rewrite soft preferences as hard exclusions.
- Do not repeat the same point in multiple places.
- Do not add preferences not supported by the confirmed categories.
- Express all edits as general principles, not references to specific papers. If a flagged paper reveals a preference, state the preference in terms of a general principle — do not insert the paper into the profile surreptitiously as an example. Create a general rule that fits naturally into the profile.
- Return ONLY the complete updated profile text, with no explanation, preamble, or code fences.
""".strip()


# ---------------------------------------------------------------------------
# LLM helpers
# ---------------------------------------------------------------------------

def _show_api_error(e):
    if e.status_code == 529:
        st.warning("Anthropic's servers are overloaded right now. Wait a moment and try again.  \nCheck status: https://status.anthropic.com")
    elif e.status_code == 429:
        st.warning("Rate limit reached. Wait a moment and try again.  \nCheck status: https://status.anthropic.com")
    elif e.status_code >= 500:
        st.warning(f"Anthropic server error ({e.status_code}). Try again.  \nCheck status: https://status.anthropic.com")
    else:
        st.error(f"API error ({e.status_code}): {e.message}")

def _calc_cost(usage):
    costs = MODEL_COSTS.get(CURATION_MODEL, {"input": 0, "output": 0})
    return (usage.input_tokens * costs["input"] + usage.output_tokens * costs["output"]) / 1_000_000


def distill_clusters_for_edit(clusters_text):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    response = client.messages.create(
        model=CURATION_MODEL,
        max_tokens=1024,
        system=DISTILL_PROMPT,
        messages=[{"role": "user", "content": clusters_text}]
    )
    return response.content[0].text, _calc_cost(response.usage)


def run_cluster_analysis(flags, profile_text):
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    max_clusters = min(5, len(flags) // 2)
    system = CLUSTER_PROMPT.format(max_clusters=max_clusters)
    items = []
    for i, f in enumerate(flags, 1):
        items.append(
            f"{i}. \"{f['title']}\"\n"
            f"   Score: {f['score']:.2f} | Journal: {f['journal']}\n"
            f"   Note: {f['note']}"
        )
    user_msg = (
        f"## Current Profile\n\n{profile_text}\n\n"
        f"## Flagged Articles ({len(flags)})\n\n" + "\n\n".join(items)
    )
    response = client.messages.create(
        model=CURATION_MODEL,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user_msg}]
    )
    return response.content[0].text, _calc_cost(response.usage)


def propose_edit(current_profile, clusters_text, work_items):
    """One loop step: propose profile edits targeting the remaining failing articles."""
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    items = []
    for i, w in enumerate(work_items, 1):
        direction = "lower" if w["direction"] == "down" else "higher"
        items.append(
            f"{i}. Score: {w['current_score']:.2f} (should be {direction})\n"
            f"   Note: {w['note']}"
        )
    user_msg = (
        f"## Current Profile\n\n{current_profile}\n\n"
        f"--- EDITING INSTRUCTIONS (do not copy this section into the profile) ---\n\n"
        f"Incorporate the following preference updates by making targeted edits to the profile above.\n\n"
        f"{clusters_text}\n\n"
        f"--- ARTICLES NEEDING IMPROVEMENT ({len(work_items)} remaining) ---\n\n"
        + "\n\n".join(items)
        + "\n\nReturn the complete updated profile text."
    )
    response = client.messages.create(
        model=CURATION_MODEL,
        max_tokens=4096,
        system=EDIT_PROMPT,
        messages=[{"role": "user", "content": user_msg}]
    )
    return response.content[0].text, _calc_cost(response.usage)


def validate(work_items, candidate_profile):
    """Re-score work items with candidate profile. Returns (updated work_items, cost)."""
    rows = [{"title": w["title"], "abstract": w["abstract"], "journal": w["journal"]} for w in work_items]
    results, usage = score_curation_batch(rows, candidate_profile, CURATION_MODEL)
    updated = []
    for w, (new_score, _) in zip(work_items, results):
        adj = JOURNAL_SCORE_ADJUSTMENTS.get(w["journal"], 0.0)
        adjusted = min(1.0, max(0.0, new_score + adj))
        updated.append({**w, "current_score": adjusted})
    return updated, _calc_cost(usage)


def did_improve(w):
    """Did the score move in the right direction relative to the original?"""
    if w["direction"] == "down":
        return w["current_score"] < w["original_score"]
    else:
        return w["current_score"] > w["original_score"]


def run_react_loop(flags, clusters_text, initial_profile, cluster_cost=0.0, on_iteration=None):
    """
    Iteratively refine candidate profile until all articles improve or MAX_ITERATIONS hit.
    Returns (final_profile_text, loop_log, total_cost).
    on_iteration(iteration, improved, total, cost_so_far) called after each iteration.
    """
    work_items = [
        {
            "feedback_id": f["feedback_id"],
            "title": f["title"],
            "abstract": f["abstract"],
            "journal": f["journal"],
            "note": f["note"],
            "original_score": f["score"],
            "current_score": f["score"],
            "direction": "down" if f["score"] >= 0.5 else "up",
        }
        for f in flags
    ]

    current_profile = initial_profile
    remaining = work_items
    log = []
    total_cost = cluster_cost

    for iteration in range(1, MAX_ITERATIONS + 1):
        candidate, edit_cost = propose_edit(current_profile, clusters_text, remaining)
        remaining, val_cost = validate(remaining, candidate)
        total_cost += edit_cost + val_cost

        improved = [w for w in remaining if did_improve(w)]
        still_failing = [w for w in remaining if not did_improve(w)]

        log.append({
            "iteration": iteration,
            "improved": len(improved),
            "total": len(remaining),
            "still_failing": [w["title"] for w in still_failing],
            "cost_so_far": total_cost,
        })

        if on_iteration:
            on_iteration(iteration, len(improved), len(remaining), total_cost)

        current_profile = candidate
        remaining = still_failing

        if not remaining:
            break

    return current_profile, log, total_cost


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(layout="wide")
st.title("Profile Builder")

conn = db.get_connection(LITCURATOR_DB)
stage = st.session_state.get("pb_stage", "select_periods")

# --- Stage: select_periods ---
if stage == "select_periods":
    periods = db.get_uningested_feedback_periods(conn)
    if not periods:
        st.info("No unprocessed flags. Flag articles in the rank viewer first.")
        conn.close()
        st.stop()
    st.subheader("Select periods to build profile from")
    options = [f"{month} ({count} flag{'s' if count != 1 else ''})" for month, count in periods]
    selected = st.multiselect("Available periods", options=options, default=options)
    if st.button("Load flags", type="primary", disabled=not selected):
        selected_months = [periods[options.index(s)][0] for s in selected]
        st.session_state["pb_selected_months"] = selected_months
        st.session_state["pb_stage"] = "show_flags"
        st.rerun()
    conn.close()
    st.stop()

selected_months = st.session_state.get("pb_selected_months")
flags = db.get_uningested_feedback(conn, months=selected_months)

if not flags and stage != "done":
    st.info("No unprocessed flags. Flag articles in the rank viewer first.")
    conn.close()
    st.stop()

# --- Stage: show_flags ---
if stage == "show_flags":
    st.markdown(f"**{len(flags)} flagged articles** ready for profile update.")
    for i, f in enumerate(flags, 1):
        with st.expander(f"{i}. {f['title']} — score: {f['score']:.2f}"):
            st.caption(f["journal"])
            st.write(f["abstract"] or "_No abstract_")
            st.markdown(f"**Your note:** {f['note']}")
    st.divider()
    if st.button("Analyze preference clusters", type="primary"):
        profile_text = PROFILE_PATH.read_text()
        try:
            with st.spinner("Analyzing flags..."):
                clusters, cluster_cost = run_cluster_analysis(flags, profile_text)
            st.session_state["pb_clusters"] = clusters
            st.session_state["pb_profile"] = profile_text
            st.session_state["pb_cluster_cost"] = cluster_cost
            st.session_state["pb_stage"] = "review_clusters"
            st.rerun()
        except anthropic.APIStatusError as e:
            _show_api_error(e)

# --- Stage: review_clusters ---
elif stage == "review_clusters":
    st.subheader("Step 1: Confirm preference categories")
    st.info("Review the LLM's interpretation of your flags below. **Edit anything it got wrong before proceeding.**")
    clusters = st.text_area(
        "Preference categories",
        value=st.session_state.get("pb_clusters", ""),
        height=400,
    )
    col1, col2 = st.columns([2, 8])
    with col1:
        if st.button("Distill", type="primary"):
            st.session_state["pb_clusters"] = clusters
            try:
                with st.spinner("Distilling principles (removing article references)..."):
                    edit_clusters, distill_cost = distill_clusters_for_edit(clusters)
                st.session_state["pb_edit_clusters"] = edit_clusters
                st.session_state["pb_distill_cost"] = distill_cost
                st.session_state["pb_stage"] = "review_distilled"
                st.rerun()
            except anthropic.APIStatusError as e:
                _show_api_error(e)
    with col2:
        if st.button("Start over"):
            for k in ["pb_stage", "pb_clusters", "pb_profile", "pb_final", "pb_log", "pb_edit_clusters", "pb_selected_months"]:
                st.session_state.pop(k, None)
            st.rerun()

# --- Stage: review_distilled ---
elif stage == "review_distilled":
    st.subheader("Step 2: Review distilled principles")
    st.info("Article references have been stripped. **This is what the profile editor will see.** Edit if needed, then run the update.")
    st.caption("Removing article details helps prevent overfitting — the editor works from general principles, not specific papers.")
    edit_clusters = st.text_area(
        "Distilled principles",
        value=st.session_state.get("pb_edit_clusters", ""),
        height=400,
        label_visibility="collapsed",
    )
    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        if st.button("Run profile update", type="primary", disabled=st.session_state.get("pb_running", False)):
            st.session_state["pb_running"] = True
            st.session_state["pb_edit_clusters"] = edit_clusters
            total_cluster_cost = st.session_state.get("pb_cluster_cost", 0.0) + st.session_state.get("pb_distill_cost", 0.0)
            try:
                with st.status(f"Updating profile (up to {MAX_ITERATIONS} iterations)...", expanded=True) as status:
                    status.write(f"Starting with {len(flags)} flagged articles...")

                    def on_iteration(iteration, improved, total, cost_so_far):
                        still = total - improved
                        status.write(
                            f"Iteration {iteration}: {improved}/{total} improved"
                            + (f", {still} still need work" if still else " — all improved!")
                            + f" | ${cost_so_far:.4f}"
                        )

                    final, log, total_cost = run_react_loop(
                        flags, edit_clusters, st.session_state["pb_profile"],
                        cluster_cost=total_cluster_cost,
                        on_iteration=on_iteration,
                    )
                    status.update(label=f"Done — total cost: ${total_cost:.4f}", state="complete")

                st.session_state["pb_final"] = final
                st.session_state["pb_log"] = log
                st.session_state["pb_cost"] = total_cost
                st.session_state["pb_running"] = False
                st.session_state["pb_stage"] = "review_diff"
                st.rerun()
            except anthropic.APIStatusError as e:
                st.session_state["pb_running"] = False
                _show_api_error(e)
    with col2:
        if st.button("Back"):
            st.session_state["pb_stage"] = "review_clusters"
            st.rerun()
    with col3:
        if st.button("Start over"):
            for k in ["pb_stage", "pb_clusters", "pb_profile", "pb_final", "pb_log", "pb_edit_clusters", "pb_selected_months"]:
                st.session_state.pop(k, None)
            st.rerun()

# --- Stage: review_diff ---
elif stage == "review_diff":
    st.subheader("Step 2: Review proposed profile update")
    total_cost = st.session_state.get("pb_cost", 0.0)
    st.caption(f"Total cost: ${total_cost:.4f}")
    log = st.session_state.get("pb_log", [])
    for entry in log:
        still = entry["still_failing"]
        suffix = ""
        if still:
            preview = "; ".join(f'"{t[:60]}"' for t in still[:2])
            if len(still) > 2:
                preview += f" (+{len(still) - 2} more)"
            suffix = f" -- still not improved: {preview}"
        st.caption(f"Iteration {entry['iteration']}: {entry['improved']}/{entry['total']} improved (${entry['cost_so_far']:.4f} cumulative){suffix}")
    st.divider()

    col_old, col_new = st.columns(2)
    with col_old:
        st.markdown("**Current profile**")
        st.text_area("Current profile", value=st.session_state.get("pb_profile", ""), height=600, disabled=True, key="pb_old", label_visibility="collapsed")
    with col_new:
        st.markdown("**Proposed profile** *(edit before accepting)*")
        final_text = st.text_area("Proposed profile", value=st.session_state.get("pb_final", ""), height=600, key="pb_new", label_visibility="collapsed")

    old_lines = st.session_state.get("pb_profile", "").splitlines(keepends=True)
    new_lines = final_text.splitlines(keepends=True)
    diff = list(difflib.unified_diff(old_lines, new_lines, lineterm=""))
    if diff:
        with st.expander("View diff", expanded=False):
            parts = []
            for line in diff[2:]:  # skip the --- / +++ header lines
                if line.startswith("+"):
                    parts.append(f'<span style="background:#1a3a1a;color:#90ee90;display:block;">{line}</span>')
                elif line.startswith("-"):
                    parts.append(f'<span style="background:#3a1a1a;color:#ff9090;display:block;">{line}</span>')
                elif line.startswith("@@"):
                    parts.append(f'<span style="color:#888888;display:block;">{line}</span>')
                else:
                    parts.append(f'<span style="display:block;">{line}</span>')
            st.markdown(
                '<pre style="font-size:0.82em;line-height:1.5;overflow-x:auto;">' + "".join(parts) + "</pre>",
                unsafe_allow_html=True,
            )

    st.divider()
    col1, col2 = st.columns([2, 8])
    with col1:
        if st.button("Accept", type="primary"):
            last_run = conn.execute(
                "SELECT profile_id FROM scoring_runs WHERE stage = 'curation' ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
            parent_id = last_run["profile_id"] if last_run else None
            new_id = db.get_or_create_profile(
                conn, final_text, parent_id=parent_id,
                notes=f"Updated from {len(flags)} flags ({date.today().isoformat()})"
            )
            PROFILE_PATH.write_text(final_text)
            pmids = [f["pmid"] for f in flags]
            all_feedback_ids = db.get_all_uningested_feedback_ids_for_pmids(conn, pmids)
            db.mark_feedback_ingested(conn, all_feedback_ids, new_id)
            st.session_state["pb_stage"] = "done"
            st.rerun()
    with col2:
        if st.button("Discard"):
            for k in ["pb_stage", "pb_clusters", "pb_profile", "pb_final", "pb_log"]:
                st.session_state.pop(k, None)
            st.rerun()

# --- Stage: done ---
elif stage == "done":
    total_cost = st.session_state.get("pb_cost", 0.0)
    st.success(f"Profile updated. {len(flags)} feedback items marked as ingested. Total cost: ${total_cost:.4f}")
    if st.button("Start new session"):
        for k in ["pb_stage", "pb_clusters", "pb_profile", "pb_final", "pb_log"]:
            st.session_state.pop(k, None)
        st.rerun()

conn.close()
