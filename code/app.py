"""Streamlit interface for the assistant.

A thin layer over answer.answer(), the same pipeline the command line and the evaluation use.
It adds stage-by-stage progress, citations highlighted by whether they passed the check,
source cards with the full narratives, the filter and retrieval details behind each answer,
and per-answer feedback saved to logs/feedback.jsonl for the peer-testing summary.

Each question is answered on its own: follow-ups do not see earlier answers.

Usage: streamlit run code/app.py      (requires ANTHROPIC_API_KEY)
"""

import json
import os
import re
import time

import anthropic
import duckdb
import streamlit as st

import answer as ans
import build_filter as bf
import preprocess
import retrieve

EXAMPLE_IDS = [1, 3, 6, 10, 11, 22]  # from evaluation/queries.json; 22 shows a designed refusal
MODES = {"Hybrid (BM25 + embeddings)": "hybrid", "Dense only (naive-RAG baseline)": "dense", "BM25 only": "bm25"}
STAGES = {"filter": "Choosing a filter over the coded fields…", "retrieve": "Searching the narratives…",
          "answer": "Writing the answer…"}
CITATION = re.compile(r"\[\s*ACN[^\]]*\]|ACN\s*#?\s*\d+")


def md(text):
    """Escape the one Markdown feature that breaks plain prose: $...$ renders as LaTeX."""
    return text.replace("$", r"\$")


def format_answer(text, citations):
    """Answer Markdown with each cited ACN highlighted: blue if verified, red if it failed the check."""
    failed = set(citations["not_shown"]) | set(citations["not_in_corpus"])

    def chip(acn):
        return f":red-background[ACN {acn} ⚠]" if acn in failed else f":blue-background[ACN {acn}]"

    return CITATION.sub(lambda m: " ".join(chip(a) for a in ans.cited_acns(m.group(0))), md(text))


def examples():
    queries = json.loads((preprocess.ROOT / "evaluation/queries.json").read_text())["queries"]
    by_id = {q["id"]: q["query"] for q in queries}
    return [by_id[i] for i in EXAMPLE_IDS]


@st.cache_resource(show_spinner="Loading the corpus and the embedding model…")
def resources():
    con = duckdb.connect(str(preprocess.DB_PATH), read_only=True)
    return con, retrieve.load(con), anthropic.Anthropic()


def save_feedback(record, rating, comment):
    path = ans.LOG_PATH.parent / "feedback.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {"time": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "question": record["question"], "mode": record["mode"],
             "rating": {0: "down", 1: "up"}.get(rating), "comment": comment, "answer": record["answer"],
             "filter": record["filter"], "citations": record["citations"]}
    with path.open("a") as f:
        f.write(json.dumps(entry) + "\n")


def render_sources(record, con):
    st.markdown(f"**Sources** ({len(record['sources'])} reports)")
    for s in record["sources"]:
        with st.expander(f"ACN {s['acn']} · {s['year']} · {s['aircraft']}"):
            st.caption(md(s["synopsis"]))
            rows = con.execute("SELECT part, text FROM narratives WHERE acn = ? ORDER BY part", [s["acn"]]).fetchall()
            for part, text in rows:
                st.markdown(f"**Narrative {part}**")
                st.text(text)


def render_details(record):
    with st.expander("How this answer was built"):
        f = record["filter"]
        st.markdown(f"**1. Filter** ({f['status']}, by {record['filter_model']})")
        if f["filters"]:
            for spec, count in zip(f["filters"], f["counts"]):
                st.code(json.dumps(spec, indent=1), language="json")
                st.caption(f"Matched {count} reports in this corpus.")
        else:
            st.caption("No valid filter was built, so the whole corpus was searched.")
        st.markdown(f"**2. Retrieval** ({record['mode']})")
        for i, side in enumerate(record["retrieved"], 1):
            if len(record["retrieved"]) > 1:
                st.caption(f"Filter {i}")
            st.dataframe([{"rank": n, "ACN": h["acn"], "narrative": h["part"], "score": round(h["score"], 4)}
                          for n, h in enumerate(side, 1)], hide_index=True)
        tokens = record["usage"].get("answer", {})
        st.markdown(f"**3. Answer** (by {record['model']}, {tokens.get('input_tokens', 0):,} tokens in, "
                    f"{tokens.get('output_tokens', 0):,} out, {record['seconds']}s total)")


def render_feedback(i, record):
    sent = st.session_state.setdefault("feedback_sent", set())
    if i in sent:
        st.caption("Feedback sent. Thank you.")
        return
    rating = st.feedback("thumbs", key=f"rating-{i}")
    comment = st.text_input("Comment (optional)", key=f"comment-{i}", placeholder="What worked or didn't?")
    if st.button("Send feedback", key=f"send-{i}", disabled=rating is None and not comment):
        save_feedback(record, rating, comment)
        sent.add(i)
        st.rerun()


def render_record(i, record, con):
    if record["status"] == "refusal":
        st.warning("The model declined to answer this request.")
    else:
        st.markdown(format_answer(record["answer"], record["citations"]))
        c = record["citations"]
        if c["not_shown"]:
            st.warning(f"Cited reports the model was not shown, so these claims are unsupported: {c['not_shown']}")
        if c["not_in_corpus"]:
            st.error(f"Cited ACNs that do not exist in the corpus: {c['not_in_corpus']}")
        if record["sources"]:
            render_sources(record, con)
    render_details(record)
    render_feedback(i, record)


def ask(question, mode, con, retriever, client):
    """Run the pipeline with live progress. Returns the record, or None after showing an error."""
    with st.status(STAGES["filter"]) as status:
        start = time.time()
        try:
            record = ans.answer(question, client=client, retriever=retriever, con=con.cursor(), mode=mode,
                                log_path=ans.LOG_PATH, filter_log_path=bf.LOG_PATH,
                                on_step=lambda stage: status.update(label=STAGES[stage]))
        except anthropic.AuthenticationError:
            error = "The API key was rejected. Check ANTHROPIC_API_KEY."
        except anthropic.RateLimitError:
            error = "Rate limited by the API. Wait a minute and try again."
        except anthropic.APIConnectionError:
            error = "Could not reach the API. Check the network connection."
        except anthropic.APIStatusError as e:
            error = f"The API returned an error ({e.status_code}). Try again."
        else:
            record |= {"mode": mode, "seconds": round(time.time() - start)}
            status.update(label=f"Answered in {record['seconds']}s", state="complete")
            return record
        status.update(label="Failed", state="error")
    st.error(error)
    return None


def main():
    st.set_page_config(page_title="ASRS Maintenance Assistant", page_icon="🔧")
    st.title("ASRS Maintenance Report Assistant")
    st.caption("Ask questions about 3,080 NASA ASRS reports filed by aviation maintenance personnel (2010–2026). "
               "Every claim in an answer cites the report it comes from.")

    with st.sidebar:
        mode = MODES[st.radio("Retrieval", list(MODES))]
        st.subheader("Try asking")
        for q in examples():
            if st.button(q, key=f"example-{q}"):
                st.session_state["pending"] = q
        st.subheader("About the data")
        st.caption("ASRS reports are voluntary and unverified, so counts describe this corpus only, not how often "
                   "events happen in aviation. The assistant will not make airworthiness determinations or "
                   "identify the people involved. Each question is answered on its own; follow-ups do not see "
                   "earlier answers.")
        if st.button("Clear conversation"):
            st.session_state["history"] = []

    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        st.error("ANTHROPIC_API_KEY is not set. See the README's Setup section, then restart the app.")
        st.stop()
    try:
        con, retriever, client = resources()
    except (duckdb.Error, OSError, RuntimeError) as e:
        st.error(f"Could not load the corpus: {e}. Run the build steps in the README's Usage section.")
        st.stop()

    history = st.session_state.setdefault("history", [])
    for i, record in enumerate(history):
        with st.chat_message("user"):
            st.markdown(md(record["question"]))
        with st.chat_message("assistant"):
            render_record(i, record, con)

    question = st.chat_input("Ask about the maintenance reports") or st.session_state.pop("pending", None)
    if question:
        with st.chat_message("user"):
            st.markdown(md(question))
        with st.chat_message("assistant"):
            record = ask(question, mode, con, retriever, client)
        if record:
            history.append(record)
            st.rerun()


if __name__ == "__main__":
    main()
