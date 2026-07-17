import os

import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()  # works locally if .env exists

# Bridge Streamlit Cloud secrets into os.environ so agent.py / ingest.py
# keep working unchanged whether running locally or on Streamlit Cloud
for _key in ["OPENAI_API_KEY", "PINECONE_API_KEY", "PINECONE_INDEX_NAME", "SUPABASE_URL", "SUPABASE_SERVICE_KEY", "TEACHER_PASSWORD"]:
    if _key in st.secrets:
        os.environ[_key] = st.secrets[_key]

from agent import evaluate_submission
from ingest import ingest_assignment_material

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
TEACHER_PASSWORD = os.environ.get("TEACHER_PASSWORD", "")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(page_title="Assignment Grader", layout="wide")

if "teacher_authenticated" not in st.session_state:
    st.session_state.teacher_authenticated = False


def teacher_login_gate():
    """Show a password prompt and block access until it matches TEACHER_PASSWORD."""
    st.title("Teacher login")

    if not TEACHER_PASSWORD:
        st.error(
            "TEACHER_PASSWORD is not set. Add it to your .env file locally, "
            "or to your Streamlit Cloud secrets, before this page can be used."
        )
        return False

    if st.session_state.teacher_authenticated:
        return True

    entered = st.text_input("Password", type="password")
    if st.button("Log in"):
        if entered == TEACHER_PASSWORD:
            st.session_state.teacher_authenticated = True
            st.rerun()
        else:
            st.error("Incorrect password.")
    return False


page = st.sidebar.radio("Navigate", ["Student: Submit", "Teacher: Setup Assignment", "Dashboard"])

if page == "Teacher: Setup Assignment":
    if not teacher_login_gate():
        st.stop()

    st.sidebar.button("Log out", on_click=lambda: st.session_state.update(teacher_authenticated=False))

    st.title("Set up an assignment")
    title = st.text_input("Assignment title")
    max_score = st.number_input("Max score", value=100)
    rubric_text = st.text_area("Rubric (paste text)", height=250)

    if st.button("Create & index assignment") and title and rubric_text:
        resp = supabase.table("assignments").insert(
            {"title": title, "max_score": max_score, "rubric_summary": rubric_text[:500]}
        ).execute()
        assignment_id = resp.data[0]["id"]
        with st.spinner("Embedding and indexing rubric into Pinecone..."):
            ingest_assignment_material(assignment_id, rubric_text, "rubric")
        st.success(f"Assignment created and indexed (id: {assignment_id})")

elif page == "Student: Submit":
    st.title("Submit your assignment")
    assignments = supabase.table("assignments").select("id, title").execute().data
    options = {a["title"]: a["id"] for a in assignments}

    if not options:
        st.info("No assignments have been created yet.")
    else:
        choice = st.selectbox("Assignment", list(options.keys()))
        submission = st.text_area("Paste your submission", height=300)
        student_id = st.text_input("Student ID / email")

        if st.button("Submit for grading") and submission:
            with st.spinner("Evaluating your submission..."):
                result = evaluate_submission(options[choice], submission, student_id)

            st.subheader(f"Score: {result['total_score']} / {result['max_score']}")
            if result["flagged_for_review"]:
                st.warning("This submission was flagged for teacher review.")

            for c in result["criteria"]:
                st.markdown(f"**{c['criterion']}**: {c['points_awarded']}/{c['points_possible']}")
                st.caption(c["comment"])

            st.markdown("### Overall feedback")
            st.write(result["overall_feedback"])

elif page == "Dashboard":
    if not teacher_login_gate():
        st.stop()

    st.sidebar.button("Log out", on_click=lambda: st.session_state.update(teacher_authenticated=False))

    st.title("Evaluation dashboard")
    evals = (
        supabase.table("evaluations")
        .select("*")
        .order("evaluated_at", desc=True)
        .execute()
        .data
    )
    st.dataframe(evals)