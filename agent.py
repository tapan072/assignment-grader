"""
LangGraph agent for evaluating student assignments.

Pipeline:
  fetch_assignment_meta -> retrieve_context -> evaluate -> save_evaluation

- Pinecone holds the rubric / reference material, namespaced per assignment.
- Supabase holds assignment metadata and the resulting evaluations.
- The LLM produces a structured (Pydantic-validated) grade + feedback object.

Required environment variables:
  OPENAI_API_KEY
  PINECONE_API_KEY
  PINECONE_INDEX_NAME   (optional, defaults to "assignments")
  SUPABASE_URL
  SUPABASE_SERVICE_KEY
"""

import os
import json
from typing import TypedDict, List, Optional
# Tapan!@#123
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field
from supabase import create_client, Client

# ---------- Config ----------
PINECONE_API_KEY = os.environ["PINECONE_API_KEY"]
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME", "assignments")
SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)
llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)

pc = Pinecone(api_key=PINECONE_API_KEY)
pinecone_index = pc.Index(PINECONE_INDEX_NAME)

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# ---------- Structured output schema ----------
class CriterionScore(BaseModel):
    criterion: str
    points_awarded: float
    points_possible: float
    comment: str


class Evaluation(BaseModel):
    total_score: float
    max_score: float
    criteria: List[CriterionScore]
    overall_feedback: str
    flagged_for_review: bool = Field(
        description="True if the submission is ambiguous, incomplete, or the grader is unsure"
    )


structured_llm = llm.with_structured_output(Evaluation)


# ---------- Graph state ----------
class GraderState(TypedDict):
    assignment_id: str
    submission_text: str
    student_id: Optional[str]
    assignment_meta: Optional[dict]
    retrieved_context: Optional[str]
    evaluation: Optional[dict]


# ---------- Nodes ----------
def fetch_assignment_meta(state: GraderState) -> GraderState:
    """Pull assignment metadata (title, max_score, rubric_summary) from Supabase."""
    resp = (
        supabase.table("assignments")
        .select("*")
        .eq("id", state["assignment_id"])
        .single()
        .execute()
    )
    state["assignment_meta"] = resp.data
    return state


def retrieve_context(state: GraderState) -> GraderState:
    """Retrieve rubric criteria + reference material from Pinecone, namespaced per assignment."""
    vectorstore = PineconeVectorStore(
        index=pinecone_index,
        embedding=embeddings,
        namespace=f"assignment_{state['assignment_id']}",
    )
    docs = vectorstore.similarity_search(state["submission_text"], k=6)
    context = "\n\n---\n\n".join(d.page_content for d in docs)
    state["retrieved_context"] = context
    return state


def evaluate(state: GraderState) -> GraderState:
    """Run the LLM grading step against rubric context + submission."""
    meta = state["assignment_meta"] or {}
    prompt = f"""You are an assignment grader.

Assignment: {meta.get('title', 'Unknown')}
Max score: {meta.get('max_score', 100)}

Rubric and reference material:
{state['retrieved_context']}

Student submission:
{state['submission_text']}

Evaluate the submission criterion-by-criterion against the rubric above.
Score each criterion, sum for the total, and explain your reasoning per criterion.
If the submission is ambiguous, incomplete, or you're unsure, set flagged_for_review=true
rather than guessing.
"""
    result: Evaluation = structured_llm.invoke(prompt)
    state["evaluation"] = result.model_dump()
    return state


def save_evaluation(state: GraderState) -> GraderState:
    """Persist the evaluation to Supabase."""
    eval_data = state["evaluation"]
    supabase.table("evaluations").insert(
        {
            "assignment_id": state["assignment_id"],
            "student_id": state.get("student_id"),
            "score": eval_data["total_score"],
            "feedback_json": eval_data,
            "flagged_for_review": eval_data["flagged_for_review"],
        }
    ).execute()
    return state


# ---------- Build graph ----------
graph = StateGraph(GraderState)
graph.add_node("fetch_assignment_meta", fetch_assignment_meta)
graph.add_node("retrieve_context", retrieve_context)
graph.add_node("evaluate", evaluate)
graph.add_node("save_evaluation", save_evaluation)

graph.set_entry_point("fetch_assignment_meta")
graph.add_edge("fetch_assignment_meta", "retrieve_context")
graph.add_edge("retrieve_context", "evaluate")
graph.add_edge("evaluate", "save_evaluation")
graph.add_edge("save_evaluation", END)

app = graph.compile()


# ---------- Public entrypoint ----------
def evaluate_submission(assignment_id: str, submission_text: str, student_id: str = None) -> dict:
    result = app.invoke(
        {
            "assignment_id": assignment_id,
            "submission_text": submission_text,
            "student_id": student_id,
        }
    )
    return result["evaluation"]


if __name__ == "__main__":
    out = evaluate_submission("demo-assignment-1", "Sample student submission text...")
    print(json.dumps(out, indent=2))
