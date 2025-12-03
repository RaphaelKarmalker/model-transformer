# agent_orchester.py

from pydantic import BaseModel
from pydantic_ai import Agent
from langgraph.graph import StateGraph, END
from typing import TypedDict, Union


# ------------------------------------------------------------
# 0. OBJECT CATALOG
# ------------------------------------------------------------

OBJECT_ID_TABLE = {
    "left_gear": "2953382",
    "right_gear": "4239423",
    "main_shaft": "9923411",
    "front_left_wheel": "1844231",
    "front_right_wheel": "1844232",
    "rear_left_wheel": "1844233",
    "rear_right_wheel": "1844234",
    "steering_column": "5539211",
    "engine_block": "3421299",
    "cooling_fan": "7712934",
}


# ------------------------------------------------------------
# 1. PYDANTIC MODELS
# ------------------------------------------------------------

class ObjectMatch(BaseModel):
    object_id: str
    object_name: str


class ModelTransformation(BaseModel):
    object_id: str
    rotate_x: float
    rotate_y: float
    rotate_z: float
    zoom: float


# ------------------------------------------------------------
# 2. AGENTS
# ------------------------------------------------------------

agent_identifier = Agent(
    model="openai:gpt-4o-mini",
    output_type=ObjectMatch,
    system_prompt=(
        "Your task is to identify the referenced object from the provided prompt.\n"
        "You receive:\n"
        "- a user prompt\n"
        "- a dictionary of known object names and IDs\n\n"
        "Rules:\n"
        "1) If the prompt clearly refers to exactly one object in the table, return it.\n"
        "2) If NO object matches, return object_id='' and object_name=''.\n"
        "3) If MORE THAN ONE object matches, return object_id='MULTI' and object_name='MULTI'.\n"
        "Return only the structured fields."
    ),
)

agent_transform = Agent(
    model="openai:gpt-4o-mini",
    output_type=Union[ModelTransformation, str],
    system_prompt=(
        "You must attempt to translate the user's transformation request into a "
        "ModelTransformation object with fields:\n"
        "- rotate_x, rotate_y, rotate_z, zoom\n\n"
        "Rules:\n"
        "1) If the transformation CAN be represented using ONLY these fields, "
        "   return a correct ModelTransformation.\n"
        "2) If the requested action CANNOT be represented using only these fields "
        "   (e.g., flip, mirror, translate, shear, bend, invert, unspecified rotation, "
        "   or anything outside the 4 allowed parameters), return the literal string "
        "   'INVALID'.\n"
        "3) Do NOT attempt workarounds.\n"
        "4) Return exactly one thing: a valid ModelTransformation OR 'INVALID'."
    ),
)


# ------------------------------------------------------------
# 3. LANGGRAPH STATE
# ------------------------------------------------------------

class GraphState(TypedDict):
    prompt: str
    object_table: dict
    object_id: str | None
    object_name: str | None
    transformation: ModelTransformation | None
    cancel_reason: str | None
    user_output: str | None


# ------------------------------------------------------------
# 4. NODES
# ------------------------------------------------------------

# ---------- IDENTIFIER NODE ----------
def node_identifier(state: GraphState):
    text = (
        f"PROMPT:\n{state['prompt']}\n\nOBJECT TABLE:\n{state['object_table']}"
    )

    result = agent_identifier.run_sync(text)
    match = result.output

    # UNKNOWN
    if match.object_id == "":
        return {
            "object_id": None,
            "object_name": None,
            "cancel_reason": "unknown_object",
        }

    # AMBIGUOUS
    if match.object_id == "MULTI":
        return {
            "object_id": None,
            "object_name": None,
            "cancel_reason": "ambiguous",
        }

    # UNIQUE MATCH
    return {
        "object_id": match.object_id,
        "object_name": match.object_name,
        "cancel_reason": None,
    }


# ---------- TRANSFORMATION NODE ----------
def node_transformation(state: GraphState):
    combined = (
        f"PROMPT: {state['prompt']}\n"
        f"OBJECT ID: {state['object_id']}"
    )

    result = agent_transform.run_sync(combined)

    # Agent returns either ModelTransformation OR "INVALID"
    if isinstance(result.output, str) and result.output.strip().upper() == "INVALID":
        return {
            "transformation": None,
            "cancel_reason": "invalid_action",
        }

    return {
        "transformation": result.output,
        "cancel_reason": None,
    }


# ---------- EXECUTE NODE ----------
def node_execute(state: GraphState):
    t = state["transformation"]
    name = state["object_name"]
    oid = state["object_id"]

    msg = (
        f"Executing transformation on '{name}' (ID: {oid}): "
        f"rotate_x={t.rotate_x}, rotate_y={t.rotate_y}, rotate_z={t.rotate_z}, zoom={t.zoom}"
    )

    return {"user_output": msg}


# ---------- CANCEL NODE ----------
def node_cancel(state: GraphState):
    r = state["cancel_reason"]

    if r == "unknown_object":
        msg = "This object does not exist. Please describe the object more clearly."

    elif r == "ambiguous":
        msg = "Your request is ambiguous. Please clarify which object you mean."

    elif r == "invalid_action":
        msg = (
            "The requested action cannot be represented by this system. "
            "Allowed fields are rotate_x, rotate_y, rotate_z, and zoom."
        )

    else:
        msg = "The request could not be completed."

    return {"user_output": msg}


# ------------------------------------------------------------
# 5. BUILD GRAPH
# ------------------------------------------------------------

graph = StateGraph(GraphState)
graph.add_node("identifier", node_identifier)
graph.add_node("transformation", node_transformation)
graph.add_node("execute", node_execute)
graph.add_node("cancel", node_cancel)
graph.set_entry_point("identifier")


# Routing: after identifier
def after_identifier(state: GraphState):
    if state["cancel_reason"] is not None:
        return "cancel"
    return "transformation"

graph.add_conditional_edges("identifier", after_identifier)


# Routing: after transformation
def after_transformation(state: GraphState):
    if state["cancel_reason"] is not None:
        return "cancel"
    return "execute"

graph.add_conditional_edges("transformation", after_transformation)


graph.add_edge("execute", END)
graph.add_edge("cancel", END)

app = graph.compile()


# ------------------------------------------------------------
# 6. MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
   # prompt = "Rotate the main_shaft by 30 degrees on the x-axis."
    prompt = "flip the main shaft by 30 degrees on the x-axis."

    result = app.invoke({"prompt": prompt, "object_table": OBJECT_ID_TABLE})

    print("\n=== RESULT ===")
    print(result["user_output"])
