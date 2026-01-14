# agent_orchester.py

import os
import json
import requests
from typing import TypedDict, Union

from pydantic import BaseModel
from langgraph.graph import StateGraph, END


# ------------------------------------------------------------
# 0. MODEL CONFIG (Ollama Native API)
# ------------------------------------------------------------
OLLAMA_BASE_URL = "http://127.0.0.1:11434"
MODEL_NAME = "llama3.2"  # oder "qwen2.5:0.5b"


# ------------------------------------------------------------
# 1. OBJECT CATALOG
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
# 2. PYDANTIC MODELS
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
# 3. OLLAMA NATIVE API FUNCTIONS
# ------------------------------------------------------------

def ollama_identify_object(prompt_text: str, model: str = MODEL_NAME) -> ObjectMatch:
    """
    Identifiziert ein Objekt aus dem Prompt.
    
    Rules:
    1) Wenn genau 1 Objekt gefunden → return object_id + object_name
    2) Wenn kein Objekt gefunden → object_id='' und object_name=''
    3) Wenn mehrere Objekte → object_id='MULTI' und object_name='MULTI'
    """
    system_prompt = (
        "Your task is to identify the referenced object from the provided prompt.\n"
        "You receive:\n"
        "- a user prompt\n"
        "- a dictionary of known object names and IDs\n\n"
        "Rules:\n"
        "1) If the prompt clearly refers to exactly one object in the table, return it.\n"
        "2) If NO object matches, return object_id='' and object_name=''.\n"
        "3) If MORE THAN ONE object matches, return object_id='MULTI' and object_name='MULTI'.\n"
        "Return only valid JSON matching this schema: {\"object_id\": \"string\", \"object_name\": \"string\"}"
    )
    
    full_prompt = f"{system_prompt}\n\n{prompt_text}"
    
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": full_prompt,
        "format": "json",
        "stream": False
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    response_text = result.get("response", "{}")
    data = json.loads(response_text)
    
    return ObjectMatch(**data)


def ollama_validate_transformation(prompt_text: str, model: str = MODEL_NAME) -> Union[ModelTransformation, str]:
    """
    Validiert, ob die Transformation mit den erlaubten Feldern darstellbar ist.
    
    Returns:
    - ModelTransformation wenn darstellbar
    - String "INVALID" wenn nicht darstellbar (flip, mirror, translate, etc.)
    """
    system_prompt = (
        "You must attempt to translate the user's transformation request into a "
        "ModelTransformation object with fields:\n"
        "- rotate_x, rotate_y, rotate_z, zoom\n\n"
        "Rules:\n"
        "1) If the transformation CAN be represented using ONLY these fields, "
        "   return valid JSON: {\"object_id\": \"string\", \"rotate_x\": float, \"rotate_y\": float, \"rotate_z\": float, \"zoom\": float}\n"
        "2) If the requested action CANNOT be represented using only these fields "
        "   (e.g., flip, mirror, translate, shear, bend, invert, unspecified rotation, "
        "   or anything outside the 4 allowed parameters), return exactly: {\"status\": \"INVALID\"}\n"
        "3) Do NOT attempt workarounds.\n"
        "Return ONLY valid JSON."
    )
    
    full_prompt = f"{system_prompt}\n\n{prompt_text}"
    
    url = f"{OLLAMA_BASE_URL}/api/generate"
    payload = {
        "model": model,
        "prompt": full_prompt,
        "format": "json",
        "stream": False
    }
    
    response = requests.post(url, json=payload)
    response.raise_for_status()
    
    result = response.json()
    response_text = result.get("response", "{}")
    data = json.loads(response_text)
    
    # Check if INVALID
    if "status" in data and data["status"] == "INVALID":
        return "INVALID"
    
    return ModelTransformation(**data)


# ------------------------------------------------------------
# 4. LANGGRAPH STATE
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
# 5. NODES
# ------------------------------------------------------------

def node_identifier(state: GraphState):
    print("[DEBUG] node_identifier START")
    text = (
        f"PROMPT:\n{state['prompt']}\n\n"
        f"OBJECT TABLE:\n{state['object_table']}"
    )
    print(f"[DEBUG] Sending to ollama_identify_object: {text[:100]}...")

    match = ollama_identify_object(text)
    print(f"[DEBUG] ollama_identify_object result: {match}")
    print(f"[DEBUG] Matched object_id={match.object_id}, object_name={match.object_name}")

    # UNKNOWN
    if match.object_id == "":
        print("[DEBUG] Object UNKNOWN")
        return {
            "object_id": None,
            "object_name": None,
            "cancel_reason": "unknown_object",
        }

    # AMBIGUOUS
    if match.object_id == "MULTI":
        print("[DEBUG] Object AMBIGUOUS")
        return {
            "object_id": None,
            "object_name": None,
            "cancel_reason": "ambiguous",
        }

    # UNIQUE MATCH
    print("[DEBUG] Object FOUND")
    return {
        "object_id": match.object_id,
        "object_name": match.object_name,
        "cancel_reason": None,
    }


def node_transformation(state: GraphState):
    print("[DEBUG] node_transformation START")
    combined = (
        f"PROMPT: {state['prompt']}\n"
        f"OBJECT ID: {state['object_id']}\n"
        f"OBJECT NAME: {state['object_name']}"
    )
    print(f"[DEBUG] Sending to ollama_validate_transformation: {combined}")

    result = ollama_validate_transformation(combined)
    print(f"[DEBUG] ollama_validate_transformation result: {result}")

    # Returns either ModelTransformation OR "INVALID"
    if isinstance(result, str) and result.strip().upper() == "INVALID":
        print("[DEBUG] Transformation INVALID")
        return {
            "transformation": None,
            "cancel_reason": "invalid_action",
        }

    t: ModelTransformation = result
    print(f"[DEBUG] Transformation: rotate_x={t.rotate_x}, rotate_y={t.rotate_y}, rotate_z={t.rotate_z}, zoom={t.zoom}")

    # Ensure object_id is set consistently (helpful for downstream)
    if not getattr(t, "object_id", None):
        t.object_id = state["object_id"]  # type: ignore[assignment]

    print("[DEBUG] Transformation VALID")
    return {
        "transformation": t,
        "cancel_reason": None,
    }


def node_execute(state: GraphState):
    t = state["transformation"]
    name = state["object_name"]
    oid = state["object_id"]

    if t is None or name is None or oid is None:
        return {"user_output": "Internal error: missing transformation or object."}

    msg = (
        f"Executing transformation on '{name}' (ID: {oid}): "
        f"rotate_x={t.rotate_x}, rotate_y={t.rotate_y}, rotate_z={t.rotate_z}, zoom={t.zoom}"
    )

    return {"user_output": msg}


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
# 6. BUILD GRAPH
# ------------------------------------------------------------

graph = StateGraph(GraphState)
graph.add_node("identifier", node_identifier)
graph.add_node("transformation", node_transformation)
graph.add_node("execute", node_execute)
graph.add_node("cancel", node_cancel)
graph.set_entry_point("identifier")


def after_identifier(state: GraphState):
    if state["cancel_reason"] is not None:
        return "cancel"
    return "transformation"


graph.add_conditional_edges("identifier", after_identifier)


def after_transformation(state: GraphState):
    if state["cancel_reason"] is not None:
        return "cancel"
    return "execute"


graph.add_conditional_edges("transformation", after_transformation)

graph.add_edge("execute", END)
graph.add_edge("cancel", END)

app = graph.compile()


# ------------------------------------------------------------
# 7. MAIN
# ------------------------------------------------------------

if __name__ == "__main__":
    print("[DEBUG] Script START")
    # prompt = "Rotate the main_shaft by 30 degrees on the x-axis."
    prompt = "flip the main shaft by 30 degrees on the x-axis."
    print(f"[DEBUG] Prompt: {prompt}")

    print("[DEBUG] Invoking graph...")
    result = app.invoke({"prompt": prompt, "object_table": OBJECT_ID_TABLE})

    print("\n=== RESULT ===")
    print(result["user_output"])
