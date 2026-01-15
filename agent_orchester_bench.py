# agent_orchester.py

import json
import yaml
from typing import TypedDict, Union, Type, TypeVar

from pydantic import BaseModel, Field, ValidationError
from langgraph.graph import StateGraph, END
from transformers import pipeline

T = TypeVar('T', bound=BaseModel)


# ------------------------------------------------------------
# 0. LOAD CONFIG
# ------------------------------------------------------------
def load_config(config_path: str = "config.yaml") -> dict:
    """Lädt die Konfiguration aus der YAML-Datei."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

config = load_config()
MODEL_ID = config['model']['id']
MAX_TOKENS = config['model'].get('max_tokens', 256)
TORCH_DTYPE = config['model'].get('torch_dtype', 'auto')
DEVICE_MAP = config['model'].get('device_map', 'auto')

# Generation parameters
TEMPERATURE = config['generation'].get('temperature', 0.7)
TOP_P = config['generation'].get('top_p', 0.9)
DO_SAMPLE = config['generation'].get('do_sample', True)

print(f"[INFO] Loading model: {MODEL_ID}")
pipe = pipeline(
    "text-generation",
    model=MODEL_ID,
    torch_dtype=TORCH_DTYPE,
    device_map=DEVICE_MAP,
)
print("[INFO] Model loaded successfully!")


# ------------------------------------------------------------
# 0.1 STRUCTURED OUTPUT HELPER
# ------------------------------------------------------------

def extract_json_from_text(text: str) -> str:
    """
    Extracts JSON object from text that may contain additional content.
    Looks for content between first { and last matching }.
    """
    import re
    
    # First try to find JSON in code blocks
    code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if code_block_match:
        return code_block_match.group(1)
    
    # Find the first { and track matching braces
    start_idx = text.find('{')
    if start_idx == -1:
        return ""
    
    brace_count = 0
    end_idx = start_idx
    
    for i, char in enumerate(text[start_idx:], start=start_idx):
        if char == '{':
            brace_count += 1
        elif char == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break
    
    if brace_count == 0:
        return text[start_idx:end_idx + 1]
    
    return ""


def generate_structured_output(
    system_prompt: str,
    user_prompt: str,
    response_model: Type[T],
    max_tokens: int = None,
    max_retries: int = 3
) -> T:
    """
    Generiert strukturierte Outputs mit automatischem Pydantic Parsing und Retry-Logic.
    
    Args:
        system_prompt: System instructions
        user_prompt: User query
        response_model: Pydantic Model class für das erwartete Response-Format
        max_tokens: Max tokens für Generation
        max_retries: Anzahl Wiederholungen bei Parse-Fehlern
    
    Returns:
        Validiertes Pydantic Model Objekt
    """
    if max_tokens is None:
        max_tokens = MAX_TOKENS
    
    # Build a simpler schema description with example
    schema = response_model.model_json_schema()
    properties = schema.get("properties", {})
    required_fields = schema.get("required", [])
    
    # Create a simple field description
    field_descriptions = []
    example_obj = {}
    for field_name, field_info in properties.items():
        field_type = field_info.get("type", "string")
        field_desc = field_info.get("description", "")
        field_descriptions.append(f'  - "{field_name}" ({field_type}): {field_desc}')
        # Create example value
        if field_type == "number":
            example_obj[field_name] = 0.0
        elif field_type == "integer":
            example_obj[field_name] = 0
        elif field_type == "boolean":
            example_obj[field_name] = False
        else:
            example_obj[field_name] = "example_value"
    
    fields_text = "\n".join(field_descriptions)
    example_json = json.dumps(example_obj, indent=2)
    
    enhanced_system_prompt = (
        f"{system_prompt}\n\n"
        f"IMPORTANT: You MUST respond with ONLY a JSON object. No explanations, no analysis, no markdown.\n\n"
        f"Required JSON fields:\n{fields_text}\n\n"
        f"Example response format:\n{example_json}\n\n"
        f"Your response must start with {{ and end with }}. Nothing else."
    )
    
    messages = [
        {"role": "system", "content": enhanced_system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    
    response_text = ""  # Initialize for error handling
    
    for attempt in range(max_retries):
        try:
            # Generate response - use deterministic settings for structured output
            outputs = pipe(
                messages,
                max_new_tokens=max_tokens,
                temperature=0.0,  # Deterministic for consistent JSON
                do_sample=False,  # No sampling for structured output
            )
            
            # Extract assistant response - handle different output formats
            generated_text = outputs[0]["generated_text"]
            if isinstance(generated_text, list):
                # Chat format: list of messages
                response_text = generated_text[-1]["content"] if generated_text else ""
            else:
                # Plain text format: need to extract only the new content
                response_text = str(generated_text)
            
            # Extract JSON from the response (handles extra text before/after JSON)
            json_text = extract_json_from_text(response_text)
            
            if not json_text:
                raise json.JSONDecodeError("No JSON object found in response", response_text, 0)
            
            # Parse JSON and validate with Pydantic
            data = json.loads(json_text)
            return response_model.model_validate(data)
            
        except (json.JSONDecodeError, ValidationError) as e:
            print(f"[WARNING] Attempt {attempt + 1}/{max_retries} failed: {e}")
            if attempt == max_retries - 1:
                raise ValueError(f"Failed to generate valid {response_model.__name__} after {max_retries} attempts")
            # Add error feedback for next attempt
            if response_text:
                messages.append({
                    "role": "assistant",
                    "content": response_text
                })
                messages.append({
                    "role": "user",
                    "content": f"Invalid response. Return ONLY a JSON object like: {example_json}"
                })


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
    """Result from object identification."""
    object_id: str = Field(description="The unique ID of the object, or empty string if not found, or 'MULTI' if ambiguous")
    object_name: str = Field(description="The name of the object, or empty string if not found, or 'MULTI' if ambiguous")


class ModelTransformation(BaseModel):
    """Represents a valid 3D model transformation."""
    object_id: str = Field(description="The unique ID of the object to transform")
    rotate_x: float = Field(description="Rotation around X-axis in degrees")
    rotate_y: float = Field(description="Rotation around Y-axis in degrees")
    rotate_z: float = Field(description="Rotation around Z-axis in degrees")
    zoom: float = Field(description="Zoom factor (1.0 = no zoom, >1 = zoom in, <1 = zoom out)")


# ------------------------------------------------------------
# 3. AGENT FUNCTIONS WITH STRUCTURED OUTPUTS
# ------------------------------------------------------------

def identify_object(prompt_text: str) -> ObjectMatch:
    """
    Identifiziert ein Objekt aus dem Prompt mit strukturiertem Output.
    
    Rules:
    1) Wenn genau 1 Objekt gefunden → return object_id + object_name
    2) Wenn kein Objekt gefunden → object_id='' und object_name=''
    3) Wenn mehrere Objekte → object_id='MULTI' und object_name='MULTI'
    """
    system_prompt = (
        "Your task is to identify the referenced object from the provided prompt.\n"
        "You receive a user prompt and a dictionary of known object names and IDs.\n\n"
        "Rules:\n"
        "1) If the prompt clearly refers to exactly one object in the table, return its object_id and object_name.\n"
        "2) If NO object matches, return object_id='' and object_name=''.\n"
        "3) If MORE THAN ONE object matches, return object_id='MULTI' and object_name='MULTI'."
    )
    
    return generate_structured_output(
        system_prompt=system_prompt,
        user_prompt=prompt_text,
        response_model=ObjectMatch,
        max_tokens=256  # More tokens for chain-of-thought models
    )


class TransformationResponse(BaseModel):
    """Union response for transformation validation."""
    # If valid transformation
    object_id: str | None = Field(default=None, description="The unique ID of the object to transform")
    rotate_x: float | None = Field(default=None, description="Rotation around X-axis in degrees")
    rotate_y: float | None = Field(default=None, description="Rotation around Y-axis in degrees")
    rotate_z: float | None = Field(default=None, description="Rotation around Z-axis in degrees")
    zoom: float | None = Field(default=None, description="Zoom factor (1.0 = no zoom)")
    # If invalid transformation
    status: str | None = Field(default=None, description="Set to 'INVALID' if transformation cannot be represented")


def validate_transformation(prompt_text: str) -> Union[ModelTransformation, str]:
    """
    Validiert eine Transformation mit strukturiertem Output.
    
    Returns:
    - ModelTransformation wenn darstellbar
    - String "INVALID" wenn nicht darstellbar
    """
    system_prompt = (
        "Translate the user's transformation request into JSON.\n"
        "Allowed: rotate_x, rotate_y, rotate_z (degrees), zoom (1.0=none).\n\n"
        "For VALID transformations (rotate, zoom only):\n"
        '{"object_id": "ID", "rotate_x": 0, "rotate_y": 0, "rotate_z": 0, "zoom": 1.0, "status": ""}\n\n'
        "For INVALID transformations (flip, mirror, move, translate, shear):\n"
        '{"status": "INVALID"}\n\n'
        "IMPORTANT: flip is NOT rotate. flip=mirror=INVALID."
    )
    
    try:
        result = generate_structured_output(
            system_prompt=system_prompt,
            user_prompt=prompt_text,
            response_model=TransformationResponse,
            max_tokens=512  # More tokens for chain-of-thought models
        )
        
        # Check if it's an INVALID response
        if result.status and result.status.upper() == "INVALID":
            return "INVALID"
        
        # Check if we have valid transformation fields
        if result.rotate_x is not None and result.rotate_y is not None and result.rotate_z is not None and result.zoom is not None:
            return ModelTransformation(
                object_id=result.object_id or "",
                rotate_x=result.rotate_x,
                rotate_y=result.rotate_y,
                rotate_z=result.rotate_z,
                zoom=result.zoom
            )
        
        # If neither valid transformation nor explicit INVALID, assume invalid
        return "INVALID"
        
    except ValueError as e:
        print(f"[ERROR] validate_transformation failed: {e}")
        return "INVALID"




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
    print(f"[DEBUG] Calling identify_object with structured output...")

    match = identify_object(text)
    print(f"[DEBUG] Structured output result: {match}")
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
    print(f"[DEBUG] Calling validate_transformation with structured output...")

    result = validate_transformation(combined)
    print(f"[DEBUG] Structured output result: {result}")

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
