"""
Generative Modeling Extension for Isaac Sim
Combines USD operations with LLM-based chat agent for transformations and color changes.

Features:
- Chat Mode: Natural language commands via LLM agent
- Static Menu Mode: Direct controls without LLM
- Memory: Keeps last 3 conversation turns for context
- Dynamic object discovery from USD stage
"""

from __future__ import annotations

import json
import os
import re
import threading
import traceback
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple, TypedDict, Union, Type, TypeVar

import omni.ext
import omni.ui as ui
import omni.usd

from pxr import Gf, Sdf, Usd, UsdGeom, UsdLux, UsdShade

# Global flags - will be set in on_startup
AGENT_AVAILABLE = False
yaml = None
BaseModel = None
Field = None
ValidationError = None
StateGraph = None
END = None
pipeline = None


T = TypeVar('T', bound=object)


# --------------------------
# Configuration
# --------------------------

KEYWORD_ALIASES = {
    "compond": "compound",
}

# keyword -> (group token in prim path, RGBA)
KEYWORD_TO_GROUP_AND_COLOR: Dict[str, Tuple[str, Gf.Vec4f]] = {
    "compound": ("compound_gear", Gf.Vec4f(0.10, 0.80, 0.20, 1.0)),  # green
    "small": ("group_small",      Gf.Vec4f(0.20, 0.50, 1.00, 1.0)),  # blue
    "big": ("group_big",          Gf.Vec4f(1.00, 0.60, 0.10, 1.0)),  # orange
    "red": ("group_red",          Gf.Vec4f(1.00, 0.20, 0.20, 1.0)),  # red
    "yellow": ("group_yellow",    Gf.Vec4f(1.00, 0.90, 0.10, 1.0)),  # yellow
}

LOOKS_ROOT_PATH = Sdf.Path("/World/Looks")

# Default object table (can be refreshed from USD)
DEFAULT_OBJECT_TABLE = {
    "left_gear": "/World/left_gear",
    "right_gear": "/World/right_gear",
    "main_shaft": "/World/main_shaft",
    "front_left_wheel": "/World/front_left_wheel",
    "front_right_wheel": "/World/front_right_wheel",
    "rear_left_wheel": "/World/rear_left_wheel",
    "rear_right_wheel": "/World/rear_right_wheel",
    "steering_column": "/World/steering_column",
    "engine_block": "/World/engine_block",
    "cooling_fan": "/World/cooling_fan",
}


# --------------------------
# Pydantic Models (lazy loaded)
# --------------------------

# These will be created dynamically when AGENT_AVAILABLE is True
ObjectMatch = None
ModelTransformation = None
ColorChange = None
AgentResponse = None

def _create_pydantic_models():
    """Create Pydantic models after dependencies are loaded."""
    global ObjectMatch, ModelTransformation, ColorChange, AgentResponse
    
    if not AGENT_AVAILABLE or BaseModel is None:
        return
    
    class _ObjectMatch(BaseModel):
        """Result from object identification."""
        object_id: str = Field(description="The USD path of the object, or empty string if not found, or 'MULTI' if ambiguous")
        object_name: str = Field(description="The name of the object, or empty string if not found, or 'MULTI' if ambiguous")

    class _ModelTransformation(BaseModel):
        """Represents a valid 3D model transformation."""
        object_id: str = Field(description="The USD path of the object to transform")
        rotate_x: float = Field(default=0.0, description="Rotation around X-axis in degrees")
        rotate_y: float = Field(default=0.0, description="Rotation around Y-axis in degrees")
        rotate_z: float = Field(default=0.0, description="Rotation around Z-axis in degrees")
        translate_x: float = Field(default=0.0, description="Translation on X-axis")
        translate_y: float = Field(default=0.0, description="Translation on Y-axis")
        translate_z: float = Field(default=0.0, description="Translation on Z-axis")
        zoom: float = Field(default=1.0, description="Scale factor (1.0 = no change)")

    class _ColorChange(BaseModel):
        """Represents a color change request."""
        object_id: str = Field(description="The USD path of the object")
        color_r: float = Field(default=1.0, description="Red component (0-1)")
        color_g: float = Field(default=1.0, description="Green component (0-1)")
        color_b: float = Field(default=1.0, description="Blue component (0-1)")

    class _AgentResponse(BaseModel):
        """Union response from agent - determines action type."""
        action_type: str = Field(description="Type of action: 'transform', 'color', 'unknown', 'ambiguous', 'invalid'")
        object_id: str = Field(default="", description="USD path of object")
        object_name: str = Field(default="", description="Name of object")
        # Transform fields
        rotate_x: float = Field(default=0.0, description="Rotation X in degrees")
        rotate_y: float = Field(default=0.0, description="Rotation Y in degrees")
        rotate_z: float = Field(default=0.0, description="Rotation Z in degrees")
        translate_x: float = Field(default=0.0, description="Translation X")
        translate_y: float = Field(default=0.0, description="Translation Y")
        translate_z: float = Field(default=0.0, description="Translation Z")
        scale: float = Field(default=1.0, description="Scale factor")
        # Color fields
        color_r: float = Field(default=1.0, description="Red (0-1)")
        color_g: float = Field(default=1.0, description="Green (0-1)")
        color_b: float = Field(default=1.0, description="Blue (0-1)")
        # Material properties
        roughness: float = Field(default=0.5, description="Surface roughness (0=smooth/shiny, 1=rough/matte)")
        metallic: float = Field(default=0.0, description="Metallic appearance (0=dielectric, 1=metal)")
        # Light fields
        light_intensity: float = Field(default=5000.0, description="Light intensity (brightness)")
        # Error info
        error_message: str = Field(default="", description="Error message if action failed")
    
    ObjectMatch = _ObjectMatch
    ModelTransformation = _ModelTransformation
    ColorChange = _ColorChange
    AgentResponse = _AgentResponse
    # removed debug


# --------------------------
# LLM Agent Manager
# --------------------------

class AgentManager:
    """Manages LLM pipeline and structured output generation."""
    
    def __init__(self, config_path: str = None):
        self._pipe = None
        self._config = None
        self._config_path = config_path
        self._loading = False
        self._loaded = False
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        if self._config is not None:
            return self._config
            
        config_path = self._config_path
        if config_path is None:
            # Try to find config relative to this file
            ext_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(ext_dir, "config.yaml")
        
        if os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)
        else:
            # Default config
            self._config = {
                'model': {
                    'id': 'microsoft/Phi-3-mini-4k-instruct',
                    'max_tokens': 256,
                    'torch_dtype': 'auto',
                    'device_map': 'auto'
                },
                'generation': {
                    'temperature': 0.1,
                    'top_p': 0.95,
                    'do_sample': True
                }
            }
        return self._config
    
    def is_loaded(self) -> bool:
        return self._loaded
    
    def is_loading(self) -> bool:
        return self._loading
    
    def load_model(self, callback=None):
        """Load the LLM model (can be called in background thread)."""
        if self._loaded or self._loading:
            if callback:
                callback(self._loaded)
            return
            
        self._loading = True
        config = self._load_config()
        model_id = config['model']['id']
        
        print(f"[INFO] Loading model: {model_id}")
        try:
            self._pipe = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype=config['model'].get('torch_dtype', 'auto'),
                device_map=config['model'].get('device_map', 'auto'),
            )
            self._loaded = True
            print("[INFO] Model loaded successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")
            self._loaded = False
        finally:
            self._loading = False
            
        if callback:
            callback(self._loaded)
    
    def extract_json_from_text(self, text: str) -> str:
        """Extract JSON object from text that may contain additional content."""
        # First try code blocks
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_block_match:
            json_str = code_block_match.group(1)
            return self._fix_json_quotes(json_str)
        
        # Find matching braces
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
            json_str = text[start_idx:end_idx + 1]
            return self._fix_json_quotes(json_str)
        
        return ""
    
    def _fix_json_quotes(self, json_str: str) -> str:
        """Fix common JSON issues like single quotes, trailing commas."""
        # Replace single quotes with double quotes (but not inside strings)
        # This is a simple fix - handles most common cases
        result = []
        in_string = False
        escape_next = False
        string_char = None
        
        for i, char in enumerate(json_str):
            if escape_next:
                result.append(char)
                escape_next = False
                continue
            
            if char == '\\':
                result.append(char)
                escape_next = True
                continue
            
            if char in '"\'':
                if not in_string:
                    in_string = True
                    string_char = char
                    result.append('"')  # Always use double quotes
                elif char == string_char:
                    in_string = False
                    string_char = None
                    result.append('"')  # Always use double quotes
                else:
                    result.append(char)
            else:
                result.append(char)
        
        fixed = ''.join(result)
        
        # Remove trailing commas before } or ]
        fixed = re.sub(r',\s*([}\]])', r'\1', fixed)
        
        # Fix True/False/None to JSON booleans
        fixed = re.sub(r'\bTrue\b', 'true', fixed)
        fixed = re.sub(r'\bFalse\b', 'false', fixed)
        fixed = re.sub(r'\bNone\b', 'null', fixed)
        
        return fixed
    
    def generate_structured_output(
        self,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        conversation_history: List[dict] = None,
        max_tokens: int = None,
        max_retries: int = 3
    ) -> T:
        """Generate structured output with Pydantic validation and retry logic."""
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        config = self._load_config()
        if max_tokens is None:
            max_tokens = config['model'].get('max_tokens', 256)
        
        # Build schema description
        schema = response_model.model_json_schema()
        properties = schema.get("properties", {})
        
        field_descriptions = []
        example_obj = {}
        for field_name, field_info in properties.items():
            field_type = field_info.get("type", "string")
            field_desc = field_info.get("description", "")
            field_descriptions.append(f'  - "{field_name}" ({field_type}): {field_desc}')
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
            f"IMPORTANT: Respond with ONLY a JSON object. No explanations.\n\n"
            f"Required JSON fields:\n{fields_text}\n\n"
            f"Example format:\n{example_json}\n\n"
            f"Start with {{ and end with }}."
        )
        
        messages = [{"role": "system", "content": enhanced_system_prompt}]
        
        # Add conversation history for context
        if conversation_history:
            messages.extend(conversation_history)
        
        messages.append({"role": "user", "content": user_prompt})
        
        response_text = ""
        
        for attempt in range(max_retries):
            try:
                import time
                start_time = time.time()
                print(f"[INFO] LLM inference starting (attempt {attempt + 1}/{max_retries})...")
                
                outputs = self._pipe(
                    messages,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                )
                
                elapsed = time.time() - start_time
                print(f"[INFO] LLM inference completed in {elapsed:.2f}s")
                
                generated_text = outputs[0]["generated_text"]
                if isinstance(generated_text, list):
                    response_text = generated_text[-1]["content"] if generated_text else ""
                else:
                    response_text = str(generated_text)
                
                # Debug: show raw response
                print(f"[DEBUG] Raw LLM response (first 500 chars): {response_text[:500]}")
                
                json_text = self.extract_json_from_text(response_text)
                
                if not json_text:
                    print(f"[DEBUG] No JSON found in response")
                    raise json.JSONDecodeError("No JSON object found", response_text, 0)
                
                print(f"[DEBUG] Extracted JSON: {json_text[:300]}")
                
                data = json.loads(json_text)
                print(f"[DEBUG] Parsed data: {data}")
                return response_model.model_validate(data)
                
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"[WARNING] Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to generate valid {response_model.__name__} after {max_retries} attempts")
                if response_text:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": f"Invalid JSON. Return ONLY valid JSON with double quotes: {example_json}"})


# --------------------------
# Main Extension Class
# --------------------------

class GenerativeModelingExtension(omni.ext.IExt):
    """
    Extension combining static USD operations with LLM-powered chat for transformations.
    """
    
    def on_startup(self, ext_id: str) -> None:
        global AGENT_AVAILABLE, yaml, BaseModel, Field, ValidationError, StateGraph, END, pipeline
        
        # removed debug
        try:
            print("[GenerativeModeling] Extension starting up...")
            
            # Lazy load external dependencies
            # removed debug
            try:
                # removed debug
                import yaml as _yaml
                # removed debug
                from pydantic import BaseModel as _BaseModel, Field as _Field, ValidationError as _ValidationError
                # removed debug
                from langgraph.graph import StateGraph as _StateGraph, END as _END
                # removed debug
                from transformers import pipeline as _pipeline
                
                yaml = _yaml
                BaseModel = _BaseModel
                Field = _Field
                ValidationError = _ValidationError
                StateGraph = _StateGraph
                END = _END
                pipeline = _pipeline
                AGENT_AVAILABLE = True
                # removed debug
                
                # Create Pydantic models now that dependencies are loaded
                _create_pydantic_models()
                
            except Exception as e:
                # removed
                import traceback as tb
                # removed
                AGENT_AVAILABLE = False
            
            self._window: Optional[ui.Window] = None
            self._status_label: Optional[ui.Label] = None
            
            # UI models
            self._keyword_model: Optional[ui.SimpleStringModel] = None
            self._dx_model: Optional[ui.SimpleFloatModel] = None
            self._dy_model: Optional[ui.SimpleFloatModel] = None
            self._dz_model: Optional[ui.SimpleFloatModel] = None
            self._rx_model: Optional[ui.SimpleFloatModel] = None
            self._ry_model: Optional[ui.SimpleFloatModel] = None
            self._rz_model: Optional[ui.SimpleFloatModel] = None
            
            # Material UI models
            self._roughness_model: Optional[ui.SimpleFloatModel] = None
            self._metallic_model: Optional[ui.SimpleFloatModel] = None
            self._color_r_model: Optional[ui.SimpleFloatModel] = None
            self._color_g_model: Optional[ui.SimpleFloatModel] = None
            self._color_b_model: Optional[ui.SimpleFloatModel] = None
            
            # Light UI models
            self._light_x_model: Optional[ui.SimpleFloatModel] = None
            self._light_y_model: Optional[ui.SimpleFloatModel] = None
            self._light_z_model: Optional[ui.SimpleFloatModel] = None
            self._light_intensity_model: Optional[ui.SimpleFloatModel] = None
            self._light_r_model: Optional[ui.SimpleFloatModel] = None
            self._light_g_model: Optional[ui.SimpleFloatModel] = None
            self._light_b_model: Optional[ui.SimpleFloatModel] = None
            self._light_counter: int = 0
            
            # Object selection
            self._object_combo: Optional[ui.ComboBox] = None
            self._object_combo_model: Optional[ui.SimpleIntModel] = None
            self._object_names_list: List[str] = []
            
            # Chat UI
            self._chat_input_model: Optional[ui.SimpleStringModel] = None
            self._chat_container: Optional[ui.VStack] = None
            self._chat_scroll: Optional[ui.ScrollingFrame] = None
            
            # Agent
            self._agent_manager: Optional[AgentManager] = None
            self._object_table: Dict[str, str] = DEFAULT_OBJECT_TABLE.copy()
            
            # Chat memory: stores last 3 turns (6 messages: 3 user + 3 assistant)
            self._chat_history: deque = deque(maxlen=6)
            
            # Pending response for main thread execution
            self._pending_response = None
            self._pending_user_message = None
            self._update_sub = None  # Subscription for main thread callbacks
            
            # Mode: 0 = Static Menu, 1 = Chat
            self._mode_index = ui.SimpleIntModel(0)
            
            # Try to load objects from USD stage
            try:
                stage = omni.usd.get_context().get_stage()
                if stage:
                    self._object_table = self._discover_scene_objects(stage)
                    print(f"[GenerativeModeling] Found {len(self._object_table)} objects from USD")
                else:
                    print("[GenerativeModeling] No USD stage loaded, using default objects")
            except Exception as e:
                print(f"[GenerativeModeling] Error discovering objects: {e}")
            
            self._build_ui()
            print("[GenerativeModeling] Extension startup complete!")
            
            # Show agent status
            if AGENT_AVAILABLE:
                print("[GenerativeModeling] Agent is available! LLM features enabled.")
            else:
                print("[GenerativeModeling] Agent NOT available - LLM features disabled")
            
        except Exception as e:
            print(f"[GenerativeModeling] FATAL ERROR during startup: {e}")
            print(traceback.format_exc())
            raise
        
    def on_shutdown(self) -> None:
        print("[GenerativeModeling] Extension shutting down...")
        
        # Clean up update subscription
        if self._update_sub:
            self._update_sub = None
        
        if self._window:
            self._window.visible = False
            self._window = None
    
    # --------------------------
    # UI Building
    # --------------------------
    
    # -- Color constants for the professional theme --
    _CLR_BG          = 0xFF191C20
    _CLR_BG_RAISED   = 0xFF22262C
    _CLR_BG_INPUT    = 0xFF1C1F24
    _CLR_BG_PANEL    = 0xFF1E2228
    _CLR_BORDER      = 0xFF2E3440
    _CLR_ACCENT      = 0xFF4C9AFF      # Primary blue
    _CLR_ACCENT_HOVER = 0xFF6CB0FF
    _CLR_ACCENT_DIM  = 0xFF2B5580
    _CLR_ACCENT_GREEN = 0xFF4CAF50
    _CLR_DANGER      = 0xFFCF6679
    _CLR_TEXT         = 0xFFD8DEE9
    _CLR_TEXT_DIM     = 0xFF7B869A
    _CLR_TEXT_BRIGHT  = 0xFFECEFF4
    _CLR_SECTION      = 0xFF81A1C1
    _CLR_TAB_ACTIVE   = 0xFF4C9AFF
    _CLR_TAB_INACTIVE = 0xFF22262C
    _CLR_STATUS_BG    = 0xFF161A1E
    _CLR_PRESET       = 0xFF2E3440

    def _build_ui(self) -> None:
        print("[GenerativeModeling] Creating main window...")

        C = GenerativeModelingExtension  # shortcut for color constants

        self._style = {
            "Window": {"background_color": C._CLR_BG},
            # Typography
            "Label": {"color": C._CLR_TEXT, "font_size": 12},
            "Label::section_title": {"color": C._CLR_SECTION, "font_size": 13},
            "Label::field_label": {"color": C._CLR_TEXT_DIM, "font_size": 11},
            "Label::axis_label": {"color": C._CLR_TEXT_DIM, "font_size": 11},
            "Label::status": {"color": C._CLR_TEXT_DIM, "font_size": 10},
            "Label::chat_system": {"color": C._CLR_TEXT_DIM, "font_size": 11},
            # Inputs
            "FloatField": {"background_color": C._CLR_BG_INPUT, "border_radius": 3,
                           "border_color": C._CLR_BORDER, "border_width": 1,
                           "color": C._CLR_TEXT_BRIGHT, "font_size": 12},
            "StringField": {"background_color": C._CLR_BG_INPUT, "border_radius": 3,
                            "border_color": C._CLR_BORDER, "border_width": 1,
                            "color": C._CLR_TEXT_BRIGHT, "font_size": 12},
            # Buttons
            "Button": {"background_color": C._CLR_BG_RAISED, "border_radius": 4,
                       "color": C._CLR_TEXT, "font_size": 11,
                       "border_color": C._CLR_BORDER, "border_width": 1},
            "Button:hovered": {"background_color": 0xFF2E3440},
            "Button::primary": {"background_color": C._CLR_ACCENT, "border_radius": 4,
                                "color": 0xFF000000, "font_size": 11, "border_width": 0},
            "Button::primary:hovered": {"background_color": C._CLR_ACCENT_HOVER},
            "Button::danger": {"background_color": 0xFF3A2030, "border_radius": 4,
                               "color": C._CLR_DANGER, "font_size": 11,
                               "border_color": 0xFF5A2030, "border_width": 1},
            "Button::danger:hovered": {"background_color": 0xFF4A2838},
            "Button::preset": {"background_color": C._CLR_PRESET, "border_radius": 3,
                               "color": C._CLR_TEXT, "font_size": 10, "border_width": 0},
            "Button::preset:hovered": {"background_color": 0xFF3B4252},
            "Button::tab_active": {"background_color": C._CLR_TAB_ACTIVE, "border_radius": 0,
                                   "color": 0xFFFFFFFF, "font_size": 12, "border_width": 0},
            "Button::tab_inactive": {"background_color": C._CLR_TAB_INACTIVE, "border_radius": 0,
                                     "color": C._CLR_TEXT_DIM, "font_size": 12, "border_width": 0},
            "Button::tab_inactive:hovered": {"background_color": 0xFF2A3040},
            "Button::obj_item": {"background_color": C._CLR_BG_INPUT, "border_radius": 3,
                                 "color": C._CLR_TEXT, "font_size": 11, "border_width": 0},
            "Button::obj_item:hovered": {"background_color": C._CLR_ACCENT_DIM},
            # Containers
            "ScrollingFrame": {"background_color": C._CLR_BG_PANEL, "border_radius": 4,
                               "border_color": C._CLR_BORDER, "border_width": 1},
        }

        self._window = ui.Window("Generative Modeling", width=460, height=520, visible=True)
        self._window.frame.set_style(self._style)

        with self._window.frame:
            with ui.VStack(spacing=0):
                # ── Tab Bar ──
                with ui.HStack(height=30, spacing=0):
                    self._static_mode_btn = ui.Button(
                        "  Manual Controls  ", clicked_fn=lambda: self._set_mode(0),
                        name="tab_active", height=30)
                    self._chat_mode_btn = ui.Button(
                        "  AI Chat  ", clicked_fn=lambda: self._set_mode(1),
                        name="tab_inactive", height=30)
                    ui.Spacer()

                # Thin accent line under tabs
                ui.Rectangle(height=2, style={"background_color": C._CLR_ACCENT})

                # ── Content ──
                with ui.ZStack():
                    self._static_frame = ui.Frame(visible=True)
                    with self._static_frame:
                        self._build_static_menu()

                    self._chat_frame = ui.Frame(visible=False)
                    with self._chat_frame:
                        self._build_chat_ui()

                # ── Status Bar ──
                ui.Rectangle(height=1, style={"background_color": C._CLR_BORDER})
                with ui.HStack(height=22, style={"margin_width": 8, "margin_height": 3,
                                                  "background_color": C._CLR_STATUS_BG}):
                    self._status_dot = ui.Rectangle(width=6, height=6,
                        style={"background_color": C._CLR_ACCENT_GREEN, "border_radius": 3})
                    ui.Spacer(width=6)
                    self._status_label = ui.Label("Ready", name="status")
    
    def _set_mode(self, mode: int) -> None:
        """Switch between Static Menu (0) and Chat (1) mode."""
        self._mode_index.set_value(mode)
        self._on_mode_changed(None)
        if mode == 0:
            self._static_mode_btn.name = "tab_active"
            self._chat_mode_btn.name = "tab_inactive"
        else:
            self._static_mode_btn.name = "tab_inactive"
            self._chat_mode_btn.name = "tab_active"
    
    def _build_section_header(self, title: str, icon: str = "") -> None:
        """Render a styled section header with optional icon."""
        C = GenerativeModelingExtension
        with ui.HStack(height=20, spacing=4):
            ui.Rectangle(width=3, height=14, style={"background_color": C._CLR_ACCENT, "border_radius": 1})
            ui.Label(f"{icon}  {title}" if icon else title, name="section_title")
            ui.Spacer()

    def _build_static_menu(self) -> None:
        """Build the static menu UI — modern, grouped, professional."""
        C = GenerativeModelingExtension

        with ui.ScrollingFrame(
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            style={"background_color": C._CLR_BG, "border_width": 0}
        ):
          with ui.VStack(spacing=0, style={"margin_width": 10, "margin_height": 6}):

            # ═══════════════════════════════════════
            #  OBJECT SELECTION
            # ═══════════════════════════════════════
            self._build_section_header("Object Selection")
            ui.Spacer(height=4)
            with ui.HStack(height=26, spacing=4):
                self._manual_path_model = ui.SimpleStringModel(
                    list(self._object_table.values())[0] if self._object_table else "/World/")
                ui.StringField(self._manual_path_model, height=24)
                ui.Button("Refresh", clicked_fn=self._on_refresh_objects_and_combo,
                          width=60, height=24, name="primary")

            ui.Spacer(height=4)
            self._object_names_list = list(self._object_table.keys()) if self._object_table else []
            self._object_paths_list = list(self._object_table.values()) if self._object_table else []
            with ui.ScrollingFrame(height=60,
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED):
                self._object_list_container = ui.VStack(spacing=2)
                self._rebuild_object_buttons()

            # Separator
            ui.Spacer(height=8)
            ui.Rectangle(height=1, style={"background_color": C._CLR_BORDER})
            ui.Spacer(height=8)

            # ═══════════════════════════════════════
            #  MATERIAL / COLOR
            # ═══════════════════════════════════════
            self._build_section_header("Material")
            ui.Spacer(height=4)

            # Color row
            with ui.HStack(height=26, spacing=6):
                ui.Label("Color", name="field_label", width=48)
                ui.Label("R", name="axis_label", width=10)
                self._color_r_model = ui.SimpleFloatModel(0.5)
                ui.FloatField(self._color_r_model, width=48, height=22)
                ui.Label("G", name="axis_label", width=10)
                self._color_g_model = ui.SimpleFloatModel(0.5)
                ui.FloatField(self._color_g_model, width=48, height=22)
                ui.Label("B", name="axis_label", width=10)
                self._color_b_model = ui.SimpleFloatModel(0.5)
                ui.FloatField(self._color_b_model, width=48, height=22)
                ui.Spacer()

            # PBR row
            with ui.HStack(height=26, spacing=6):
                ui.Label("PBR", name="field_label", width=48)
                ui.Label("Rough", name="axis_label", width=34)
                self._roughness_model = ui.SimpleFloatModel(0.5)
                ui.FloatField(self._roughness_model, width=52, height=22)
                ui.Spacer(width=4)
                ui.Label("Metal", name="axis_label", width=34)
                self._metallic_model = ui.SimpleFloatModel(0.0)
                ui.FloatField(self._metallic_model, width=52, height=22)
                ui.Spacer()

            ui.Spacer(height=4)
            with ui.HStack(height=26, spacing=6):
                ui.Button("Apply Material", clicked_fn=self._on_apply_material_clicked,
                          width=110, height=24, name="primary")
                ui.Button("Reset Material", clicked_fn=self._on_reset_material_clicked,
                          width=100, height=24, name="danger")
                ui.Spacer()

            # Presets
            ui.Spacer(height=4)
            ui.Label("Presets", name="field_label")
            ui.Spacer(height=2)
            with ui.HStack(height=22, spacing=3):
                ui.Button("Gold", clicked_fn=lambda: self._set_quick_material(1.0, 0.84, 0.0, 0.3, 1.0), name="preset")
                ui.Button("Silver", clicked_fn=lambda: self._set_quick_material(0.75, 0.75, 0.75, 0.2, 1.0), name="preset")
                ui.Button("Chrome", clicked_fn=lambda: self._set_quick_material(0.55, 0.55, 0.55, 0.05, 1.0), name="preset")
                ui.Button("Plastic", clicked_fn=lambda: self._set_quick_material(0.8, 0.2, 0.2, 0.4, 0.0), name="preset")
                ui.Button("Glass", clicked_fn=lambda: self._set_quick_material(0.9, 0.95, 1.0, 0.0, 0.0), name="preset")
            with ui.HStack(height=22, spacing=3):
                ui.Button("Red", clicked_fn=lambda: self._set_quick_color(1,0,0), name="preset",
                          style={"background_color": 0xFF5C2020})
                ui.Button("Green", clicked_fn=lambda: self._set_quick_color(0,1,0), name="preset",
                          style={"background_color": 0xFF205C20})
                ui.Button("Blue", clicked_fn=lambda: self._set_quick_color(0,0,1), name="preset",
                          style={"background_color": 0xFF20205C})
                ui.Spacer()

            # Separator
            ui.Spacer(height=8)
            ui.Rectangle(height=1, style={"background_color": C._CLR_BORDER})
            ui.Spacer(height=8)

            # ═══════════════════════════════════════
            #  TRANSFORM
            # ═══════════════════════════════════════
            self._build_section_header("Transform")
            ui.Spacer(height=4)

            # Translate
            with ui.HStack(height=26, spacing=6):
                ui.Label("Translate", name="field_label", width=52)
                ui.Label("X", name="axis_label", width=10)
                self._dx_model = ui.SimpleFloatModel(0.0)
                ui.FloatField(self._dx_model, width=50, height=22)
                ui.Label("Y", name="axis_label", width=10)
                self._dy_model = ui.SimpleFloatModel(0.0)
                ui.FloatField(self._dy_model, width=50, height=22)
                ui.Label("Z", name="axis_label", width=10)
                self._dz_model = ui.SimpleFloatModel(0.0)
                ui.FloatField(self._dz_model, width=50, height=22)
                ui.Button("Apply", clicked_fn=self._on_translate_object_clicked,
                          width=50, height=22, name="primary")

            # Rotate
            with ui.HStack(height=26, spacing=6):
                ui.Label("Rotate", name="field_label", width=52)
                ui.Label("X", name="axis_label", width=10)
                self._rx_model = ui.SimpleFloatModel(0.0)
                ui.FloatField(self._rx_model, width=50, height=22)
                ui.Label("Y", name="axis_label", width=10)
                self._ry_model = ui.SimpleFloatModel(0.0)
                ui.FloatField(self._ry_model, width=50, height=22)
                ui.Label("Z", name="axis_label", width=10)
                self._rz_model = ui.SimpleFloatModel(0.0)
                ui.FloatField(self._rz_model, width=50, height=22)
                ui.Button("Apply", clicked_fn=self._on_rotate_object_clicked,
                          width=50, height=22, name="primary")

            ui.Spacer(height=4)
            with ui.HStack(height=22, spacing=3):
                ui.Button("+90° X", clicked_fn=lambda: self._quick_rotate(90,0,0), name="preset")
                ui.Button("+90° Y", clicked_fn=lambda: self._quick_rotate(0,90,0), name="preset")
                ui.Button("+90° Z", clicked_fn=lambda: self._quick_rotate(0,0,90), name="preset")
                ui.Spacer(width=8)
                ui.Button("Reset Rotation", clicked_fn=self._on_reset_rotation_clicked,
                          height=22, name="danger")
                ui.Spacer()

            # Separator
            ui.Spacer(height=8)
            ui.Rectangle(height=1, style={"background_color": C._CLR_BORDER})
            ui.Spacer(height=8)

            # ═══════════════════════════════════════
            #  LIGHTING
            # ═══════════════════════════════════════
            self._build_section_header("Lighting")
            ui.Spacer(height=4)

            with ui.HStack(height=26, spacing=6):
                ui.Label("Position", name="field_label", width=52)
                ui.Label("X", name="axis_label", width=10)
                self._light_x_model = ui.SimpleFloatModel(0.0)
                ui.FloatField(self._light_x_model, width=50, height=22)
                ui.Label("Y", name="axis_label", width=10)
                self._light_y_model = ui.SimpleFloatModel(300.0)
                ui.FloatField(self._light_y_model, width=50, height=22)
                ui.Label("Z", name="axis_label", width=10)
                self._light_z_model = ui.SimpleFloatModel(0.0)
                ui.FloatField(self._light_z_model, width=50, height=22)
                ui.Spacer()

            with ui.HStack(height=26, spacing=6):
                ui.Label("Intensity", name="field_label", width=52)
                self._light_intensity_model = ui.SimpleFloatModel(5000.0)
                ui.FloatField(self._light_intensity_model, width=60, height=22)
                ui.Spacer(width=4)
                ui.Label("R", name="axis_label", width=10)
                self._light_r_model = ui.SimpleFloatModel(1.0)
                ui.FloatField(self._light_r_model, width=40, height=22)
                ui.Label("G", name="axis_label", width=10)
                self._light_g_model = ui.SimpleFloatModel(1.0)
                ui.FloatField(self._light_g_model, width=40, height=22)
                ui.Label("B", name="axis_label", width=10)
                self._light_b_model = ui.SimpleFloatModel(1.0)
                ui.FloatField(self._light_b_model, width=40, height=22)

            ui.Spacer(height=4)
            with ui.HStack(height=26, spacing=6):
                ui.Button("Create Light", clicked_fn=self._on_create_light_clicked,
                          width=100, height=24, name="primary")
                ui.Button("Delete All Lights", clicked_fn=self._on_delete_lights_clicked,
                          width=120, height=24, name="danger")
                ui.Spacer()

            ui.Spacer(height=6)

            # Legacy (hidden)
            self._legacy_visible = False
            self._legacy_frame = ui.Frame(visible=False)
            with self._legacy_frame:
                with ui.VStack(spacing=2):
                    self._keyword_model = ui.SimpleStringModel("compound")
                    ui.StringField(self._keyword_model, height=20)
                    with ui.HStack(height=20, spacing=3):
                        ui.Button("Apply Color", clicked_fn=self._on_apply_color_clicked, width=65)
                        ui.Button("Translate", clicked_fn=self._on_translate_clicked, width=60)
                        ui.Button("Rotate", clicked_fn=self._on_rotate_clicked, width=50)
    
    def _build_chat_ui(self) -> None:
        """Build the chat interface UI — clean, modern, professional."""
        C = GenerativeModelingExtension

        with ui.VStack(spacing=0, style={"margin_width": 10, "margin_height": 8}):
            # ── Agent header ──
            with ui.HStack(height=28, spacing=8):
                ui.Rectangle(width=3, height=14,
                    style={"background_color": C._CLR_ACCENT, "border_radius": 1})
                ui.Label("AI Assistant", name="section_title")
                ui.Spacer()
                # Status indicator
                agent_status = "Ready" if AGENT_AVAILABLE else "Unavailable"
                status_clr = C._CLR_ACCENT_GREEN if AGENT_AVAILABLE else C._CLR_DANGER
                ui.Rectangle(width=8, height=8,
                    style={"background_color": status_clr, "border_radius": 4})
                ui.Spacer(width=2)
                ui.Label(agent_status, style={"color": status_clr, "font_size": 11})
                ui.Spacer(width=6)
                if AGENT_AVAILABLE:
                    ui.Button("Load Model", clicked_fn=self._on_load_model,
                              width=85, height=24, name="primary")

            ui.Spacer(height=6)

            # ── Chat messages area ──
            self._chat_scroll = ui.ScrollingFrame(
                height=240,
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
                style={"background_color": C._CLR_BG_PANEL, "border_radius": 6,
                       "border_color": C._CLR_BORDER, "border_width": 1}
            )
            with self._chat_scroll:
                self._chat_container = ui.VStack(spacing=4, style={"margin": 8})
                with self._chat_container:
                    ui.Label("Send a message to get started...", name="chat_system")

            ui.Spacer(height=6)

            # ── Input row ──
            with ui.HStack(height=34, spacing=6):
                self._chat_input_model = ui.SimpleStringModel("")
                ui.StringField(self._chat_input_model, height=32, multiline=True,
                    style={"border_radius": 6, "background_color": C._CLR_BG_INPUT,
                           "border_color": C._CLR_ACCENT_DIM, "border_width": 1,
                           "color": C._CLR_TEXT_BRIGHT, "font_size": 12})
                ui.Button("Send", clicked_fn=self._on_send_chat,
                          width=60, height=32, name="primary")

            ui.Spacer(height=6)

            # ── Quick actions ──
            ui.Label("Quick commands", name="field_label")
            ui.Spacer(height=3)
            with ui.HStack(height=22, spacing=3):
                ui.Button("Rotate 45°", clicked_fn=lambda: self._insert_example("Rotate the gear by 45 degrees"), name="preset")
                ui.Button("Make Red", clicked_fn=lambda: self._insert_example("Make it red"), name="preset")
                ui.Button("Shiny Metal", clicked_fn=lambda: self._insert_example("Make it shiny metal"), name="preset")
                ui.Button("Add Light", clicked_fn=lambda: self._insert_example("Add a bright light above the scene at y=500"), name="preset")
            with ui.HStack(height=22, spacing=3):
                ui.Button("Clear Chat", clicked_fn=self._on_clear_chat, name="danger")
                ui.Button("List Objects", clicked_fn=self._on_show_objects_in_chat, name="preset")
                ui.Spacer()
    
    def _on_mode_changed(self, model) -> None:
        """Handle mode switch between Static and Chat."""
        # Get mode from _mode_index since model might be None (when called from _set_mode)
        if model is not None:
            mode = model.as_int
        else:
            mode = self._mode_index.as_int
        
        if mode == 0:
            self._static_frame.visible = True
            self._chat_frame.visible = False
        else:
            self._static_frame.visible = False
            self._chat_frame.visible = True
    
    # --------------------------
    # Chat Actions
    # --------------------------
    
    def _on_load_model(self) -> None:
        """Load the LLM model in background thread."""
        if not AGENT_AVAILABLE:
            self._set_status("Agent dependencies not available.")
            return
        
        if self._agent_manager is None:
            ext_dir = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(ext_dir, "config.yaml")
            self._agent_manager = AgentManager(config_path)
        
        if self._agent_manager.is_loaded():
            self._set_status("Model already loaded.")
            return
        
        if self._agent_manager.is_loading():
            self._set_status("Model is currently loading...")
            return
        
        self._set_status("Loading LLM model... (this may take a while)")
        
        def load_callback(success):
            if success:
                self._set_status("Model loaded successfully! You can now use chat.")
            else:
                self._set_status("Failed to load model. Check console for errors.")
        
        # Load in background thread
        thread = threading.Thread(target=self._agent_manager.load_model, args=(load_callback,))
        thread.start()
    
    def _on_send_chat(self) -> None:
        """Process chat message through the agent."""
        if not AGENT_AVAILABLE:
            self._set_status("Agent not available.")
            return
        
        if self._agent_manager is None or not self._agent_manager.is_loaded():
            self._set_status("Please load the model first.")
            return
        
        user_message = self._chat_input_model.as_string.strip() if self._chat_input_model else ""
        if not user_message:
            self._set_status("Please enter a message.")
            return
        
        # Add user message to UI
        self._add_chat_message("You", user_message, is_user=True)
        
        # Clear input
        self._chat_input_model.set_value("")
        
        # Process in background to avoid UI freeze
        self._set_status("Processing...")
        thread = threading.Thread(target=self._process_chat_message, args=(user_message,))
        thread.start()
    
    def _process_chat_message(self, user_message: str) -> None:
        """Process the chat message through the LLM agent."""
        try:
            import time
            start_time = time.time()
            # removed debug
            print(f"[INFO] Processing chat message...")
            
            # Build the system prompt
            system_prompt = self._build_agent_system_prompt()
            
            # Build user prompt with object table
            user_prompt = f"USER REQUEST: {user_message}\n\nAVAILABLE OBJECTS:\n{json.dumps(self._object_table, indent=2)}"
            
            # Convert deque to list for the conversation history
            history_list = list(self._chat_history)
            
            # removed
            print(f"[INFO] Calling LLM with {len(self._object_table)} objects...")
            
            # Generate structured response
            response = self._agent_manager.generate_structured_output(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=AgentResponse,
                conversation_history=history_list,
                max_tokens=256  # Reduced from 512 for faster response
            )
            
            elapsed = time.time() - start_time
            # removed debug
            print(f"[INFO] Got response in {elapsed:.2f}s: action={response.action_type}")
            
            # Store response for main thread execution
            self._pending_response = response
            self._pending_user_message = user_message
            
            # Subscribe to update events to execute on main thread
            if self._update_sub is None:
                import omni.kit.app
                self._update_sub = omni.kit.app.get_app().get_update_event_stream().create_subscription_to_pop(
                    self._on_update_execute_pending,
                    name="GenerativeModeling_PendingExecution"
                )
            
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}"
            # removed debug
            # removed
            print(f"[ERROR] Chat failed: {error_msg}")
            self._add_chat_message("Agent", error_msg, is_user=False)
            self._set_status(error_msg)
    
    def _on_update_execute_pending(self, event) -> None:
        """Called on main thread update - execute pending response if any."""
        if self._pending_response is None:
            return
        
        try:
            response = self._pending_response
            user_message = self._pending_user_message
            self._pending_response = None
            self._pending_user_message = None
            
            # removed debug
            print(f"[INFO] Executing pending response on main thread: {response.action_type}")
            
            # Process the response (USD operations happen here, on main thread)
            result_message = self._execute_agent_response(response)
            
            # Update chat history (memory)
            self._chat_history.append({"role": "user", "content": user_message})
            self._chat_history.append({"role": "assistant", "content": result_message})
            
            # Update UI
            self._add_chat_message("Agent", result_message, is_user=False)
            self._set_status("Done.")
            
        except Exception as e:
            import traceback
            error_msg = f"Error executing response: {str(e)}"
            # removed debug
            # removed
            print(f"[ERROR] Execute failed: {error_msg}")
            self._add_chat_message("Agent", error_msg, is_user=False)
            self._set_status(error_msg)

    def _build_agent_system_prompt(self) -> str:
        """Build the system prompt for the agent."""
        return """You are a 3D modeling assistant for Isaac Sim. Analyze user requests and determine the appropriate action.

ACTIONS YOU CAN PERFORM:
1. TRANSFORM: rotate, translate, or scale objects
2. COLOR: change object colors
3. UNKNOWN: object not found in the list
4. AMBIGUOUS: multiple objects could match
5. INVALID: request cannot be fulfilled (e.g., flip, mirror, delete)

RULES:
- "flip" is NOT the same as rotate - flip is INVALID
- "mirror" is INVALID
- "delete" or "remove" is INVALID
- Identify the object from the AVAILABLE OBJECTS list by matching names
- For rotations: specify degrees in rotate_x, rotate_y, rotate_z
- For translations: specify units in translate_x, translate_y, translate_z
- For colors: specify RGB values (0-1) in color_r, color_g, color_b
- For scale: use the scale field (1.0 = no change)

Return action_type as one of: 'transform', 'color', 'material', 'light', 'unknown', 'ambiguous', 'invalid'
Set error_message to explain issues if action_type is unknown/ambiguous/invalid.

For LIGHT placement:
- action_type = 'light'
- Use translate_x, translate_y, translate_z for the light POSITION
- Set light_intensity (default 5000, range 100-50000)
- Set color_r, color_g, color_b for light color (default white 1,1,1)
- object_name will be used as the light name

For MATERIAL changes (shiny, matte, metallic, chrome, silver, gold, etc.):
- action_type = 'material'
- Set roughness: 0.0 = very shiny/glossy, 1.0 = very matte/rough
- Set metallic: 0.0 = plastic/dielectric, 1.0 = pure metal
- IMPORTANT: For metallic materials, you MUST also set the color! Metallic materials with black color will appear black.

MATERIAL PRESETS (always set color_r, color_g, color_b along with roughness and metallic):
- Chrome: color_r=0.55, color_g=0.55, color_b=0.55, roughness=0.05, metallic=1.0
- Silver: color_r=0.75, color_g=0.75, color_b=0.75, roughness=0.2, metallic=1.0
- Gold: color_r=1.0, color_g=0.84, color_b=0.0, roughness=0.3, metallic=1.0
- Copper: color_r=0.95, color_g=0.64, color_b=0.54, roughness=0.3, metallic=1.0
- Bronze: color_r=0.8, color_g=0.5, color_b=0.2, roughness=0.4, metallic=1.0
- Shiny plastic: color_r=0.8, color_g=0.2, color_b=0.2, roughness=0.2, metallic=0.0
- Matte plastic: color_r=0.5, color_g=0.5, color_b=0.5, roughness=0.8, metallic=0.0"""
    
    def _execute_agent_response(self, response: 'AgentResponse') -> str:
        """Execute the agent's response and return a result message."""
        action = response.action_type.lower()
        # removed debug
        print(f"[INFO] Executing action: {action} for object: {response.object_name}")
        
        if action == "unknown":
            return f"Object not found. {response.error_message or 'Please specify a valid object from the scene.'}"
        
        if action == "ambiguous":
            return f"Ambiguous request. {response.error_message or 'Please be more specific about which object you mean.'}"
        
        if action == "invalid":
            return f"Cannot perform this action. {response.error_message or 'Only rotate, translate, scale, and color changes are supported.'}"
        
        stage = self._get_stage()
        if stage is None:
            return "No USD stage is loaded."
        
        # Find the prim
        prim = None
        if response.object_id:
            prim = stage.GetPrimAtPath(response.object_id)
        
        if not prim or not prim.IsValid():
            # Try to find by name in object table
            for name, path in self._object_table.items():
                if name.lower() in response.object_name.lower() or response.object_name.lower() in name.lower():
                    prim = stage.GetPrimAtPath(path)
                    if prim and prim.IsValid():
                        break
        
        if not prim or not prim.IsValid():
            return f"Could not find object '{response.object_name}' in the scene."
        
        if action == "transform":
            results = []
            
            # Apply rotation
            if response.rotate_x != 0 or response.rotate_y != 0 or response.rotate_z != 0:
                if self._apply_rotation_delta(prim, Gf.Vec3d(response.rotate_x, response.rotate_y, response.rotate_z)):
                    results.append(f"rotated ({response.rotate_x}, {response.rotate_y}, {response.rotate_z}) degrees")
            
            # Apply translation
            if response.translate_x != 0 or response.translate_y != 0 or response.translate_z != 0:
                if self._apply_translate_delta(prim, Gf.Vec3d(response.translate_x, response.translate_y, response.translate_z)):
                    results.append(f"translated ({response.translate_x}, {response.translate_y}, {response.translate_z})")
            
            # Apply scale
            if response.scale != 1.0:
                if self._apply_scale(prim, response.scale):
                    results.append(f"scaled to {response.scale}")
            
            if results:
                return f"Applied to '{response.object_name}': {', '.join(results)}"
            else:
                return f"No transformation applied to '{response.object_name}'."
        
        elif action == "color":
            rgba = Gf.Vec4f(response.color_r, response.color_g, response.color_b, 1.0)
            roughness = getattr(response, 'roughness', 0.5)
            metallic = getattr(response, 'metallic', 0.0)
            material_path = LOOKS_ROOT_PATH.AppendChild(f"mat_{response.object_name.replace(' ', '_')}")
            self._ensure_xform(stage, LOOKS_ROOT_PATH)
            material = self._get_or_create_preview_material(
                stage, material_path, rgba,
                roughness=roughness,
                metallic=metallic
            )
            
            # Bind to prim and all its mesh children (same as manual mode)
            bound = 0
            if self._bind_material_to_prim(prim, material):
                bound += 1
            for mesh in self._collect_mesh_descendants(prim):
                if self._bind_material_to_prim(mesh, material):
                    bound += 1
            
            if bound > 0:
                return f"Applied color RGB({response.color_r:.2f}, {response.color_g:.2f}, {response.color_b:.2f}) to '{response.object_name}' (rough={roughness:.2f}, metal={metallic:.2f}, {bound} prims)."
            else:
                return f"Failed to apply color to '{response.object_name}'."
        
        elif action == "light":
            # Create a light source
            position = (response.translate_x, response.translate_y, response.translate_z)
            intensity = getattr(response, 'light_intensity', 5000.0)
            color = (response.color_r, response.color_g, response.color_b)
            
            self._light_counter += 1
            light_name = response.object_name.replace(' ', '_') if response.object_name else f"Light_{self._light_counter}"
            
            if self._create_light(stage, position, intensity, color, light_name):
                return f"Created light '{light_name}' at ({position[0]}, {position[1]}, {position[2]}), intensity={intensity}, color=({color[0]:.2f}, {color[1]:.2f}, {color[2]:.2f})"
            else:
                return "Failed to create light."
        
        elif action == "material":
            # Material change without necessarily changing color
            rgba = Gf.Vec4f(response.color_r, response.color_g, response.color_b, 1.0)
            roughness = getattr(response, 'roughness', 0.5)
            metallic = getattr(response, 'metallic', 0.0)
            
            material_path = LOOKS_ROOT_PATH.AppendChild(f"mat_{response.object_name.replace(' ', '_')}_mat")
            self._ensure_xform(stage, LOOKS_ROOT_PATH)
            material = self._get_or_create_preview_material(stage, material_path, rgba, roughness, metallic)
            
            # Bind to prim and all its mesh children (same as manual mode)
            bound = 0
            if self._bind_material_to_prim(prim, material):
                bound += 1
            for mesh in self._collect_mesh_descendants(prim):
                if self._bind_material_to_prim(mesh, material):
                    bound += 1
            
            if bound > 0:
                return f"Applied material (roughness={roughness:.2f}, metallic={metallic:.2f}) to '{response.object_name}' ({bound} prims)."
            else:
                return f"Failed to apply material to '{response.object_name}'."
        
        return "Unknown action type."
    
    def _add_chat_message(self, sender: str, message: str, is_user: bool = True) -> None:
        """Add a message to the chat UI (thread-safe)."""
        def _add_to_ui():
            if not self._chat_container:
                return

            C = GenerativeModelingExtension
            if is_user:
                tag_color = C._CLR_ACCENT
                msg_color = C._CLR_TEXT_BRIGHT
                bg_color = 0xFF1A2636
            else:
                tag_color = C._CLR_ACCENT_GREEN
                msg_color = C._CLR_TEXT
                bg_color = 0xFF1A2A1E

            try:
                with self._chat_container:
                    with ui.HStack(height=0, spacing=6,
                                   style={"background_color": bg_color, "border_radius": 4,
                                          "margin_height": 1, "margin_width": 2}):
                        ui.Rectangle(width=3, style={"background_color": tag_color, "border_radius": 1})
                        ui.Label(f"{sender}", width=48,
                                 style={"color": tag_color, "font_size": 11})
                        ui.Label(message, word_wrap=True,
                                 style={"color": msg_color, "font_size": 12})
            except Exception as e:
                print(f"[WARNING] Could not add chat message: {e}")
        
        # Try to schedule on main thread using asyncio
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(_add_to_ui)
            else:
                _add_to_ui()
        except:
            # Fallback: direct call
            _add_to_ui()
    
    def _on_clear_chat(self) -> None:
        """Clear chat history."""
        self._chat_history.clear()
        if self._chat_container:
            self._chat_container.clear()
            with self._chat_container:
                ui.Label("Chat cleared.", name="chat_system")
        self._set_status("Chat history cleared.")
    
    def _insert_example(self, text: str) -> None:
        """Insert example text into chat input."""
        if self._chat_input_model:
            self._chat_input_model.set_value(text)
    
    def _on_show_objects_in_chat(self) -> None:
        """Show available objects in chat."""
        obj_list = ", ".join(self._object_table.keys())
        self._add_chat_message("System", f"Available objects: {obj_list}", is_user=False)
    
    # --------------------------
    # Static Menu Actions - New Object-based
    # --------------------------
    
    def _rebuild_object_buttons(self) -> None:
        """Rebuild the object selection buttons."""
        if self._object_list_container is None:
            return
        C = GenerativeModelingExtension
        self._object_list_container.clear()
        with self._object_list_container:
            if not self._object_table:
                ui.Label("No objects found. Click Refresh.",
                         style={"color": C._CLR_DANGER, "font_size": 11})
            else:
                for name, path in list(self._object_table.items())[:30]:
                    ui.Button(
                        f"  {name}",
                        clicked_fn=lambda p=path, n=name: self._select_object(p, n),
                        height=22, name="obj_item"
                    )
                if len(self._object_table) > 30:
                    ui.Label(f"  + {len(self._object_table) - 30} more...",
                            style={"color": C._CLR_TEXT_DIM, "font_size": 11})
    
    def _select_object(self, path: str, name: str) -> None:
        """Select an object by setting the path field."""
        if self._manual_path_model:
            self._manual_path_model.set_value(path)
        self._set_status(f"Selected: {name} ({path})")
    
    def _get_selected_object_path(self) -> Optional[str]:
        """Get the USD path of the currently selected object."""
        if self._manual_path_model:
            return self._manual_path_model.as_string
        return None
    
    def _get_selected_object_name(self) -> str:
        """Get the name of the currently selected object."""
        path = self._get_selected_object_path()
        if path:
            # Find name from path
            for name, p in self._object_table.items():
                if p == path:
                    return name
            # Return last part of path
            return path.split("/")[-1]
        return ""
    
    def _on_object_combo_changed(self, model) -> None:
        """Handle object selection from ComboBox."""
        idx = model.as_int
        if 0 <= idx < len(self._object_paths_list):
            selected_path = self._object_paths_list[idx]
            self._manual_path_model.set_value(selected_path)
            self._set_status(f"Selected: {self._object_names_list[idx]}")
    
    def _on_refresh_objects_and_combo(self) -> None:
        """Refresh objects and rebuild ComboBox."""
        self._on_refresh_objects()
        
        # Update lists
        self._object_names_list = list(self._object_table.keys()) if self._object_table else ["(none)"]
        self._object_paths_list = list(self._object_table.values()) if self._object_table else ["/World/"]
        
        # Rebuild the UI to update ComboBox
        if hasattr(self, '_static_frame') and self._static_frame:
            self._static_frame.clear()
            with self._static_frame:
                self._build_static_menu()
        
        self._set_status(f"Found {len(self._object_table)} objects")
    
    def _update_object_combo(self) -> None:
        """Update the object list (legacy, now uses buttons)."""
        self._object_names_list = list(self._object_table.keys())
        self._rebuild_object_buttons()
        # ComboBox in omni.ui needs to be rebuilt - this is a limitation
        self._set_status(f"Object list updated: {len(self._object_names_list)} objects. Re-open window to see changes in dropdown.")
    
    def _on_show_materials_info(self) -> None:
        """Show popup with all available material presets."""
        material_info = """
╔════════════════════════════════════════════════════════════╗
║                 AVAILABLE MATERIAL PRESETS                  ║
╠════════════════════════════════════════════════════════════╣
║  METALS (Metallic = 1.0)                                   ║
║  ─────────────────────────────────────────────────────────  ║
║  Gold     │ RGB(1.00, 0.84, 0.00) │ Roughness: 0.30       ║
║  Silver   │ RGB(0.75, 0.75, 0.75) │ Roughness: 0.20       ║
║  Copper   │ RGB(0.72, 0.45, 0.20) │ Roughness: 0.30       ║
║  Bronze   │ RGB(0.55, 0.47, 0.33) │ Roughness: 0.35       ║
║  Iron     │ RGB(0.30, 0.30, 0.35) │ Roughness: 0.50       ║
║  Chrome   │ RGB(0.55, 0.55, 0.55) │ Roughness: 0.05       ║
╠════════════════════════════════════════════════════════════╣
║  NON-METALS (Metallic = 0.0)                               ║
║  ─────────────────────────────────────────────────────────  ║
║  Plastic  │ RGB(0.80, 0.20, 0.20) │ Roughness: 0.40       ║
║  Rubber   │ RGB(0.10, 0.10, 0.10) │ Roughness: 0.90       ║
║  Glass    │ RGB(0.90, 0.95, 1.00) │ Roughness: 0.00       ║
║  Wood     │ RGB(0.55, 0.35, 0.15) │ Roughness: 0.70       ║
║  Ceramic  │ RGB(0.95, 0.95, 0.90) │ Roughness: 0.30       ║
║  Matte    │ RGB(0.50, 0.50, 0.50) │ Roughness: 1.00       ║
╠════════════════════════════════════════════════════════════╣
║  COLORS                                                     ║
║  ─────────────────────────────────────────────────────────  ║
║  Red, Green, Blue, Yellow, Cyan, Magenta                   ║
╚════════════════════════════════════════════════════════════╝

LLM Chat Commands:
  • "make [object] gold/silver/copper/etc"
  • "make [object] shiny" (roughness=0.1)
  • "make [object] matte" (roughness=0.9)
  • "make [object] metallic" (metallic=1.0)
"""
        print(material_info)
        self._set_status("Material info printed to console. Check terminal output.")
    
    def _set_quick_color(self, r: float, g: float, b: float) -> None:
        """Set quick color values."""
        if self._color_r_model:
            self._color_r_model.set_value(r)
        if self._color_g_model:
            self._color_g_model.set_value(g)
        if self._color_b_model:
            self._color_b_model.set_value(b)
    
    def _set_quick_material(self, r: float, g: float, b: float, roughness: float, metallic: float) -> None:
        """Set quick material preset."""
        self._set_quick_color(r, g, b)
        if self._roughness_model:
            self._roughness_model.set_value(roughness)
        if self._metallic_model:
            self._metallic_model.set_value(metallic)
    
    def _quick_rotate(self, rx: float, ry: float, rz: float) -> None:
        """Apply quick rotation to selected object."""
        path = self._get_selected_object_path()
        if not path:
            self._set_status("No object selected.")
            return
        
        stage = self._get_stage()
        if not stage:
            self._set_status("No USD stage loaded.")
            return
        
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            self._set_status(f"Object not found: {path}")
            return
        
        if self._apply_rotation_delta(prim, Gf.Vec3d(rx, ry, rz)):
            self._set_status(f"Rotated '{self._get_selected_object_name()}' by ({rx}, {ry}, {rz})°")
        else:
            self._set_status("Rotation failed.")
    
    def _on_apply_material_clicked(self) -> None:
        """Apply material with color and PBR properties to selected object."""
        path = self._get_selected_object_path()
        name = self._get_selected_object_name()
        if not path:
            self._set_status("No object selected.")
            return
        
        stage = self._get_stage()
        if not stage:
            self._set_status("No USD stage loaded.")
            return
        
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            self._set_status(f"Object not found: {path}")
            return
        
        r = self._color_r_model.as_float if self._color_r_model else 0.5
        g = self._color_g_model.as_float if self._color_g_model else 0.5
        b = self._color_b_model.as_float if self._color_b_model else 0.5
        roughness = self._roughness_model.as_float if self._roughness_model else 0.5
        metallic = self._metallic_model.as_float if self._metallic_model else 0.0
        
        rgba = Gf.Vec4f(r, g, b, 1.0)
        self._ensure_xform(stage, LOOKS_ROOT_PATH)
        material_path = LOOKS_ROOT_PATH.AppendChild(f"mat_{name}")
        material = self._get_or_create_preview_material(stage, material_path, rgba, roughness, metallic)
        
        # Bind to prim and all its mesh children
        bound = 0
        if self._bind_material_to_prim(prim, material):
            bound += 1
        for mesh in self._collect_mesh_descendants(prim):
            if self._bind_material_to_prim(mesh, material):
                bound += 1
        
        self._set_status(f"Applied material to '{name}': RGB({r:.2f},{g:.2f},{b:.2f}), rough={roughness:.2f}, metal={metallic:.2f} ({bound} prims)")
    
    def _on_reset_material_clicked(self) -> None:
        """Reset material on selected object."""
        path = self._get_selected_object_path()
        name = self._get_selected_object_name()
        if not path:
            self._set_status("No object selected.")
            return
        
        stage = self._get_stage()
        if not stage:
            self._set_status("No USD stage loaded.")
            return
        
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            self._set_status(f"Object not found: {path}")
            return
        
        cleared = 0
        if self._clear_material_binding(prim):
            cleared += 1
        for mesh in self._collect_mesh_descendants(prim):
            if self._clear_material_binding(mesh):
                cleared += 1
        
        self._set_status(f"Cleared material bindings on '{name}' ({cleared} prims)")
    
    def _on_translate_object_clicked(self) -> None:
        """Translate selected object."""
        path = self._get_selected_object_path()
        name = self._get_selected_object_name()
        if not path:
            self._set_status("No object selected.")
            return
        
        stage = self._get_stage()
        if not stage:
            self._set_status("No USD stage loaded.")
            return
        
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            self._set_status(f"Object not found: {path}")
            return
        
        dx = self._dx_model.as_float if self._dx_model else 0.0
        dy = self._dy_model.as_float if self._dy_model else 0.0
        dz = self._dz_model.as_float if self._dz_model else 0.0
        
        if self._apply_translate_delta(prim, Gf.Vec3d(dx, dy, dz)):
            self._set_status(f"Translated '{name}' by ({dx}, {dy}, {dz})")
        else:
            self._set_status("Translation failed.")
    
    def _on_rotate_object_clicked(self) -> None:
        """Rotate selected object."""
        path = self._get_selected_object_path()
        name = self._get_selected_object_name()
        if not path:
            self._set_status("No object selected.")
            return
        
        stage = self._get_stage()
        if not stage:
            self._set_status("No USD stage loaded.")
            return
        
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            self._set_status(f"Object not found: {path}")
            return
        
        rx = self._rx_model.as_float if self._rx_model else 0.0
        ry = self._ry_model.as_float if self._ry_model else 0.0
        rz = self._rz_model.as_float if self._rz_model else 0.0
        
        if self._apply_rotation_delta(prim, Gf.Vec3d(rx, ry, rz)):
            self._set_status(f"Rotated '{name}' by ({rx}, {ry}, {rz})°")
        else:
            self._set_status("Rotation failed.")
    
    def _on_reset_rotation_clicked(self) -> None:
        """Reset rotation on selected object."""
        path = self._get_selected_object_path()
        name = self._get_selected_object_name()
        if not path:
            self._set_status("No object selected.")
            return
        
        stage = self._get_stage()
        if not stage:
            self._set_status("No USD stage loaded.")
            return
        
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsValid():
            self._set_status(f"Object not found: {path}")
            return
        
        try:
            xf = UsdGeom.Xformable(prim)
            ops = xf.GetOrderedXformOps()
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                    op.Set(Gf.Vec3f(0.0, 0.0, 0.0))
                    self._set_status(f"Reset rotation on '{name}'")
                    return
            self._set_status(f"No rotation op found on '{name}'")
        except Exception as e:
            self._set_status(f"Failed to reset rotation: {e}")
    
    # --------------------------
    # Static Menu Actions - Legacy Keyword-based
    # --------------------------
    
    def _normalize_keyword(self, raw: str) -> str:
        k = raw.strip().lower()
        return KEYWORD_ALIASES.get(k, k)
    
    def _on_apply_color_clicked(self) -> None:
        """Apply color based on keyword (static mode)."""
        keyword = self._normalize_keyword(self._keyword_model.as_string if self._keyword_model else "")
        if keyword not in KEYWORD_TO_GROUP_AND_COLOR:
            self._set_status(f"Unsupported keyword '{keyword}'. Use: compound/small/big/red/yellow.")
            return
        
        stage = self._get_stage()
        if stage is None:
            self._set_status("No USD stage is loaded.")
            return
        
        group_token, rgba = KEYWORD_TO_GROUP_AND_COLOR[keyword]
        mesh_prims = self._find_meshes_under_groups(stage, group_token)
        if not mesh_prims:
            self._set_status(f"No Mesh prims found under groups matching '{group_token}'.")
            return
        
        self._ensure_xform(stage, LOOKS_ROOT_PATH)
        material_path = LOOKS_ROOT_PATH.AppendChild(f"mat_{keyword}")
        material = self._get_or_create_preview_material(stage, material_path, rgba)
        
        bound_count = 0
        for prim in mesh_prims:
            if self._bind_material_to_prim(prim, material):
                bound_count += 1
        
        self._set_status(f"Applied '{keyword}' color to {bound_count} mesh prim(s).")
    
    def _on_translate_clicked(self) -> None:
        """Translate objects (static mode)."""
        keyword = self._normalize_keyword(self._keyword_model.as_string if self._keyword_model else "")
        if keyword not in KEYWORD_TO_GROUP_AND_COLOR:
            self._set_status(f"Unsupported keyword '{keyword}'. Use: compound/small/big/red/yellow.")
            return
        
        stage = self._get_stage()
        if stage is None:
            self._set_status("No USD stage is loaded.")
            return
        
        dx = float(self._dx_model.as_float if self._dx_model else 0.0)
        dy = float(self._dy_model.as_float if self._dy_model else 0.0)
        dz = float(self._dz_model.as_float if self._dz_model else 0.0)
        
        group_token, _ = KEYWORD_TO_GROUP_AND_COLOR[keyword]
        roots = self._find_group_roots(stage, group_token)
        if not roots:
            self._set_status(f"No group root prim found matching '{group_token}'.")
            return
        
        moved = 0
        for root_prim in roots:
            if self._apply_translate_delta(root_prim, Gf.Vec3d(dx, dy, dz)):
                moved += 1
        
        self._set_status(f"Translated '{keyword}' on {moved} group root prim(s) by ({dx}, {dy}, {dz}).")
    
    def _on_rotate_clicked(self) -> None:
        """Rotate objects (static mode)."""
        keyword = self._normalize_keyword(self._keyword_model.as_string if self._keyword_model else "")
        if keyword not in KEYWORD_TO_GROUP_AND_COLOR:
            self._set_status(f"Unsupported keyword '{keyword}'. Use: compound/small/big/red/yellow.")
            return
        
        stage = self._get_stage()
        if stage is None:
            self._set_status("No USD stage is loaded.")
            return
        
        rx = float(self._rx_model.as_float if self._rx_model else 0.0)
        ry = float(self._ry_model.as_float if self._ry_model else 0.0)
        rz = float(self._rz_model.as_float if self._rz_model else 0.0)
        
        group_token, _ = KEYWORD_TO_GROUP_AND_COLOR[keyword]
        roots = self._find_group_roots(stage, group_token)
        if not roots:
            self._set_status(f"No group root prim found matching '{group_token}'.")
            return
        
        rotated = 0
        for root_prim in roots:
            if self._apply_rotation_delta(root_prim, Gf.Vec3d(rx, ry, rz)):
                rotated += 1
        
        self._set_status(f"Rotated '{keyword}' on {rotated} group root prim(s) by ({rx}, {ry}, {rz}) degrees.")
    
    def _on_reset_clicked(self) -> None:
        """Reset material bindings (static mode)."""
        stage = self._get_stage()
        if stage is None:
            self._set_status("No USD stage is loaded.")
            return
        
        total_cleared = 0
        for _, (group_token, _) in KEYWORD_TO_GROUP_AND_COLOR.items():
            for prim in self._find_meshes_under_groups(stage, group_token):
                if self._clear_material_binding(prim):
                    total_cleared += 1
        
        self._set_status(f"Cleared bindings on {total_cleared} mesh prim(s).")
    
    def _on_refresh_objects(self) -> None:
        """Refresh object table from USD stage."""
        stage = self._get_stage()
        if stage is None:
            self._set_status("No USD stage is loaded.")
            return
        
        self._object_table = self._discover_scene_objects(stage)
        count = len(self._object_table)
        self._set_status(f"Refreshed object table: found {count} objects.")
    
    # --------------------------
    # Light Actions
    # --------------------------
    
    def _on_create_light_clicked(self) -> None:
        """Create a SphereLight from the UI fields."""
        stage = self._get_stage()
        if not stage:
            self._set_status("No USD stage loaded.")
            return
        
        x = self._light_x_model.as_float if self._light_x_model else 0.0
        y = self._light_y_model.as_float if self._light_y_model else 300.0
        z = self._light_z_model.as_float if self._light_z_model else 0.0
        intensity = self._light_intensity_model.as_float if self._light_intensity_model else 5000.0
        r = self._light_r_model.as_float if self._light_r_model else 1.0
        g = self._light_g_model.as_float if self._light_g_model else 1.0
        b = self._light_b_model.as_float if self._light_b_model else 1.0
        
        self._light_counter += 1
        name = f"Light_{self._light_counter}"
        
        if self._create_light(stage, (x, y, z), intensity, (r, g, b), name):
            self._set_status(f"Created '{name}' at ({x}, {y}, {z}), intensity={intensity}")
        else:
            self._set_status("Failed to create light.")
    
    def _on_delete_lights_clicked(self) -> None:
        """Delete all generated lights under /World/Lights."""
        stage = self._get_stage()
        if not stage:
            self._set_status("No USD stage loaded.")
            return
        
        lights_prim = stage.GetPrimAtPath("/World/Lights")
        if lights_prim and lights_prim.IsValid():
            children = list(lights_prim.GetChildren())
            for child in children:
                stage.RemovePrim(child.GetPath())
            self._light_counter = 0
            self._set_status(f"Deleted {len(children)} light(s).")
        else:
            self._set_status("No lights found.")
    
    def _create_light(self, stage, position, intensity=5000.0, color=(1.0, 1.0, 1.0), name="Light") -> bool:
        """Create a SphereLight at the given position."""
        try:
            light_path = Sdf.Path(f"/World/Lights/{name}")
            self._ensure_xform(stage, Sdf.Path("/World/Lights"))
            
            light = UsdLux.SphereLight.Define(stage, light_path)
            light.CreateIntensityAttr(float(intensity))
            light.CreateColorAttr(Gf.Vec3f(float(color[0]), float(color[1]), float(color[2])))
            light.CreateRadiusAttr(1.0)
            
            # Set position
            xf = UsdGeom.Xformable(light.GetPrim())
            xf.AddTranslateOp().Set(Gf.Vec3d(float(position[0]), float(position[1]), float(position[2])))
            
            return True
        except Exception as e:
            print(f"[ERROR] Light creation failed: {e}")
            return False
    
    # --------------------------
    # USD Helpers
    # --------------------------
    
    def _get_stage(self) -> Optional[Usd.Stage]:
        return omni.usd.get_context().get_stage()
    
    def _discover_scene_objects(self, stage: Usd.Stage) -> Dict[str, str]:
        """Discover objects from the USD stage dynamically."""
        objects = {}
        for prim in stage.Traverse():
            if not prim.IsValid():
                continue
            # Include Xform and Mesh prims
            if prim.IsA(UsdGeom.Xform) or prim.IsA(UsdGeom.Mesh):
                name = prim.GetName()
                path = prim.GetPath().pathString
                # Skip internal paths
                if name.startswith("_") or "/Looks/" in path:
                    continue
                objects[name] = path
        return objects
    
    def _find_group_roots(self, stage: Usd.Stage, group_token: str) -> List[Usd.Prim]:
        """Find Xform prims that match the group token."""
        candidates: List[Usd.Prim] = []
        for prim in stage.Traverse():
            if not prim.IsValid():
                continue
            if group_token not in prim.GetPath().pathString:
                continue
            if prim.IsA(UsdGeom.Xform):
                candidates.append(prim)
        
        if not candidates:
            for prim in stage.Traverse():
                if not prim.IsValid():
                    continue
                if group_token not in prim.GetPath().pathString:
                    continue
                if UsdGeom.Xformable(prim):
                    candidates.append(prim)
        
        seen = set()
        roots: List[Usd.Prim] = []
        for p in candidates:
            ps = p.GetPath().pathString
            if ps not in seen:
                seen.add(ps)
                roots.append(p)
        return roots
    
    def _find_meshes_under_groups(self, stage: Usd.Stage, group_token: str) -> List[Usd.Prim]:
        """Find all Mesh prims under groups matching the token."""
        targets: List[Usd.Prim] = []
        for prim in stage.Traverse():
            if not prim.IsValid():
                continue
            if group_token not in prim.GetPath().pathString:
                continue
            targets.extend(self._collect_mesh_descendants(prim))
        
        seen = set()
        unique: List[Usd.Prim] = []
        for p in targets:
            ps = p.GetPath().pathString
            if ps not in seen:
                seen.add(ps)
                unique.append(p)
        return unique
    
    def _collect_mesh_descendants(self, root: Usd.Prim) -> List[Usd.Prim]:
        """Collect all Mesh prims under a root prim."""
        meshes: List[Usd.Prim] = []
        if root.IsA(UsdGeom.Mesh):
            meshes.append(root)
        for child in Usd.PrimRange(root):
            if child == root:
                continue
            if child.IsA(UsdGeom.Mesh):
                meshes.append(child)
        return meshes
    
    def _apply_translate_delta(self, prim: Usd.Prim, delta: Gf.Vec3d) -> bool:
        """Apply translation delta to a prim."""
        if not prim or not prim.IsValid():
            return False
        
        try:
            xf = UsdGeom.Xformable(prim)
            ops = xf.GetOrderedXformOps()
            translate_op = None
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                    translate_op = op
                    break
            
            if translate_op is None:
                translate_op = xf.AddTranslateOp()
            
            current = translate_op.Get()
            if current is None:
                current = Gf.Vec3d(0.0, 0.0, 0.0)
            
            translate_op.Set(Gf.Vec3d(
                current[0] + delta[0],
                current[1] + delta[1],
                current[2] + delta[2]
            ))
            return True
        except Exception as e:
            print(f"[ERROR] Translation failed: {e}")
            return False
    
    def _apply_rotation_delta(self, prim: Usd.Prim, delta: Gf.Vec3d) -> bool:
        """Apply rotation delta (in degrees) to a prim."""
        if not prim or not prim.IsValid():
            return False
        
        try:
            xf = UsdGeom.Xformable(prim)
            ops = xf.GetOrderedXformOps()
            
            # Look for existing rotation op
            rotate_op = None
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeRotateXYZ:
                    rotate_op = op
                    break
            
            if rotate_op is None:
                rotate_op = xf.AddRotateXYZOp()
            
            current = rotate_op.Get()
            if current is None:
                current = Gf.Vec3f(0.0, 0.0, 0.0)
            
            rotate_op.Set(Gf.Vec3f(
                float(current[0] + delta[0]),
                float(current[1] + delta[1]),
                float(current[2] + delta[2])
            ))
            return True
        except Exception as e:
            print(f"[ERROR] Rotation failed: {e}")
            return False
    
    def _apply_scale(self, prim: Usd.Prim, scale: float) -> bool:
        """Apply uniform scale to a prim."""
        if not prim or not prim.IsValid():
            return False
        
        try:
            xf = UsdGeom.Xformable(prim)
            ops = xf.GetOrderedXformOps()
            
            scale_op = None
            for op in ops:
                if op.GetOpType() == UsdGeom.XformOp.TypeScale:
                    scale_op = op
                    break
            
            if scale_op is None:
                scale_op = xf.AddScaleOp()
            
            scale_op.Set(Gf.Vec3f(scale, scale, scale))
            return True
        except Exception as e:
            print(f"[ERROR] Scale failed: {e}")
            return False
    
    def _ensure_xform(self, stage: Usd.Stage, path: Sdf.Path) -> Usd.Prim:
        """Ensure an Xform prim exists at the given path."""
        prim = stage.GetPrimAtPath(path)
        if prim and prim.IsValid():
            return prim
        return UsdGeom.Xform.Define(stage, path).GetPrim()
    
    def _get_or_create_preview_material(
        self, stage: Usd.Stage, material_path: Sdf.Path, rgba: Gf.Vec4f,
        roughness: float = 0.5, metallic: float = 0.0
    ) -> UsdShade.Material:
        """Get or create a UsdPreviewSurface material with full PBR properties."""
        material = UsdShade.Material.Get(stage, material_path)
        if not material:
            material = UsdShade.Material.Define(stage, material_path)
        
        shader_path = material_path.AppendChild("PreviewShader")
        shader = UsdShade.Shader.Get(stage, shader_path)
        if not shader:
            shader = UsdShade.Shader.Define(stage, shader_path)
            shader.CreateIdAttr("UsdPreviewSurface")
        
        # Color
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(
            Gf.Vec3f(rgba[0], rgba[1], rgba[2])
        )
        shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(float(rgba[3]))
        
        # PBR Properties
        shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(float(roughness))
        shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(float(metallic))
        
        # For metallic materials, use specular workflow hints
        if metallic > 0.5:
            shader.CreateInput("useSpecularWorkflow", Sdf.ValueTypeNames.Int).Set(0)
        
        material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
        return material
    
    def _bind_material_to_prim(self, prim: Usd.Prim, material: UsdShade.Material) -> bool:
        """Bind a material to a prim."""
        if not prim or not prim.IsValid():
            return False
        try:
            UsdShade.MaterialBindingAPI(prim).Bind(material)
            return True
        except Exception:
            return False
    
    def _clear_material_binding(self, prim: Usd.Prim) -> bool:
        """Clear material binding from a prim."""
        if not prim or not prim.IsValid():
            return False
        try:
            UsdShade.MaterialBindingAPI(prim).UnbindDirectBinding()
            return True
        except Exception:
            return False
    
    def _set_status(self, msg: str) -> None:
        """Update the status label (thread-safe)."""
        C = GenerativeModelingExtension

        def _update():
            try:
                if self._status_label:
                    self._status_label.text = msg
                # Update status dot color based on message content
                if hasattr(self, '_status_dot') and self._status_dot:
                    lower = msg.lower()
                    if 'error' in lower or 'fail' in lower:
                        self._status_dot.set_style({"background_color": C._CLR_DANGER, "border_radius": 3})
                    elif 'processing' in lower or 'loading' in lower:
                        self._status_dot.set_style({"background_color": 0xFFFFB74D, "border_radius": 3})
                    else:
                        self._status_dot.set_style({"background_color": C._CLR_ACCENT_GREEN, "border_radius": 3})
            except Exception as e:
                print(f"[WARNING] Could not update status: {e}")
        
        print(f"[GenerativeModeling] {msg}")
        
        # Try to schedule on main thread using asyncio
        try:
            import asyncio
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.call_soon_threadsafe(_update)
            else:
                _update()
        except:
            # Fallback: direct call
            _update()
