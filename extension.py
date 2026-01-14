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
from collections import deque
from typing import Dict, List, Optional, Tuple, TypedDict, Union, Type, TypeVar

import omni.ext
import omni.ui as ui
import omni.usd

from pxr import Gf, Sdf, Usd, UsdGeom, UsdShade

# Pydantic and LangGraph imports (lazy loaded)
try:
    import yaml
    from pydantic import BaseModel, Field, ValidationError
    from langgraph.graph import StateGraph, END
    from transformers import pipeline
    AGENT_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING] Agent dependencies not available: {e}")
    AGENT_AVAILABLE = False


T = TypeVar('T', bound='BaseModel')


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
# Pydantic Models (if available)
# --------------------------

if AGENT_AVAILABLE:
    class ObjectMatch(BaseModel):
        """Result from object identification."""
        object_id: str = Field(description="The USD path of the object, or empty string if not found, or 'MULTI' if ambiguous")
        object_name: str = Field(description="The name of the object, or empty string if not found, or 'MULTI' if ambiguous")

    class ModelTransformation(BaseModel):
        """Represents a valid 3D model transformation."""
        object_id: str = Field(description="The USD path of the object to transform")
        rotate_x: float = Field(default=0.0, description="Rotation around X-axis in degrees")
        rotate_y: float = Field(default=0.0, description="Rotation around Y-axis in degrees")
        rotate_z: float = Field(default=0.0, description="Rotation around Z-axis in degrees")
        translate_x: float = Field(default=0.0, description="Translation on X-axis")
        translate_y: float = Field(default=0.0, description="Translation on Y-axis")
        translate_z: float = Field(default=0.0, description="Translation on Z-axis")
        zoom: float = Field(default=1.0, description="Scale factor (1.0 = no change)")

    class ColorChange(BaseModel):
        """Represents a color change request."""
        object_id: str = Field(description="The USD path of the object")
        color_r: float = Field(default=1.0, description="Red component (0-1)")
        color_g: float = Field(default=1.0, description="Green component (0-1)")
        color_b: float = Field(default=1.0, description="Blue component (0-1)")

    class AgentResponse(BaseModel):
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
        # Error info
        error_message: str = Field(default="", description="Error message if action failed")


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
            return code_block_match.group(1)
        
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
            return text[start_idx:end_idx + 1]
        
        return ""
    
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
                outputs = self._pipe(
                    messages,
                    max_new_tokens=max_tokens,
                    temperature=0.0,
                    do_sample=False,
                )
                
                generated_text = outputs[0]["generated_text"]
                if isinstance(generated_text, list):
                    response_text = generated_text[-1]["content"] if generated_text else ""
                else:
                    response_text = str(generated_text)
                
                json_text = self.extract_json_from_text(response_text)
                
                if not json_text:
                    raise json.JSONDecodeError("No JSON object found", response_text, 0)
                
                data = json.loads(json_text)
                return response_model.model_validate(data)
                
            except (json.JSONDecodeError, ValidationError) as e:
                print(f"[WARNING] Attempt {attempt + 1}/{max_retries} failed: {e}")
                if attempt == max_retries - 1:
                    raise ValueError(f"Failed to generate valid {response_model.__name__} after {max_retries} attempts")
                if response_text:
                    messages.append({"role": "assistant", "content": response_text})
                    messages.append({"role": "user", "content": f"Invalid response. Return ONLY JSON: {example_json}"})


# --------------------------
# Main Extension Class
# --------------------------

class GenerativeModelingExtension(omni.ext.IExt):
    """
    Extension combining static USD operations with LLM-powered chat for transformations.
    """
    
    def on_startup(self, ext_id: str) -> None:
        print("[GenerativeModeling] Extension starting up...")
        
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
        
        # Mode: 0 = Static Menu, 1 = Chat
        self._mode_index = ui.SimpleIntModel(0)
        
        self._build_ui()
        
    def on_shutdown(self) -> None:
        print("[GenerativeModeling] Extension shutting down...")
        if self._window:
            self._window.visible = False
            self._window = None
    
    # --------------------------
    # UI Building
    # --------------------------
    
    def _build_ui(self) -> None:
        self._window = ui.Window("Generative Modeling", width=650, height=600, visible=True)
        
        with self._window.frame:
            with ui.VStack(spacing=6):
                # Mode selector
                ui.Label("Mode:", height=20)
                with ui.HStack(height=30, spacing=10):
                    ui.RadioButton(
                        text="Static Menu",
                        radio_collection=self._mode_index,
                        width=120
                    )
                    ui.RadioButton(
                        text="Chat (LLM)",
                        radio_collection=self._mode_index,
                        width=120
                    )
                    ui.Button("Refresh Objects from USD", clicked_fn=self._on_refresh_objects, width=180)
                
                ui.Separator(height=8)
                
                # Stack for mode-specific content
                with ui.ZStack():
                    # Static Menu Mode
                    self._static_frame = ui.Frame(visible=True)
                    with self._static_frame:
                        self._build_static_menu()
                    
                    # Chat Mode
                    self._chat_frame = ui.Frame(visible=False)
                    with self._chat_frame:
                        self._build_chat_ui()
                
                # Status label at bottom
                ui.Separator(height=4)
                self._status_label = ui.Label("Ready", height=40, word_wrap=True)
        
        # Mode change callback
        self._mode_index.add_value_changed_fn(self._on_mode_changed)
    
    def _build_static_menu(self) -> None:
        """Build the static menu UI (no LLM)."""
        with ui.VStack(spacing=6):
            # Object Selection Section
            ui.Label("Select Object:", height=18, style={"font_size": 14})
            with ui.HStack(height=28, spacing=8):
                self._object_combo_model = ui.SimpleIntModel(0)
                self._object_names_list = list(self._object_table.keys())
                self._object_combo = ui.ComboBox(
                    self._object_combo_model, 
                    *self._object_names_list,
                    width=300
                )
                ui.Button("↻ Refresh", clicked_fn=self._on_refresh_objects_and_combo, width=80)
            
            ui.Separator(height=8)
            
            # === MATERIAL SECTION ===
            ui.Label("Material Properties:", height=18, style={"font_size": 14})
            
            # Color RGB
            with ui.HStack(height=28, spacing=6):
                ui.Label("Color:", width=50)
                ui.Label("R", width=15)
                self._color_r_model = ui.SimpleFloatModel(0.5)
                ui.FloatField(self._color_r_model, width=60)
                ui.Label("G", width=15)
                self._color_g_model = ui.SimpleFloatModel(0.5)
                ui.FloatField(self._color_g_model, width=60)
                ui.Label("B", width=15)
                self._color_b_model = ui.SimpleFloatModel(0.5)
                ui.FloatField(self._color_b_model, width=60)
            
            # Roughness & Metallic
            with ui.HStack(height=28, spacing=6):
                ui.Label("Roughness:", width=70)
                self._roughness_model = ui.SimpleFloatModel(0.5)
                ui.FloatField(self._roughness_model, width=60)
                ui.Label("(0=shiny, 1=matte)", width=100, style={"color": 0xFF888888})
                
                ui.Label("Metallic:", width=55)
                self._metallic_model = ui.SimpleFloatModel(0.0)
                ui.FloatField(self._metallic_model, width=60)
                ui.Label("(0-1)", width=40, style={"color": 0xFF888888})
            
            with ui.HStack(height=30, spacing=8):
                ui.Button("Apply Material", clicked_fn=self._on_apply_material_clicked, width=120)
                ui.Button("Reset Material", clicked_fn=self._on_reset_material_clicked, width=120)
            
            # Quick color presets
            ui.Label("Quick Colors:", height=16, style={"color": 0xFF888888})
            with ui.HStack(height=26, spacing=4):
                ui.Button("Red", clicked_fn=lambda: self._set_quick_color(1,0,0), width=50)
                ui.Button("Green", clicked_fn=lambda: self._set_quick_color(0,1,0), width=50)
                ui.Button("Blue", clicked_fn=lambda: self._set_quick_color(0,0,1), width=50)
                ui.Button("Yellow", clicked_fn=lambda: self._set_quick_color(1,1,0), width=50)
                ui.Button("Metal", clicked_fn=lambda: self._set_quick_material(0.7,0.7,0.7,0.2,1.0), width=50)
                ui.Button("Plastic", clicked_fn=lambda: self._set_quick_material(0.8,0.2,0.2,0.4,0.0), width=55)
            
            ui.Separator(height=8)
            
            # === TRANSFORM SECTION ===
            ui.Label("Transform:", height=18, style={"font_size": 14})
            
            # Translation
            ui.Label("Translate (dx, dy, dz):", height=16)
            with ui.HStack(height=28, spacing=6):
                ui.Label("dx", width=20)
                self._dx_model = ui.SimpleFloatModel(0.0)
                ui.FloatField(self._dx_model, width=70)
                
                ui.Label("dy", width=20)
                self._dy_model = ui.SimpleFloatModel(0.0)
                ui.FloatField(self._dy_model, width=70)
                
                ui.Label("dz", width=20)
                self._dz_model = ui.SimpleFloatModel(0.0)
                ui.FloatField(self._dz_model, width=70)
                
                ui.Button("Translate", clicked_fn=self._on_translate_object_clicked, width=80)
            
            # Rotation
            ui.Label("Rotate (rx, ry, rz) in degrees:", height=16)
            with ui.HStack(height=28, spacing=6):
                ui.Label("rx", width=20)
                self._rx_model = ui.SimpleFloatModel(0.0)
                ui.FloatField(self._rx_model, width=70)
                
                ui.Label("ry", width=20)
                self._ry_model = ui.SimpleFloatModel(0.0)
                ui.FloatField(self._ry_model, width=70)
                
                ui.Label("rz", width=20)
                self._rz_model = ui.SimpleFloatModel(0.0)
                ui.FloatField(self._rz_model, width=70)
                
                ui.Button("Rotate", clicked_fn=self._on_rotate_object_clicked, width=80)
            
            # Quick rotation presets
            ui.Label("Quick Rotations:", height=16, style={"color": 0xFF888888})
            with ui.HStack(height=26, spacing=4):
                ui.Button("+90° X", clicked_fn=lambda: self._quick_rotate(90,0,0), width=55)
                ui.Button("+90° Y", clicked_fn=lambda: self._quick_rotate(0,90,0), width=55)
                ui.Button("+90° Z", clicked_fn=lambda: self._quick_rotate(0,0,90), width=55)
                ui.Button("Reset Rot", clicked_fn=self._on_reset_rotation_clicked, width=70)
            
            ui.Separator(height=8)
            
            # === LEGACY KEYWORD MODE ===
            with ui.CollapsableFrame("Legacy: Keyword-based Operations", collapsed=True, height=0):
                with ui.VStack(spacing=4):
                    ui.Label("Keywords: compound / small / big / red / yellow", height=16, 
                             style={"color": 0xFF888888})
                    self._keyword_model = ui.SimpleStringModel("compound")
                    ui.StringField(self._keyword_model, height=24)
                    with ui.HStack(height=26, spacing=6):
                        ui.Button("Apply Color (Keyword)", clicked_fn=self._on_apply_color_clicked)
                        ui.Button("Translate (Keyword)", clicked_fn=self._on_translate_clicked)
                        ui.Button("Rotate (Keyword)", clicked_fn=self._on_rotate_clicked)
    
    def _build_chat_ui(self) -> None:
        """Build the chat interface UI."""
        with ui.VStack(spacing=6):
            # Agent status
            agent_status = "Available" if AGENT_AVAILABLE else "Not Available (missing dependencies)"
            ui.Label(f"Agent Status: {agent_status}", height=18,
                     style={"color": 0xFF00FF00 if AGENT_AVAILABLE else 0xFFFF0000})
            
            if AGENT_AVAILABLE:
                ui.Button("Load LLM Model", clicked_fn=self._on_load_model, height=30)
            
            ui.Separator(height=4)
            
            # Chat history (scrollable)
            ui.Label("Chat History:", height=18)
            self._chat_scroll = ui.ScrollingFrame(
                height=280,
                horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_OFF,
                vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED
            )
            with self._chat_scroll:
                self._chat_container = ui.VStack(spacing=4)
                with self._chat_container:
                    ui.Label("Chat messages will appear here...", 
                             style={"color": 0xFF888888})
            
            ui.Separator(height=4)
            
            # Example commands
            ui.Label("Example Commands:", height=16, style={"color": 0xFF888888})
            with ui.HStack(height=24, spacing=4):
                ui.Button("Rotate gear 45°", clicked_fn=lambda: self._insert_example("Rotate the left_gear by 45 degrees on the Z axis"), width=110)
                ui.Button("Make red", clicked_fn=lambda: self._insert_example("Make the main_shaft red"), width=75)
                ui.Button("Scale 2x", clicked_fn=lambda: self._insert_example("Scale the engine_block to 2x size"), width=70)
                ui.Button("Shiny metal", clicked_fn=lambda: self._insert_example("Make the right_gear look like shiny metal"), width=85)
            
            ui.Separator(height=4)
            
            # Input area
            ui.Label("Your message:", height=18)
            with ui.HStack(height=50, spacing=8):
                self._chat_input_model = ui.SimpleStringModel("")
                ui.StringField(self._chat_input_model, height=45, multiline=True)
                ui.Button("Send", clicked_fn=self._on_send_chat, width=80, height=45)
            
            with ui.HStack(height=26, spacing=8):
                ui.Button("Clear Chat History", clicked_fn=self._on_clear_chat, width=140)
                ui.Button("Show Objects", clicked_fn=self._on_show_objects_in_chat, width=100)
    
    def _on_mode_changed(self, model) -> None:
        """Handle mode switch between Static and Chat."""
        mode = model.as_int
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
            # Build the system prompt
            system_prompt = self._build_agent_system_prompt()
            
            # Build user prompt with object table
            user_prompt = f"USER REQUEST: {user_message}\n\nAVAILABLE OBJECTS:\n{json.dumps(self._object_table, indent=2)}"
            
            # Convert deque to list for the conversation history
            history_list = list(self._chat_history)
            
            # Generate structured response
            response = self._agent_manager.generate_structured_output(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                response_model=AgentResponse,
                conversation_history=history_list,
                max_tokens=512
            )
            
            # Process the response
            result_message = self._execute_agent_response(response)
            
            # Update chat history (memory)
            self._chat_history.append({"role": "user", "content": user_message})
            self._chat_history.append({"role": "assistant", "content": result_message})
            
            # Update UI
            self._add_chat_message("Agent", result_message, is_user=False)
            self._set_status("Done.")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
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

Return action_type as one of: 'transform', 'color', 'material', 'unknown', 'ambiguous', 'invalid'
Set error_message to explain issues if action_type is unknown/ambiguous/invalid.

For MATERIAL changes (shiny, matte, metallic, plastic):
- action_type = 'material'
- Set roughness: 0.0 = very shiny/glossy, 1.0 = very matte/rough
- Set metallic: 0.0 = plastic/dielectric, 1.0 = pure metal
- Common presets: shiny metal (roughness=0.2, metallic=1.0), matte plastic (roughness=0.8, metallic=0.0)"""
    
    def _execute_agent_response(self, response: 'AgentResponse') -> str:
        """Execute the agent's response and return a result message."""
        action = response.action_type.lower()
        
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
            material_path = LOOKS_ROOT_PATH.AppendChild(f"mat_{response.object_name.replace(' ', '_')}")
            self._ensure_xform(stage, LOOKS_ROOT_PATH)
            material = self._get_or_create_preview_material(
                stage, material_path, rgba,
                roughness=getattr(response, 'roughness', 0.5),
                metallic=getattr(response, 'metallic', 0.0)
            )
            
            if self._bind_material_to_prim(prim, material):
                return f"Applied color RGB({response.color_r:.2f}, {response.color_g:.2f}, {response.color_b:.2f}) to '{response.object_name}'."
            else:
                return f"Failed to apply color to '{response.object_name}'."
        
        elif action == "material":
            # Material change without necessarily changing color
            rgba = Gf.Vec4f(response.color_r, response.color_g, response.color_b, 1.0)
            roughness = getattr(response, 'roughness', 0.5)
            metallic = getattr(response, 'metallic', 0.0)
            
            material_path = LOOKS_ROOT_PATH.AppendChild(f"mat_{response.object_name.replace(' ', '_')}_mat")
            self._ensure_xform(stage, LOOKS_ROOT_PATH)
            material = self._get_or_create_preview_material(stage, material_path, rgba, roughness, metallic)
            
            if self._bind_material_to_prim(prim, material):
                return f"Applied material (roughness={roughness:.2f}, metallic={metallic:.2f}) to '{response.object_name}'."
            else:
                return f"Failed to apply material to '{response.object_name}'."
        
        return "Unknown action type."
    
    def _add_chat_message(self, sender: str, message: str, is_user: bool = True) -> None:
        """Add a message to the chat UI."""
        if not self._chat_container:
            return
        
        color = 0xFF4488FF if is_user else 0xFF44FF88
        
        with self._chat_container:
            with ui.HStack(height=0, spacing=4):
                ui.Label(f"[{sender}]:", width=60, style={"color": color})
                ui.Label(message, word_wrap=True, style={"color": 0xFFFFFFFF})
    
    def _on_clear_chat(self) -> None:
        """Clear chat history."""
        self._chat_history.clear()
        if self._chat_container:
            self._chat_container.clear()
            with self._chat_container:
                ui.Label("Chat cleared.", style={"color": 0xFF888888})
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
    
    def _get_selected_object_path(self) -> Optional[str]:
        """Get the USD path of the currently selected object."""
        if not self._object_combo_model or not self._object_names_list:
            return None
        idx = self._object_combo_model.as_int
        if 0 <= idx < len(self._object_names_list):
            name = self._object_names_list[idx]
            return self._object_table.get(name)
        return None
    
    def _get_selected_object_name(self) -> str:
        """Get the name of the currently selected object."""
        if not self._object_combo_model or not self._object_names_list:
            return ""
        idx = self._object_combo_model.as_int
        if 0 <= idx < len(self._object_names_list):
            return self._object_names_list[idx]
        return ""
    
    def _on_refresh_objects_and_combo(self) -> None:
        """Refresh objects and update ComboBox."""
        self._on_refresh_objects()
        self._update_object_combo()
    
    def _update_object_combo(self) -> None:
        """Update the object ComboBox with current object table."""
        self._object_names_list = list(self._object_table.keys())
        # ComboBox in omni.ui needs to be rebuilt - this is a limitation
        self._set_status(f"Object list updated: {len(self._object_names_list)} objects. Re-open window to see changes in dropdown.")
    
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
        """Update the status label."""
        if self._status_label:
            self._status_label.text = msg
        print(f"[GenerativeModeling] {msg}")
