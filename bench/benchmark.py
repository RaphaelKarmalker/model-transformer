# bench/benchmark.py
"""
Benchmarking Script für Model Transformer Agent
- Misst Inference-Zeit
- Testet ob Modelle die Aufgaben korrekt erfüllen
- Speichert Ergebnisse in ./bench/results/
"""

import json
import time
import yaml
import os
from datetime import datetime
from typing import TypedDict, Union, Type, TypeVar
from pathlib import Path

from pydantic import BaseModel, Field, ValidationError
from transformers import pipeline

# Ensure results directory exists
BENCH_DIR = Path(__file__).parent
RESULTS_DIR = BENCH_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ------------------------------------------------------------
# TEST CASES
# ------------------------------------------------------------

TEST_CASES = [
    # Format: (prompt, expected_outcome, expected_cancel_reason)
    # expected_outcome: "success" | "cancel"
    # expected_cancel_reason: None | "unknown_object" | "ambiguous" | "invalid_action"
    
    # === VALID TRANSFORMATIONS ===
    {
        "id": 1,
        "prompt": "Rotate the main_shaft by 30 degrees on the x-axis.",
        "expected_outcome": "success",
        "expected_cancel_reason": None,
        "description": "Valid rotation on known object"
    },
    {
        "id": 2,
        "prompt": "Zoom in on the engine_block by a factor of 2.",
        "expected_outcome": "success",
        "expected_cancel_reason": None,
        "description": "Valid zoom on known object"
    },
    {
        "id": 3,
        "prompt": "Rotate the left_gear 45 degrees around the y-axis.",
        "expected_outcome": "success",
        "expected_cancel_reason": None,
        "description": "Valid Y-axis rotation"
    },
    {
        "id": 4,
        "prompt": "Turn the cooling_fan 90 degrees on z-axis and zoom out to 0.5.",
        "expected_outcome": "success",
        "expected_cancel_reason": None,
        "description": "Combined rotation and zoom"
    },
    
    # === INVALID TRANSFORMATIONS ===
    {
        "id": 5,
        "prompt": "Flip the main_shaft horizontally.",
        "expected_outcome": "cancel",
        "expected_cancel_reason": "invalid_action",
        "description": "Flip is not allowed (invalid)"
    },
    {
        "id": 6,
        "prompt": "Mirror the engine_block along the x-axis.",
        "expected_outcome": "cancel",
        "expected_cancel_reason": "invalid_action",
        "description": "Mirror is not allowed (invalid)"
    },
    {
        "id": 7,
        "prompt": "Move the left_gear 10 units to the right.",
        "expected_outcome": "cancel",
        "expected_cancel_reason": "invalid_action",
        "description": "Move/translate is not allowed"
    },
    
    # === UNKNOWN OBJECTS ===
    {
        "id": 8,
        "prompt": "Rotate the banana by 45 degrees.",
        "expected_outcome": "cancel",
        "expected_cancel_reason": "unknown_object",
        "description": "Object not in catalog"
    },
    {
        "id": 9,
        "prompt": "Zoom in on the spaceship.",
        "expected_outcome": "cancel",
        "expected_cancel_reason": "unknown_object",
        "description": "Unknown object"
    },
    
    # === AMBIGUOUS REQUESTS ===
    {
        "id": 10,
        "prompt": "Rotate the wheel by 30 degrees.",
        "expected_outcome": "cancel",
        "expected_cancel_reason": "ambiguous",
        "description": "Multiple wheels exist - ambiguous"
    },
]


# Object catalog (same as in agent_orchester_bench.py)
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
# PYDANTIC MODELS (copied from main agent)
# ------------------------------------------------------------

T = TypeVar('T', bound=BaseModel)

class ObjectMatch(BaseModel):
    object_id: str = Field(description="The unique ID of the object, or empty string if not found, or 'MULTI' if ambiguous")
    object_name: str = Field(description="The name of the object, or empty string if not found, or 'MULTI' if ambiguous")

class TransformationResponse(BaseModel):
    object_id: str | None = Field(default=None, description="The unique ID of the object to transform")
    rotate_x: float | None = Field(default=None, description="Rotation around X-axis in degrees")
    rotate_y: float | None = Field(default=None, description="Rotation around Y-axis in degrees")
    rotate_z: float | None = Field(default=None, description="Rotation around Z-axis in degrees")
    zoom: float | None = Field(default=None, description="Zoom factor (1.0 = no zoom)")
    status: str | None = Field(default=None, description="Set to 'INVALID' if transformation cannot be represented")

class ModelTransformation(BaseModel):
    object_id: str = Field(description="The unique ID of the object to transform")
    rotate_x: float = Field(description="Rotation around X-axis in degrees")
    rotate_y: float = Field(description="Rotation around Y-axis in degrees")
    rotate_z: float = Field(description="Rotation around Z-axis in degrees")
    zoom: float = Field(description="Zoom factor (1.0 = no zoom, >1 = zoom in, <1 = zoom out)")


# ------------------------------------------------------------
# BENCHMARK CLASS
# ------------------------------------------------------------

class ModelBenchmark:
    def __init__(self, model_id: str, max_tokens: int = 256):
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.pipe = None
        self.results = []
        
    def load_model(self):
        """Load the model pipeline."""
        print(f"[BENCH] Loading model: {self.model_id}")
        start_time = time.time()
        self.pipe = pipeline(
            "text-generation",
            model=self.model_id,
            torch_dtype="auto",
            device_map="auto",
        )
        load_time = time.time() - start_time
        print(f"[BENCH] Model loaded in {load_time:.2f}s")
        return load_time
    
    def extract_json_from_text(self, text: str) -> str:
        """Extract JSON object from text."""
        import re
        
        code_block_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if code_block_match:
            return code_block_match.group(1)
        
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
        max_retries: int = 2
    ) -> tuple[T | None, float, str]:
        """
        Generate structured output and measure time.
        Returns: (result, inference_time, raw_response)
        """
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
            f"IMPORTANT: You MUST respond with ONLY a JSON object. No explanations.\n\n"
            f"Required JSON fields:\n{fields_text}\n\n"
            f"Example format:\n{example_json}\n\n"
            f"Your response must start with {{ and end with }}."
        )
        
        messages = [
            {"role": "system", "content": enhanced_system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        
        response_text = ""
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()
                outputs = self.pipe(
                    messages,
                    max_new_tokens=self.max_tokens,
                    temperature=0.0,
                    do_sample=False,
                )
                inference_time = time.time() - start_time
                
                generated_text = outputs[0]["generated_text"]
                if isinstance(generated_text, list):
                    response_text = generated_text[-1]["content"] if generated_text else ""
                else:
                    response_text = str(generated_text)
                
                json_text = self.extract_json_from_text(response_text)
                
                if not json_text:
                    raise json.JSONDecodeError("No JSON found", response_text, 0)
                
                data = json.loads(json_text)
                result = response_model.model_validate(data)
                return result, inference_time, response_text
                
            except (json.JSONDecodeError, ValidationError) as e:
                if attempt == max_retries - 1:
                    return None, inference_time if 'inference_time' in locals() else 0.0, response_text
        
        return None, 0.0, ""
    
    def run_single_test(self, test_case: dict) -> dict:
        """Run a single test case and return results."""
        print(f"\n[TEST {test_case['id']}] {test_case['description']}")
        print(f"  Prompt: {test_case['prompt']}")
        
        total_time = 0.0
        actual_outcome = None
        actual_cancel_reason = None
        raw_responses = []
        
        # Step 1: Object Identification
        id_system_prompt = (
            "Your task is to identify the referenced object from the provided prompt.\n"
            "Rules:\n"
            "1) If the prompt clearly refers to exactly one object in the table, return its object_id and object_name.\n"
            "2) If NO object matches, return object_id='' and object_name=''.\n"
            "3) If MORE THAN ONE object matches, return object_id='MULTI' and object_name='MULTI'."
        )
        
        id_user_prompt = f"PROMPT:\n{test_case['prompt']}\n\nOBJECT TABLE:\n{OBJECT_ID_TABLE}"
        
        id_result, id_time, id_response = self.generate_structured_output(
            id_system_prompt, id_user_prompt, ObjectMatch
        )
        total_time += id_time
        raw_responses.append({"step": "identification", "response": id_response, "time": id_time})
        
        if id_result is None:
            actual_outcome = "error"
            actual_cancel_reason = "parse_error"
        elif id_result.object_id == "":
            actual_outcome = "cancel"
            actual_cancel_reason = "unknown_object"
        elif id_result.object_id == "MULTI":
            actual_outcome = "cancel"
            actual_cancel_reason = "ambiguous"
        else:
            # Step 2: Transformation Validation
            trans_system_prompt = (
                "Translate the user's transformation request into JSON.\n"
                "Allowed: rotate_x, rotate_y, rotate_z (degrees), zoom (1.0=none).\n\n"
                "For VALID transformations (rotate, zoom only):\n"
                '{"object_id": "ID", "rotate_x": 0, "rotate_y": 0, "rotate_z": 0, "zoom": 1.0, "status": ""}\n\n'
                "For INVALID transformations (flip, mirror, move, translate, shear):\n"
                '{"status": "INVALID"}\n\n'
                "IMPORTANT: flip is NOT rotate. flip=mirror=INVALID."
            )
            
            trans_user_prompt = (
                f"PROMPT: {test_case['prompt']}\n"
                f"OBJECT ID: {id_result.object_id}\n"
                f"OBJECT NAME: {id_result.object_name}"
            )
            
            trans_result, trans_time, trans_response = self.generate_structured_output(
                trans_system_prompt, trans_user_prompt, TransformationResponse
            )
            total_time += trans_time
            raw_responses.append({"step": "transformation", "response": trans_response, "time": trans_time})
            
            if trans_result is None:
                actual_outcome = "error"
                actual_cancel_reason = "parse_error"
            elif trans_result.status and trans_result.status.upper() == "INVALID":
                actual_outcome = "cancel"
                actual_cancel_reason = "invalid_action"
            elif (trans_result.rotate_x is not None and trans_result.rotate_y is not None and 
                  trans_result.rotate_z is not None and trans_result.zoom is not None):
                actual_outcome = "success"
                actual_cancel_reason = None
            else:
                actual_outcome = "cancel"
                actual_cancel_reason = "invalid_action"
        
        # Determine if test passed
        expected_outcome = test_case["expected_outcome"]
        expected_cancel_reason = test_case["expected_cancel_reason"]
        
        passed = (actual_outcome == expected_outcome and actual_cancel_reason == expected_cancel_reason)
        
        result = {
            "test_id": test_case["id"],
            "description": test_case["description"],
            "prompt": test_case["prompt"],
            "expected_outcome": expected_outcome,
            "expected_cancel_reason": expected_cancel_reason,
            "actual_outcome": actual_outcome,
            "actual_cancel_reason": actual_cancel_reason,
            "passed": passed,
            "total_inference_time": total_time,
            "raw_responses": raw_responses,
        }
        
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"  Result: {status} (expected: {expected_outcome}/{expected_cancel_reason}, got: {actual_outcome}/{actual_cancel_reason})")
        print(f"  Inference time: {total_time:.3f}s")
        
        return result
    
    def run_benchmark(self, test_cases: list = None) -> dict:
        """Run all benchmark tests."""
        if test_cases is None:
            test_cases = TEST_CASES
        
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {self.model_id}")
        print(f"{'='*60}")
        
        # Load model
        load_time = self.load_model()
        
        # Run tests
        results = []
        for test_case in test_cases:
            result = self.run_single_test(test_case)
            results.append(result)
        
        # Calculate summary
        passed = sum(1 for r in results if r["passed"])
        failed = len(results) - passed
        total_time = sum(r["total_inference_time"] for r in results)
        avg_time = total_time / len(results) if results else 0
        
        summary = {
            "model_id": self.model_id,
            "timestamp": datetime.now().isoformat(),
            "model_load_time": load_time,
            "total_tests": len(results),
            "passed": passed,
            "failed": failed,
            "accuracy": passed / len(results) if results else 0,
            "total_inference_time": total_time,
            "average_inference_time": avg_time,
            "results": results,
        }
        
        print(f"\n{'='*60}")
        print(f"SUMMARY: {self.model_id}")
        print(f"{'='*60}")
        print(f"  Tests: {len(results)}")
        print(f"  Passed: {passed} ({summary['accuracy']*100:.1f}%)")
        print(f"  Failed: {failed}")
        print(f"  Total inference time: {total_time:.2f}s")
        print(f"  Average inference time: {avg_time:.3f}s")
        
        return summary
    
    def save_results(self, summary: dict, filename: str = None):
        """Save results to JSON file."""
        if filename is None:
            # Create filename from model ID and timestamp
            safe_model_name = self.model_id.replace("/", "_").replace(":", "_")
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{safe_model_name}_{timestamp}.json"
        
        filepath = RESULTS_DIR / filename
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        print(f"\n[BENCH] Results saved to: {filepath}")
        return filepath


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------

def run_benchmark_for_model(model_id: str) -> dict:
    """Run benchmark for a specific model."""
    benchmark = ModelBenchmark(model_id)
    summary = benchmark.run_benchmark()
    benchmark.save_results(summary)
    return summary


def run_all_benchmarks(model_ids: list) -> list:
    """Run benchmarks for multiple models."""
    all_results = []
    for model_id in model_ids:
        try:
            result = run_benchmark_for_model(model_id)
            all_results.append(result)
        except Exception as e:
            print(f"[ERROR] Failed to benchmark {model_id}: {e}")
            all_results.append({
                "model_id": model_id,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
    
    # Save combined results
    combined_filepath = RESULTS_DIR / f"combined_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(combined_filepath, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n[BENCH] Combined results saved to: {combined_filepath}")
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark Model Transformer Agent")
    parser.add_argument("--model", "-m", type=str, help="Model ID to benchmark (uses config.yaml if not provided)")
    parser.add_argument("--models", "-M", nargs="+", help="Multiple model IDs to benchmark")
    parser.add_argument("--config", "-c", type=str, default="config.yaml", help="Path to config.yaml")
    
    args = parser.parse_args()
    
    if args.models:
        # Multiple models specified
        run_all_benchmarks(args.models)
    elif args.model:
        # Single model specified
        run_benchmark_for_model(args.model)
    else:
        # Use config.yaml
        config_path = Path(__file__).parent.parent / args.config
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            model_id = config['model']['id']
            run_benchmark_for_model(model_id)
        else:
            print(f"[ERROR] Config not found: {config_path}")
            print("Usage: python benchmark.py --model <model_id>")
