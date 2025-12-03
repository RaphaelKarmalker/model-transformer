from pydantic import BaseModel, Field
from pydantic_ai import Agent, RunContext

class ModelTransforamtion(BaseModel):
    object_id: str = Field(..., description="Unique identifier for the object")
    rotate_x: float = Field(..., description="Value between 0-360")
    rotate_y: float = Field(..., description="Value between 0-360")
    rotate_z: float = Field(..., description="Value between 0-360")
    zoom: float = Field(..., description="Value between 0.1-2.0")

agent = Agent(model="openai:gpt-4o-mini",  
    output_type=ModelTransforamtion,
    system_prompt=(
        'fill out all variables'
    ),
)

if __name__ == "__main__":
    prompt = "Please create a Model Transformation with a 180° rotation on X and a slightly increased zoom"
    result = agent.run_sync(prompt)
    print(result.output) 

