"""
DeepLenseSim Agentic Workflow — Pydantic AI Agent

An agentic system that wraps the DeepLenseSim simulation pipeline,
allowing users to generate strong gravitational lensing images
through natural language interaction with human-in-the-loop refinement.
"""

from __future__ import annotations

import json
import os
import sys

from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.openai import OpenAIProvider

from models import (
    ClarificationNeeded,
    ModelConfig,
    SimulationParams,
    SimulationResult,
    SubstructureType,
)
from tools.simulator import run_simulation, visualize_results


# ---------------------------------------------------------------------------
# Agent Dependencies (shared state passed via RunContext)
# ---------------------------------------------------------------------------
class AgentDeps:
    """Dependencies injected into the agent's run context."""

    def __init__(self, output_dir: str = "outputs"):
        self.output_dir = output_dir
        self.confirmed_params: SimulationParams | None = None
        os.makedirs(output_dir, exist_ok=True)


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are DeepLens Agent, an expert assistant for generating strong gravitational
lensing simulations using the DeepLenseSim pipeline. You help users create
realistic simulated images of gravitationally lensed galaxies.

## Your Capabilities
You can generate lensing images with three types of dark matter substructure:
- **no_substructure**: Clean Einstein ring/arc (no dark matter subhalos)
- **axion_vortex**: Axion/vortex-type perturbations (requires axion mass)
- **cdm_subhalo**: Cold dark matter point-mass subhalos

Using three telescope/model configurations:
- **Model_I**: 150×150 pixels, custom Gaussian PSF, high SNR
- **Model_II**: 64×64 pixels, Euclid-like realistic instrument simulation
- **Model_III**: 64×64 pixels, HST-like realistic instrument simulation

## Key Physical Parameters
- **Halo mass**: Main lens mass (~10^12 solar masses typical)
- **z_lens**: Lens redshift (typically 0.2–1.0)
- **z_source**: Source galaxy redshift (must be > z_lens, typically 0.5–3.0)
- **Axion mass**: Dark matter particle mass in eV (10^-25 to 10^-21 range)
- **Vortex mass**: Total vortex substructure mass (~3×10^10 solar masses)
- **Cosmology**: H0, Omega_m, Omega_b (defaults: 70, 0.3, 0.05)

## Workflow
1. **Parse** the user's request to understand what simulation they want.
2. **Clarify** any ambiguous or missing parameters by asking follow-up questions.
   Always confirm key choices before running: substructure type, model config,
   and number of images. If the user is vague, suggest reasonable defaults.
3. **Validate** parameters using the SimulationParams model.
4. **Execute** the simulation by calling the `execute_simulation` tool.
5. **Visualize** results by calling the `create_visualization` tool.
6. **Report** the results with structured metadata.

## Human-in-the-Loop Rules
- ALWAYS ask for confirmation before running a simulation with >5 images.
- If the user doesn't specify a model config, ask which one they prefer.
- If they request axion/vortex but don't give an axion mass, suggest 1e-23 eV.
- If parameters seem physically unreasonable, warn the user and suggest fixes.
- Present a summary of planned parameters and ask "Shall I proceed?" before executing.
"""


# ---------------------------------------------------------------------------
# Create Agent
# ---------------------------------------------------------------------------
# Use local llama.cpp server (Qwen3.5-9B) as OpenAI-compatible endpoint
# Start with: llama-server -m Qwen3.5-9B-Q5_K_M.gguf --host 127.0.0.1 --port 8080 -ngl 99
provider = OpenAIProvider(base_url="http://127.0.0.1:8080/v1", api_key="local")
model = OpenAIChatModel("Qwen3.5-9B-Q5_K_M", provider=provider)

agent = Agent(
    model=model,
    system_prompt=SYSTEM_PROMPT,
    deps_type=AgentDeps,
    retries=2,
)


# ---------------------------------------------------------------------------
# Tool: Execute Simulation
# ---------------------------------------------------------------------------
@agent.tool
async def execute_simulation(
    ctx: RunContext[AgentDeps],
    model_config_name: str,
    substructure_type: str,
    num_images: int = 1,
    halo_mass: float = 1e12,
    z_lens: float = 0.5,
    z_source: float = 1.0,
    axion_mass: float | None = None,
    vortex_mass: float = 3e10,
    vortex_resolution: int = 100,
    H0: float = 70.0,
    Om0: float = 0.3,
    Ob0: float = 0.05,
) -> str:
    """Execute a DeepLenseSim simulation with the given parameters.

    Call this tool after confirming parameters with the user.
    Returns a JSON string with simulation results and file paths.
    """
    try:
        params = SimulationParams(
            model_config_name=ModelConfig(model_config_name),
            substructure_type=SubstructureType(substructure_type),
            num_images=num_images,
            halo_mass=halo_mass,
            z_lens=z_lens,
            z_source=z_source,
            axion_mass=axion_mass,
            vortex_mass=vortex_mass,
            vortex_resolution=vortex_resolution,
            H0=H0,
            Om0=Om0,
            Ob0=Ob0,
        )
    except Exception as e:
        return json.dumps({"error": f"Invalid parameters: {e}"})

    try:
        result = run_simulation(params, output_dir=ctx.deps.output_dir)
        ctx.deps.confirmed_params = params
        return result.model_dump_json(indent=2)
    except Exception as e:
        return json.dumps({"error": f"Simulation failed: {e}"})


# ---------------------------------------------------------------------------
# Tool: Visualize Results
# ---------------------------------------------------------------------------
@agent.tool
async def create_visualization(
    ctx: RunContext[AgentDeps],
    result_json: str,
    save_filename: str = "simulation_grid.png",
) -> str:
    """Create a visualization grid from simulation results.

    Takes the JSON output of execute_simulation and creates a PNG plot.
    Returns the path to the saved plot.
    """
    try:
        result = SimulationResult.model_validate_json(result_json)
        save_path = os.path.join("plots", save_filename)
        path = visualize_results(result, save_path=save_path)
        return json.dumps({"plot_path": path, "num_images_shown": min(result.num_generated, 12)})
    except Exception as e:
        return json.dumps({"error": f"Visualization failed: {e}"})


# ---------------------------------------------------------------------------
# Tool: Validate Parameters (dry run)
# ---------------------------------------------------------------------------
@agent.tool
async def validate_parameters(
    ctx: RunContext[AgentDeps],
    model_config_name: str,
    substructure_type: str,
    num_images: int = 1,
    halo_mass: float = 1e12,
    z_lens: float = 0.5,
    z_source: float = 1.0,
    axion_mass: float | None = None,
) -> str:
    """Validate simulation parameters WITHOUT running the simulation.

    Use this to check if parameters are valid before asking the user to confirm.
    Returns a summary of validated parameters or validation errors.
    """
    try:
        params = SimulationParams(
            model_config_name=ModelConfig(model_config_name),
            substructure_type=SubstructureType(substructure_type),
            num_images=num_images,
            halo_mass=halo_mass,
            z_lens=z_lens,
            z_source=z_source,
            axion_mass=axion_mass,
        )
        model_info = {
            "Model_I": "150×150 px, custom PSF",
            "Model_II": "64×64 px, Euclid-like",
            "Model_III": "64×64 px, HST-like",
        }
        return json.dumps({
            "valid": True,
            "summary": {
                "model": f"{params.model_config_name.value} ({model_info[params.model_config_name.value]})",
                "substructure": params.substructure_type.value,
                "num_images": params.num_images,
                "halo_mass": f"{params.halo_mass:.1e} solar masses",
                "z_lens": params.z_lens,
                "z_source": params.z_source,
                "axion_mass": f"{params.axion_mass:.1e} eV" if params.axion_mass else "N/A",
            },
        })
    except Exception as e:
        return json.dumps({"valid": False, "error": str(e)})


# ---------------------------------------------------------------------------
# Interactive CLI Loop (Human-in-the-Loop)
# ---------------------------------------------------------------------------
async def run_interactive():
    """Run the agent in an interactive terminal session with human-in-the-loop."""
    deps = AgentDeps(output_dir="outputs")

    print("=" * 70)
    print("  DeepLens Agent — Gravitational Lensing Simulation Assistant")
    print("=" * 70)
    print()
    print("Describe the lensing simulation you'd like to generate.")
    print("Examples:")
    print('  "Generate 5 images with vortex substructure using Euclid"')
    print('  "Simulate CDM subhalos at redshift 0.3 with Model_I"')
    print('  "Create a no-substructure lensing image with HST"')
    print()
    print("Type 'quit' or 'exit' to end the session.")
    print("-" * 70)

    # Keep conversation history for multi-turn interaction
    message_history = []

    while True:
        try:
            user_input = input("\n[You] > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        # Run agent with conversation history for multi-turn support
        result = await agent.run(
            user_input,
            deps=deps,
            message_history=message_history,
        )

        # Update history for next turn
        message_history = result.all_messages()

        # Display response
        print(f"\n[Agent] {result.output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import asyncio
    asyncio.run(run_interactive())
