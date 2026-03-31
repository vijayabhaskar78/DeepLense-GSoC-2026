# Specific Test II: Agentic AI

An agentic workflow using Pydantic AI that takes natural language requests and generates strong gravitational lensing images via the DeepLenseSim pipeline.

## What it does

A user types something like "generate 5 CDM images with the Euclid model at lens redshift 0.5" and the agent:
1. Parses and validates the parameters against physical constraints
2. Summarizes what it's about to run and asks for confirmation
3. Executes the simulation and saves the output arrays

**Physics-aware validation** is enforced at the Pydantic schema level — not in the agent logic, not in if-statements. Constraints like `z_source > z_lens` (the source galaxy must be behind the lens) are built into `SimulationParams` itself, so invalid configurations are rejected before any simulation runs, with clear feedback explaining what went wrong and how to fix it. This guarantees that every generated image is physically meaningful regardless of how the agent is called.
4. Generates a visualization grid

Before running anything expensive, the agent always does a dry-run validation and presents the parsed parameters to the user — the human-in-the-loop step. Physical validity is enforced at the Pydantic model level: source must be behind the lens (z_source > z_lens), halo mass within a sane range, axion mass auto-defaulted to 1e-23 eV when not specified.

All 3 model configs × 3 substructure types = **9/9 combinations verified working**.

## Example session

```
[You] > Generate 5 vortex images with the Euclid model at lens redshift 0.3

[Agent] Here's what I'll run:
  Model:       Model_II (64×64 px, Euclid-like)
  Substructure: axion_vortex
  Axion mass:  1e-23 eV  (default)
  Lens z:      0.3
  Source z:    1.0
  Halo mass:   1.0e+12 M_sun
  N images:    5

  Shall I proceed?

[You] > yes

[Agent] Done. Saved to outputs/.
  img_0: mean=0.0726, std=0.0482
  img_1: mean=0.0893, std=0.0531
  ...
  Visualization saved to plots/simulation_grid.png
```

## Architecture

```
User (natural language)
       │
       ▼
 Pydantic AI Agent  ←── system prompt: physics context + confirmation rules
       │
       ├── validate_parameters()    dry-run: parse + Pydantic validate, no simulation
       ├── execute_simulation()     run DeepLenseSim, save .npy, return JSON metadata
       └── create_visualization()   load .npy arrays, save PNG grid
                  │
                  ▼
       Pydantic models
       ├── SimulationParams         validated input (physical constraints enforced)
       ├── ImageMetadata            per-image stats and instrument info
       └── SimulationResult         full output with paths + summary
                  │
                  ▼
       DeepLenseSim pipeline
       Model_I (150×150, custom PSF) / Model_II (64×64, Euclid) / Model_III (64×64, HST)
                  │
                  ▼
       .npy arrays  +  PNG visualization  +  JSON metadata
```

## Evaluation against the task criteria

| Criterion | Implementation |
|---|---|
| Agent architecture quality | Pydantic AI with typed tools; clean separation across models.py / tools/simulator.py / agent.py |
| Tool design | 3 tools with clear single responsibilities; validate-then-execute pattern |
| Correctness of simulations | All 9 model × substructure combos verified; physical params validated before DeepLenseSim is called |
| Structured output validation | SimulationParams, ImageMetadata, SimulationResult — full Pydantic schemas with field constraints |
| Human-in-the-loop | Enforced in system prompt; validate_parameters always runs first as dry-run before execution |
| Code modularity | models.py (schemas), tools/simulator.py (pipeline wrapper), agent.py (orchestration) — each independently testable |
| Model configs supported | 3 of 3 (Model_I, Model_II, Model_III) |

## Design decisions

**Why Pydantic AI**: Tool definitions are clean Python functions with type annotations — the framework handles JSON schema generation and validation automatically. Pydantic model integration is first-class, so the agent can return structured objects rather than raw string JSON. Built-in retry logic handles transient LLM tool-call errors without extra code.

**Validate-then-execute pattern**: The `validate_parameters` tool does everything `execute_simulation` does *except* actually run the simulation. The agent always calls it first and presents the result to the user. This means parameter errors are caught before any compute is spent, and the user sees exactly what will run. The alternative (combining into one tool) hides this from the user until it's too late.

**Local LLM**: Qwen3.5-9B via llama.cpp with an OpenAI-compatible endpoint. No paid API, no internet dependency during simulation runs. The model supports tool calling and handles multi-turn conversation history correctly.

**Pydantic validation at the model boundary**: Physical constraints are enforced in the Pydantic model itself, not scattered through tool code. `z_source > z_lens` is a `field_validator`. `axion_mass` default is a `model_validator`. This means any code path that creates a `SimulationParams` object — whether from the agent, a script, or a test — gets the same validation.

## Files

```
Specific_Test_II/
├── Test2_Agentic_AI.ipynb    # main notebook, fully executed with outputs
├── requirements.txt
├── agent.py                  # agent definition, tools, interactive CLI
├── models.py                 # Pydantic schemas (SimulationParams, ImageMetadata, SimulationResult)
├── tools/
│   ├── __init__.py
│   └── simulator.py          # DeepLenseSim wrapper (run_simulation, visualize_results)
├── outputs/                  # saved .npy simulation arrays
└── plots/                    # generated visualization grids
    ├── all_simulations.png
    ├── model_comparison.png
    ├── sample_images.png
    ├── simulation_grid.png
    └── ...                   # additional per-run grids
```

## Setup

```bash
pip install -r requirements.txt

# DeepLenseSim
git clone https://github.com/mwt5345/DeepLenseSim.git
cd DeepLenseSim && pip install -e . && cd ..

# pyHalo (required for CDM subhalo substructure)
git clone https://github.com/dangilman/pyHalo.git
cd pyHalo && pip install -e . && cd ..

# Start local LLM server (requires llama.cpp and a GGUF model)
llama-server -m Qwen3.5-9B-Q5_K_M.gguf --host 127.0.0.1 --port 8080 -ngl 99

# Interactive agent
python agent.py

# Or run the notebook
jupyter notebook Test2_Agentic_AI.ipynb
```
