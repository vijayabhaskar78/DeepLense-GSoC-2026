"""Tool functions that the agent can invoke to run DeepLenseSim simulations."""

from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from models import (
    ImageMetadata,
    ModelConfig,
    SimulationParams,
    SimulationResult,
    SubstructureType,
)


def _run_single_simulation(params: SimulationParams, idx: int) -> tuple[np.ndarray, ImageMetadata]:
    """Run one DeepLenseSim simulation and return image + metadata."""
    from deeplense.lens import DeepLens

    # Determine axion mass for vortex simulations
    axion_mass = None
    if params.substructure_type == SubstructureType.AXION_VORTEX:
        axion_mass = params.axion_mass or 1e-23

    # Create lens instance
    lens = DeepLens(
        axion_mass=axion_mass,
        H0=params.H0,
        Om0=params.Om0,
        Ob0=params.Ob0,
        z_halo=params.z_lens,
        z_gal=params.z_source,
    )

    # Setup main halo
    lens.make_single_halo(params.halo_mass)

    # Add substructure
    if params.substructure_type == SubstructureType.NO_SUBSTRUCTURE:
        lens.make_no_sub()
    elif params.substructure_type == SubstructureType.AXION_VORTEX:
        lens.make_vortex(params.vortex_mass, res=params.vortex_resolution)
    elif params.substructure_type == SubstructureType.CDM_SUBHALO:
        lens.make_old_cdm()

    # Configure based on model
    if params.model_config_name == ModelConfig.MODEL_I:
        lens.make_source_light()
        lens.simple_sim()
        resolution = (150, 150)
        pixel_scale = 0.05
        instrument = None
    elif params.model_config_name == ModelConfig.MODEL_II:
        lens.set_instrument("Euclid")
        lens.make_source_light_mag()
        lens.simple_sim_2()
        resolution = (64, 64)
        pixel_scale = 0.101
        instrument = "Euclid"
    elif params.model_config_name == ModelConfig.MODEL_III:
        lens.set_instrument("hst")
        lens.make_source_light_mag()
        lens.simple_sim_2()
        resolution = (64, 64)
        pixel_scale = 0.08
        instrument = "HST"
    else:
        raise ValueError(f"Unsupported model: {params.model_config_name}")

    image = lens.image_real

    # Build metadata
    meta = ImageMetadata(
        image_index=idx,
        model_config_name=params.model_config_name.value,
        substructure_type=params.substructure_type.value,
        resolution=resolution,
        z_lens=params.z_lens,
        z_source=params.z_source,
        halo_mass=params.halo_mass,
        axion_mass=axion_mass,
        vortex_mass=params.vortex_mass if params.substructure_type == SubstructureType.AXION_VORTEX else None,
        instrument=instrument,
        pixel_scale=pixel_scale,
        image_stats={
            "mean": float(np.mean(image)),
            "std": float(np.std(image)),
            "min": float(np.min(image)),
            "max": float(np.max(image)),
        },
    )
    return image, meta


def run_simulation(params: SimulationParams, output_dir: str = "outputs") -> SimulationResult:
    """Run a batch of simulations with the given parameters.

    This is the main tool function the agent invokes.
    """
    os.makedirs(output_dir, exist_ok=True)
    timestamp = int(time.time())
    sub_label = params.substructure_type.value
    model_label = params.model_config_name.value

    image_paths: list[str] = []
    metadata_list: list[ImageMetadata] = []

    for i in range(params.num_images):
        image, meta = _run_single_simulation(params, i)

        # Save image as .npy
        fname = f"{model_label}_{sub_label}_{timestamp}_{i}.npy"
        fpath = os.path.join(output_dir, fname)
        np.save(fpath, image)
        image_paths.append(fpath)
        metadata_list.append(meta)

    # Build summary
    res = metadata_list[0].resolution
    summary = (
        f"Generated {params.num_images} image(s) using {model_label} "
        f"({res[0]}x{res[1]} px) with {sub_label} substructure. "
        f"Lens z={params.z_lens}, Source z={params.z_source}, "
        f"Halo mass={params.halo_mass:.1e} M_sun."
    )
    if params.substructure_type == SubstructureType.AXION_VORTEX:
        summary += f" Axion mass={params.axion_mass or 1e-23:.1e} eV."

    return SimulationResult(
        params=params,
        num_generated=params.num_images,
        image_paths=image_paths,
        metadata=metadata_list,
        summary=summary,
    )


def visualize_results(result: SimulationResult, save_path: str = "plots/simulation_grid.png") -> str:
    """Create a visualization grid of generated images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    n = min(result.num_generated, 12)  # show max 12
    cols = min(n, 4)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for i in range(n):
        ax = axes[i // cols, i % cols]
        img = np.load(result.image_paths[i])
        ax.imshow(img, cmap="inferno", origin="lower")
        meta = result.metadata[i]
        ax.set_title(
            f"{meta.substructure_type}\nz_L={meta.z_lens}, z_S={meta.z_source}",
            fontsize=9,
        )
        ax.axis("off")

    # Hide unused axes
    for i in range(n, rows * cols):
        axes[i // cols, i % cols].axis("off")

    fig.suptitle(
        f"{result.params.model_config_name.value} — {result.params.substructure_type.value}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    return save_path
