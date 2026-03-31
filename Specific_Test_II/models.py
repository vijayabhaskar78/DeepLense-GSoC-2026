"""Pydantic models for DeepLenseSim simulation parameters and outputs."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator, model_validator


class SubstructureType(str, Enum):
    """Types of dark matter substructure in gravitational lensing."""
    NO_SUBSTRUCTURE = "no_substructure"
    AXION_VORTEX = "axion_vortex"
    CDM_SUBHALO = "cdm_subhalo"


class ModelConfig(str, Enum):
    """DeepLenseSim model configurations."""
    MODEL_I = "Model_I"      # 150x150, custom PSF
    MODEL_II = "Model_II"    # 64x64, Euclid-like
    MODEL_III = "Model_III"  # 64x64, HST-like


class SimulationParams(BaseModel):
    """Validated parameters for a single gravitational lensing simulation."""

    model_config_name: ModelConfig = Field(
        default=ModelConfig.MODEL_I,
        description="Which DeepLenseSim model to use (Model_I, Model_II, Model_III)"
    )
    substructure_type: SubstructureType = Field(
        default=SubstructureType.NO_SUBSTRUCTURE,
        description="Type of dark matter substructure"
    )
    num_images: int = Field(
        default=1, ge=1, le=100,
        description="Number of images to generate (1-100)"
    )
    halo_mass: float = Field(
        default=1e12, gt=1e10, lt=1e15,
        description="Main lens halo mass in solar masses"
    )
    z_lens: float = Field(
        default=0.5, gt=0.01, lt=5.0,
        description="Redshift of the lens (foreground halo)"
    )
    z_source: float = Field(
        default=1.0, gt=0.01, lt=10.0,
        description="Redshift of the source galaxy"
    )
    axion_mass: Optional[float] = Field(
        default=None,
        description="Axion particle mass in eV (only for axion_vortex substructure)"
    )
    vortex_mass: float = Field(
        default=3e10, gt=1e8, lt=1e13,
        description="Total vortex mass in solar masses (for axion_vortex)"
    )
    vortex_resolution: int = Field(
        default=100, ge=10, le=500,
        description="Number of point masses forming the vortex ring"
    )
    H0: float = Field(default=70.0, description="Hubble constant (km/s/Mpc)")
    Om0: float = Field(default=0.3, description="Matter density parameter")
    Ob0: float = Field(default=0.05, description="Baryon density parameter")

    @field_validator("z_source")
    @classmethod
    def source_behind_lens(cls, v, info):
        z_lens = info.data.get("z_lens", 0.5)
        if v <= z_lens:
            raise ValueError(
                f"Source redshift ({v}) must be greater than lens redshift ({z_lens})"
            )
        return v

    @model_validator(mode="after")
    def set_axion_mass_default(self):
        if self.substructure_type == SubstructureType.AXION_VORTEX and self.axion_mass is None:
            self.axion_mass = 1e-23
        return self


class ImageMetadata(BaseModel):
    """Structured metadata for a generated lensing image."""
    image_index: int
    model_config_name: str
    substructure_type: str
    resolution: tuple[int, int]
    z_lens: float
    z_source: float
    halo_mass: float
    axion_mass: Optional[float] = None
    vortex_mass: Optional[float] = None
    instrument: Optional[str] = None
    pixel_scale: float
    image_stats: dict = Field(default_factory=dict)


class SimulationResult(BaseModel):
    """Complete result from a simulation run."""
    params: SimulationParams
    num_generated: int
    image_paths: list[str] = Field(default_factory=list)
    metadata: list[ImageMetadata] = Field(default_factory=list)
    summary: str = ""


class ClarificationNeeded(BaseModel):
    """When the agent needs to ask the user for clarification."""
    question: str
    suggestions: list[str] = Field(default_factory=list)
    current_params: Optional[SimulationParams] = None
