#!/usr/bin/env python3
"""
Dump MJCF XML for a brittle-star morphology using the project's Hydra configs.

Usage examples:

  # Use a named morphology config from configs/morphology (Hydra style)
  uv run python scripts/analysis/dump_mjcf.py morphology=3_arms

  # Use a morphology override YAML (same key as simulation.morphology_override)
  uv run python scripts/analysis/dump_mjcf.py \
    simulation.morphology_override=configs/morphology/3_arms.yaml

Output path:
    Provide `dump_out=path/to/file.xml` on the command line, otherwise writes `morphology.xml` in
    current directory or `runs/morphologies/<name>.xml`.
"""

from __future__ import annotations

import dataclasses
import logging
import sys
from pathlib import Path
from typing import Any, Optional

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from brittle_star_project.configs.register_configs import register_configs
from brittle_star_project.environment.env_config import MorphologyConfig
from brittle_star_project.environment.factory import BrittleStarEnvFactory

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def extract_xml_string(obj: Any) -> Optional[str]:
    """
    Attempts to serialize the morphology object to an XML string by checking
    common dm_control and internal API methods.
    """
    serialization_methods = [
        "to_xml_string",
        "to_xml",
        "to_string",
        "to_mjcf",
        "to_mjcf_string",
        "get_mjcf",
        "get_mjcf_str",
        "export_to_xml_string",
    ]

    # If the object itself has an 'mjcf' attribute, try to serialize that instead
    target_obj = getattr(obj, "mjcf", obj)

    for method_name in serialization_methods:
        method = getattr(target_obj, method_name, None)
        if callable(method):
            try:
                xml_data = method()
                # Safely handle both string and byte responses
                if isinstance(xml_data, str):
                    return xml_data
                elif isinstance(xml_data, bytes):
                    return xml_data.decode("utf-8")
            except Exception as e:
                logger.debug(f"Method {method_name}() failed during serialization: {e}")

    return None


def resolve_output_path(cfg: DictConfig) -> Path:
    """Determines the appropriate output path for the MJCF XML."""
    dump_out = cfg.get("dump_out", None)
    if dump_out is not None:
        return Path(hydra.utils.to_absolute_path(str(dump_out)))

    morph_name = "morphology"
    for arg in sys.argv[1:]:
        if arg.startswith("morphology="):
            morph_name = arg.split("=", 1)[1]
            break

    default_out = (
        f"runs/morphologies/{morph_name}.xml" if morph_name != "morphology" else "morphology.xml"
    )
    return Path(hydra.utils.to_absolute_path(default_out))


@hydra.main(config_path="../../configs", config_name="main_config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    """Main entry point to construct the morphology and dump its XML."""
    logger.info("Initializing morphology construction...")

    # Extract morphology config safely using dict `.get()` to avoid OmegaConf AttributeErrors
    simulation_cfg = cfg.get("simulation", cfg)
    override_path = simulation_cfg.get("morphology_override", None)

    if override_path:
        logger.info(f"Using morphology override: {override_path}")
        with open(hydra.utils.to_absolute_path(override_path), "r") as f:
            data = yaml.safe_load(f) or {}
        morph_cfg = MorphologyConfig(**data)
    else:
        # Fallback to default simulation morphology, or an empty base config
        morph_node = simulation_cfg.get("morphology", cfg.get("morphology", None))

        if morph_node is not None:
            # Convert OmegaConf node to dict and instantiate MorphologyConfig.
            # This ensures any missing keys gracefully fall back to the dataclass defaults.
            morph_dict = OmegaConf.to_container(morph_node, resolve=True)
            if isinstance(morph_dict, dict):
                # Filter to avoid unexpected kwargs if the dataclass is strictly defined
                if dataclasses.is_dataclass(MorphologyConfig):
                    valid_keys = {f.name for f in dataclasses.fields(MorphologyConfig)}
                    morph_dict = {k: v for k, v in morph_dict.items() if k in valid_keys}
                morph_cfg = MorphologyConfig(**morph_dict)
            else:
                morph_cfg = MorphologyConfig()
        else:
            morph_cfg = MorphologyConfig()

    morphology = BrittleStarEnvFactory.create_morphology(morph_cfg)

    xml_text = extract_xml_string(morphology)
    if not xml_text:
        raise RuntimeError("Failed to serialize morphology to MJCF/XML. ")

    out_path = resolve_output_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(xml_text)

    logger.info(f"Successfully exported MJCF XML to: {out_path}")


if __name__ == "__main__":
    register_configs()
    main()
