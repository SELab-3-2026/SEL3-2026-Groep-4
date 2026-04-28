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
    current directory.
"""

from __future__ import annotations

from pathlib import Path
import sys
import xml.etree.ElementTree as ET

import hydra
from omegaconf import DictConfig, OmegaConf

from brittle_star_project.configs.main_config import BrittleStarConfig
from brittle_star_project.configs.register_configs import register_configs
from brittle_star_project.environment.env_config import MorphologyConfig
from brittle_star_project.environment.factory import BrittleStarEnvFactory


def try_serialize(obj):
    """Try multiple common accessors to obtain an XML string from the morphology object."""
    candidates = [
        "to_xml_string",
        "to_xml",
        "to_string",
        "to_mjcf",
        "to_mjcf_string",
        "get_mjcf",
        "get_mjcf_str",
        "get_mjcf_assets",
        "export_to_xml_with_assets",
        "get_xml",
        "xml",
        "mjcf",
        "mjcf_model",
        "mjcf_body",
        "model",
        "root",
    ]

    def normalize(out):
        if out is None:
            return None
        # lxml element
        try:
            import lxml.etree as lxml_et

            if isinstance(out, lxml_et._Element):
                return lxml_et.tostring(out, encoding="unicode")
        except Exception:
            pass

        if isinstance(out, ET.Element):
            return ET.tostring(out, encoding="unicode")

        if isinstance(out, bytes):
            try:
                return out.decode()
            except Exception:
                return None

        if hasattr(out, "toxml") and callable(out.toxml):
            try:
                return out.toxml()
            except Exception:
                pass

        try:
            s = str(out)
            if s.lstrip().startswith("<"):
                return s
            return s
        except Exception:
            return None

    for name in candidates:
        attr = getattr(obj, name, None)
        if callable(attr):
            try:
                out = attr()
            except Exception:
                out = None
            if out:
                norm = normalize(out)
                if norm:
                    return norm
        elif attr is not None:
            norm = normalize(attr)
            if norm:
                return norm

    if hasattr(obj, "mjcf"):
        nested = getattr(obj, "mjcf")
        if nested is not None:
            return try_serialize(nested)

    return None


@hydra.main(config_path="../../configs", config_name="main_config", version_base="1.3")
def main(dict_cfg: DictConfig) -> None:
    # Compose typed config like the rest of the project
    cfg = OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(BrittleStarConfig), dict_cfg))

    # Check for a simulation morphology override (points to a YAML file)
    sim_override = None
    try:
        sim_override = dict_cfg.get("simulation", {}).get("morphology_override", None)
    except Exception:
        sim_override = getattr(getattr(cfg, "simulation", None), "morphology_override", None)

    if sim_override:
        override_path = Path(hydra.utils.to_absolute_path(sim_override))
        if not override_path.exists():
            raise FileNotFoundError(f"Could not find morphology override YAML at {override_path}")
        import yaml

        with open(override_path, "r") as f:
            override_dict = yaml.safe_load(f)
        env_morphology = OmegaConf.to_object(
            OmegaConf.merge(OmegaConf.structured(MorphologyConfig), override_dict)
        )
    else:
        env_morphology = cfg.morphology

    # Ensure we have a MorphologyConfig instance
    if isinstance(env_morphology, dict):
        morph_cfg = MorphologyConfig(**env_morphology)
    else:
        morph_cfg = env_morphology

    # Build morphology via project factory (same as runtime)
    morph = BrittleStarEnvFactory.create_morphology(morph_cfg)

    xml_text = try_serialize(morph)
    if xml_text is None and hasattr(morph, "mjcf"):
        xml_text = try_serialize(morph.mjcf)

    if xml_text is None:
        raise RuntimeError(
            "Failed to serialize morphology to MJCF/XML. Inspect the `morph` object interactively."
        )

    # Prefer explicit CLI override `dump_out=...` if provided, otherwise choose a sensible default.
    dump_out = None
    try:
        dump_out = dict_cfg.get("dump_out", None)
    except Exception:
        dump_out = None

    if dump_out is None:
        # If the user passed a morphology group on the CLI (e.g. morphology=3_arms),
        # use a descriptive default path under `runs/morphologies/`.
        morph_name = None
        for a in sys.argv[1:]:
            if a.startswith("morphology="):
                morph_name = a.split("=", 1)[1]
                break

        default_out = f"runs/morphologies/{morph_name}.xml" if morph_name else "morphology.xml"
        out_path = Path(hydra.utils.to_absolute_path(default_out))
    else:
        out_path = Path(hydra.utils.to_absolute_path(str(dump_out)))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(xml_text)
    print(f"Wrote MJCF XML to {out_path}")


if __name__ == "__main__":
    register_configs()
    main()
