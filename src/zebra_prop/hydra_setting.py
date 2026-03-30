"""Helpers for adapting Hydra configs to legacy attribute-style arguments."""

from omegaconf import DictConfig, OmegaConf


def config_to_args(cfg: DictConfig):
    """Convert a Hydra config object into an attribute-style args object."""

    class Args:
        def __init__(self, config_dict):
            """Populate attributes from the resolved config dictionary."""
            # Mirror argparse-style access (args.foo) across existing code paths.
            for key, value in config_dict.items():
                setattr(self, key, value)

    return Args(OmegaConf.to_container(cfg, resolve=True))
