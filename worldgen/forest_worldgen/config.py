"""
Configuration loading and path resolution utilities
"""

import os
import yaml
from copy import deepcopy
from functools import lru_cache


@lru_cache(maxsize=256)
def _load_config_cached(abs_path):
    with open(abs_path, 'r') as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=32)
def _load_template_cached(abs_template_path):
    with open(abs_template_path, 'r') as f:
        return f.read()


def load_config(config_path):
    """
    Load config from yaml file
    
    Returns parsed configuration dict
    """
    abs_path = os.path.abspath(config_path)
    return deepcopy(_load_config_cached(abs_path))


def load_template(template_name, worldgen_root):
    """
    Load template file from templates directory
    
    Returns template contents as string
    """
    template_path = os.path.abspath(os.path.join(worldgen_root, 'templates', template_name))
    return _load_template_cached(template_path)


def resolve_path(ref_path, project_root):
    """
    Resolve a config-relative path against the project root
    
    Returns absolute resolved path
    """
    if os.path.isabs(ref_path):
        return ref_path
    return os.path.normpath(os.path.join(project_root, ref_path))


def load_distribution(dist_ref, project_root):
    """
    Load a distribution yaml by its filename reference
    
    Returns parsed distribution configuration dict
    """
    dist_dir = os.path.join(project_root, 'configs', 'worldgen', 'distributions')
    dist_path = os.path.abspath(os.path.join(dist_dir, dist_ref))
    return load_config(dist_path)
