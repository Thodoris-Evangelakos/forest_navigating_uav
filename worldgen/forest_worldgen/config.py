"""
Configuration loading and path resolution utilities
"""

import os
import yaml


def load_config(config_path):
    """
    Load config from yaml file
    
    Returns parsed configuration dict
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_template(template_name, worldgen_root):
    """
    Load template file from templates directory
    
    Returns template contents as string
    """
    template_path = os.path.join(worldgen_root, 'templates', template_name)
    with open(template_path, 'r') as f:
        return f.read()


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
    dist_path = os.path.join(dist_dir, dist_ref)
    return load_config(dist_path)
