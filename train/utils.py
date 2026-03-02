"""
Utility functions for training scripts.
"""

import os
import glob
from typing import Optional


def find_demo_file(env_id: str, demo_dir: Optional[str] = None) -> str:
    """
    Auto-detect demonstration file for a given environment.
    
    Searches in the following locations (in order):
    1. gymnasium_mpail/demos/{env_id}/*/expert*.pt
    2. demos/{env_id}/*/expert*.pt
    3. ./demos/{env_id}/*.pt
    
    Args:
        env_id: Gymnasium environment ID (e.g., "Ant-v5")
        demo_dir: Optional base directory to search in
        
    Returns:
        Path to the demonstration file
        
    Raises:
        FileNotFoundError: If no demo file is found
    """
    # Get the script directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    
    # Possible search patterns
    search_paths = []
    
    if demo_dir:
        search_paths.append(os.path.join(demo_dir, env_id, "**", "*.pt"))
        search_paths.append(os.path.join(demo_dir, env_id, "*.pt"))
    
    # Default search locations
    search_paths.extend([
        os.path.join(project_root, "gymnasium_mpail", "demos", env_id, "*", "expert*.pt"),
        os.path.join(project_root, "gymnasium_mpail", "demos", env_id, "*.pt"),
        os.path.join(project_root, "demos", env_id, "*", "expert*.pt"),
        os.path.join(project_root, "demos", env_id, "*.pt"),
        os.path.join(script_dir, "demos", env_id, "*.pt"),
    ])
    
    for pattern in search_paths:
        matches = glob.glob(pattern, recursive=True)
        if matches:
            # Return the first match (or most recent if multiple)
            demo_path = sorted(matches)[-1]  # Sort to get consistent ordering
            print(f"[INFO] Auto-detected demo file: {demo_path}")
            return demo_path
    
    raise FileNotFoundError(
        f"No demonstration file found for environment '{env_id}'.\n"
        f"Searched patterns:\n" + "\n".join(f"  - {p}" for p in search_paths) +
        f"\n\nPlease provide a demo file with --demo-path or place demos in:\n"
        f"  {os.path.join(project_root, 'gymnasium_mpail', 'demos', env_id, '/')}"
    )
