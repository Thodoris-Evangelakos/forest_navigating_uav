"""
Top-down preview image generation
"""

try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def generate_preview(positions, world_config, output_path):
    """
    Generate a top-down preview image of the world
    """
    if not HAS_MATPLOTLIB:
        print("Warning: matplotlib not available, skipping preview generation")
        return

    K = world_config['generation']['area_size']
    
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot positions
    if positions:
        xs, ys = zip(*positions)
        ax.scatter(xs, ys, s=20, c='green', alpha=0.6, marker='o')
    
    # Set limits and aspect
    ax.set_xlim(-K/2, K/2)
    ax.set_ylim(-K/2, K/2)
    ax.set_aspect('equal')
    
    # Labels and grid
    ax.set_xlabel('X (m)')
    ax.set_ylabel('Y (m)')
    ax.set_title(f"World: {world_config['world_name']} ({len(positions)} objects)")
    ax.grid(True, alpha=0.3)
    
    # Save
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
