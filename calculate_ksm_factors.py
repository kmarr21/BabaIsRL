import numpy as np
import torch
from collections import deque
import argparse

from template_lifo_corridors import TemplateLIFOCorridorsEnv
from dqn_agent_enhanced import DQNAgentEnhanced

def calculate_ksm_for_all_templates():
    """Calculate and display the KSM factor for all environment templates."""
    # List of all available templates
    templates = [
        "basic_med", 
        "sparse_med", 
        "zipper_med", 
        "bottleneck_med", 
        "bottleneck_hard", 
        "corridors_med"
    ]
    
    print("\n===== KSM Factor Analysis for All Templates =====\n")
    print(f"{'Template':<15} {'Walls':<8} {'Path':<8} {'Strategy':<10} {'KSM':<8}")
    print("-" * 55)
    
    results = {}
    
    # Create a temporary agent for analysis
    agent = DQNAgentEnhanced(0, 0, use_augmented_state=True, ksm_mode="adaptive")
    
    # Analyze each template
    for template_name in templates:
        # Create environment with the template
        env = TemplateLIFOCorridorsEnv(template_name=template_name, render_enabled=False, verbose=False)
        
        # Get initial state
        state, _ = env.reset()
        
        # Set template context for logging
        agent.set_template_context(template_name)
        
        # Calculate the KSM factor (capture output for parsing)
        import io
        import sys
        from contextlib import redirect_stdout
        
        # Capture the detailed output
        detailed_output = io.StringIO()
        with redirect_stdout(detailed_output):
            ksm_factor = agent.calculate_environment_ksm_factor(state)
        
        # Parse the captured output to get constraint values
        output_lines = detailed_output.getvalue().strip().split('\n')
        constraints = {}
        wall_count = 0
        
        for line in output_lines:
            if ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    if "walls" in key:
                        # Extract wall count
                        value_parts = parts[1].strip().split('-')
                        if len(value_parts) > 0:
                            try:
                                wall_count = int(value_parts[0].strip())
                                constraints["wall"] = float(value_parts[1].strip().split()[1])
                            except:
                                wall_count = 0
                                constraints["wall"] = 0.0
                    elif "path" in key:
                        constraints["path"] = float(parts[1].strip())
                    elif "strategy" in key:
                        constraints["strategy"] = float(parts[1].strip())
                    elif "final" in key:
                        constraints["ksm"] = float(parts[1].strip())
        
        # Store results
        results[template_name] = {
            "walls": wall_count,
            "wall": constraints.get("wall", 0.0),
            "path": constraints.get("path", 0.0),
            "strategy": constraints.get("strategy", 0.0),
            "ksm": constraints.get("ksm", 0.0)
        }
        
        # Display in table format
        print(f"{template_name:<15} {wall_count:<8d} {constraints.get('path', 0.0):<8.2f} {constraints.get('strategy', 0.0):<10.2f} {constraints.get('ksm', 0.0):<8.2f}")
        
        # Close the environment
        env.close()
    
    print("\n===== Analysis Summary =====\n")
    # Sort templates by KSM factor (highest to lowest)
    sorted_templates = sorted(results.items(), key=lambda x: x[1]["ksm"], reverse=True)
    
    print("Templates ranked by KSM factor (highest to lowest):")
    for i, (template, values) in enumerate(sorted_templates):
        print(f"{i+1}. {template:<15} - KSM: {values['ksm']:.2f}")
    
    print("\nDominant constraints by template:")
    for template, values in results.items():
        # Find the highest constraint
        max_constraint = max(
            ("Wall", values["wall"]), 
            ("Path", values["path"]),
            ("Strategy", values["strategy"]),
            key=lambda x: x[1]
        )
        print(f"{template:<15} - {max_constraint[0]} constraint: {max_constraint[1]:.2f}")
    
    print("\nExpected KSM Ranking (Based on Environment Complexity):")
    print("1. bottleneck_hard - Vertical enemy at critical bottleneck")
    print("2. zipper_med - Vertical enemy with corridor constraints")
    print("3. corridors_med - Maze-like pattern with multiple compartments")
    print("4. bottleneck_med - Horizontal barrier with bottleneck")
    print("5. basic_med - Simple layout with few walls")
    print("6. sparse_med - Very open layout with minimal constraints")
    
    return results

if __name__ == "__main__":
    calculate_ksm_for_all_templates()
