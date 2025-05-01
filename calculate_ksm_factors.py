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
    print(f"{'Template':<15} {'Wall':<8} {'Path':<8} {'LIFO':<8} {'Enemy':<8} {'KSM':<8}")
    print("-" * 60)
    
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
        for line in output_lines:
            if ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    if "wall" in key:
                        constraints["wall"] = float(parts[1].strip().split()[0])
                    elif "path" in key:
                        constraints["path"] = float(parts[1].strip().split()[0])
                    elif "lifo" in key:
                        constraints["lifo"] = float(parts[1].strip().split()[0])
                    elif "enemy" in key:
                        constraints["enemy"] = float(parts[1].strip().split()[0])
                    elif "final" in key:
                        constraints["ksm"] = float(parts[1].strip().split()[0])
        
        # Store results
        results[template_name] = {
            "wall": constraints.get("wall", 0.0),
            "path": constraints.get("path", 0.0),
            "lifo": constraints.get("lifo", 0.0),
            "enemy": constraints.get("enemy", 0.0),
            "ksm": constraints.get("ksm", 0.0)
        }
        
        # Display in table format
        print(f"{template_name:<15} {constraints.get('wall', 0.0):<8.2f} {constraints.get('path', 0.0):<8.2f} {constraints.get('lifo', 0.0):<8.2f} {constraints.get('enemy', 0.0):<8.2f} {constraints.get('ksm', 0.0):<8.2f}")
        
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
            ("LIFO", values["lifo"]),
            ("Enemy", values["enemy"]),
            key=lambda x: x[1]
        )
        print(f"{template:<15} - {max_constraint[0]} constraint: {max_constraint[1]:.2f}")
    
    return results

if __name__ == "__main__":
    calculate_ksm_for_all_templates()
