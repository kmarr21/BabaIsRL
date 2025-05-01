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
    print(f"{'Template':<15} {'Walls':<8} {'Path':<8} {'Strategy':<10} {'LIFO':<8} {'KSM':<8}")
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
        wall_count = 0
        key0_viable = False
        key1_viable = False
        strategy1_cost = 0.0
        strategy2_cost = 0.0
        
        for line in output_lines:
            if ":" in line:
                parts = line.split(":")
                if len(parts) == 2:
                    key = parts[0].strip().lower()
                    value = parts[1].strip()
                    
                    if "walls" in key:
                        try:
                            wall_count = int(value)
                        except:
                            wall_count = 0
                    elif "key0 first viable" in key:
                        key0_viable = value.lower() == "true"
                    elif "key1 first viable" in key:
                        key1_viable = value.lower() == "true"
                    elif "strategy1 cost" in key:
                        try:
                            strategy1_cost = float(value)
                        except:
                            strategy1_cost = 0.0
                    elif "strategy2 cost" in key:
                        try:
                            strategy2_cost = float(value)
                        except:
                            strategy2_cost = 0.0
                    elif "path complexity" in key:
                        try:
                            constraints["path"] = float(value)
                        except:
                            constraints["path"] = 0.0
                    elif "strategy importance" in key:
                        try:
                            constraints["strategy"] = float(value)
                        except:
                            constraints["strategy"] = 0.0
                    elif "lifo constraint" in key:
                        try:
                            constraints["lifo"] = float(value)
                        except:
                            constraints["lifo"] = 0.0
                    elif "final ksm factor" in key:
                        try:
                            constraints["ksm"] = float(value)
                        except:
                            constraints["ksm"] = 0.0
        
        # Store results
        results[template_name] = {
            "walls": wall_count,
            "path": constraints.get("path", 0.0),
            "strategy": constraints.get("strategy", 0.0),
            "lifo": constraints.get("lifo", 0.0),
            "ksm": constraints.get("ksm", 0.0),
            "key0_viable": key0_viable,
            "key1_viable": key1_viable,
            "strategy1_cost": strategy1_cost,
            "strategy2_cost": strategy2_cost
        }
        
        # Display in table format
        print(f"{template_name:<15} {wall_count:<8d} {constraints.get('path', 0.0):<8.2f} {constraints.get('strategy', 0.0):<10.2f} {constraints.get('lifo', 0.0):<8.2f} {constraints.get('ksm', 0.0):<8.2f}")
        
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
            ("Path", values["path"]), 
            ("Strategy", values["strategy"]),
            ("LIFO", values["lifo"]),
            key=lambda x: x[1]
        )
        print(f"{template:<15} - {max_constraint[0]} constraint: {max_constraint[1]:.2f}")
    
    print("\nStrategy Viability Analysis:")
    for template, values in results.items():
        key0_viable = values["key0_viable"]
        key1_viable = values["key1_viable"]
        strategy1_cost = values["strategy1_cost"]
        strategy2_cost = values["strategy2_cost"]
        
        # Determine if both strategies are viable and which is better
        if key0_viable and key1_viable:
            if strategy1_cost < strategy2_cost:
                better = "Key0 first"
                diff = (strategy2_cost - strategy1_cost) / strategy1_cost * 100
            else:
                better = "Key1 first"
                diff = (strategy1_cost - strategy2_cost) / strategy2_cost * 100
            
            print(f"{template:<15} - Both strategies viable. Better: {better} (by {diff:.1f}%)")
        elif key0_viable:
            print(f"{template:<15} - Only Key0 first viable")
        elif key1_viable:
            print(f"{template:<15} - Only Key1 first viable")
        else:
            print(f"{template:<15} - No viable strategies detected")
    
    return results

if __name__ == "__main__":
    check_results = calculate_ksm_for_all_templates()
    
    print("\nExpected Ranking:")
    print("1. bottleneck_hard/zipper_med - Complex environments with LIFO constraints")
    print("2. bottleneck_med - Forced strategy but high path complexity")
    print("3. basic_med - Moderate constraints")
    print("4. corridors_med/sparse_med - Simpler environments with minimal constraints")
