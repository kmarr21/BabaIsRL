import numpy as np
import torch
from collections import deque
import argparse

from template_lifo_corridors import TemplateLIFOCorridorsEnv
from dqn_agent_enhanced import DQNAgentEnhanced

def analyze_paths(agent, state_dict):
    """Analyze path characteristics for more detailed metrics."""
    # Extract key components
    agent_pos = state_dict['agent']
    keys = state_dict['keys']
    doors = state_dict['doors']
    walls = []
    
    for wall in state_dict['walls']:
        if wall[0] >= 0:  # Filter out -1 placeholders
            walls.append((wall[0], wall[1]))
    
    # Dictionary to store path analysis metrics
    path_metrics = {}
    
    # Calculate paths
    key0_path = agent._simplified_path(state_dict, agent_pos, keys[0], consider_doors=True)
    key1_path = agent._simplified_path(state_dict, agent_pos, keys[1], consider_doors=True)
    key0_to_door0_path = agent._simplified_path(state_dict, keys[0], doors[0], consider_doors=True, available_keys=[0])
    key1_to_door1_path = agent._simplified_path(state_dict, keys[1], doors[1], consider_doors=True, available_keys=[1])
    key0_to_key1_path = agent._simplified_path(state_dict, keys[0], keys[1], consider_doors=True, available_keys=[0])
    door0_to_key1_path = agent._simplified_path(state_dict, doors[0], keys[1], consider_doors=True, available_keys=[0])
    door1_to_key0_path = agent._simplified_path(state_dict, doors[1], keys[0], consider_doors=True, available_keys=[1])
    
    # All paths for analysis
    all_paths = [
        ("agent_to_key0", key0_path),
        ("agent_to_key1", key1_path),
        ("key0_to_door0", key0_to_door0_path),
        ("key1_to_door1", key1_to_door1_path),
        ("key0_to_key1", key0_to_key1_path),
        ("door0_to_key1", door0_to_key1_path),
        ("door1_to_key0", door1_to_key0_path)
    ]
    
    # 1. Path lengths
    path_lengths = {name: len(path) for name, path in all_paths if path}
    path_metrics["path_lengths"] = path_lengths
    
    # 2. Calculate Manhattan distances for comparison
    manhattan_distances = {}
    for name, path in all_paths:
        if not path:
            continue
        start = path[0]
        end = path[-1]
        manhattan = abs(start[0] - end[0]) + abs(start[1] - end[1])
        manhattan_distances[name] = manhattan
    
    path_metrics["manhattan_distances"] = manhattan_distances
    
    # 3. Path efficiency (actual/manhattan)
    path_efficiency = {}
    for name in path_lengths:
        if name in manhattan_distances and manhattan_distances[name] > 0:
            efficiency = path_lengths[name] / manhattan_distances[name]
            path_efficiency[name] = efficiency
    
    path_metrics["path_efficiency"] = path_efficiency
    avg_efficiency = np.mean(list(path_efficiency.values())) if path_efficiency else 0
    path_metrics["avg_path_efficiency"] = avg_efficiency
    
    # 4. Direction changes (corners/turns in path)
    direction_changes = {}
    total_changes = 0
    
    for name, path in all_paths:
        if not path or len(path) < 3:
            continue
            
        changes = 0
        for i in range(1, len(path) - 1):
            prev_dir = (path[i][0] - path[i-1][0], path[i][1] - path[i-1][1])
            next_dir = (path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            if prev_dir != next_dir:
                changes += 1
        
        direction_changes[name] = changes
        total_changes += changes
    
    path_metrics["direction_changes"] = direction_changes
    path_metrics["total_direction_changes"] = total_changes
    
    # 5. Identify choke points (spaces with limited access)
    # A choke point is a cell with only 2 adjacent navigable cells
    grid_size = 6
    navigable_cells = set()
    
    # Add all grid cells
    for x in range(grid_size):
        for y in range(grid_size):
            navigable_cells.add((x, y))
    
    # Remove walls
    for wall in walls:
        if tuple(wall) in navigable_cells:
            navigable_cells.remove(tuple(wall))
    
    # Find choke points
    choke_points = []
    for cell in navigable_cells:
        x, y = cell
        # Check adjacent cells
        adjacent = 0
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if (nx, ny) in navigable_cells:
                adjacent += 1
        
        # Cell with only 2 adjacent navigable cells is a choke point
        if adjacent == 2:
            choke_points.append(cell)
    
    path_metrics["choke_points"] = choke_points
    path_metrics["num_choke_points"] = len(choke_points)
    
    # 6. Check which paths go through choke points
    choke_point_traversals = {}
    for name, path in all_paths:
        if not path:
            continue
            
        traversals = 0
        for point in choke_points:
            if point in path:
                traversals += 1
        
        choke_point_traversals[name] = traversals
    
    path_metrics["choke_point_traversals"] = choke_point_traversals
    path_metrics["total_choke_traversals"] = sum(choke_point_traversals.values())
    
    # 7. Enemy path analysis
    enemy_zones = set()
    # Fix for accessing enemy types - we need to get them correctly from the state_dict
    enemy_types = state_dict['enemy_types']  # This is directly in state_dict, not in 'enemies'
    
    for i in range(len(state_dict['enemies'])):
        enemy_pos = state_dict['enemies'][i]
        # Convert numeric type to string type expected by _find_enemy_patrol_path
        enemy_type_str = 'vertical' if enemy_types[i] == 1 else 'horizontal'
        
        # Get enemy patrol path
        patrol_path = agent._find_enemy_patrol_path(state_dict, enemy_pos, 0 if enemy_type_str == 'horizontal' else 1)
        for pos in patrol_path:
            enemy_zones.add(pos)
    
    path_metrics["enemy_zones"] = enemy_zones
    
    # 8. Check path overlap with enemy zones
    enemy_overlaps = {}
    for name, path in all_paths:
        if not path:
            continue
            
        overlaps = 0
        for point in path:
            if point in enemy_zones:
                overlaps += 1
        
        enemy_overlaps[name] = overlaps
    
    path_metrics["enemy_overlaps"] = enemy_overlaps
    path_metrics["total_enemy_overlaps"] = sum(enemy_overlaps.values())
    
    # 9. Check for backtracking (revisiting cells)
    backtracking = {}
    for name, path in all_paths:
        if not path:
            continue
            
        # Count cells that appear multiple times in path
        visited = set()
        revisits = 0
        
        for point in path:
            if point in visited:
                revisits += 1
            else:
                visited.add(point)
        
        backtracking[name] = revisits
    
    path_metrics["backtracking"] = backtracking
    path_metrics["total_backtracking"] = sum(backtracking.values())
    
    # 10. Path variance
    if path_lengths:
        path_metrics["path_length_variance"] = np.var(list(path_lengths.values()))
    else:
        path_metrics["path_length_variance"] = 0
    
    return path_metrics

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
    print(f"{'Template':<15} {'Walls':<8} {'Path':<8} {'Strategy':<10} {'LIFO':<8} {'KSM':<8} {'Key0 Viable':<12} {'Key1 Viable':<12}")
    print("-" * 90)
    
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
        
        # Calculate detailed path metrics
        path_metrics = analyze_paths(agent, state)
        
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
        wall_density = 0.0
        detour_ratio = 0.0
        kdk_complexity = 0.0
        
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
                    elif "wall density" in key:
                        try:
                            wall_density = float(value)
                        except:
                            wall_density = 0.0
                    elif "detour ratio" in key:
                        try:
                            detour_ratio = float(value)
                        except:
                            detour_ratio = 0.0
                    elif "k-d-k complexity" in key:
                        try:
                            kdk_complexity = float(value)
                        except:
                            kdk_complexity = 0.0
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
                    elif "ksm factor" in key:
                        try:
                            constraints["ksm"] = float(value)
                        except:
                            constraints["ksm"] = 0.0
        
        # Store results
        results[template_name] = {
            "walls": wall_count,
            "wall_density": wall_density,
            "detour_ratio": detour_ratio,
            "kdk_complexity": kdk_complexity,
            "path": constraints.get("path", 0.0),
            "strategy": constraints.get("strategy", 0.0),
            "lifo": constraints.get("lifo", 0.0),
            "ksm": constraints.get("ksm", 0.0),
            "key0_viable": key0_viable,
            "key1_viable": key1_viable,
            "strategy1_cost": strategy1_cost,
            "strategy2_cost": strategy2_cost,
            "path_metrics": path_metrics
        }
        
        # Display in table format
        print(f"{template_name:<15} {wall_count:<8d} {constraints.get('path', 0.0):<8.2f} {constraints.get('strategy', 0.0):<10.2f} {constraints.get('lifo', 0.0):<8.2f} {constraints.get('ksm', 0.0):<8.2f} {str(key0_viable):<12} {str(key1_viable):<12}")
        
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
    
    print("\nDetailed Path Analysis:")
    for template, values in results.items():
        path_metrics = values["path_metrics"]
        print(f"\n{template} Path Metrics:")
        print(f"  Choke Points: {path_metrics['num_choke_points']}")
        print(f"  Choke Point Traversals: {path_metrics['total_choke_traversals']}")
        print(f"  Direction Changes: {path_metrics['total_direction_changes']}")
        print(f"  Enemy Zone Overlaps: {path_metrics['total_enemy_overlaps']}")
        print(f"  Backtracking Instances: {path_metrics['total_backtracking']}")
        print(f"  Average Path Efficiency: {path_metrics['avg_path_efficiency']:.2f}")
        print(f"  Path Length Variance: {path_metrics['path_length_variance']:.2f}")
    
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
    
    print("\nExpected Results:")
    print("1. Templates with high KSM factors should have meaningful strategic choices")
    print("2. Templates with forced path order (like bottleneck_med) should show only one viable strategy")
    print("3. Templates with both keys accessible (like bottleneck_hard) should show both strategies as viable")
    print("4. Templates with complex paths (like zipper_med) should have higher path complexity")
