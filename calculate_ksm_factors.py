import numpy as np
import torch
from collections import deque
import argparse
import math

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
    total_enemy_overlaps = 0
    
    for name, path in all_paths:
        if not path:
            continue
            
        overlaps = 0
        for point in path:
            if point in enemy_zones:
                overlaps += 1
        
        enemy_overlaps[name] = overlaps
        total_enemy_overlaps += overlaps
    
    path_metrics["enemy_overlaps"] = enemy_overlaps
    path_metrics["total_enemy_overlaps"] = total_enemy_overlaps
    
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

def calculate_ksm_factor_components(agent, state_dict, path_metrics):
    """Calculate the components of the KSM factor using the mathematical transformation."""
    # Extract key components
    agent_pos = state_dict['agent']
    keys = state_dict['keys']
    doors = state_dict['doors']
    walls = []
    
    for wall in state_dict['walls']:
        if wall[0] >= 0:  # Filter out -1 placeholders
            walls.append((wall[0], wall[1]))
    
    # Get key metrics from path analysis
    choke_points = path_metrics['num_choke_points']
    choke_traversals = path_metrics['total_choke_traversals']
    total_direction_changes = path_metrics['total_direction_changes']
    path_length_variance = path_metrics['path_length_variance']
    enemy_overlaps = path_metrics['total_enemy_overlaps']
    
    # Get path complexity from agent's calculation
    ksm_factor = agent.calculate_environment_ksm_factor(state_dict)
    
    # Extract the components from the agent's method
    # These are printed during the agent's calculation
    path_complexity = 0.0
    strategy_importance = 0.0
    lifo_constraint = 0.0
    key0_first_viable = False
    key1_first_viable = False
    strategy1_cost = 0.0
    strategy2_cost = 0.0
    
    # Calculate BFS distances for strategy costs
    agent_key0 = agent._bfs_distance(state_dict, agent_pos, keys[0], consider_doors=True)
    agent_key1 = agent._bfs_distance(state_dict, agent_pos, keys[1], consider_doors=True)
    key0_door0 = agent._bfs_distance(state_dict, keys[0], doors[0], consider_doors=True, available_keys=[0])
    key1_door1 = agent._bfs_distance(state_dict, keys[1], doors[1], consider_doors=True, available_keys=[1])
    door0_key1 = agent._bfs_distance(state_dict, doors[0], keys[1], consider_doors=True, available_keys=[0])
    door1_key0 = agent._bfs_distance(state_dict, doors[1], keys[0], consider_doors=True, available_keys=[1])
    
    # Handle infinite distances with Manhattan estimates
    if agent_key0 == float('inf'): agent_key0 = agent._manhattan_distance(agent_pos, keys[0]) * 1.5
    if agent_key1 == float('inf'): agent_key1 = agent._manhattan_distance(agent_pos, keys[1]) * 1.5
    if key0_door0 == float('inf'): key0_door0 = agent._manhattan_distance(keys[0], doors[0]) * 1.5
    if key1_door1 == float('inf'): key1_door1 = agent._manhattan_distance(keys[1], doors[1]) * 1.5
    if door0_key1 == float('inf'): door0_key1 = agent._manhattan_distance(doors[0], keys[1]) * 1.5
    if door1_key0 == float('inf'): door1_key0 = agent._manhattan_distance(doors[1], keys[0]) * 1.5
    
    # Calculate strategy costs
    strategy1_cost = agent_key0 + key0_door0 + door0_key1 + key1_door1  # Key0 -> Door0 -> Key1 -> Door1
    strategy2_cost = agent_key1 + key1_door1 + door1_key0 + key0_door0  # Key1 -> Door1 -> Key0 -> Door0
    
    # Check direct accessibility to keys
    can_reach_key0 = agent._bfs_path_exists(state_dict, agent_pos, keys[0], consider_doors=True)
    can_reach_key1 = agent._bfs_path_exists(state_dict, agent_pos, keys[1], consider_doors=True)
    both_keys_accessible = can_reach_key0 and can_reach_key1
    
    # For Key0 first viability:
    key0_first_viable = (
        can_reach_key0 and
        key0_door0 != float('inf') and
        (
            (both_keys_accessible and 
             (agent._bfs_path_exists(state_dict, keys[1], doors[1], consider_doors=True, available_keys=[0, 1]) or
              agent._bfs_path_exists(state_dict, doors[0], doors[1], consider_doors=True, available_keys=[0, 1]))) or
            (not both_keys_accessible and
             door0_key1 != float('inf') and key1_door1 != float('inf'))
        )
    )
    
    # For Key1 first viability:
    key1_first_viable = (
        can_reach_key1 and
        key1_door1 != float('inf') and
        (
            (both_keys_accessible and 
             (agent._bfs_path_exists(state_dict, keys[0], doors[0], consider_doors=True, available_keys=[0, 1]) or
              agent._bfs_path_exists(state_dict, doors[1], doors[0], consider_doors=True, available_keys=[0, 1]))) or
            (not both_keys_accessible and
             door1_key0 != float('inf') and key0_door0 != float('inf'))
        )
    )
    
    # Calculate wall density
    wall_density = len(walls) / 12.0  # Normalize by a value that gives good spread
    
    # Calculate detour ratio
    direct_paths = (agent._manhattan_distance(agent_pos, keys[0]) +
                   agent._manhattan_distance(agent_pos, keys[1]) +
                   agent._manhattan_distance(keys[0], doors[0]) +
                   agent._manhattan_distance(keys[1], doors[1]) +
                   agent._manhattan_distance(keys[0], keys[1]))
    
    actual_paths = (agent_key0 + agent_key1 + key0_door0 + key1_door1 +
                   agent._bfs_distance(state_dict, keys[0], keys[1], consider_doors=True, available_keys=[0]))
    
    if direct_paths > 0:
        detour_ratio = (actual_paths - direct_paths) / direct_paths
    else:
        detour_ratio = 0.0
        
    # Calculate K-D-K sequence complexity
    key_door_key_complexity = 0.0
    
    if not both_keys_accessible:
        # If one key is behind a door, there's a dependency
        key_door_key_complexity = 0.3
    elif door0_key1 < agent_key1 or door1_key0 < agent_key0:
        # If going through a door provides a shorter path to the other key
        # This indicates interesting level design with dependencies
        key_door_key_complexity = 0.2
    
    # Calculate Path complexity
    path_complexity = min(1.0, (
        0.4 * wall_density +
        0.3 * detour_ratio + 
        0.2 * key_door_key_complexity +
        0.1 * min(1.0, enemy_overlaps / 10)
    ))
    
    # Calculate strategy importance
    if key0_first_viable and key1_first_viable:
        # Both strategies are viable, compare costs
        if min(strategy1_cost, strategy2_cost) > 0:
            # Calculate cost difference ratio
            strategy_diff_pct = abs(strategy1_cost - strategy2_cost) / min(strategy1_cost, strategy2_cost) * 100
            strategy_importance = min(1.0, strategy_diff_pct / 100)
        else:
            strategy_importance = 0.0
    elif key0_first_viable or key1_first_viable:
        # Only one strategy is viable - low KSM value since no choice needed
        strategy_importance = 0.1
    else:
        # No viable strategies - something is wrong
        strategy_importance = 0.0
    
    # Calculate LIFO constraint
    lifo_constraint = 0.3  # Base constraint value
    
    # Keys being close to each other makes LIFO more important
    if agent._bfs_distance(state_dict, keys[0], keys[1], consider_doors=True, available_keys=[0]) <= 3:
        lifo_constraint += 0.3
    
    # Keys being close to their own doors makes order LESS critical
    if key0_door0 <= 3 and key1_door1 <= 3:
        lifo_constraint -= 0.2
    
    # Check if one key is locked behind the other's door
    if not both_keys_accessible:
        # One key is locked - reduces KSM importance (forced order)
        lifo_constraint = 0.1
    
    # Ensure LIFO constraint is in [0,1] range
    lifo_constraint = max(0.0, min(1.0, lifo_constraint))
    
    # Return all components and related metrics
    components = {
        "path_complexity": path_complexity,
        "wall_density": wall_density,
        "detour_ratio": detour_ratio,
        "kdk_complexity": key_door_key_complexity,
        "strategy_importance": strategy_importance,
        "lifo_constraint": lifo_constraint,
        "key0_viable": key0_first_viable,
        "key1_viable": key1_first_viable,
        "strategy1_cost": strategy1_cost,
        "strategy2_cost": strategy2_cost,
        "strategy_diff_pct": strategy_diff_pct if key0_first_viable and key1_first_viable and min(strategy1_cost, strategy2_cost) > 0 else 0,
        "walls": len(walls),
        "original_ksm": ksm_factor
    }
    
    return components

def calculate_enhanced_ksm_factor(components, path_metrics):
    """Calculate the enhanced KSM factor using a mathematically principled transformation."""
    # Get key metrics
    path_complexity = components["path_complexity"]
    choke_points = path_metrics["num_choke_points"]
    choke_traversals = path_metrics["total_choke_traversals"]
    total_direction_changes = path_metrics["total_direction_changes"]
    path_length_variance = path_metrics["path_length_variance"]
    enemy_overlaps = path_metrics["total_enemy_overlaps"]
    wall_count = components["walls"]
    strategy_diff_pct = components["strategy_diff_pct"]
    lifo_constraint = components["lifo_constraint"]
    
    # CRITICAL: Wall count as exponential factor
    # This creates a dramatic difference between 2 walls and higher counts
    wall_exp = 0.2 * (1 - math.exp(-0.4 * (wall_count - 1.5)))
    
    # Path complexity component
    path_exp = 0.05 * (1 - math.exp(-0.6 * path_complexity))
    
    # Choke points - more important for corridors_med
    choke_exp = 0.15 * (1 - math.exp(-0.01 * (choke_points * choke_traversals)))
    
    # Direction changes weighted by variance
    # This heavily penalizes templates with high directions but low variance (sparse_med)
    variance_weight = math.sqrt(path_length_variance) / 2.0
    direction_exp = 0.15 * (1 - math.exp(-0.1 * total_direction_changes)) * variance_weight
    
    # Variance exponential - emphasized for zipper_med
    variance_exp = 0.2 * (1 - math.exp(-0.15 * path_length_variance))
    
    # Strategy importance - emphasize meaningful choice
    strategy_coef = 0.0
    if components["key0_viable"] and components["key1_viable"]:
        # Enhanced quadratic scaling for strategy differences
        strategy_coef = 0.2 * math.pow(strategy_diff_pct / 100, 0.35)
    else:
        strategy_coef = 0.05  # Single viable strategy penalty
    
    # LIFO component
    lifo_factor = 0.1 * lifo_constraint
    
    # Combined KSM with stronger separation
    enhanced_ksm = wall_exp + path_exp + choke_exp + direction_exp + variance_exp + strategy_coef + lifo_factor
    
    return enhanced_ksm

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
    print(f"{'Template':<15} {'Walls':<8} {'Path':<8} {'Strategy':<10} {'LIFO':<8} {'KSM':<8} {'Enhanced':<10} {'Key0 Viable':<12} {'Key1 Viable':<12}")
    print("-" * 100)
    
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
        
        # Calculate KSM factor components
        components = calculate_ksm_factor_components(agent, state, path_metrics)
        
        # Calculate enhanced KSM factor
        enhanced_ksm = calculate_enhanced_ksm_factor(components, path_metrics)
        
        # Store results
        results[template_name] = {
            "components": components,
            "path_metrics": path_metrics,
            "enhanced_ksm": enhanced_ksm
        }
        
        # Display in table format
        print(f"{template_name:<15} {components['walls']:<8d} {components['path_complexity']:<8.2f} {components['strategy_importance']:<10.2f} {components['lifo_constraint']:<8.2f} {components['original_ksm']:<8.2f} {enhanced_ksm:<10.2f} {str(components['key0_viable']):<12} {str(components['key1_viable']):<12}")
        
        # Close the environment
        env.close()
    
    print("\n===== Enhanced KSM Factor Analysis =====\n")
    
    # Sort templates by enhanced KSM factor (highest to lowest)
    sorted_templates = sorted(results.items(), key=lambda x: x[1]["enhanced_ksm"], reverse=True)
    
    print("Templates ranked by enhanced KSM factor (highest to lowest):")
    for i, (template, values) in enumerate(sorted_templates):
        enhanced_ksm = values["enhanced_ksm"]
        print(f"{i+1}. {template:<15} - Enhanced KSM: {enhanced_ksm:.2f}")
    
    print("\n===== Component Breakdown by Template =====\n")
    for template, values in results.items():
        components = values["components"]
        path_metrics = values["path_metrics"]
        
        # Calculate the individual components for display
        path_complexity = components["path_complexity"]
        choke_points = path_metrics["num_choke_points"]
        choke_traversals = path_metrics["total_choke_traversals"]
        total_direction_changes = path_metrics["total_direction_changes"]
        path_length_variance = path_metrics["path_length_variance"]
        enemy_overlaps = path_metrics["total_enemy_overlaps"]
        wall_count = components["walls"]
        strategy_diff_pct = components["strategy_diff_pct"]
        lifo_constraint = components["lifo_constraint"]
        
        # Calculate each component
        wall_quadratic = 0.03 * ((wall_count - 2) ** 2) / 16
        path_exp = 0.08 * (1 - math.exp(-0.5 * path_complexity))
        choke_exp = 0.08 * (1 - math.exp(-0.02 * (choke_points * choke_traversals)))
        direction_variance_ratio = min(1.0, path_length_variance / 3.0)
        direction_exp = 0.15 * (1 - math.exp(-0.12 * total_direction_changes)) * direction_variance_ratio
        variance_exp = 0.25 * (1 - math.exp(-0.2 * path_length_variance))
        
        strategy_cube = 0
        if components["key0_viable"] and components["key1_viable"]:
            strategy_cube = 0.2 * (strategy_diff_pct / 100) ** 0.5
        else:
            strategy_cube = 0.05
            
        lifo_factor = 0.1 * lifo_constraint
        
        print(f"\n{template} Component Breakdown:")
        print(f"  Wall quadratic:         {wall_quadratic:.2f}")
        print(f"  Path exponential:       {path_exp:.2f}")
        print(f"  Choke exponential:      {choke_exp:.2f}")
        print(f"  Direction exponential:  {direction_exp:.2f}")
        print(f"  Variance exponential:   {variance_exp:.2f}")
        print(f"  Strategy factor:        {strategy_cube:.2f}")
        print(f"  LIFO factor:            {lifo_factor:.2f}")
        print(f"  TOTAL Enhanced KSM:     {values['enhanced_ksm']:.2f}")
        print(f"  Original KSM:           {components['original_ksm']:.2f}")
    
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
        components = values["components"]
        key0_viable = components["key0_viable"]
        key1_viable = components["key1_viable"]
        strategy1_cost = components["strategy1_cost"]
        strategy2_cost = components["strategy2_cost"]
        
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
