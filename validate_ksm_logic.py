import numpy as np
import torch
from collections import deque
import argparse

from template_lifo_corridors import TemplateLIFOCorridorsEnv
from dqn_agent_enhanced import DQNAgentEnhanced

def validate_bfs_paths(env_state, template_name):
    """Validate that BFS paths are working correctly."""
    # Create a temporary agent for BFS calculation
    agent = DQNAgentEnhanced(0, 0, use_augmented_state=True)
    
    # Extract key components
    agent_pos = env_state['agent']
    keys = env_state['keys']
    doors = env_state['doors']
    
    print(f"\n=== Path Analysis for {template_name} ===")
    print(f"Agent position: {agent_pos}")
    print(f"Key positions: {keys}")
    print(f"Door positions: {doors}")
    
    # Calculate all important paths using BFS - with door consideration
    paths = [
        ("Agent -> Key0", agent._bfs_distance(env_state, agent_pos, keys[0], consider_doors=True)),
        ("Agent -> Key1", agent._bfs_distance(env_state, agent_pos, keys[1], consider_doors=True)),
        ("Key0 -> Door0", agent._bfs_distance(env_state, keys[0], doors[0], consider_doors=True, available_keys=[0])),
        ("Key1 -> Door1", agent._bfs_distance(env_state, keys[1], doors[1], consider_doors=True, available_keys=[1])),
        ("Key0 -> Key1", agent._bfs_distance(env_state, keys[0], keys[1], consider_doors=True, available_keys=[0])),
        ("Door0 -> Key1", agent._bfs_distance(env_state, doors[0], keys[1], consider_doors=True, available_keys=[0])),
        ("Door1 -> Key0", agent._bfs_distance(env_state, doors[1], keys[0], consider_doors=True, available_keys=[1]))
    ]
    
    # Check which paths exist and display their lengths
    print("\nPath Existence:")
    for name, distance in paths:
        if distance == float('inf'):
            exists = "NO"
        else:
            exists = "YES"
        print(f"{name:<15}: {exists} - Distance: {distance if distance != float('inf') else 'inf'}")
    
    # Path existence checks with correct door accessibility
    # When checking if agent can reach doors, we need to have the corresponding key available!
    can_reach_key0 = agent._bfs_path_exists(env_state, agent_pos, keys[0], consider_doors=True)
    can_reach_key1 = agent._bfs_path_exists(env_state, agent_pos, keys[1], consider_doors=True)
    can_reach_door0 = agent._bfs_path_exists(env_state, agent_pos, doors[0], consider_doors=True, available_keys=[0])
    can_reach_door1 = agent._bfs_path_exists(env_state, agent_pos, doors[1], consider_doors=True, available_keys=[1])
    
    print("\nReachability from Agent:")
    print(f"Can reach Key0: {can_reach_key0}")
    print(f"Can reach Key1: {can_reach_key1}")
    print(f"Can reach Door0: {can_reach_door0}")
    print(f"Can reach Door1: {can_reach_door1}")
    
    # Calculate strategy costs with door consideration
    agent_key0 = agent._bfs_distance(env_state, agent_pos, keys[0], consider_doors=True)
    agent_key1 = agent._bfs_distance(env_state, agent_pos, keys[1], consider_doors=True)
    key0_door0 = agent._bfs_distance(env_state, keys[0], doors[0], consider_doors=True, available_keys=[0])
    key1_door1 = agent._bfs_distance(env_state, keys[1], doors[1], consider_doors=True, available_keys=[1])
    door0_key1 = agent._bfs_distance(env_state, doors[0], keys[1], consider_doors=True, available_keys=[0])
    door1_key0 = agent._bfs_distance(env_state, doors[1], keys[0], consider_doors=True, available_keys=[1])
    
    # Handle infinite distances
    if agent_key0 == float('inf'): agent_key0 = agent._manhattan_distance(agent_pos, keys[0]) * 1.5
    if agent_key1 == float('inf'): agent_key1 = agent._manhattan_distance(agent_pos, keys[1]) * 1.5
    if key0_door0 == float('inf'): key0_door0 = agent._manhattan_distance(keys[0], doors[0]) * 1.5
    if key1_door1 == float('inf'): key1_door1 = agent._manhattan_distance(keys[1], doors[1]) * 1.5
    if door0_key1 == float('inf'): door0_key1 = agent._manhattan_distance(doors[0], keys[1]) * 1.5
    if door1_key0 == float('inf'): door1_key0 = agent._manhattan_distance(doors[1], keys[0]) * 1.5
    
    # Calculate strategy costs
    strategy1 = agent_key0 + key0_door0 + door0_key1 + key1_door1  # Key0 -> Door0 -> Key1 -> Door1
    strategy2 = agent_key1 + key1_door1 + door1_key0 + key0_door0  # Key1 -> Door1 -> Key0 -> Door0
    
    print("\nStrategy Analysis:")
    print(f"Key0 first strategy cost: {strategy1:.1f}")
    print(f"Key1 first strategy cost: {strategy2:.1f}")
    
    # Comprehensive viability checks considering all possibilities
    
    # Check if the agent can reach both keys directly
    both_keys_accessible = can_reach_key0 and can_reach_key1
    
    # For Key0 first viability:
    # 1. Agent must be able to reach Key0
    # 2. Key0 must be able to reach Door0
    # 3a. If both keys are directly accessible:
    #   - Key1 must be able to reach Door1 (may need to go through Door0 first)
    # 3b. If Key1 is not directly accessible:
    #   - Door0 must be able to reach Key1, and Key1 must be able to reach Door1
    key0_first_viable = (
        can_reach_key0 and
        key0_door0 != float('inf') and
        (
            (both_keys_accessible and 
             (agent._bfs_path_exists(env_state, keys[1], doors[1], consider_doors=True, available_keys=[0, 1]) or
              agent._bfs_path_exists(env_state, doors[0], doors[1], consider_doors=True, available_keys=[0, 1]))) or
            (not both_keys_accessible and
             door0_key1 != float('inf') and key1_door1 != float('inf'))
        )
    )
    
    # For Key1 first viability:
    # 1. Agent must be able to reach Key1
    # 2. Key1 must be able to reach Door1
    # 3a. If both keys are directly accessible:
    #   - Key0 must be able to reach Door0 (may need to go through Door1 first)
    # 3b. If Key0 is not directly accessible:
    #   - Door1 must be able to reach Key0, and Key0 must be able to reach Door0
    key1_first_viable = (
        can_reach_key1 and
        key1_door1 != float('inf') and
        (
            (both_keys_accessible and 
             (agent._bfs_path_exists(env_state, keys[0], doors[0], consider_doors=True, available_keys=[0, 1]) or
              agent._bfs_path_exists(env_state, doors[1], doors[0], consider_doors=True, available_keys=[0, 1]))) or
            (not both_keys_accessible and
             door1_key0 != float('inf') and key0_door0 != float('inf'))
        )
    )
    
    # Debugging: print intermediate checks for bottleneck_hard
    if template_name == "bottleneck_hard":
        print("\nDebugging bottleneck_hard viability:")
        print(f"Both keys directly accessible: {both_keys_accessible}")
        
        # Key0 first debugging
        k0_cond1 = can_reach_key0
        k0_cond2 = key0_door0 != float('inf')
        k0_cond3a = both_keys_accessible and (
            agent._bfs_path_exists(env_state, keys[1], doors[1], consider_doors=True, available_keys=[0, 1]) or
            agent._bfs_path_exists(env_state, doors[0], doors[1], consider_doors=True, available_keys=[0, 1])
        )
        k0_cond3b = (not both_keys_accessible and door0_key1 != float('inf') and key1_door1 != float('inf'))
        print(f"Key0 first - Can reach Key0: {k0_cond1}")
        print(f"Key0 first - Key0 can reach Door0: {k0_cond2}")
        print(f"Key0 first - Path to Door1 with both keys: {k0_cond3a}")
        print(f"Key0 first - Sequential path if needed: {k0_cond3b}")
        
        # Key1 first debugging
        k1_cond1 = can_reach_key1
        k1_cond2 = key1_door1 != float('inf')
        k1_cond3a = both_keys_accessible and (
            agent._bfs_path_exists(env_state, keys[0], doors[0], consider_doors=True, available_keys=[0, 1]) or
            agent._bfs_path_exists(env_state, doors[1], doors[0], consider_doors=True, available_keys=[0, 1])
        )
        k1_cond3b = (not both_keys_accessible and door1_key0 != float('inf') and key0_door0 != float('inf'))
        print(f"Key1 first - Can reach Key1: {k1_cond1}")
        print(f"Key1 first - Key1 can reach Door1: {k1_cond2}")
        print(f"Key1 first - Path to Door0 with both keys: {k1_cond3a}")
        print(f"Key1 first - Sequential path if needed: {k1_cond3b}")
    
    print(f"\nKey0 first viable: {key0_first_viable}")
    print(f"Key1 first viable: {key1_first_viable}")
    
    # Calculate strategy importance
    if key0_first_viable and key1_first_viable:
        # Both strategies are viable, compare costs
        if min(strategy1, strategy2) > 0:
            # Calculate cost difference ratio
            strategy_diff = abs(strategy1 - strategy2) / min(strategy1, strategy2)
            strategy_importance = min(1.0, strategy_diff)
        else:
            strategy_importance = 0.0
    elif key0_first_viable or key1_first_viable:
        # Only one strategy is viable - lower importance since no choice needed
        strategy_importance = 0.1
    else:
        # No viable strategies - something is wrong
        strategy_importance = 0.0
    
    print(f"Strategy importance: {strategy_importance:.2f}")
    
    # LIFO constraint analysis
    lifo_constraint = 0.3  # Base constraint value
    
    # Keys being close to each other makes LIFO more important
    if agent._bfs_distance(env_state, keys[0], keys[1], consider_doors=True, available_keys=[0]) <= 3:
        lifo_constraint += 0.3
        print("Keys are close to each other: +0.3 to LIFO")
    
    # Keys being close to their own doors makes order LESS critical
    if key0_door0 <= 3 and key1_door1 <= 3:
        lifo_constraint -= 0.2
        print("Both keys close to doors: -0.2 to LIFO")
    
    # Check if one key is locked behind the other's door
    if both_keys_accessible:
        # Both keys are accessible directly - LIFO is more important
        pass
    else:
        # One key is locked - reduces LIFO importance (forced order)
        lifo_constraint = 0.1
        print("One key not directly reachable: LIFO reduced to 0.1")
    
    # Ensure LIFO constraint is in [0,1] range
    lifo_constraint = max(0.0, min(1.0, lifo_constraint))
    
    print(f"LIFO constraint: {lifo_constraint:.2f}")
    
    return {
        "template": template_name,
        "key0_viable": key0_first_viable,
        "key1_viable": key1_first_viable,
        "strategy1": strategy1,
        "strategy2": strategy2,
        "strategy_importance": strategy_importance,
        "lifo_constraint": lifo_constraint
    }

def check_all_templates():
    """Check KSM logic for all templates."""
    templates = [
        "basic_med", 
        "sparse_med", 
        "zipper_med", 
        "bottleneck_med", 
        "bottleneck_hard", 
        "corridors_med"
    ]
    
    results = {}
    
    for template_name in templates:
        # Create environment with the template
        env = TemplateLIFOCorridorsEnv(template_name=template_name, render_enabled=False, verbose=False)
        
        # Get initial state
        state, _ = env.reset()
        
        # Validate paths and strategies
        results[template_name] = validate_bfs_paths(state, template_name)
        
        # Close environment
        env.close()
        
        # Add a separator
        print("\n" + "-" * 60 + "\n")
    
    # Summary
    print("\n=== Summary for All Templates ===\n")
    print(f"{'Template':<15} {'Key0 Viable':<12} {'Key1 Viable':<12} {'Key0 First':<10} {'Key1 First':<10} {'Strategy':<10} {'LIFO':<8}")
    print("-" * 80)
    
    for template, data in results.items():
        print(f"{template:<15} {str(data['key0_viable']):<12} {str(data['key1_viable']):<12} {data['strategy1']:<10.1f} {data['strategy2']:<10.1f} {data['strategy_importance']:<10.2f} {data['lifo_constraint']:<8.2f}")
    
    return results

if __name__ == "__main__":
    check_all_templates()
