# neural_symbolic_tree.py
import numpy as np
import torch
from collections import deque

# neural-guided decision tree for LIFO key-door environment; integrates symbolic reasoning w/ neural network guidance
class NeuralGuidedDecisionTree:
    # initialize DT w/ reference to DQN model (& device to run tensor operations on)
    def __init__(self, dqn_model, device="cpu"):
        self.dqn = dqn_model
        self.device = device
        
        # dict of decision nodes defining the tree structure
        self.decision_nodes = self._build_tree()

    # build DT structure
    def _build_tree(self):
        nodes = {}
        
        # root node checks the number of keys collected
        nodes['root'] = {
            'condition': "check_keys_collected",
            'branches': {
                'none': 'select_first_key',
                'some': 'follow_lifo',
                'all': 'terminal'
            }
        }
        
        # Wwen no keys collected, decide which to go for first
        nodes['select_first_key'] = {
            'condition': "check_key_accessibility",
            'branches': {
                'both_accessible': 'compare_key_distances',
                'only_key0': 'prefer_key0',
                'only_key1': 'prefer_key1',
                'none_accessible': 'neural_exploration'
            }
        }
        
        # compare distances to keys when both are accessible
        nodes['compare_key_distances'] = {
            'condition': "compare_distances_to_keys",
            'branches': {
                'key0_closer': 'check_door0_accessibility',
                'key1_closer': 'check_door1_accessibility',
                'similar_distance': 'compare_door_distances'
            }
        }
        
        # check if key0's door will be accessible after getting key0
        nodes['check_door0_accessibility'] = {
            'condition': "check_door_accessibility",
            'door_idx': 0,
            'branches': {
                'accessible': 'prefer_key0',
                'inaccessible': 'check_strategic_value'
            }
        }
        
        # check if key1's door will be accessible after getting key1
        nodes['check_door1_accessibility'] = {
            'condition': "check_door_accessibility",
            'door_idx': 1,
            'branches': {
                'accessible': 'prefer_key1',
                'inaccessible': 'check_strategic_value'
            }
        }
        
        # compare distances to doors when key distances are similar
        nodes['compare_door_distances'] = {
            'condition': "compare_key_door_pairs",
            'branches': {
                'key0_door0_better': 'prefer_key0',
                'key1_door1_better': 'prefer_key1',
                'similar': 'check_strategic_value'
            }
        }
        
        # determine strategic value when simple distance metrics aren't decisive
        nodes['check_strategic_value'] = {
            'condition': "evaluate_full_strategy",
            'branches': {
                'key0_first_better': 'prefer_key0',
                'key1_first_better': 'prefer_key1',
                'similar': 'neural_decision'
            }
        }
        
        # when some keys are collected, follow LIFO constraints
        nodes['follow_lifo'] = {
            'condition': "get_latest_key",
            'branches': {
                'key0': 'target_door0',
                'key1': 'target_door1',
                'no_keys': 'select_first_key' # fallback if stack is empty!
            }
        }
        
        # when following LIFO for key0 . . .
        nodes['target_door0'] = {
            'condition': "check_door_open",
            'door_idx': 0,
            'branches': {
                'open': 'select_next_key',
                'closed': 'navigate_to_door0'
            }
        }
        
        # when following LIFO for key1 . . .
        nodes['target_door1'] = {
            'condition': "check_door_open",
            'door_idx': 1,
            'branches': {
                'open': 'select_next_key',
                'closed': 'navigate_to_door1'
            }
        }
        
        # after using a key, select the next one
        nodes['select_next_key'] = {
            'condition': "check_remaining_keys",
            'branches': {
                'key0_remaining': 'prefer_key0',
                'key1_remaining': 'prefer_key1',
                'both_remaining': 'compare_key_distances',
                'none_remaining': 'target_victory'
            }
        }
        
        # terminal nodes (these return action preferences)
        nodes['prefer_key0'] = {'terminal': True, 'preference': 'key0'}
        nodes['prefer_key1'] = {'terminal': True, 'preference': 'key1'}
        nodes['navigate_to_door0'] = {'terminal': True, 'preference': 'door0'}
        nodes['navigate_to_door1'] = {'terminal': True, 'preference': 'door1'}
        nodes['target_victory'] = {'terminal': True, 'preference': 'victory'}
        nodes['neural_exploration'] = {'terminal': True, 'preference': 'neural_explore'}
        nodes['neural_decision'] = {'terminal': True, 'preference': 'neural_decide'}
        nodes['terminal'] = {'terminal': True, 'preference': 'complete'}
        
        return nodes

    # calculate manhattan distances between positions
    def _calculate_distances(self, state_dict, from_pos=None, to_positions=None):
        if from_pos is None:
            from_pos = state_dict['agent']
            
        if to_positions is None:
            # calc distances to keys and doors
            to_positions = {
                'key0': state_dict['keys'][0],
                'key1': state_dict['keys'][1],
                'door0': state_dict['doors'][0],
                'door1': state_dict['doors'][1]
            }
        
        distances = {}
        for name, pos in to_positions.items():
            dist = np.abs(from_pos[0] - pos[0]) + np.abs(from_pos[1] - pos[1])
            distances[name] = dist
            
        return distances

    # calculate the shortest path distance (accounting for walls and locked doors) using BFS
    def _bfs_distance(self, state_dict, start_pos, target_pos, available_keys=None):
        # extract walls from state_dict
        walls = []
        for wall in state_dict['walls']:
            if wall[0] >= 0:  # filter out -1 placeholders
                walls.append((wall[0], wall[1]))
        
        # extract doors if needed
        doors = {}
        for i, door_pos in enumerate(state_dict['doors']):
            if state_dict['door_status'][i] == 0: # only include closed doors
                doors[(door_pos[0], door_pos[1])] = i # map position to door index
        
        # if no keys provided, initialize empty set
        if available_keys is None:
            available_keys = set()
        else:
            available_keys = set(available_keys)
        
        # frid size (assumed to be 6x6 for this environment)
        grid_size = 6
        
        # check if positions are valid
        if not (0 <= start_pos[0] < grid_size and 0 <= start_pos[1] < grid_size and
                0 <= target_pos[0] < grid_size and 0 <= target_pos[1] < grid_size):
            return float('inf')  # invalid position
        
        # if start and target are the same:
        if np.array_equal(start_pos, target_pos):
            return 0
        
        # convert positions to tuples for use in sets/dictionaries
        start = tuple(start_pos)
        target = tuple(target_pos)
        
        # BFS queue
        queue = deque([(start, 0)])  # (position, distance)
        visited = {start}
        
        # possible movements: up, right, down, left
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            (x, y), dist = queue.popleft()
            
            # check all possible moves
            for dx, dy in moves:
                nx, ny = x + dx, y + dy
                new_pos = (nx, ny)
                
                # check if position is a locked door we can't open
                is_locked_door = new_pos in doors and doors[new_pos] not in available_keys
                
                # check if new position is valid, not a wall + not a locked door we can't pass
                if (0 <= nx < grid_size and 0 <= ny < grid_size and 
                    new_pos not in visited and 
                    new_pos not in walls and
                    not is_locked_door):
                    
                    if new_pos == target:
                        return dist + 1  # Found the target
                    
                    visited.add(new_pos)
                    queue.append((new_pos, dist + 1))
        
        # if no path found
        return float('inf')

    # calculate distances to keys and doors using BFS
    def _bfs_distances(self, state_dict, from_pos=None, consider_keys=None):
        if from_pos is None:
            from_pos = state_dict['agent']
            
        # fet key positions with status
        key_positions = {}
        for i, key_pos in enumerate(state_dict['keys']):
            if state_dict['key_status'][i] == 0:  # only include uncollected keys
                key_positions[f'key{i}'] = (tuple(key_pos), False) # false = not collected

        # get door positions with status
        door_positions = {}
        for i, door_pos in enumerate(state_dict['doors']):
            if state_dict['door_status'][i] == 0:  # only include unopened doors
                door_positions[f'door{i}'] = (tuple(door_pos), False)  # false = not open
        
        # calculate BFS distances to all relevant points
        distances = {}
        for name, (pos, _) in {**key_positions, **door_positions}.items():
            # Use the BFS distance method
            dist = self._bfs_distance(state_dict, from_pos, pos, consider_keys)
            distances[name] = dist
            
        return distances

    # manhattan distance between two positions
    def _manhattan_distance(self, pos1, pos2):
        return np.sum(np.abs(np.array(pos1) - np.array(pos2)))

    # evaluate a condition at a decision node
    def _evaluate_condition(self, condition, state_dict, node, state_tensor=None):
        # track number of keys collected
        if condition == "check_keys_collected":
            collected = sum(state_dict['key_status'])
            total_keys = len(state_dict['key_status'])
            
            if collected == 0:
                return 'none'
            elif collected == total_keys:
                return 'all'
            else:
                return 'some'
        
        # check if keys are accessible
        elif condition == "check_key_accessibility":
            key0_accessible = self._bfs_distance(state_dict, state_dict['agent'], state_dict['keys'][0]) < float('inf')
            key1_accessible = self._bfs_distance(state_dict, state_dict['agent'], state_dict['keys'][1]) < float('inf')
            
            if key0_accessible and key1_accessible:
                return 'both_accessible'
            elif key0_accessible:
                return 'only_key0'
            elif key1_accessible:
                return 'only_key1'
            else:
                return 'none_accessible'
        
        # compare distances to keys
        elif condition == "compare_distances_to_keys":
            distances = self._bfs_distances(state_dict)
            
            # check if keys exist in distances dict
            if 'key0' not in distances or 'key1' not in distances:
                # Fallback to Manhattan distances
                manhattan = self._calculate_distances(state_dict)
                key0_dist = manhattan.get('key0', float('inf'))
                key1_dist = manhattan.get('key1', float('inf'))
            else:
                key0_dist = distances['key0']
                key1_dist = distances['key1']
            
            # compare distances with a threshold for "significantly closer"
            threshold = 0.8
            if key0_dist < key1_dist * threshold:
                return 'key0_closer'
            elif key1_dist < key0_dist * threshold:
                return 'key1_closer'
            else:
                return 'similar_distance'
        
        # check if a door will be accessible after getting its key
        elif condition == "check_door_accessibility":
            door_idx = node.get('door_idx', 0)
            key_pos = state_dict['keys'][door_idx]
            door_pos = state_dict['doors'][door_idx]
            
            # check if there's a path from key to door using that key
            path_exists = self._bfs_distance(state_dict, key_pos, door_pos, available_keys=[door_idx]) < float('inf')
            
            return 'accessible' if path_exists else 'inaccessible'
        
        # compare key-door pair distances
        elif condition == "compare_key_door_pairs":
            # calculate distances: agent->key0->door0 vs agent->key1->door1
            agent_to_key0 = self._bfs_distance(state_dict, state_dict['agent'], state_dict['keys'][0])
            key0_to_door0 = self._bfs_distance(state_dict, state_dict['keys'][0], state_dict['doors'][0], available_keys=[0])
            
            agent_to_key1 = self._bfs_distance(state_dict, state_dict['agent'], state_dict['keys'][1])
            key1_to_door1 = self._bfs_distance(state_dict, state_dict['keys'][1], state_dict['doors'][1], available_keys=[1])
            
            # if any distance is infinite, use manhattan as fallback
            if agent_to_key0 == float('inf') or key0_to_door0 == float('inf'):
                manhattan = self._calculate_distances(state_dict)
                agent_to_key0 = manhattan.get('key0', float('inf'))
                key0_to_door0 = self._manhattan_distance(state_dict['keys'][0], state_dict['doors'][0])
            
            if agent_to_key1 == float('inf') or key1_to_door1 == float('inf'):
                manhattan = self._calculate_distances(state_dict)
                agent_to_key1 = manhattan.get('key1', float('inf'))
                key1_to_door1 = self._manhattan_distance(state_dict['keys'][1], state_dict['doors'][1])
            
            path0_cost = agent_to_key0 + key0_to_door0
            path1_cost = agent_to_key1 + key1_to_door1
            
            # compare path costs with a threshold
            threshold = 0.8
            if path0_cost < path1_cost * threshold:
                return 'key0_door0_better'
            elif path1_cost < path0_cost * threshold:
                return 'key1_door1_better'
            else:
                return 'similar'
        
        # evaluate full strategy costs
        elif condition == "evaluate_full_strategy":
            # strategy 1: Key0 first then Key1
            key0_first_cost = self._evaluate_strategy_cost(state_dict, [0, 1])
            
            # strategy 2: key1 first then Key0
            key1_first_cost = self._evaluate_strategy_cost(state_dict, [1, 0])
            
            # compare costs
            threshold = 0.8
            if key0_first_cost < key1_first_cost * threshold:
                return 'key0_first_better'
            elif key1_first_cost < key0_first_cost * threshold:
                return 'key1_first_better'
            else:
                return 'similar'
        
        # get the latest key from the key stack
        elif condition == "get_latest_key":
            key_stack = state_dict['key_stack']
            
            if key_stack[0] >= 0:  # There's a key on the stack
                if key_stack[0] == 0:
                    return 'key0'
                else:
                    return 'key1'
            else:
                return 'no_keys'
        
        # check if a door is open
        elif condition == "check_door_open":
            door_idx = node.get('door_idx', 0)
            if state_dict['door_status'][door_idx] == 1:
                return 'open'
            else:
                return 'closed'
        
        # check remaining uncollected keys
        elif condition == "check_remaining_keys":
            key0_remaining = state_dict['key_status'][0] == 0
            key1_remaining = state_dict['key_status'][1] == 0
            
            if key0_remaining and key1_remaining:
                return 'both_remaining'
            elif key0_remaining:
                return 'key0_remaining'
            elif key1_remaining:
                return 'key1_remaining'
            else:
                return 'none_remaining'
        
        # default to neural decision if condition not recognized!
        return 'neural_decision'

    # evaluate cost of a complete key collection strategy; return estimated cost of the strategy
    def _evaluate_strategy_cost(self, state_dict, key_order):
        agent_pos = state_dict['agent']
        total_cost = 0
        current_pos = agent_pos
        available_keys = []
        
        for key_idx in key_order:
            # skip if key already collected
            if state_dict['key_status'][key_idx] == 1:
                continue
                
            # calc cost to reach the key
            key_pos = state_dict['keys'][key_idx]
            key_cost = self._bfs_distance(state_dict, current_pos, key_pos, available_keys)
            
            # if no valid path, use manhattan distance as heuristic
            if key_cost == float('inf'):
                key_cost = self._manhattan_distance(current_pos, key_pos) * 1.5
                
            total_cost += key_cost
            
            # update current position and available keys
            current_pos = key_pos
            available_keys.append(key_idx)
            
            # calc cost to reach the corresponding door
            door_pos = state_dict['doors'][key_idx]
            door_cost = self._bfs_distance(state_dict, current_pos, door_pos, available_keys)
            
            # if no valid path, use manhattan distance as heuristic
            if door_cost == float('inf'):
                door_cost = self._manhattan_distance(current_pos, door_pos) * 1.5
                
            total_cost += door_cost
            
            # update current position
            current_pos = door_pos
        
        return total_cost

    # traverse DT to get strategic recommendation
    def traverse(self, state_dict, state_tensor=None):
        current_node = 'root'
        reasoning = ["Starting decision process at root"]
        
        # keep traversing until reaching a terminal node
        while True:
            node = self.decision_nodes[current_node]
            
            # check if reached a terminal node
            if 'terminal' in node and node['terminal']:
                preference = node['preference']
                reasoning.append(f"Reached terminal node: {current_node} with preference {preference}")
                return {
                    'node': current_node,
                    'preference': preference,
                    'reasoning': reasoning
                }
            
            # evaluate the condition to determine which branch to take
            condition = node['condition']
            branch = self._evaluate_condition(condition, state_dict, node, state_tensor)
            
            reasoning.append(f"At node '{current_node}', condition '{condition}' evaluated to branch '{branch}'")
            
            # move to the next node based on the branch
            if branch in node['branches']:
                current_node = node['branches'][branch]
            else:
                # default to neural decision if branch not found
                reasoning.append(f"Branch '{branch}' not found, defaulting to neural decision")
                return {
                    'node': 'neural_decision',
                    'preference': 'neural_decide',
                    'reasoning': reasoning
                }

    # guide action selection process using DT and neural network
    def guide_action_selection(self, state_dict, state_tensor, possible_actions):
        # traverse the decision tree to get a recommendation
        decision = self.traverse(state_dict, state_tensor)
        preference = decision['preference']
        
        if preference == 'neural_decide' or preference == 'neural_explore':
            # let the neural network decide:
            return {
                'mode': 'neural',
                'reasoning': decision['reasoning'],
                'action_mask': None  # No mask, use all actions
            }
        
        # for key preferences, mask actions that don't lead toward the preferred key
        if preference in ['key0', 'key1']:
            key_idx = 0 if preference == 'key0' else 1
            key_pos = state_dict['keys'][key_idx]
            
            # if the key is already collected, update preference to the door
            if state_dict['key_status'][key_idx] == 1:
                preference = f'door{key_idx}'
                door_pos = state_dict['doors'][key_idx]
                target_pos = door_pos
            else:
                target_pos = key_pos
                
            # create action mask that prefers actions moving toward the target
            action_mask = self._create_direction_mask(state_dict['agent'], target_pos)
            
            return {
                'mode': 'guided',
                'preference': preference,
                'target': target_pos,
                'reasoning': decision['reasoning'],
                'action_mask': action_mask
            }
        
        # for door preferences, mask actions that don't lead toward the door
        elif preference in ['door0', 'door1']:
            door_idx = 0 if preference == 'door0' else 1
            door_pos = state_dict['doors'][door_idx]
            
            # create action mask that prefers actions moving toward the door
            action_mask = self._create_direction_mask(state_dict['agent'], door_pos)
            
            return {
                'mode': 'guided',
                'preference': preference,
                'target': door_pos,
                'reasoning': decision['reasoning'],
                'action_mask': action_mask
            }
        
        # for victory target, find the nearest unopened door
        elif preference == 'victory':
            # Find nearest unopened door
            nearest_door = None
            min_dist = float('inf')
            
            for i, door_pos in enumerate(state_dict['doors']):
                if state_dict['door_status'][i] == 0:  #door is closed
                    dist = self._manhattan_distance(state_dict['agent'], door_pos)
                    if dist < min_dist:
                        min_dist = dist
                        nearest_door = door_pos
            
            if nearest_door is not None:
                action_mask = self._create_direction_mask(state_dict['agent'], nearest_door)
            else:
                action_mask = None  # no specific preference if all doors are open
            
            return {
                'mode': 'guided',
                'preference': 'victory',
                'target': nearest_door,
                'reasoning': decision['reasoning'],
                'action_mask': action_mask
            }
        
        # default to neural network
        return {
            'mode': 'neural',
            'reasoning': decision['reasoning'],
            'action_mask': None
        }

    # create an action mask that prefers actions moving toward the target
    def _create_direction_mask(self, current_pos, target_pos):
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        
        # initialize neutral mask
        mask = np.ones(5)  # [up, right, down, left, stay]
        
        # boost actions that move in the right direction
        if dy > 0:  # Target is above
            mask[0] += 0.5  # prefer up
        elif dy < 0:  # target is below
            mask[2] += 0.5  # prefer down
            
        if dx > 0:  # target is to the right
            mask[1] += 0.5  # prefer right
        elif dx < 0:  # target is to the left
            mask[3] += 0.5  # prefer left
        
        # slightly reduce preference for staying
        mask[4] -= 0.2
        
        return mask
