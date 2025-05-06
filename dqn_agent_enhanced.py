import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from collections import deque, namedtuple

from prioritized_replay_buffer import PrioritizedReplayBuffer, Transition

# SET DEVICE! (for running on cloud instance)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN w/ improved architecture for custom LIFO environment
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        
        # network sizing
        hidden_size1 = 128
        hidden_size2 = 128
        
        self.fc1 = nn.Linear(state_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, hidden_size2 // 2)
        self.fc4 = nn.Linear(hidden_size2 // 2, action_size)
        
        # initialize weights with xavier initialization!
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.xavier_uniform_(self.fc3.weight)
        nn.init.xavier_uniform_(self.fc4.weight)

    # single sample (during action selection)
    def forward(self, x):
        if x.dim() == 1: x = x.unsqueeze(0)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)

# ENHANCED DQN agent class using prioritized replay buffer, etc.
class DQNAgentEnhanced:
    # initialize agent params
    def __init__(self, state_size, action_size, seed=0, 
                 learning_rate=0.0003, gamma=0.99, tau=0.0005,
                 buffer_size=100000, batch_size=128, update_every=8,
                 use_augmented_state=True, ksm_mode="off"):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.gamma = gamma
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.use_augmented_state = use_augmented_state
        self.ksm_mode = ksm_mode  # "off", "standard", or "adaptive"
        
        # for adaptive success bias:
        self.current_success_rate = 0.0
        
        # for environment-based KSM:
        self.env_ksm_factor = None  # will be calculated once on first call!
        
        # Q-networks (policy and target)
        self.policy_net = DQN(state_size, action_size).to(device)
        self.target_net = DQN(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # target network in evaluation mode
        
        # optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        
        # prioritized replay memory (call custom PRB)
        self.memory = PrioritizedReplayBuffer(buffer_size, device=device)
        
        # initialize step counter
        self.t_step = 0
        
        # for tracking training progress:
        self.loss_list = []
        
        # success tracking
        self.current_episode_success = False
        
        # template for logging
        self.template_name = "unknown"

    # set current template context (just used for logging)
    def set_template_context(self, template_name):
        self.template_name = template_name

    # choose appropriate state preprocessing based on flag
    def preprocess_state(self, state_dict):
        if self.use_augmented_state:
            state_vector = self.preprocess_state_augmented(state_dict)
            
            # add KSM features if enabled!
            if self.ksm_mode == "standard":
                ksm = self._calculate_key_selection_metric(state_dict)
                state_vector = np.append(state_vector, ksm)
            elif self.ksm_mode == "adaptive":
                ksm = self._calculate_adaptive_ksm(state_dict)
                state_vector = np.append(state_vector, ksm)
            
            return state_vector
        else:
            return self.preprocess_state_basic(state_dict)

    # convert dict observation to flat vector w/o advanced features
    def preprocess_state_basic(self, state_dict):
        # get basic components
        agent_pos = state_dict['agent']
        enemies_pos = state_dict['enemies'].flatten()
        enemy_directions = state_dict['enemy_directions']
        keys_pos = state_dict['keys'].flatten()
        key_status = state_dict['key_status']
        doors_pos = state_dict['doors'].flatten()
        door_status = state_dict['door_status']
        key_stack = state_dict['key_stack']
        
        # combine basic features into a vector
        state_vector = np.concatenate([
            agent_pos, # agent position (2)
            enemies_pos, # flattened enemy positions (2)
            enemy_directions, # enemy directions (1)
            keys_pos, # key positions (4)
            key_status, # key status (2)
            doors_pos, # door positions (4)
            door_status, # door status (2)
            key_stack # key stack (2)
        ])
        
        return state_vector

    # convert dict observation to flat vector w/ IMPROVED features for LIFO env.
    def preprocess_state_augmented(self, state_dict):
        # gets components
        agent_pos = state_dict['agent']
        enemies_pos = state_dict['enemies'].flatten()
        enemy_directions = state_dict['enemy_directions']
        keys_pos = state_dict['keys'].flatten()
        key_status = state_dict['key_status']
        doors_pos = state_dict['doors'].flatten()
        door_status = state_dict['door_status']
        key_stack = state_dict['key_stack']
        
        # calculates distances to keys and doors using Manhattan distance
        distances_to_keys = []
        for i, key_pos in enumerate(state_dict['keys']):
            if key_status[i] == 0:  # only include uncollected keys!
                dist = np.abs(agent_pos[0] - key_pos[0]) + np.abs(agent_pos[1] - key_pos[1])
                distances_to_keys.append(dist)
            else:
                distances_to_keys.append(-1)  # -1 indicates collected key
        
        distances_to_doors = []
        for i, door_pos in enumerate(state_dict['doors']):
            if door_status[i] == 0:  # cnly include unopened doors!
                dist = np.abs(agent_pos[0] - door_pos[0]) + np.abs(agent_pos[1] - door_pos[1])
                distances_to_doors.append(dist)
            else:
                distances_to_doors.append(-1)  # -1 indicates opened door
        
        # calculatee distances to enemies
        distances_to_enemies = []
        for enemy_pos in state_dict['enemies']:
            dist = np.abs(agent_pos[0] - enemy_pos[0]) + np.abs(agent_pos[1] - enemy_pos[1])
            distances_to_enemies.append(dist)
        
        # features related to key-door relationship
        keys_collected = np.sum(key_status)
        doors_opened = np.sum(door_status)
        keys_remaining = len(key_status) - keys_collected
        doors_remaining = len(door_status) - doors_opened
        
        # LIFO-specific features
        has_key = 1.0 if len(key_stack) > 0 and key_stack[0] >= 0 else 0.0
        
        # next usable door features
        next_usable_door_dist = -1
        next_usable_door_idx = -1
        if len(key_stack) > 0 and key_stack[0] >= 0:  # if we have a key in the stack
            top_key = key_stack[0]
            if top_key < len(door_status) and door_status[top_key] == 0:  # if matching door exists and is not open
                door_pos = state_dict['doors'][top_key]
                next_usable_door_dist = np.abs(agent_pos[0] - door_pos[0]) + np.abs(agent_pos[1] - door_pos[1])
                next_usable_door_idx = top_key
        
        # one-hot encoding of top key in stack
        top_key_onehot = np.zeros(3)  # 2 keys + no key
        if len(key_stack) > 0 and key_stack[0] >= 0:
            top_key_onehot[key_stack[0]] = 1
        else:
            top_key_onehot[2] = 1  # No key
        
        # combine all features into a single vector
        state_vector = np.concatenate([
            agent_pos, # agent position (2)
            enemies_pos, # flattened enemy positions (2)
            enemy_directions, # enemy directions (1)
            key_status, # key status (2)
            door_status, # door status (2)
            np.array(distances_to_keys, dtype=np.float32), # distances to keys (2)
            np.array(distances_to_doors, dtype=np.float32), # distances to doors (2)
            np.array(distances_to_enemies, dtype=np.float32), # distances to enemies (1)
            np.array([keys_collected, doors_opened, keys_remaining, doors_remaining], dtype=np.float32),  # summary stats (4)
            np.array([has_key, next_usable_door_dist, next_usable_door_idx], dtype=np.float32),  # LIFO features (3)
            top_key_onehot, # one-hot encoding of top key (3)
            key_stack # full key stack (2)
        ])
        
        return state_vector

    # calc shortest path distance accounting for walls and locked doors using BFS
    def _bfs_distance(self, state_dict, start_pos, target_pos, consider_doors=True, available_keys=None):
        # extract walls from state_dict
        walls = []
        for wall in state_dict['walls']:
            if wall[0] >= 0:  # filter out -1 placeholders
                walls.append((wall[0], wall[1]))
        
        # extract doors if needed
        doors = {}
        if consider_doors:
            for i, door_pos in enumerate(state_dict['doors']):
                if state_dict['door_status'][i] == 0:  # only include closed doors
                    doors[(door_pos[0], door_pos[1])] = i  # map position to door index
        
        # if no keys provided, initialize empty set
        if available_keys is None:
            available_keys = set()
        else:
            available_keys = set(available_keys)
        
        # grid size (assumed to be 6x6 as in the environment)
        grid_size = 6
        
        # check if positions are valid
        if not (0 <= start_pos[0] < grid_size and 0 <= start_pos[1] < grid_size and
                0 <= target_pos[0] < grid_size and 0 <= target_pos[1] < grid_size):
            return float('inf')  # Invalid position
        
        # if start and target are the same
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
                
                # check if new position is valid, not a wall, and not a locked door we can't pass
                if (0 <= nx < grid_size and 0 <= ny < grid_size and 
                    new_pos not in visited and 
                    new_pos not in walls and
                    not is_locked_door):
                    
                    if new_pos == target:
                        return dist + 1  # found the target!
                    
                    visited.add(new_pos)
                    queue.append((new_pos, dist + 1))
        
        # if no path found
        return float('inf')

    # return a simplified path betw/ positions for analysis
    def _simplified_path(self, state_dict, start_pos, target_pos, consider_doors=True, available_keys=None):
        # get walls
        walls = []
        for wall in state_dict['walls']:
            if wall[0] >= 0:  # filter out -1 placeholders
                walls.append((wall[0], wall[1]))
        
        # extract doors if needed
        doors = {}
        if consider_doors:
            for i, door_pos in enumerate(state_dict['doors']):
                if state_dict['door_status'][i] == 0:  # pnly include closed doors
                    doors[(door_pos[0], door_pos[1])] = i  # map position to door index
        
        # if no keys provided, initialize empty set
        if available_keys is None:
            available_keys = set()
        else:
            available_keys = set(available_keys)
        
        grid_size = 6
        
        # check if positions are valid
        if not (0 <= start_pos[0] < grid_size and 0 <= start_pos[1] < grid_size and
                0 <= target_pos[0] < grid_size and 0 <= target_pos[1] < grid_size):
            return []  # Invalid position
        
        # if start and target are the same
        if np.array_equal(start_pos, target_pos):
            return [tuple(start_pos)]
        
        # convert positions to tuples
        start = tuple(start_pos)
        target = tuple(target_pos)
        
        # BFS queue and path tracking
        queue = deque([(start, [start])])
        visited = {start}
        
        # possible movements: up, right, down, left
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            pos, path = queue.popleft()
            
            for dx, dy in moves:
                new_pos = (pos[0] + dx, pos[1] + dy)
                
                # check if position is a locked door we can't open
                is_locked_door = new_pos in doors and doors[new_pos] not in available_keys
                
                if (0 <= new_pos[0] < grid_size and 0 <= new_pos[1] < grid_size and
                    new_pos not in visited and new_pos not in walls and
                    not is_locked_door):
                    
                    new_path = path + [new_pos]
                    
                    if new_pos == target:
                        return new_path
                    
                    visited.add(new_pos)
                    queue.append((new_pos, new_path))
        
        return []  # no path found

    # check if a path exists betw/ two positions using BFS
    def _bfs_path_exists(self, state_dict, start_pos, target_pos, consider_doors=True, available_keys=None):
        # get walls
        walls = []
        for wall in state_dict['walls']:
            if wall[0] >= 0:  # filter out -1 placeholders
                walls.append((wall[0], wall[1]))
        
        # extract doors if needed
        doors = {}
        if consider_doors:
            for i, door_pos in enumerate(state_dict['doors']):
                if state_dict['door_status'][i] == 0:  # only include closed doors
                    doors[(door_pos[0], door_pos[1])] = i
        
        # if no keys provided, initialize empty set
        if available_keys is None:
            available_keys = set()
        else:
            available_keys = set(available_keys)
        
        grid_size = 6
        
        # skip if positions are invalid
        if not (0 <= start_pos[0] < grid_size and 0 <= start_pos[1] < grid_size and
                0 <= target_pos[0] < grid_size and 0 <= target_pos[1] < grid_size):
            return False
        
        # if start and target are the same
        if np.array_equal(start_pos, target_pos):
            return True
        
        # convert positions to tuples
        start = tuple(start_pos)
        target = tuple(target_pos)
        
        # BFS
        queue = deque([start])
        visited = {start}
        
        # possible movements
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        
        while queue:
            pos = queue.popleft()
            
            for dx, dy in moves:
                new_pos = (pos[0] + dx, pos[1] + dy)
                
                # Check if position is a locked door we can't open
                is_locked_door = new_pos in doors and doors[new_pos] not in available_keys
                
                if (0 <= new_pos[0] < grid_size and 0 <= new_pos[1] < grid_size and
                    new_pos not in visited and new_pos not in walls and
                    not is_locked_door):
                    
                    if new_pos == target:
                        return True
                    
                    visited.add(new_pos)
                    queue.append(new_pos)
        
        return False

    # find patrol path of enemy based on position and type (Vertical, horizontal)
    def _find_enemy_patrol_path(self, state_dict, enemy_pos, enemy_type):
        walls = []
        for wall in state_dict['walls']:
            if wall[0] >= 0:  # filter out -1 placeholders
                walls.append((wall[0], wall[1]))
        
        grid_size = 6
        patrol_path = []
        
        if enemy_type == 0:  # horizontal movement
            row = enemy_pos[1]
            # find leftmost and rightmost positions in this row
            left_bound, right_bound = 0, grid_size - 1
            
            # check for walls that limit horizontal movement
            for wall_pos in walls:
                if wall_pos[1] == row:  # wall in same row
                    if wall_pos[0] < enemy_pos[0] and wall_pos[0] + 1 > left_bound:
                        left_bound = wall_pos[0] + 1
                    if wall_pos[0] > enemy_pos[0] and wall_pos[0] - 1 < right_bound:
                        right_bound = wall_pos[0] - 1
            
            # create patrol path
            for x in range(left_bound, right_bound + 1):
                patrol_path.append((x, row))
                
        else:  # vertical movement
            col = enemy_pos[0]
            # find bottom and top positions in this column
            bottom_bound, top_bound = 0, grid_size - 1
            
            # check for walls that limit vertical movement
            for wall_pos in walls:
                if wall_pos[0] == col:  # wall in same column
                    if wall_pos[1] < enemy_pos[1] and wall_pos[1] + 1 > bottom_bound:
                        bottom_bound = wall_pos[1] + 1
                    if wall_pos[1] > enemy_pos[1] and wall_pos[1] - 1 < top_bound:
                        top_bound = wall_pos[1] - 1
            
            # create patrol path
            for y in range(bottom_bound, top_bound + 1):
                patrol_path.append((col, y))
                
        return patrol_path

    # analyze path characteristics for enhanced KSM calc!
    def _analyze_path_metrics(self, state_dict):
        agent_pos = state_dict['agent']
        keys = state_dict['keys']
        doors = state_dict['doors']
        walls = []
        
        for wall in state_dict['walls']:
            if wall[0] >= 0:  # filter out -1 placeholders
                walls.append((wall[0], wall[1]))
        
        # dict to store path analysis metrics
        path_metrics = {}
        
        # calc paths
        key0_path = self._simplified_path(state_dict, agent_pos, keys[0], consider_doors=True)
        key1_path = self._simplified_path(state_dict, agent_pos, keys[1], consider_doors=True)
        key0_to_door0_path = self._simplified_path(state_dict, keys[0], doors[0], consider_doors=True, available_keys=[0])
        key1_to_door1_path = self._simplified_path(state_dict, keys[1], doors[1], consider_doors=True, available_keys=[1])
        key0_to_key1_path = self._simplified_path(state_dict, keys[0], keys[1], consider_doors=True, available_keys=[0])
        door0_to_key1_path = self._simplified_path(state_dict, doors[0], keys[1], consider_doors=True, available_keys=[0])
        door1_to_key0_path = self._simplified_path(state_dict, doors[1], keys[0], consider_doors=True, available_keys=[1])
        
        # all paths
        all_paths = [
            ("agent_to_key0", key0_path),
            ("agent_to_key1", key1_path),
            ("key0_to_door0", key0_to_door0_path),
            ("key1_to_door1", key1_to_door1_path),
            ("key0_to_key1", key0_to_key1_path),
            ("door0_to_key1", door0_to_key1_path),
            ("door1_to_key0", door1_to_key0_path)
        ]
        
        # calculate path lengths
        path_lengths = {name: len(path) for name, path in all_paths if path}
        
        # direction changes (corners/turns in path)
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
            
            total_changes += changes
        
        path_metrics["total_direction_changes"] = total_changes
        
        # choke points (spaces with limited access)
        grid_size = 6
        navigable_cells = set()
        
        # add all grid cells
        for x in range(grid_size):
            for y in range(grid_size):
                navigable_cells.add((x, y))
        
        # remove walls
        for wall in walls:
            if tuple(wall) in navigable_cells:
                navigable_cells.remove(tuple(wall))
        
        # find choke points
        choke_points = []
        for cell in navigable_cells:
            x, y = cell
            # Check adjacent cells
            adjacent = 0
            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if (nx, ny) in navigable_cells:
                    adjacent += 1
            
            # cell with only 2 adjacent navigable cells is a choke point
            if adjacent == 2:
                choke_points.append(cell)
        
        path_metrics["num_choke_points"] = len(choke_points)
        
        # check which paths go through choke points
        choke_traversals = 0
        for name, path in all_paths:
            if not path:
                continue
                
            traversals = 0
            for point in choke_points:
                if point in path:
                    traversals += 1
            
            choke_traversals += traversals
        
        path_metrics["total_choke_traversals"] = choke_traversals
        
        # enemy path
        enemy_zones = set()
        
        # handle both possible enemy formats in state_dict
        if 'enemy_types' in state_dict:
            # Format from environment directly
            enemy_types = state_dict['enemy_types']
            for i in range(len(state_dict['enemies'])):
                enemy_pos = state_dict['enemies'][i]
                enemy_type = 0 if enemy_types[i] == 0 else 1  # 0 for horizontal, 1 for vertical
                
                # get enemy patrol path
                patrol_path = self._find_enemy_patrol_path(state_dict, enemy_pos, enemy_type)
                for pos in patrol_path:
                    enemy_zones.add(pos)
        elif 'enemies' in state_dict and isinstance(state_dict['enemies'], dict) and 'types' in state_dict['enemies']:
            # format from calculate_ksm_factors.py
            for i, enemy_type in enumerate(state_dict['enemies']['types']):
                enemy_pos = state_dict['enemies']['positions'][i]
                
                # get enemy patrol path
                patrol_path = self._find_enemy_patrol_path(state_dict, enemy_pos, 0 if enemy_type == 'horizontal' else 1)
                for pos in patrol_path:
                    enemy_zones.add(pos)
        
        # check path overlap with enemy zones
        enemy_overlaps = 0
        for name, path in all_paths:
            if not path:
                continue
                
            overlaps = 0
            for point in path:
                if point in enemy_zones:
                    overlaps += 1
            
            enemy_overlaps += overlaps
        
        path_metrics["total_enemy_overlaps"] = enemy_overlaps
        
        # path variance
        if path_lengths:
            path_metrics["path_length_variance"] = np.var(list(path_lengths.values()))
        else:
            path_metrics["path_length_variance"] = 0
        
        return path_metrics

    # calc KSM to guide key collection strategy
    def _calculate_key_selection_metric(self, state_dict):
        agent_pos = state_dict['agent']
        keys = state_dict['keys']
        doors = state_dict['doors']
        key_status = state_dict['key_status']
        door_status = state_dict['door_status']
        
        # if one or both keys already collected, no need for selection strategy!
        if key_status[0] == 1 or key_status[1] == 1:
            return 0.0
        
        # check direct accessibility to keys
        can_reach_key0 = self._bfs_path_exists(state_dict, agent_pos, keys[0], consider_doors=True)
        can_reach_key1 = self._bfs_path_exists(state_dict, agent_pos, keys[1], consider_doors=True)
        both_keys_accessible = can_reach_key0 and can_reach_key1
        
        # calculate BFS distances betw/ relevant positions
        agent_to_key0 = self._bfs_distance(state_dict, agent_pos, keys[0], consider_doors=True)
        agent_to_key1 = self._bfs_distance(state_dict, agent_pos, keys[1], consider_doors=True)
        
        # if we collect key0 . . .
        key0_to_door0 = self._bfs_distance(state_dict, keys[0], doors[0], consider_doors=True, available_keys=[0])
        door0_to_key1 = self._bfs_distance(state_dict, doors[0], keys[1], consider_doors=True, available_keys=[0])
        
        # if we collect key1 . . . 
        key1_to_door1 = self._bfs_distance(state_dict, keys[1], doors[1], consider_doors=True, available_keys=[1])
        door1_to_key0 = self._bfs_distance(state_dict, doors[1], keys[0], consider_doors=True, available_keys=[1])
        
        # other relevant distances
        key0_to_key1 = self._bfs_distance(state_dict, keys[0], keys[1], consider_doors=True, available_keys=[0])
        key0_to_door1 = self._bfs_distance(state_dict, keys[0], doors[1], consider_doors=True, available_keys=[0])
        key1_to_door0 = self._bfs_distance(state_dict, keys[1], doors[0], consider_doors=True, available_keys=[1])
        
        # handle cases where no valid path exists: use Manhattan as fallback!
        if agent_to_key0 == float('inf'):
            agent_to_key0 = self._manhattan_distance(agent_pos, keys[0]) * 1.5
        if agent_to_key1 == float('inf'):
            agent_to_key1 = self._manhattan_distance(agent_pos, keys[1]) * 1.5
        if key0_to_door0 == float('inf'):
            key0_to_door0 = self._manhattan_distance(keys[0], doors[0]) * 1.5
        if key1_to_door1 == float('inf'):
            key1_to_door1 = self._manhattan_distance(keys[1], doors[1]) * 1.5
        if key0_to_key1 == float('inf'):
            key0_to_key1 = self._manhattan_distance(keys[0], keys[1]) * 1.5
        if door0_to_key1 == float('inf'):
            door0_to_key1 = self._manhattan_distance(doors[0], keys[1]) * 1.5
        if door1_to_key0 == float('inf'):
            door1_to_key0 = self._manhattan_distance(doors[1], keys[0]) * 1.5
        if key0_to_door1 == float('inf'):
            key0_to_door1 = self._manhattan_distance(keys[0], doors[1]) * 1.5
        if key1_to_door0 == float('inf'):
            key1_to_door0 = self._manhattan_distance(keys[1], doors[0]) * 1.5
        
        # calculate costs for different collection strategies
        # Strategy 1: Key0 → Door0 → Key1 → Door1
        strategy1_cost = agent_to_key0 + key0_to_door0 + door0_to_key1 + key1_to_door1
        
        # Strategy 2: Key1 → Door1 → Key0 → Door0
        strategy2_cost = agent_to_key1 + key1_to_door1 + door1_to_key0 + key0_to_door0
        
        # Strategy 3: Key0 → Key1 → Door1 → Door0
        strategy3_cost = agent_to_key0 + key0_to_key1 + key1_to_door1 + key1_to_door0
        
        # Strategy 4: Key1 → Key0 → Door0 → Door1
        strategy4_cost = agent_to_key1 + key0_to_key1 + key0_to_door0 + key0_to_door1
        
        # check viability of strategies
        #     for Key0 first viability:
        key0_first_viable = (
            can_reach_key0 and
            key0_to_door0 != float('inf') and
            (
                (both_keys_accessible and 
                 (self._bfs_path_exists(state_dict, keys[1], doors[1], consider_doors=True, available_keys=[0, 1]) or
                  self._bfs_path_exists(state_dict, doors[0], doors[1], consider_doors=True, available_keys=[0, 1]))) or
                (not both_keys_accessible and
                 door0_to_key1 != float('inf') and key1_to_door1 != float('inf'))
            )
        )
        
        # for Key1 first viability
        key1_first_viable = (
            can_reach_key1 and
            key1_to_door1 != float('inf') and
            (
                (both_keys_accessible and 
                 (self._bfs_path_exists(state_dict, keys[0], doors[0], consider_doors=True, available_keys=[0, 1]) or
                  self._bfs_path_exists(state_dict, doors[1], doors[0], consider_doors=True, available_keys=[0, 1]))) or
                (not both_keys_accessible and
                 door1_to_key0 != float('inf') and key0_to_door0 != float('inf'))
            )
        )
        
        # keys are close to each other? (using BFS distance)
        keys_are_close = key0_to_key1 <= 3
        
        # key is near its door?
        key0_near_door0 = key0_to_door0 <= 3
        key1_near_door1 = key1_to_door1 <= 3
        
        # base score on which key to collect first (positive for Key0, negative for Key1)
        score = 0.0
        
        # if only one strategy viable, strongly bias towards it
        if key0_first_viable and not key1_first_viable:
            score = 0.9  # Strong preference for Key0 first
        elif key1_first_viable and not key0_first_viable:
            score = -0.9  # Strong preference for Key1 first
        else:
            # but if BOTH strategies viable, use LIFO strategic thinking:
            if keys_are_close:
                # determine which door is closer to its key
                if key0_to_door0 < key1_to_door1:
                    # Door0 is closer to Key0 than Door1 is to Key1
                    # with LIFO constraint, collect Key0 LAST (so Key1 first)
                    score -= 0.7
                elif key1_to_door1 < key0_to_door0:
                    # Door1 is closer to Key1 than Door0 is to Key0
                    # with LIFO constraint, collect Key1 LAST (so Key0 first)
                    score += 0.7
            
            # consider overall strategy costs
            key0_first_cost = min(strategy1_cost, strategy3_cost) if key0_first_viable else float('inf')
            key1_first_cost = min(strategy2_cost, strategy4_cost) if key1_first_viable else float('inf')
            
            # adjust score based on strategy costs
            if key0_first_cost < key1_first_cost:
                diff = key1_first_cost - key0_first_cost
                score += min(0.5, diff / 10)  # scale by difference but cap at 0.5
            elif key1_first_cost < key0_first_cost:
                diff = key0_first_cost - key1_first_cost
                score -= min(0.5, diff / 10)  # scale by difference but cap at 0.5
            
            # if one strategy is significantly better, strengthen the signal
            if abs(key0_first_cost - key1_first_cost) > 10:
                score *= 1.5  # Amplify the signal for clearly superior strategies
        
        # ensure the score is in [-1, 1] range
        return np.clip(score, -1.0, 1.0)

    # calc adaptive KSM based on env structure with enhanced formula
    def _calculate_adaptive_ksm(self, state_dict):
        # calc the base KSM value
        base_ksm = self._calculate_key_selection_metric(state_dict)
        
        # use cached environment factor or calculate it
        if self.env_ksm_factor is None:
            self.env_ksm_factor = self._calculate_enhanced_ksm_factor(state_dict)
        
        # apply environment factor to the KSM value
        adaptive_ksm = base_ksm * self.env_ksm_factor
        
        return adaptive_ksm

    # calculate enhanced KSM 
    def _calculate_enhanced_ksm_factor(self, state_dict):
        # first, calculate path metrics
        path_metrics = self._analyze_path_metrics(state_dict)
        
        agent_pos = state_dict['agent']
        keys = state_dict['keys']
        doors = state_dict['doors']
        walls = []
        
        for wall in state_dict['walls']:
            if wall[0] >= 0:  # Filter out -1 placeholders
                walls.append((wall[0], wall[1]))
        wall_count = len(walls)
        
        # calculate BFS distances for strategy costs
        agent_key0 = self._bfs_distance(state_dict, agent_pos, keys[0], consider_doors=True)
        agent_key1 = self._bfs_distance(state_dict, agent_pos, keys[1], consider_doors=True)
        key0_door0 = self._bfs_distance(state_dict, keys[0], doors[0], consider_doors=True, available_keys=[0])
        key1_door1 = self._bfs_distance(state_dict, keys[1], doors[1], consider_doors=True, available_keys=[1])
        door0_key1 = self._bfs_distance(state_dict, doors[0], keys[1], consider_doors=True, available_keys=[0])
        door1_key0 = self._bfs_distance(state_dict, doors[1], keys[0], consider_doors=True, available_keys=[1])
        
        # for infinite distances, use manhattan estimates
        if agent_key0 == float('inf'): agent_key0 = self._manhattan_distance(agent_pos, keys[0]) * 1.5
        if agent_key1 == float('inf'): agent_key1 = self._manhattan_distance(agent_pos, keys[1]) * 1.5
        if key0_door0 == float('inf'): key0_door0 = self._manhattan_distance(keys[0], doors[0]) * 1.5
        if key1_door1 == float('inf'): key1_door1 = self._manhattan_distance(keys[1], doors[1]) * 1.5
        if door0_key1 == float('inf'): door0_key1 = self._manhattan_distance(doors[0], keys[1]) * 1.5
        if door1_key0 == float('inf'): door1_key0 = self._manhattan_distance(doors[1], keys[0]) * 1.5
        
        # calculate strategy costs
        strategy1_cost = agent_key0 + key0_door0 + door0_key1 + key1_door1  # Key0 -> Door0 -> Key1 -> Door1
        strategy2_cost = agent_key1 + key1_door1 + door1_key0 + key0_door0  # Key1 -> Door1 -> Key0 -> Door0
        
        # check direct accessibility to keys
        can_reach_key0 = self._bfs_path_exists(state_dict, agent_pos, keys[0], consider_doors=True)
        can_reach_key1 = self._bfs_path_exists(state_dict, agent_pos, keys[1], consider_doors=True)
        both_keys_accessible = can_reach_key0 and can_reach_key1
        
        # for Key0 first viability:
        key0_first_viable = (
            can_reach_key0 and
            key0_door0 != float('inf') and
            (
                (both_keys_accessible and 
                 (self._bfs_path_exists(state_dict, keys[1], doors[1], consider_doors=True, available_keys=[0, 1]) or
                  self._bfs_path_exists(state_dict, doors[0], doors[1], consider_doors=True, available_keys=[0, 1]))) or
                (not both_keys_accessible and
                 door0_key1 != float('inf') and key1_door1 != float('inf'))
            )
        )
        
        # for Key1 first viability:
        key1_first_viable = (
            can_reach_key1 and
            key1_door1 != float('inf') and
            (
                (both_keys_accessible and 
                 (self._bfs_path_exists(state_dict, keys[0], doors[0], consider_doors=True, available_keys=[0, 1]) or
                  self._bfs_path_exists(state_dict, doors[1], doors[0], consider_doors=True, available_keys=[0, 1]))) or
                (not both_keys_accessible and
                 door1_key0 != float('inf') and key0_door0 != float('inf'))
            )
        )
        
        # calculate wall density
        wall_density = len(walls) / 12.0  # Normalize by a value that gives good spread
        
        # calculate detour ratio
        direct_paths = (self._manhattan_distance(agent_pos, keys[0]) +
                       self._manhattan_distance(agent_pos, keys[1]) +
                       self._manhattan_distance(keys[0], doors[0]) +
                       self._manhattan_distance(keys[1], doors[1]) +
                       self._manhattan_distance(keys[0], keys[1]))
        
        actual_paths = (agent_key0 + agent_key1 + key0_door0 + key1_door1 +
                       self._bfs_distance(state_dict, keys[0], keys[1], consider_doors=True, available_keys=[0]))
        
        if direct_paths > 0:
            detour_ratio = (actual_paths - direct_paths) / direct_paths
        else:
            detour_ratio = 0.0
            
        # calculate K-D-K sequence complexity
        key_door_key_complexity = 0.0
        
        if not both_keys_accessible:
            # if one key is behind a door, there's a dependency
            key_door_key_complexity = 0.3
        elif door0_key1 < agent_key1 or door1_key0 < agent_key0:
            # if going through a door provides a shorter path to the other key, this indicates level with dependencies
            key_door_key_complexity = 0.2
        
        # calculate path complexity
        path_complexity = min(1.0, (
            0.4 * wall_density +
            0.3 * detour_ratio + 
            0.2 * key_door_key_complexity +
            0.1 * min(1.0, path_metrics.get('total_enemy_overlaps', 0) / 10)
        ))
        
        # calculate strategy importance
        strategy_diff_pct = 0
        if key0_first_viable and key1_first_viable:
            # Both strategies are viable, compare costs
            if min(strategy1_cost, strategy2_cost) > 0:
                # Calculate cost difference ratio
                strategy_diff_pct = abs(strategy1_cost - strategy2_cost) / min(strategy1_cost, strategy2_cost) * 100
                
        # calculate LIFO constraint
        lifo_constraint = 0.3  # Base constraint value
        
        # keys being close to each other makes LIFO more important
        if self._bfs_distance(state_dict, keys[0], keys[1], consider_doors=True, available_keys=[0]) <= 3:
            lifo_constraint += 0.3
        
        # keys being close to their own doors makes order LESS critical
        if key0_door0 <= 3 and key1_door1 <= 3:
            lifo_constraint -= 0.2
        
        # check if one key is locked behind the other's door
        if not both_keys_accessible:
            # lne key is locked, reduces LIFO importance (forced order)
            lifo_constraint = 0.1
        
        # ensure LIFO constraint is in [0,1] range
        lifo_constraint = max(0.0, min(1.0, lifo_constraint))
        
        # get path metrics for enhanced calculation
        choke_points = path_metrics.get('num_choke_points', 0)
        choke_traversals = path_metrics.get('total_choke_traversals', 0)
        total_direction_changes = path_metrics.get('total_direction_changes', 0)
        path_length_variance = path_metrics.get('path_length_variance', 0)
        enemy_overlaps = path_metrics.get('total_enemy_overlaps', 0)
        
        # WALL FACTOR: quadratic penalty for low wall count specifically targeting sparse_med
        # threshold around 3 walls
        wall_factor = 0.2 * (1 - math.exp(-1.0 * (wall_count - 2.5)))
        
        # path complexity component
        path_exp = 0.05 * (1 - math.exp(-0.6 * path_complexity))
        
        # choke points
        choke_exp = 0.15 * (1 - math.exp(-0.01 * (choke_points * choke_traversals)))
        
        # direction changes weighted by variance: strongly penalize high directions with low variance
        variance_factor = math.sqrt(max(0.1, path_length_variance)) / 2.0
        direction_exp = 0.15 * (1 - math.exp(-0.1 * total_direction_changes)) * variance_factor
        
        # variance exponential
        variance_exp = 0.2 * (1 - math.exp(-0.15 * path_length_variance))
        
        # strategy component
        strategy_coef = 0.0
        if key0_first_viable and key1_first_viable:
            # enhanced weighting for strategy differences; less penalty for higher differences
            strategy_coef = 0.25 * math.pow(strategy_diff_pct / 100, 0.3)
        else:
            strategy_coef = 0.05  # Single viable strategy penalty
        
        # LIFO component with stronger weighting
        lifo_factor = 0.1 * lifo_constraint
        
        # combined KSM with stronger separation
        enhanced_ksm = wall_factor + path_exp + choke_exp + direction_exp + variance_exp + strategy_coef + lifo_factor
        
        # print the KSM analysis using the template context for identification
        template_context = getattr(self, 'template_name', 'unknown')
        print(f"Environment analysis for template '{template_context}':")
        print(f"  Walls: {len(walls)}")
        print(f"  Key0 first viable: {key0_first_viable}")
        print(f"  Key1 first viable: {key1_first_viable}")
        print(f"  Strategy1 cost: {strategy1_cost:.1f}")
        print(f"  Strategy2 cost: {strategy2_cost:.1f}")
        print(f"  Wall density: {wall_density:.2f}")
        print(f"  Detour ratio: {detour_ratio:.2f}")
        print(f"  K-D-K complexity: {key_door_key_complexity:.2f}")
        print(f"  Path complexity: {path_complexity:.2f}")
        print(f"  Strategy importance: {strategy_diff_pct/100 if key0_first_viable and key1_first_viable and min(strategy1_cost, strategy2_cost) > 0 else 0.1:.2f}")
        print(f"  LIFO constraint: {lifo_constraint:.2f}")
        print(f"  KSM factor: {enhanced_ksm:.2f}")
        
        return enhanced_ksm

    # get "critical paths" in the env
    def _calculate_critical_paths(self, state_dict):
        keys = state_dict['keys']
        doors = state_dict['doors']
        
        # store all paths between keys and doors
        critical_paths = []
        
        # find paths between all keys and doors
        for i in range(2):  # each key
            for j in range(2): # each door
                path = self._simplified_path(state_dict, keys[i], doors[j], consider_doors=True, available_keys=[i])
                if path:  # If path exists
                    critical_paths.append(path)
        
        # find path between keys
        key_path = self._simplified_path(state_dict, keys[0], keys[1], consider_doors=True)
        if key_path:
            critical_paths.append(key_path)
            
        return critical_paths

    # calculate KSM factor based on strategy imp. and path constraints
    # Used by calculate_ksm_factors.py as well, for outside checking! 
    def calculate_environment_ksm_factor(self, state_dict):
        walls = []
        for wall in state_dict['walls']:
            if wall[0] >= 0:  # filter out -1 placeholders
                walls.append(tuple(wall))
        
        grid_size = 6
        agent_pos = state_dict['agent']
        keys = state_dict['keys']
        doors = state_dict['doors']
        
        # check direct accessibility to keys
        can_reach_key0 = self._bfs_path_exists(state_dict, agent_pos, keys[0], consider_doors=True)
        can_reach_key1 = self._bfs_path_exists(state_dict, agent_pos, keys[1], consider_doors=True)
        both_keys_accessible = can_reach_key0 and can_reach_key1
        
        # calculate all relevant distances using BFS with door consideration
        agent_key0 = self._bfs_distance(state_dict, agent_pos, keys[0], consider_doors=True)
        agent_key1 = self._bfs_distance(state_dict, agent_pos, keys[1], consider_doors=True)
        key0_door0 = self._bfs_distance(state_dict, keys[0], doors[0], consider_doors=True, available_keys=[0])
        key1_door1 = self._bfs_distance(state_dict, keys[1], doors[1], consider_doors=True, available_keys=[1])
        key0_key1 = self._bfs_distance(state_dict, keys[0], keys[1], consider_doors=True, available_keys=[0])
        door0_key1 = self._bfs_distance(state_dict, doors[0], keys[1], consider_doors=True, available_keys=[0])
        door1_key0 = self._bfs_distance(state_dict, doors[1], keys[0], consider_doors=True, available_keys=[1])
        
        # handle infinite distances with manhattan estimates
        if agent_key0 == float('inf'): agent_key0 = self._manhattan_distance(agent_pos, keys[0]) * 1.5
        if agent_key1 == float('inf'): agent_key1 = self._manhattan_distance(agent_pos, keys[1]) * 1.5
        if key0_door0 == float('inf'): key0_door0 = self._manhattan_distance(keys[0], doors[0]) * 1.5
        if key1_door1 == float('inf'): key1_door1 = self._manhattan_distance(keys[1], doors[1]) * 1.5
        if key0_key1 == float('inf'): key0_key1 = self._manhattan_distance(keys[0], keys[1]) * 1.5
        if door0_key1 == float('inf'): door0_key1 = self._manhattan_distance(doors[0], keys[1]) * 1.5
        if door1_key0 == float('inf'): door1_key0 = self._manhattan_distance(doors[1], keys[0]) * 1.5
        
        # 1. STRATEGY VIABILITY: check if both key collection orders are viable
        # for Key0 first to be viable:
        key0_first_viable = (
            can_reach_key0 and
            key0_door0 != float('inf') and
            (
                (both_keys_accessible and 
                 (self._bfs_path_exists(state_dict, keys[1], doors[1], consider_doors=True, available_keys=[0, 1]) or
                  self._bfs_path_exists(state_dict, doors[0], doors[1], consider_doors=True, available_keys=[0, 1]))) or
                (not both_keys_accessible and
                 door0_key1 != float('inf') and key1_door1 != float('inf'))
            )
        )
        
        # for Key1 first to be viable:
        key1_first_viable = (
            can_reach_key1 and
            key1_door1 != float('inf') and
            (
                (both_keys_accessible and 
                 (self._bfs_path_exists(state_dict, keys[0], doors[0], consider_doors=True, available_keys=[0, 1]) or
                  self._bfs_path_exists(state_dict, doors[1], doors[0], consider_doors=True, available_keys=[0, 1]))) or
                (not both_keys_accessible and
                 door1_key0 != float('inf') and key0_door0 != float('inf'))
            )
        )
        
        # calculate path costs for different strategies
        strategy1 = agent_key0 + key0_door0 + door0_key1 + key1_door1  # Key0 → Door0 → Key1 → Door1
        strategy2 = agent_key1 + key1_door1 + door1_key0 + key0_door0  # Key1 → Door1 → Key0 → Door0
        
        # 2. PATH COMPLEXITY: measure how walls affect path length
        # calculate direct Manhattan distances
        manhattan_agent_key0 = self._manhattan_distance(agent_pos, keys[0])
        manhattan_agent_key1 = self._manhattan_distance(agent_pos, keys[1])
        manhattan_key0_door0 = self._manhattan_distance(keys[0], doors[0])
        manhattan_key1_door1 = self._manhattan_distance(keys[1], doors[1])
        manhattan_key0_key1 = self._manhattan_distance(keys[0], keys[1])
        
        # path complexity:
        
        # 1. wall density factor (more walls = higher complexity)
        wall_density = len(walls) / 12.0  # Normalize by a value that gives good spread
        
        # 2. detour factor (how much walls force longer paths)
        actual_paths = agent_key0 + agent_key1 + key0_door0 + key1_door1 + key0_key1
        direct_paths = manhattan_agent_key0 + manhattan_agent_key1 + manhattan_key0_door0 + manhattan_key1_door1 + manhattan_key0_key1
        
        if direct_paths > 0:
            detour_ratio = (actual_paths - direct_paths) / direct_paths
        else:
            detour_ratio = 0.0
            
        # 3. Key-Door-Key sequence complexity 
        # (if the level requires navigating from key to door to other key)
        # (higher values indicate more complex path dependencies)
        key_door_key_complexity = 0.0
        
        if not both_keys_accessible:
            # if one key is behind a door, there's a dependency
            key_door_key_complexity = 0.3
        elif door0_key1 < agent_key1 or door1_key0 < agent_key0:
            # if going through a door provides a shorter path to the other key, indicates level with dependencies
            key_door_key_complexity = 0.2
        
        # 4. enemy interaction (approximate the effect of enemies on path planning)
        # using a simple approximation based on path length to estimate the likelihood of enemy interaction
        enemy_interaction = min(1.0, (actual_paths / 30.0))
        
        # combine all path complexity factors
        path_complexity = min(1.0, (
            0.4 * wall_density +
            0.3 * detour_ratio + 
            0.2 * key_door_key_complexity +
            0.1 * enemy_interaction
        ))
        
        # 3. STRATEGY IMPORTANCE (how much the key order matters)
        strategy_importance = 0.0
        strategy_diff = 0.0
        if key0_first_viable and key1_first_viable:
            # Both strategies are viable, compare costs
            if min(strategy1, strategy2) > 0:
                # calculate cost difference ratio
                strategy_diff = abs(strategy1 - strategy2) / min(strategy1, strategy2)
                strategy_importance = min(1.0, strategy_diff)
            else:
                strategy_importance = 0.0
        elif key0_first_viable or key1_first_viable:
            # only one strategy is viable = low KSM value since no choice needed
            strategy_importance = 0.1
        else:
            # no viable strategies = something is wrong?
            strategy_importance = 0.0
        
        # 4. LIFO CONSTRAINT (key proximity affects key order importance)
        lifo_constraint = 0.3  # Base constraint value
        
        # keys being close to each other makes LIFO more important
        if key0_key1 <= 3:
            lifo_constraint += 0.3
        
        # keys being close to their own doors makes order LESS critical (can use quickly)
        if key0_door0 <= 3 and key1_door1 <= 3:
            lifo_constraint -= 0.2  # reduce LIFO importance
        
        # check if one key is locked behind the other's door
        if not both_keys_accessible:
            # one key is locked => reduces KSM importance (forced order)
            lifo_constraint = 0.1
        
        # ensure LIFO constraint is in [0,1] range
        lifo_constraint = max(0.0, min(1.0, lifo_constraint))
        
        # 5. COMBINED KSM FACTOR (weighted combination with higher path weight)
        ksm_factor = (
            0.30 * strategy_importance +  # strategy cost difference (30%)
            0.25 * lifo_constraint +      # LIFO-specific constraints (25%)
            0.45 * path_complexity        # path planning difficulty (45%)
        )
        
        # ensure the factor is in [0, 1] range
        ksm_factor = max(0.0, min(1.0, ksm_factor))
        
        # print detailed analysis
        template_name = getattr(self, 'template_name', 'unknown')
        print(f"Environment analysis for template '{template_name}':")
        print(f"  Walls: {len(walls)}")
        # Print viability as True/False strings for consistent parsing
        print(f"  Key0 first viable: {str(key0_first_viable)}")
        print(f"  Key1 first viable: {str(key1_first_viable)}")
        print(f"  Strategy1 cost: {strategy1:.1f}")
        print(f"  Strategy2 cost: {strategy2:.1f}")
        print(f"  Wall density: {wall_density:.2f}")
        print(f"  Detour ratio: {detour_ratio:.2f}")
        print(f"  K-D-K complexity: {key_door_key_complexity:.2f}")
        print(f"  Path complexity: {path_complexity:.2f}")
        print(f"  Strategy importance: {strategy_importance:.2f}")
        print(f"  LIFO constraint: {lifo_constraint:.2f}")
        print(f"  KSM factor: {ksm_factor:.2f}")
        
        # use enhanced KSM factor to replace the original KSM factor
        enhanced_ksm = self._calculate_enhanced_ksm_factor(state_dict)
        
        return enhanced_ksm

    # calculate manhattan distance between 2 positions
    def _manhattan_distance(self, pos1, pos2):
        if isinstance(pos1, np.ndarray) and isinstance(pos2, np.ndarray):
            return np.sum(np.abs(pos1 - pos2))
        else:
            # handle when positions are lists or other iterables
            return sum(abs(a - b) for a, b in zip(pos1, pos2))

    # process a step and learn if appropriate
    def step(self, state, action, reward, next_state, done, info=None):
        # check if episode was successful
        if done and info is not None and 'success' in info:
            self.current_episode_success = info['success']
        elif done:
            self.current_episode_success = False
            
        # preprocess states
        state_vector = self.preprocess_state(state)
        next_state_vector = self.preprocess_state(next_state)
        
        # store experience in replay memory
        state_tensor = torch.FloatTensor(state_vector).to(device)
        next_state_tensor = torch.FloatTensor(next_state_vector).to(device)
        action_tensor = action
        reward_tensor = reward
        done_tensor = done
        
        self.memory.push(state_tensor, action_tensor, next_state_tensor, reward_tensor, done_tensor,
                        is_success=self.current_episode_success if done else False)
        
        # learn every update_every steps
        self.t_step = (self.t_step + 1) % self.update_every
        if self.t_step == 0 and len(self.memory) > self.batch_size:
            # determine success bias based on current success rate
            if self.current_success_rate >= 0.7:
                # high success rate => focus more on successful episodes
                success_bias = 0.5
            else:
                # low success rate => default exploration balance
                success_bias = 0.3
                
            self.learn(success_bias=success_bias)
        
        # reset episode success if episode ended
        if done: self.current_episode_success = False

    # select action using epsilon-greedy policy
    def act(self, state, eps=0.0):
        # preprocess state
        state_vector = self.preprocess_state(state)
        state_tensor = torch.FloatTensor(state_vector).to(device)
        
        # epsilon-greedy action selection
        if random.random() > eps:
            with torch.no_grad():
                self.policy_net.eval()
                action_values = self.policy_net(state_tensor)
                self.policy_net.train()
                return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    # update value params using batchs of experience
    def learn(self, success_bias=0.4):
        # sample a batch from memory with priorities and increased success_bias
        (state_batch, action_batch, next_state_batch, reward_batch, done_batch), importance_weights, indices = self.memory.sample(self.batch_size, success_bias=success_bias)
        
        # convert to appropriate tensor shapes (if needed)
        if isinstance(action_batch, (list, tuple)):
            action_batch = torch.tensor(action_batch, dtype=torch.long, device=device)
        if isinstance(reward_batch, (list, tuple)):
            reward_batch = torch.tensor(reward_batch, dtype=torch.float, device=device)
        if isinstance(done_batch, (list, tuple)):
            done_batch = torch.tensor(done_batch, dtype=torch.float, device=device)
        
        # compute Q values for current states
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # compute next state values (with target network) using double DQN
        with torch.no_grad():
            # use policy network to select actions
            next_q_values = self.policy_net(next_state_batch)
            next_actions = next_q_values.max(1)[1].unsqueeze(1)
            
            # use target network to evaluate chosen actions
            next_state_values = self.target_net(next_state_batch).gather(1, next_actions).squeeze(1)
            next_state_values = next_state_values * (1 - done_batch)
        
        # compute expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        
        # calculate loss (with importance sampling weights for prioritized replay)
        td_errors = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1), reduction='none')
        weighted_td_errors = importance_weights.unsqueeze(1) * td_errors
        loss = weighted_td_errors.mean()
        
        # update priorities in replay buffer based on TD errors
        new_priorities = td_errors.detach().cpu().numpy() + 1e-6  # small constant to avoid zero priorities
        self.memory.update_priorities(indices, new_priorities.reshape(-1))
        
        # track loss
        self.loss_list.append(loss.item())
        
        # optimize
        self.optimizer.zero_grad()
        loss.backward()
        
        # gradient clipping
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.optimizer.step()
        
        # soft update target network
        self.soft_update()

    # soft update target network params
    def soft_update(self):
        for target_param, policy_param in zip(self.target_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(self.tau * policy_param.data + (1.0 - self.tau) * target_param.data)

    # save agent model and params
    def save(self, filename):
        checkpoint = {
            'policy_state_dict': self.policy_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss_list': self.loss_list,
            'env_ksm_factor': self.env_ksm_factor
        }
        torch.save(checkpoint, filename)

    # load agent model and params
    def load(self, filename):
        checkpoint = torch.load(filename, map_location=device)
        self.policy_net.load_state_dict(checkpoint['policy_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.loss_list = checkpoint['loss_list']
        
        # load KSM effectiveness data if it exists
        if 'env_ksm_factor' in checkpoint:
            self.env_ksm_factor = checkpoint['env_ksm_factor']
