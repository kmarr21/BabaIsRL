import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pygame
import os
import time

# custom KeyCorridors env w/ LIFO constraint using fixed templates
class TemplateLIFOCorridorsEnv(gym.Env):
    # initialize; fallback to basic_med template if none given
    def __init__(self, template_name="basic_med", render_enabled=True, verbose=False, 
                 use_reward_shaping=True):
        super().__init__()
        
        # grid size (6x6)
        self.size = 6
        
        # template name
        self.template_name = template_name
        if template_name not in ["basic_med", "sparse_med", "zipper_med", "bottleneck_med", "corridors_med", "bottleneck_hard"]:
            print(f"Warning: Unknown template '{template_name}'. Defaulting to 'basic_med'.")
            self.template_name = "basic_med"
        
        # rendering flag
        self.render_enabled = render_enabled
        self.verbose = verbose
        
        # reward shaping flag
        self.use_reward_shaping = use_reward_shaping
        
        # action space: move in 4 directions or stay
        self.action_space = spaces.Discrete(5)
        
        # observation space (includes enemy_types and walls)
        self.observation_space = spaces.Dict({
            'agent': spaces.Box(low=0, high=self.size-1, shape=(2,), dtype=np.int32),
            'enemies': spaces.Box(low=0, high=self.size-1, shape=(1, 2), dtype=np.int32),
            'enemy_directions': spaces.Box(low=0, high=3, shape=(1,), dtype=np.int32),
            'enemy_types': spaces.MultiDiscrete([2]), # 0: horizontal, 1: vertical
            'keys': spaces.Box(low=0, high=self.size-1, shape=(2, 2), dtype=np.int32),
            'key_status': spaces.MultiBinary(2),
            'doors': spaces.Box(low=0, high=self.size-1, shape=(2, 2), dtype=np.int32),
            'door_status': spaces.MultiBinary(2),
            'key_stack': spaces.Box(low=-1, high=1, shape=(2,), dtype=np.int32), # LIFO stack of keys
            'walls': spaces.Box(low=0, high=self.size-1, shape=(10, 2), dtype=np.int32) # up to 10 walls
        })
        
        # visualization settings
        self.window_size = 600
        self.panel_height = 80  # height for key inventory panel
        self.cell_size = self.window_size // self.size
        self.window = None
        
        # define colors for keys and doors
        self.key_door_colors = {
            0: (255, 165, 0), # orange
            1: (255, 100, 255) # purple
        }
        
        # enemy colors
        self.enemy_colors = {
            'horizontal': (0, 0, 255), # blue for horizontal
            'vertical': (255, 0, 0) # red for vertical
        }
        
        # initialize pygame for rendering
        if render_enabled:
            pygame.init()
            pygame.display.init()
            temp_surface = pygame.display.set_mode((1, 1))
            pygame.font.init()
            
            # load images
            self.images = {}
            image_files = {
                'bunny': os.path.join(os.path.dirname(__file__), 'icons', 'bunny.png'),
                'robot': os.path.join(os.path.dirname(__file__), 'icons', 'robot.png'),
                'key': os.path.join(os.path.dirname(__file__), 'icons', 'key.png'),
                'door': os.path.join(os.path.dirname(__file__), 'icons', 'door.png')
            }
            
            for name, file in image_files.items():
                try:
                    img = pygame.image.load(file)
                    img = pygame.transform.scale(img, (self.cell_size - 4, self.cell_size - 4))
                    if name in ['key', 'door', 'robot']:  # Convert black to white for tintable images
                        img = img.convert_alpha()
                        self._convert_black_to_white(img)
                    self.images[name] = img
                except pygame.error as e:
                    print(f"Couldn't load image {file}: {e}")
                    self.images = {}
                    break
            
            pygame.display.quit()
        
        # episode stats
        self.steps_taken = 0
        self.total_reward = 0
        
        # key stack for LIFO constraint
        self.key_stack = []
        
        # message display for wrong key attempts
        self.show_wrong_key = False
        self.wrong_key_time = 0
        self.wrong_key_duration = 1.0  # seconds

        # add to the init method:
        self.show_game_over = False
        self.game_over_message = "Game Over!"
        
        # load template definitions
        self.templates = self._define_templates()

    # convert black pixels to white for recoloring (while maintaining alpha channel)
    def _convert_black_to_white(self, surface):
        arr = pygame.surfarray.pixels3d(surface)
        black_pixels = (arr[..., 0] < 30) & (arr[..., 1] < 30) & (arr[..., 2] < 30)
        arr[black_pixels] = [255, 255, 255]
        del arr

    # define fixed templates
    def _define_templates(self):
        templates = {
            # basic template
            "basic_med": {
                "name": "Basic",
                "description": "Simple layout with a few walls",
                "walls": [
                    [1, 3], [2, 3], [2, 2]
                ],
                "agent_pos": [0, 0], # bunny starts in bottom-left
                "enemies": {
                    "positions": [[3, 4]],  # blue robot
                    "directions": [1],  # right
                    "types": ["horizontal"]  # blue moves horizontally
                },
                "keys": {
                    "positions": [[2, 1], [3, 1]]  # orange, purple keys
                },
                "doors": {
                    "positions": [[1, 2], [5, 5]]  # orange, purple doors
                }
            },
            
            # sparse template
            "sparse_med": {
                "name": "Sparse",
                "description": "Very few walls with mostly open space",
                "walls": [
                    [1, 1], [3, 4]
                ],
                "agent_pos": [0, 0],  # bunny starts in bottom-left
                "enemies": {
                    "positions": [[2, 2]],  # blue robot
                    "directions": [1],  # right
                    "types": ["horizontal"]
                },
                "keys": {
                    "positions": [[2, 4], [4, 1]]  # orange, purple keys
                },
                "doors": {
                    "positions": [[4, 4], [1, 2]]  # orange, purple doors
                }
            },
            
            # zipper template
            "zipper_med": {
                "name": "Zipper",
                "description": "Vertical corridor requiring careful timing",
                "walls": [
                    [1, 0], [1, 1], [1, 2], [1, 3], [3, 4], [3, 5]
                ],
                "agent_pos": [0, 0],  # bunny starts in bottom-left
                "enemies": {
                    "positions": [[3, 2]],  # red robot
                    "directions": [0],  # up
                    "types": ["vertical"]
                },
                "keys": {
                    "positions": [[5, 5], [5, 4]]  # orange, purple keys
                },
                "doors": {
                    "positions": [[0, 5], [5, 0]]  # orange, purple doors
                }
            },
            
            # bottleneck (medium difficulty) template
            "bottleneck_med": {
                "name": "Bottleneck",
                "description": "Horizontal wall with a single gap",
                "walls": [
                    [0, 2], [1, 2], [2, 2], [4, 2], [5, 2]
                ],
                "agent_pos": [0, 0],  # bunny starts in bottom-left
                "enemies": {
                    "positions": [[2, 1]],  # red robot
                    "directions": [1],  # up
                    "types": ["horizontal"]
                },
                "keys": {
                    "positions": [[2, 0], [5, 4]]  # orange, purple keys
                },
                "doors": {
                    "positions": [[3, 2], [0, 5]]  # orange, purple doors
                }
            },
            
            # bottleneck hard template
            "bottleneck_hard": {
                "name": "Bottleneck Hard",
                "description": "Horizontal wall with a single gap and vertical moving enemy",
                "walls": [
                    [0, 3], [1, 3], [2, 3], [4, 3], [5, 3]
                ],
                "agent_pos": [0, 0],  # bunny starts in bottom-left
                "enemies": {
                    "positions": [[2, 2]],  # red robot
                    "directions": [0],  # up
                    "types": ["vertical"]
                },
                "keys": {
                    "positions": [[2, 0], [5, 0]]  # orange, purple keys
                },
                "doors": {
                    "positions": [[3, 3], [0, 5]]  # orange, purple doors
                }
            },
            
            # corridors template
            "corridors_med": {
                "name": "Corridors",
                "description": "Maze-like pattern with vertical corridors",
                "walls": [
                    [1, 1], [1, 2], [1, 4], [3, 1], [3, 2], [3, 4]
                ],
                "agent_pos": [0, 0],  # bunny starts in bottom-left
                "enemies": {
                    "positions": [[3, 3]],  # red robot
                    "directions": [1],  # up
                    "types": ["horizontal"]
                },
                "keys": {
                    "positions": [[4, 1], [1, 3]]  # orange, purple keys
                },
                "doors": {
                    "positions": [[5, 5], [0, 5]]  # orange, purple doors
                }
            }
        }
        
        return templates

    # reset env to initial state (using selected template)
    def reset(self, seed=None):
        if seed is not None:
            super().reset(seed=seed)
        
        # reset stats
        self.steps_taken = 0
        self.total_reward = 0
        
        # reset key stack
        self.key_stack = []
        
        # reset wrong key message
        self.show_wrong_key = False
        
        # get the template for the current template
        template = self.templates[self.template_name]
        
        # load walls from template
        self.walls = np.array(template["walls"])
        
        # set agent position
        self.agent_pos = np.array(template["agent_pos"])
        
        # set enemy positions and directions
        self.enemies = {
            "positions": np.array(template["enemies"]["positions"]),
            "directions": np.array(template["enemies"]["directions"]),
            "types": template["enemies"]["types"]
        }
        
        # set key positions
        self.keys = {
            "positions": np.array(template["keys"]["positions"]),
            "collected": np.zeros(2, dtype=np.int32)
        }
        
        # set door positions
        self.doors = {
            "positions": np.array(template["doors"]["positions"]),
            "open": np.zeros(2, dtype=np.int32)
        }
        
        return self._get_obs(), {}

    # take step in env w/ given action
    def step(self, action):
        # update step counter
        self.steps_taken += 1
        
        # default small step penalty to encourage efficiency
        reward = -0.01
        done = False
        info = {}
        
        # reset wrong key message
        self.show_wrong_key = False
        
        # reset game over message
        self.show_game_over = False
        
        # FIRST calculate where robots WILL be after movement
        next_enemy_positions = []
        current_enemy_positions = []
        next_enemy_directions = self.enemies['directions'].copy()
        
        for i, enemy_type in enumerate(self.enemies['types']):
            enemy_pos = self.enemies['positions'][i].copy()
            current_enemy_positions.append(enemy_pos.copy())  # store current position
            direction = self.enemies['directions'][i]
            
            # calculate intended movement
            move = {
                0: np.array([0, 1]),   # up
                1: np.array([1, 0]),   # right
                2: np.array([0, -1]),  # down
                3: np.array([-1, 0]),  # left
            }[direction]
            
            # check if robot can move to new position
            new_pos = enemy_pos + move
            if (0 <= new_pos[0] < self.size and 
                0 <= new_pos[1] < self.size and
                not self._is_wall(new_pos)):
                # robot can move:
                next_enemy_positions.append(new_pos)
            else:
                # robot hits wall, changes direction but STAYS IN PLACE
                if enemy_type == 'horizontal':
                    next_enemy_directions[i] = 1 if direction == 3 else 3
                else:  # vertical
                    next_enemy_directions[i] = 0 if direction == 2 else 2
                # robot remains in current position
                next_enemy_positions.append(enemy_pos)
        
        # agent movement
        moved = False
        if action < 4:  # Movement action (not stay)
            move = {
                0: np.array([0, 1]),   # up
                1: np.array([1, 0]),   # right
                2: np.array([0, -1]),  # down
                3: np.array([-1, 0]),  # left
            }[action]
            new_pos = self.agent_pos + move
            
            # check boundaries and walls
            valid_position = (0 <= new_pos[0] < self.size and 
                            0 <= new_pos[1] < self.size and
                            not self._is_wall(new_pos))
            
            if valid_position:
                # check for all collision types:
                # 1. check if moving onto a robot's NEXT position
                future_robot_collision = any(np.array_equal(new_pos, enemy_pos) for enemy_pos in next_enemy_positions)
                
                # 2. check for "head-on" collisions (trading places with robot)
                head_on_collision = False
                head_on_enemy_idx = -1
                for i, (curr_pos, next_pos) in enumerate(zip(current_enemy_positions, next_enemy_positions)):
                    # If robot is moving from curr_pos to next_pos, and agent is moving from current to new_pos
                    # Check if they're crossing paths (agent to robot's current, robot to agent's current)
                    if np.array_equal(new_pos, curr_pos) and np.array_equal(next_pos, self.agent_pos):
                        head_on_collision = True
                        head_on_enemy_idx = i
                        break
                
                # collision detected
                if future_robot_collision or head_on_collision:
                    # for future collisions, allow agent to move (to show overlap)
                    if future_robot_collision and not head_on_collision:
                        self.agent_pos = new_pos  # Move to show collision
                    elif head_on_collision:
                        # for head-on collisions, don't move either
                        pass
                    
                    reward = -5.0
                    done = True
                    self.show_game_over = True  # show "Game Over!" message
                    info['terminated_reason'] = 'enemy_collision'
                # moving onto a door
                elif any(np.array_equal(new_pos, door_pos) for door_pos in self.doors['positions']):
                    door_idx = self._get_door_at_position(new_pos)
                    if door_idx is not None:
                        if not self.doors['open'][door_idx]:
                            # door is closed, check if we have the right key
                            if len(self.key_stack) > 0 and self.key_stack[-1] == door_idx:
                                # correct key => open door
                                self.doors['open'][door_idx] = 1
                                self.key_stack.pop()  # use the key
                                reward += 3.0
                                info['opened_door'] = door_idx
                                self.agent_pos = new_pos
                                moved = True
                            else:
                                # wrong key for door
                                self.show_wrong_key = True
                                self.wrong_key_time = time.time()
                                reward -= 1.0
                                info['wrong_key_attempt'] = True
                        else:
                            # door is already open, move through
                            self.agent_pos = new_pos
                            moved = True
                # normal move
                else:
                    self.agent_pos = new_pos
                    moved = True
        
        # if agent successfully moved, check for key collection
        if moved:
            key_idx = self._get_key_at_position(self.agent_pos)
            if key_idx is not None and not self.keys['collected'][key_idx]:
                self.keys['collected'][key_idx] = 1
                self.key_stack.append(key_idx)
                reward += 2.0
                info['collected_key'] = key_idx
        
        # update enemy positions and directions (only if no future collision)
        if not done or (done and self.show_game_over and any(np.array_equal(self.agent_pos, enemy_pos) for enemy_pos in next_enemy_positions)):
            self.enemies['positions'] = np.array(next_enemy_positions)
            self.enemies['directions'] = next_enemy_directions
                
            # check if any robot moved onto the agent's position
            if not done and self._check_enemy_collision():
                reward = -5.0
                done = True
                self.show_game_over = True  # show "Game Over!" message
                info['terminated_reason'] = 'enemy_collision'
        
        # check win condition (all doors open)
        if np.all(self.doors['open']):
            reward += 10.0
            done = True
            info['terminated_reason'] = 'success'
            info['success'] = True
        
        # check timeout
        if self.steps_taken >= 100:
            done = True
            info['terminated_reason'] = 'timeout'
        
        # add distance-based reward shaping if enabled
        if self.use_reward_shaping:
            reward += self._calculate_distance_reward() * 0.25  # CAN INCREASE to shape rewards more
        
        # update total reward
        self.total_reward += reward
        info['total_reward'] = self.total_reward
        info['steps'] = self.steps_taken
        
        if done and self.verbose:
            if 'success' in info and info['success']:
                print(f"Episode successful after {self.steps_taken} steps with total reward {self.total_reward:.2f}")
            else:
                print(f"Episode failed after {self.steps_taken} steps with total reward {self.total_reward:.2f}")
        
        return self._get_obs(), reward, done, False, info

    # check if position contains a wall
    def _is_wall(self, pos):
        return any(np.array_equal(pos, wall) for wall in self.walls)

    # check if there's a key at the given position and return index
    def _get_key_at_position(self, pos):
        for i, key_pos in enumerate(self.keys['positions']):
            if np.array_equal(pos, key_pos) and not self.keys['collected'][i]:
                return i
        return None

    # check if door at given position and return index
    def _get_door_at_position(self, pos):
        for i, door_pos in enumerate(self.doors['positions']):
            if np.array_equal(pos, door_pos):
                return i
        return None

    # check if agent has collided with an enemy
    def _check_enemy_collision(self):
        return any(np.array_equal(self.agent_pos, enemy_pos) 
                  for enemy_pos in self.enemies['positions'])

    # calc reward based on distance to objectives
    def _calculate_distance_reward(self):
        shaping_reward = 0.0
        
        # find closest uncollected key
        min_key_dist = float('inf')
        for i, key_pos in enumerate(self.keys['positions']):
            if not self.keys['collected'][i]:
                dist = self._manhattan_distance(self.agent_pos, key_pos)
                min_key_dist = min(min_key_dist, dist)
        
        # find the door that matches the top key in stack (if any)
        if len(self.key_stack) > 0:
            top_key = self.key_stack[-1]
            if not self.doors['open'][top_key]:
                door_dist = self._manhattan_distance(self.agent_pos, self.doors['positions'][top_key])
                shaping_reward += 0.05 * (1.0 / (door_dist + 1))
        
        # if no matching door (or no keys), find any unopened door
        elif min_key_dist == float('inf'):
            min_door_dist = float('inf')
            for i, door_pos in enumerate(self.doors['positions']):
                if not self.doors['open'][i]:
                    dist = self._manhattan_distance(self.agent_pos, door_pos)
                    min_door_dist = min(min_door_dist, dist)
            
            if min_door_dist != float('inf'):
                shaping_reward += 0.02 * (1.0 / (min_door_dist + 1))
        
        # reward for being close to keys
        if min_key_dist != float('inf'):
            shaping_reward += 0.05 * (1.0 / (min_key_dist + 1))
        
        # enemy avoidance reward
        enemy_dist = float('inf')
        for enemy_pos in self.enemies['positions']:
            dist = self._manhattan_distance(self.agent_pos, enemy_pos)
            enemy_dist = min(enemy_dist, dist)
        
        shaping_reward += 0.02 * min(enemy_dist, 3)  # Cap at distance of 3
        
        return shaping_reward

    # manhattan distance betw/ 2 positions
    def _manhattan_distance(self, pos1, pos2):
        return np.sum(np.abs(pos1 - pos2))

    # get current observation
    def _get_obs(self):
        # Create a fixed-size representation of the key stack
        key_stack_obs = np.ones(2, dtype=np.int32) * -1  # -1 indicates no key
        for i, key_idx in enumerate(self.key_stack[-2:]):  # only include last 2 keys
            key_stack_obs[i] = key_idx
            
        # convert enemy types to numerical representation
        enemy_types_enum = np.array([
            1 if enemy_type == 'vertical' else 0 
            for enemy_type in self.enemies['types']
        ])
        
        # pad walls array to fixed size
        wall_array = np.ones((10, 2), dtype=np.int32) * -1  # -1 indicates no wall
        for i, wall in enumerate(self.walls):
            if i < 10:  #only include up to 10 walls
                wall_array[i] = wall
        
        return {
            'agent': self.agent_pos,
            'enemies': self.enemies['positions'],
            'enemy_directions': self.enemies['directions'],
            'enemy_types': enemy_types_enum,
            'keys': self.keys['positions'],
            'key_status': self.keys['collected'],
            'doors': self.doors['positions'],
            'door_status': self.doors['open'],
            'key_stack': key_stack_obs,  # LIFO stack
            'walls': wall_array  # wall positions
        }

    # render environment
    def render(self):
        if not self.render_enabled:
            return
            
        if self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size + self.panel_height))
            template_name = self.templates[self.template_name]["name"]
            pygame.display.set_caption(f"LIFO Corridors - {template_name} Template")
        
        # fill background
        self.window.fill((255, 255, 255))  # White background
        
        # draw gridlines
        for i in range(self.size + 1):
            pygame.draw.line(
                self.window,
                (200, 200, 200),
                (i * self.cell_size, 0),
                (i * self.cell_size, self.window_size)
            )
            pygame.draw.line(
                self.window,
                (200, 200, 200),
                (0, i * self.cell_size),
                (self.window_size, i * self.cell_size)
            )
        
        # draw walls
        for wall_pos in self.walls:
            pygame.draw.rect(
                self.window,
                (128, 128, 128),
                (wall_pos[0] * self.cell_size, 
                 (self.size - 1 - wall_pos[1]) * self.cell_size,
                 self.cell_size,
                 self.cell_size)
            )
        
        # draw doors
        for i, door_pos in enumerate(self.doors['positions']):
            if not self.doors['open'][i]:
                door_x = door_pos[0] * self.cell_size + 2
                door_y = (self.size - 1 - door_pos[1]) * self.cell_size + 2
                
                if hasattr(self, 'images') and 'door' in self.images and len(self.images) > 0:
                    # use image with tint
                    tinted_door = self.images['door'].copy()
                    tinted_door.fill(self.key_door_colors[i], special_flags=pygame.BLEND_RGBA_MULT)
                    self.window.blit(tinted_door, (door_x, door_y))
                else:
                    # fallback to rectangle (if icons not working/loading for whatever reason)
                    pygame.draw.rect(
                        self.window,
                        self.key_door_colors[i],
                        (door_x + self.cell_size//4, door_y + self.cell_size//4,
                         self.cell_size//2, self.cell_size//2)
                    )
        
        # draw keys
        for i, key_pos in enumerate(self.keys['positions']):
            if not self.keys['collected'][i]:
                key_x = key_pos[0] * self.cell_size + 2
                key_y = (self.size - 1 - key_pos[1]) * self.cell_size + 2
                
                if hasattr(self, 'images') and 'key' in self.images and len(self.images) > 0:
                    # use image with tint
                    tinted_key = self.images['key'].copy()
                    tinted_key.fill(self.key_door_colors[i], special_flags=pygame.BLEND_RGBA_MULT)
                    self.window.blit(tinted_key, (key_x, key_y))
                else:
                    # fallback to circle
                    pygame.draw.circle(
                        self.window,
                        self.key_door_colors[i],
                        (key_x + self.cell_size//2 - 2, key_y + self.cell_size//2 - 2),
                        self.cell_size // 4
                    )
        
        # draw enemies
        for i, (enemy_pos, enemy_type) in enumerate(zip(self.enemies['positions'], self.enemies['types'])):
            enemy_x = enemy_pos[0] * self.cell_size + 2
            enemy_y = (self.size - 1 - enemy_pos[1]) * self.cell_size + 2
            
            # get color based on movement type
            color = self.enemy_colors[enemy_type]
            
            if hasattr(self, 'images') and 'robot' in self.images and len(self.images) > 0:
                # use image with tint
                tinted_robot = self.images['robot'].copy()
                tinted_robot.fill(color, special_flags=pygame.BLEND_RGBA_MULT)
                self.window.blit(tinted_robot, (enemy_x, enemy_y))
            else:
                # fallback to circle
                pygame.draw.circle(
                    self.window,
                    color,
                    (enemy_x + self.cell_size//2 - 2, enemy_y + self.cell_size//2 - 2),
                    self.cell_size // 3
                )
        
        # draw agent (bunny)
        agent_x = self.agent_pos[0] * self.cell_size + 2
        agent_y = (self.size - 1 - self.agent_pos[1]) * self.cell_size + 2
        
        if hasattr(self, 'images') and 'bunny' in self.images and len(self.images) > 0:
            self.window.blit(self.images['bunny'], (agent_x, agent_y))
        else:
            # fallback to circle
            pygame.draw.circle(
                self.window,
                (0, 0, 0), # black for bunny
                (agent_x + self.cell_size//2 - 2, agent_y + self.cell_size//2 - 2),
                self.cell_size // 3
            )
        
        # draw keys in stack above bunny (only those still in possession)
        key_spacing = 12
        mini_key_size = 16
        
        if hasattr(self, 'images') and 'key' in self.images and len(self.images) > 0:
            key_img = pygame.transform.scale(self.images['key'], (mini_key_size, mini_key_size))
            
            # only show keys that are in the stack
            start_x = agent_x + (self.cell_size - len(self.key_stack) * key_spacing) // 2
            
            for key_idx in self.key_stack:
                mini_key = key_img.copy()
                mini_key.fill(self.key_door_colors[key_idx], special_flags=pygame.BLEND_RGBA_MULT)
                self.window.blit(mini_key, (start_x, agent_y - mini_key_size - 2))
                start_x += key_spacing
        
        # draw key inventory panel
        panel_rect = pygame.Rect(0, self.window_size, self.window_size, self.panel_height)
        pygame.draw.rect(self.window, (240, 240, 240), panel_rect)  # light gray background
        pygame.draw.line(self.window, (200, 200, 200), (0, self.window_size), (self.window_size, self.window_size), 2)
        
        # draw text "Key Stack (Must Use Top Key First):"
        font = pygame.font.SysFont('Arial', 16)
        text = font.render("Key Stack (Must Use Top Key First):", True, (0, 0, 0))
        self.window.blit(text, (10, self.window_size + 5))
        
        # draw keys in the stack (most recent first)
        key_size = 32  # Size for inventory keys
        stack_x = 10
        
        if hasattr(self, 'images') and 'key' in self.images and len(self.images) > 0:
            key_img = pygame.transform.scale(self.images['key'], (key_size, key_size))
            
            # draw keys from newest to oldest
            for idx, key_idx in enumerate(reversed(self.key_stack)):
                tinted_key = key_img.copy()
                tinted_key.fill(self.key_door_colors[key_idx], special_flags=pygame.BLEND_RGBA_MULT)
                key_pos = (stack_x, self.window_size + 25)
                self.window.blit(tinted_key, key_pos)
                
                # draw red outline around top (usable) key
                if idx == 0:  # This is the top key
                    pygame.draw.rect(self.window, (255, 0, 0), 
                                   (key_pos[0], key_pos[1], key_size, key_size), 2)
                
                stack_x += key_size + 10
        
        # draw template, key count, etc.
        template_name = self.templates[self.template_name]["name"]
        template_text = f"Template: {template_name}"
        key_text = f"Keys: {int(np.sum(self.keys['collected']))}/2"
        door_text = f"Doors: {int(np.sum(self.doors['open']))}/2"
        step_text = f"Steps: {self.steps_taken}"
        reward_text = f"Reward: {self.total_reward:.1f}"
        stack_text = f"Stack: {[i for i in self.key_stack]}" if self.key_stack else "Stack: []"
        shaping_text = f"Reward Shaping: {'ON' if self.use_reward_shaping else 'OFF'}"
        
        template_surf = font.render(template_text, True, (0, 0, 0))
        key_surf = font.render(key_text, True, (0, 0, 0))
        door_surf = font.render(door_text, True, (0, 0, 0))
        step_surf = font.render(step_text, True, (0, 0, 0))
        reward_surf = font.render(reward_text, True, (0, 0, 0))
        stack_surf = font.render(stack_text, True, (0, 0, 0))
        shaping_surf = font.render(shaping_text, True, (0, 0, 0))
        
        self.window.blit(template_surf, (10, 10))
        self.window.blit(key_surf, (10, 30))
        self.window.blit(door_surf, (10, 50))
        self.window.blit(step_surf, (10, 70))
        self.window.blit(reward_surf, (10, 90))
        self.window.blit(stack_surf, (200, self.window_size + 40))
        self.window.blit(shaping_surf, (200, self.window_size + 5))
        
        # show "WRONG KEY!" message if needed
        if self.show_wrong_key:
            current_time = time.time()
            if current_time - self.wrong_key_time < self.wrong_key_duration:
                font_large = pygame.font.SysFont('Arial', 48)
                text = font_large.render("WRONG KEY!", True, (255, 0, 0))
                text_rect = text.get_rect(center=(self.window_size // 2, self.window_size // 2))
                self.window.blit(text, text_rect)
            else:
                self.show_wrong_key = False

        # show "Game Over!" message if needed
        if self.show_game_over:
            font_large = pygame.font.SysFont('Arial', 72)
            text = font_large.render("Game Over!", True, (255, 0, 0))
            text_rect = text.get_rect(center=(self.window_size // 2, self.window_size // 2))
            self.window.blit(text, text_rect)
        
        pygame.display.flip()

    # close environment
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
