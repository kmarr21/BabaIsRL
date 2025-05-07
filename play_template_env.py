import pygame
import time
import numpy as np
import argparse
import random
from template_lifo_corridors import TemplateLIFOCorridorsEnv

# play template LIFO corridors env. using keyboard controls
def play_template_environment(template_name="basic_med"):
    # create environment with the chosen template
    env = TemplateLIFOCorridorsEnv(template_name=template_name, render_enabled=True, verbose=True)
    
    # reset to get initial state
    state, _ = env.reset()
    
    # initialize pygame for input handling
    pygame.init()
    
    # define color mapping for keys and doors (correctly map the colors!)
    key_door_colors = {
        0: "Orange", 
        1: "Purple"
    }
    
    # print instructions
    template_info = env.templates[template_name]
    template_display_name = template_info["name"]
    print("\n=== Template LIFO Corridors Environment ===")
    print(f"Template: {template_display_name}")
    print(f"Description: {template_info['description']}")
    print("\nKeyboard Controls:")
    print("Arrow Up    - Move Up")
    print("Arrow Right - Move Right")
    print("Arrow Down  - Move Down")
    print("Arrow Left  - Move Left")
    print("Space       - Stay in place")
    print("R           - Reset environment")
    print("Q or ESC    - Quit\n")
    print("\nLIFO RULE: Only the most recently collected key can be used!")
    print("Keys and doors are color-coded (orange, purple).")
    
    # print robot information based on the template
    robot_type = env.enemies["types"][0]
    if robot_type == "horizontal":
        print("Blue robot moves horizontally.")
    else:
        print("Red robot moves vertically.")
    
    # game state variables
    done = False
    total_reward = 0
    step_count = 0
    keys_collected = 0
    doors_opened = 0
    
    # MAIN game loop
    running = True
    while running:
        # render environment
        env.render()
        
        # process input
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action = 0  # Move up
                elif event.key == pygame.K_RIGHT:
                    action = 1  # Move right
                elif event.key == pygame.K_DOWN:
                    action = 2  # Move down
                elif event.key == pygame.K_LEFT:
                    action = 3  # Move left
                elif event.key == pygame.K_SPACE:
                    action = 4  # Stay in place
                elif event.key == pygame.K_r:
                    # reset environment
                    state, _ = env.reset()
                    total_reward = 0
                    step_count = 0
                    keys_collected = 0
                    doors_opened = 0
                    done = False
                    print("\nEnvironment reset!")
                elif event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
        
        # if a valid action was chosen, take step
        if action is not None and not done:
            next_state, reward, done, _, info = env.step(action)
            
            # update state and stats
            state = next_state
            total_reward += reward
            step_count += 1
            
            # track key and door status
            current_keys = np.sum(state['key_status'])
            current_doors = np.sum(state['door_status'])
            
            # display info when key collected or door opened
            if 'collected_key' in info:
                collected_key = info['collected_key']
                color_name = key_door_colors[collected_key]
                print(f"Collected {color_name} key! +2.0 reward")
                
            if 'opened_door' in info:
                opened_door = info['opened_door']
                color_name = key_door_colors[opened_door]
                print(f"Opened {color_name} door! +3.0 reward")
                
            if 'wrong_key_attempt' in info:
                print(f"Wrong key! -1.0 reward")
            
            # print step information / keeping key_stack info on separate line
            action_names = ["Up", "Right", "Down", "Left", "Stay"]
            key_stack_info = f"Key stack: {env.key_stack}" if hasattr(env, 'key_stack') else ""
            
            print(f"Step {step_count}: Action={action_names[action]}, Reward={reward:.2f}, Total={total_reward:.2f}")
            print(f"  Keys: {int(current_keys)}/2, Doors: {int(current_doors)}/2 {key_stack_info}")
            
            # check if episode is done
            if done:
                if np.all(state['door_status']):
                    print("\nCongratulations! You won by opening all doors!")
                    print(f"Bonus reward: +10.0")
                elif 'terminated_reason' in info and info['terminated_reason'] == 'timeout':
                    print("\nTime's up! Maximum steps reached.")
                else:
                    print("\nGame over! You collided with an enemy.")
                    print(f"Penalty: -5.0")
                    
                print(f"Total steps: {step_count}, Total reward: {total_reward:.2f}")
        
        # small delay to limit frame rate
        time.sleep(0.05)
    
    # close env
    env.close()
    pygame.quit()
    print("Game closed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Play Template LIFO Corridors environment.')
    parser.add_argument('--template', type=str, default="basic_med", 
                       choices=["basic_med", "sparse_med", "zipper_med", "bottleneck_med", "bottleneck_hard", "corridors_med"],
                       help='Template name to use')
    parser.add_argument('--seed', type=int, default=None, help='Random seed')
    
    args = parser.parse_args()
    
    # set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    play_template_environment(template_name=args.template)
