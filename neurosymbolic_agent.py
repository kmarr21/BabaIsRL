import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque

from dqn_agent_enhanced import DQNAgentEnhanced
from neural_symbolic_tree import NeuralGuidedDecisionTree

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# neurosymbolic DQN agent that extends enhanced DQN agent (or base DQN agent) with a neural-guided DT for symbolic reasoning
class NeurosymbolicDQNAgent:
    def __init__(self, state_size, action_size, seed=0, 
                 learning_rate=0.0003, gamma=0.99, tau=0.0005,
                 buffer_size=100000, batch_size=128, update_every=8,
                 use_augmented_state=True, ksm_mode="off",
                 symbolic_guidance_weight=0.65,
                 use_base_dqn=False,
                 gradual_guidance_decrease=False,
                 min_guidance_weight=0.3,
                 guidance_decay=0.9999):
                     
        # for base DQN, turn off augmented state and KSM
        if use_base_dqn:
            use_augmented_state = False
            ksm_mode = "off"
        
        # initialize the DQN agent
        self.dqn_agent = DQNAgentEnhanced(
            state_size=state_size, 
            action_size=action_size, 
            seed=seed,
            learning_rate=learning_rate, 
            gamma=gamma, 
            tau=tau,
            buffer_size=buffer_size, 
            batch_size=batch_size, 
            update_every=update_every,
            use_augmented_state=use_augmented_state, 
            ksm_mode=ksm_mode
        )
        
        # store action and state dimensions
        self.state_size = state_size
        self.action_size = action_size
        
        # create the NDT
        self.symbolic_tree = NeuralGuidedDecisionTree(self.dqn_agent.policy_net, device)
        
        # guidance weight parameters
        self.initial_guidance_weight = symbolic_guidance_weight
        self.symbolic_guidance_weight = symbolic_guidance_weight
        self.gradual_guidance_decrease = gradual_guidance_decrease
        self.min_guidance_weight = min_guidance_weight
        self.guidance_decay = guidance_decay
        
        # track symbolic reasoning for analysis
        self.decision_history = []
        
        # for state preprocessing compatibility
        self.use_augmented_state = use_augmented_state

    # preprocess state using underlying DQN agent's method
    def preprocess_state(self, state):
        return self.dqn_agent.preprocess_state(state)

    # select action using hybrid of symbolic reasoning & epsilon-greedy policy
    def act(self, state, eps=0.0):
        # preprocess state
        state_vector = self.dqn_agent.preprocess_state(state)
        state_tensor = torch.FloatTensor(state_vector).to(device)
        
        # list of possible actions
        possible_actions = list(range(self.action_size))
        
        # get symbolic guidance
        guidance = self.symbolic_tree.guide_action_selection(state, state_tensor, possible_actions)
        
        # store decision for analysis
        self.decision_history.append(guidance)
        
        # check if exploring randomly
        if random.random() < eps:
            if guidance['mode'] == 'guided' and random.random() < 0.3:
                # 30% of exploration uses guided randomness based on the symbolic mask
                if guidance['action_mask'] is not None:
                    # sample from action mask probabilities
                    probs = guidance['action_mask'] / guidance['action_mask'].sum()
                    return np.random.choice(possible_actions, p=probs)
            # 70% of exploration is pure random
            return random.choice(possible_actions)
        
        #if not exploring, use hybrid decision making
        with torch.no_grad():
            self.dqn_agent.policy_net.eval()
            action_values = self.dqn_agent.policy_net(state_tensor).cpu().data.numpy()
            self.dqn_agent.policy_net.train()
            
            if guidance['mode'] == 'guided' and guidance['action_mask'] is not None:
                # blend neural and symbolic preferences
                # normalize neural values to [0,1] range for blending
                neural_prefs = action_values - action_values.min()
                if neural_prefs.max() > 0:
                    neural_prefs = neural_prefs / neural_prefs.max()
                
                # normalize symbolic mask
                symbolic_prefs = guidance['action_mask']
                symbolic_prefs = symbolic_prefs / symbolic_prefs.max()
                
                # weighted blend = using the symbolic guidance weight
                w = self.symbolic_guidance_weight
                blended_prefs = (1 - w) * neural_prefs + w * symbolic_prefs
                
                # select action with highest blended preference
                return np.argmax(blended_prefs)
            else:
                # pure neural decision
                return np.argmax(action_values)

    # process a step and learn if appropriate
    def step(self, state, action, reward, next_state, done, info=None):
        # store symbolic decision
        symbolic_decision = self.decision_history[-1] if self.decision_history else None
        
        # call the DQN agent's step method
        self.dqn_agent.step(state, action, reward, next_state, done, info)
        
        # reset symbolic decision history if episode ended
        if done:
            self.decision_history = []
            
            # decrease guidance weight if using gradual decrease
            if self.gradual_guidance_decrease:
                self.symbolic_guidance_weight = max(
                    self.min_guidance_weight,
                    self.symbolic_guidance_weight * self.guidance_decay
                )

    # get stats about symbolic decision making
    def get_symbolic_stats(self):
        if not hasattr(self, 'symbolic_stats'):
            self.symbolic_stats = {
                'neural_decisions': 0,
                'guided_decisions': 0,
                'key0_preference': 0,
                'key1_preference': 0,
                'door0_preference': 0,
                'door1_preference': 0
            }
            
        return self.symbolic_stats

    # set template context for DQN agent
    def set_template_context(self, template_name):
        if hasattr(self.dqn_agent, 'set_template_context'):
            self.dqn_agent.set_template_context(template_name)

    # save the model
    def save(self, filename):
        self.dqn_agent.save(filename)
        
        # add neurosymbolic-specific parameters
        checkpoint = torch.load(filename)
        checkpoint['symbolic_guidance_weight'] = self.symbolic_guidance_weight
        checkpoint['initial_guidance_weight'] = self.initial_guidance_weight
        checkpoint['gradual_guidance_decrease'] = self.gradual_guidance_decrease
        torch.save(checkpoint, filename)

    # load the model
    def load(self, filename):
        self.dqn_agent.load(filename)
        
        # load neurosymbolic-specific parameters (if available)
        checkpoint = torch.load(filename, map_location=device)
        if 'symbolic_guidance_weight' in checkpoint:
            self.symbolic_guidance_weight = checkpoint['symbolic_guidance_weight']
        if 'initial_guidance_weight' in checkpoint:
            self.initial_guidance_weight = checkpoint['initial_guidance_weight']
        if 'gradual_guidance_decrease' in checkpoint:
            self.gradual_guidance_decrease = checkpoint['gradual_guidance_decrease']
