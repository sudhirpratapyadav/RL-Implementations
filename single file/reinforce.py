import os
import gym
import numpy as np


import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt 

import argparse

class PolicyCategorical():
  def __init__(self, num_inputs, num_outputs, device):
    self.device = device
    self.network = nn.Sequential(
      nn.Linear(num_inputs, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, num_outputs),
    ).to(self.device)
  
  def get_action(self, x, deterministic=False):
    logits = self.network(torch.from_numpy(x).to(device=self.device))

    action_log_prob = None

    # Call get_action with determinstic=True during test time
    if deterministic:
      action_int = int(torch.argmax(logits).item())
    else:
      pi = Categorical(logits=logits)
      action = pi.sample() 
      action_log_prob = pi.log_prob(action)
      action_int = int(action.item())

    return action_int, action_log_prob

class ReinforcePolicyGradient:

  def __init__(self,
         env,
         policy,
         device = None,
         render=False,
         num_epochs=50,
         max_sampling_steps=5000,
         lr=1e-2,
         ):
     
    self.env = env
    self.policy = policy
    self.optimizer = optim.AdamW(self.policy.network.parameters(), lr=lr, amsgrad=True)

    if device is None:
      self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
      self.device = device
    print('Device:', self.device)

    self.render = render
    self.num_epochs = num_epochs
    self.max_sampling_steps = max_sampling_steps

    self.avg_returns = []

    # Printing all Parameters
    print("-------------------------")
    print(self.__dict__)
    print("\n\n---NN Parameters---")
    for name, param in self.policy.network.named_parameters():
      if param.requires_grad:
        print (name, param.shape)
    print()
    print("-------------------------")
    
  
  def collect_trajectories(self):
    log_probs = []
    total_return = []

    ep_return_lst = []
    ep_len_lst = []

    total_steps = 10
    obs = self.env.reset()
    ep_rews = []

    finished_rendering_this_epoch = False

    while True:
      if (not finished_rendering_this_epoch) and self.render:
        self.env.render()

      a, a_log_prob = self.policy.get_action(obs)
      obs, r, done, info = self.env.step(a)

      ep_rews.append(r)
      log_probs.append(a_log_prob)

      total_steps +=1
      if done:
        ep_return, ep_len = sum(ep_rews), len(ep_rews)
        ep_return_lst.append(ep_return)
        ep_len_lst.append(ep_len)

        total_return.extend([ep_return] * ep_len)

        obs = self.env.reset()
        ep_rews = []
        done = False

        finished_rendering_this_epoch = True

        if total_steps>self.max_sampling_steps:
          break

    return log_probs, total_return, ep_return_lst, ep_len_lst

  @staticmethod
  def compute_policy_loss(log_probs, weights):
    return -(log_probs * weights).mean()
    
  @staticmethod
  def update_policy(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  def train(self):
    print(f"Starting training for {self.num_epochs} epochs\n")

    for epoch_num in range(1, self.num_epochs+1):
      
      log_probs, total_return, ep_return_lst, ep_len_lst = self.collect_trajectories()
      
      log_prob_tensor = torch.stack(log_probs).to(device=self.device)
      '''
      log_prob_tensor = torch.as_tensor(log_probs, dtype=torch.float32, device=device)
      note: above expression can not be used, we have to use either torch.stack or torch.cat otherwise, new tensor
        will be created and it will be disconnected from the neural-net graph and thus won't have require_grad
        as true and auto-grad won't be able to compute its gradient when calling loss.backword()
      '''
      weight_tensor = torch.as_tensor(total_return, dtype=torch.float32, device=self.device)

      loss = self.compute_policy_loss(log_prob_tensor, weight_tensor)
      self.update_policy(self.optimizer, loss)

      avg_return = np.array(ep_return_lst).mean()
      avg_ep_len = np.array(ep_len_lst).mean()
      self.avg_returns.append([epoch_num, avg_return])

      print(f"Epoch: [{epoch_num}/{self.num_epochs}]\t Loss: {loss:.2f}\t ep_reward: {avg_return:.2f}\t ep_len: {avg_ep_len:.2f}")
    
    return np.array(self.avg_returns)

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--env_name', '--env', type=str, default='CartPole-v0')
  parser.add_argument('--render', action='store_true')
  parser.add_argument('--num_epochs', type=int, default=100)
  parser.add_argument('--max_sampling_steps', type=int, default=5000)
  parser.add_argument('--lr', type=float, default=1e-2)
  args = parser.parse_args()

  env = gym.make(args.env_name)
  obs_dim = env.observation_space.shape[0] # observation_space is continuous in gym env, this will get number of dimension
  action_dim = env.action_space.n # action_space is discrete in gym env, this will get number of dimension

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  policy = PolicyCategorical(num_inputs=obs_dim, num_outputs=action_dim, device=device)
  agent = ReinforcePolicyGradient(env=env,
                                  policy=policy,
                                  device=device,
                                  render=args.render,
                                  num_epochs=args.num_epochs,
                                  max_sampling_steps=args.max_sampling_steps,
                                  lr=args.lr)
  # Train the agent
  result = agent.train()

  # Plotting the returns
  epoch_num = result[:, 0]
  rewards = result[:, 1]
  plt.plot(epoch_num, rewards)
  plt.xlabel('Epoch')
  plt.ylabel('Reward')
  plt.title('Reward vs. Epoch')
  plt.show()
    


if __name__=="__main__":
  main()
