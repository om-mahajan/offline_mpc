from collections import deque
import numpy as np
import random
import torch

from wrappers.atari_wrapper import LazyFrames
from dataset.expert_dataset import ExpertDataset


class Memory(object):
    def __init__(self, memory_size: int, seed: int = 0) -> None:
        random.seed(seed)
        self.memory_size = memory_size
        self.buffer = deque(maxlen=self.memory_size)
        self.shift = 0
        self.scale = 1

    def add(self, experience) -> None:
        self.buffer.append(experience)

    def size(self):
        return len(self.buffer)

    def sample(self, batch_size: int, continuous: bool = True):
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        if continuous:
            rand = random.randint(0, len(self.buffer) - batch_size)
            return [self.buffer[i] for i in range(rand, rand + batch_size)]
        else:
            indexes = np.random.randint(low=0, high=len(self.buffer), size=batch_size)
            return [self.buffer[i] for i in indexes]

    def clear(self):
        self.buffer.clear()

    def save(self, path):
        b = np.asarray(self.buffer)
        print(b.shape)
        np.save(path, b)

    def load(self, path, num_trajs, sample_freq, seed):
        # If path has no extension add npy
        if not path.endswith("pkl"):
            path += '.npy'
        data = ExpertDataset(path, num_trajs, sample_freq, seed)
        # data = np.load(path, allow_pickle=True)
        for i in range(len(data)):
            self.add(data[i])
            
    def load_from_data(self, data):
        self.memory_size = data['states'].shape[0]
        self.buffer = deque(maxlen=self.memory_size)
        
        for i in range(self.memory_size):
            self.add((
                data['states'][i],
                data['next_states'][i],
                data['actions'][i],
                data['rewards'][i],
                data['costs'][i],
                data['dones'][i],
                data['is_constrained'][i],
            ))

    def get_samples(self, batch_size, device,noise=0.0):
        batch = self.sample(batch_size, False)

        batch_state, batch_next_state, batch_action, batch_reward, batch_cost, batch_done,is_constrained = zip(
            *batch)

        batch_state = np.array(batch_state)
        batch_next_state = np.array(batch_next_state)
        batch_action = np.array(batch_action)
        
        batch_state = (batch_state + self.shift) * self.scale
        batch_next_state = (batch_next_state + self.shift) * self.scale

        batch_state = torch.as_tensor(batch_state, dtype=torch.float, device=device)
        batch_next_state = torch.as_tensor(batch_next_state, dtype=torch.float, device=device)
        batch_action = torch.as_tensor(batch_action, dtype=torch.float, device=device).clip(-1, 1)
        if batch_action.ndim == 1:
            batch_action = batch_action.unsqueeze(1)
        batch_reward = torch.as_tensor(batch_reward, dtype=torch.float, device=device).unsqueeze(1)
        batch_cost = torch.as_tensor(batch_cost, dtype=torch.float, device=device).unsqueeze(1)
        batch_done = torch.as_tensor(batch_done, dtype=torch.float, device=device).unsqueeze(1)
        batch_is_constrained = torch.as_tensor(is_constrained, dtype=torch.bool, device=device).unsqueeze(1)

        return batch_state, batch_next_state, batch_action, batch_reward,batch_cost, batch_done, batch_is_constrained