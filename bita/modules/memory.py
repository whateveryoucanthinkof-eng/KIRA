import torch
from torch import nn

from collections import defaultdict
from copy import deepcopy


class Memory(nn.Module):

  def __init__(self, n_nodes, memory_dimension, input_dimension, message_dimension=None,
               device="cpu", combination_method='sum'):
    super(Memory, self).__init__()
    self.n_nodes = n_nodes
    self.memory_dimension = memory_dimension
    self.input_dimension = input_dimension
    self.message_dimension = message_dimension
    self.device = device

    self.combination_method = combination_method

    self.__init_memory__()

  def __init_memory__(self):
    """
    Initializes the memory to all zeros. It should be called at the start of each epoch.
    """
    # Treat memory as parameter so that it is saved and loaded together with the model
    self.memory = nn.Parameter(torch.zeros((self.n_nodes, self.memory_dimension)).to(self.device),
                               requires_grad=False)
    # float64: at Unix-epoch scale (~1.5e9 s) float32 resolves only 128 s, so
    # "time since last update" of anything under two minutes came out as 0.
    self.last_update = nn.Parameter(torch.zeros(self.n_nodes, dtype=torch.float64).to(self.device),
                                    requires_grad=False)

    self.messages = defaultdict(list)

  def ensure_capacity(self, n_nodes):
    """Grow the memory table to hold node ids < n_nodes. New rows start at zero.

    Serving and trajectory extraction assign node ids as hosts appear, so the
    table cannot be sized once at construction the way it is in batch training.
    """
    if n_nodes <= self.n_nodes:
      return
    extra = n_nodes - self.n_nodes
    dev = self.memory.device
    self.memory = nn.Parameter(
      torch.cat([self.memory.data, torch.zeros(extra, self.memory_dimension, device=dev)]),
      requires_grad=False)
    self.last_update = nn.Parameter(
      torch.cat([self.last_update.data, torch.zeros(extra, dtype=self.last_update.dtype, device=dev)]),
      requires_grad=False)
    self.n_nodes = n_nodes

  def store_raw_messages(self, nodes, node_id_to_messages):
    for node in nodes:
      self.messages[node].extend(node_id_to_messages[node])

  def get_memory(self, node_idxs):
    return self.memory[node_idxs, :]

  def set_memory(self, node_idxs, values):
    self.memory[node_idxs, :] = values

  def get_last_update(self, node_idxs):
    return self.last_update[node_idxs]

  def backup_memory(self):
    # Messages are (raw, t) or (raw, t, peer); keep any trailing fields.
    messages_clone = {}
    for k, v in self.messages.items():
      messages_clone[k] = [(x[0].clone(), x[1].clone()) + tuple(x[2:]) for x in v]

    return self.memory.data.clone(), self.last_update.data.clone(), messages_clone

  def restore_memory(self, memory_backup):
    self.memory.data, self.last_update.data = memory_backup[0].clone(), memory_backup[1].clone()

    self.messages = defaultdict(list)
    for k, v in memory_backup[2].items():
      self.messages[k] = [(x[0].clone(), x[1].clone()) + tuple(x[2:]) for x in v]

  def detach_memory(self):
    self.memory.detach_()

    # Detach all stored messages
    for k, v in self.messages.items():
      new_node_messages = []
      for message in v:
        new_node_messages.append((message[0].detach(), message[1]) + tuple(message[2:]))

      self.messages[k] = new_node_messages

  def clear_messages(self, nodes):
    for node in nodes:
      self.messages[node] = []
