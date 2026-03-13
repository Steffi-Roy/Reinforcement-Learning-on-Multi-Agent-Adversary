##Implement Replay Buffer and DQN network Arch

"""Transition is a namedtuple used to store a transition.

The structure of Transition looks like this:
    (state, action, reward, next_state, done)
"""
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:

    def __init__(self, capacity=100_000):

        self._storage = deque([], maxlen=capacity)

    def add(self, state, action, reward, next_state, done):

        transition = Transition(state, action, reward, next_state, done)
        self._storage.append(transition)

    def sample(self, batch_size):

        transitions = random.sample(self._storage, batch_size)
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*transitions)
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        """Return the length of the buffer."""
        return len(self._storage)
