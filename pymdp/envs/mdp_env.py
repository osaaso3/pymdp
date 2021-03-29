import numpy as np


from pymdp.envs import Env
from pymdp import maths
from pymdp.utils import get_model_dimensions


class MDPEnv(Env):
    def __init__(self, A, B):
        self.A = A.copy()
        self.B = B.copy()
        self.num_obs, self.num_states, _, self.num_factors = get_model_dimensions(A, B)
        self.states = []
        self.obs = []

    def reset(self, states=None):
        self.states = []
        if states is None:
            for ns in self.num_states:
                state = np.random.randint(0, ns)
                self.states.append(state)
        else:
            self.states = states

        self.obs = []
        for g, _ in enumerate(self.num_obs):
            # TODO variable number of states
            obs = maths.sample(self.A[g][:, self.states[0], self.states[1]])
            self.obs.append(obs)

        return self.obs

    def step(self, action):
        for f, s in enumerate(self.states):
            self.states[f] = maths.sample(self.B[f][:, s, action[f]])

        self.obs = []
        for g, _ in enumerate(self.num_obs):
            # TODO variable number of states
            obs = maths.sample(self.A[g][:, self.states[0], self.states[1]])
            self.obs.append(obs)

        return self.obs

    def __str__(self):
        return "<{}>".format(type(self).__name__)
