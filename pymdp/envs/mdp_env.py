import numpy as np


from pymdp.envs import Env
from pymdp.maths import sample, spm_dot
from pymdp.utils import get_model_dimensions, onehot


class MDPEnv(Env):
    def __init__(self, A, B):
        self.A = A.copy()
        self.B = B.copy()
        self.num_obs, self.num_states, _, self.num_factors = get_model_dimensions(A, B)
        self.states = []
        self.obs = []

    def reset(self, states=None):
    
        self.states = utils.obj_array(self.num_factors)
        if states is None:
            for f, ns in enumerate(self.num_states):
                state_idx = np.random.randint(0, ns)
                self.states[f] = onehot(state_idx,ns)
        else:
            self.states = states

        self.obs = []
        for g, _ in enumerate(self.num_obs):
            obs = sample(spm_dot(self.A[g], self.states))
            self.obs.append(obs)

        return self.obs

    def step(self, action):
        for f, ns in enumerate(self.num_states):
            ps = self.B[f][:,:,action[f]].dot(self.states[f])
            self.states[f] = onehot(sample(ps), ns)

        self.obs = []
        for g, _ in enumerate(self.num_obs):
            obs = sample(spm_dot(self.A[g], self.states))
            self.obs.append(obs)

        return self.obs

    def __str__(self):
        return "<{}>".format(type(self).__name__)
