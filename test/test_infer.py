import unittest

import numpy as np

from pymdp import utils, maths
from pymdp.infer import infer_states_mmp


def rand_onehot_obs(num_obs):
    if type(num_obs) is int:
        num_obs = [num_obs]
    obs = utils.obj_array(len(num_obs))
    for i in range(len(num_obs)):
        ob = np.random.randint(num_obs[i])
        obs[i] = utils.onehot(ob, num_obs[i])
    return obs


def rand_controls(num_controls):
    if type(num_controls) is int:
        num_controls = [num_controls]
    controls = np.zeros(len(num_controls))
    for i in range(len(num_controls)):
        controls[i] = np.random.randint(num_controls[i])
    return controls


def rand_state_dist(num_states):
    if type(num_states) is int:
        num_states = [num_states]
    states = utils.obj_array(len(num_states))
    for i in range(len(num_states)):
        states[i] = maths.norm_dist(np.random.rand(num_states[i]))
    return states


class TestInference(unittest.TestCase):
    def test_update_syayes(self):

        past_len, future_len, num_policies = 3, 4, 5
        num_states = [6, 7, 8]
        num_controls = [9, 10, 11]
        num_obs = [12, 13, 14]
        num_modalities = len(num_obs)

        A = utils.rand_A_mat(num_obs, num_states)
        B = utils.rand_B_mat(num_states, num_controls)
        prev_obs = [rand_onehot_obs(num_obs) for _ in range(past_len)]
        prev_actions = np.array([rand_controls(num_controls) for _ in range(past_len)])
        policies = [
            np.array([rand_controls(num_controls) for _ in range(future_len)])
            for _ in range(num_policies)
        ]
        prior = rand_state_dist(num_states)

        pi_qs_seq, vfe = infer_states_mmp(A, B, prev_obs, policies, prev_actions, prior=prior)

if __name__ == "__main__":
    unittest.main()

