import os
import unittest

import numpy as np
from scipy.io import loadmat


from pymdp import utils
from pymdp.algos import mmp
from pymdp.maths import get_joint_likelihood_seq

from test.spm_utils import load_spm_data, convert_obs_arr

DATA_PATH = "test/spm_output/"


class TestMMP(unittest.TestCase):
    def test_mmp_a(self):
        """ Testing `algos.mmp` with 1 hidden state factor & 
            1 outcome modality, at a random fixed point during the generative process
        """

        spm_path = os.path.join(os.getcwd(), DATA_PATH + "mmp_a.mat")
        data = load_spm_data(spm_path)

        num_obs, num_states, _, num_factors = utils.get_model_dimensions(data.A, data.B)
        prev_obs = data.prev_obs[:, max(0, data.curr_t - data.horizon) : (data.curr_t + 1)]
        prev_obs = convert_obs_arr(prev_obs, num_obs)
        prev_actions = data.prev_actions[(max(0, data.curr_t - data.horizon) - 1) :, :]
        prior = utils.obj_array(num_factors)
        for f in range(num_factors):
            uniform = np.ones(num_states[f]) / num_states[f]
            prior[f] = data.B[f][:, :, prev_actions[0, f]].dot(uniform)

        lh_seq = get_joint_likelihood_seq(data.A, prev_obs, num_states)
        qs_seq, _ = mmp(
            lh_seq,
            data.B,
            data.policy,
            prev_actions[1:],
            prior=prior,
            num_iter=5,
            grad_descent=True,
        )

        result_pymdp = qs_seq[-1]
        for f in range(num_factors):
            self.assertTrue(np.isclose(data.result[f].squeeze(), result_pymdp[f]).all())

    def test_mmp_b(self):
        """ Testing `algos.mmp` with 2 hidden state factors & 
            2 outcome modalities, at a random fixed point during the generative process
        """

        spm_path = os.path.join(os.getcwd(), DATA_PATH + "mmp_b.mat")
        data = load_spm_data(spm_path)

        num_obs, num_states, _, num_factors = utils.get_model_dimensions(data.A, data.B)
        prev_obs = data.prev_obs[:, max(0, data.curr_t - data.horizon) : (data.curr_t + 1)]
        prev_obs = convert_obs_arr(prev_obs, num_obs)

        prev_actions = data.prev_actions[(max(0, data.curr_t - data.horizon)) :, :]
        lh_seq = get_joint_likelihood_seq(data.A, prev_obs, num_states)
        qs_seq, _ = mmp(
            lh_seq,
            data.B,
            data.policy,
            prev_actions=prev_actions,
            prior=None,
            num_iter=5,
            grad_descent=True,
        )

        result_pymdp = qs_seq[-1]
        for f in range(num_factors):
            self.assertTrue(np.isclose(data.result[f].squeeze(), result_pymdp[f]).all())


if __name__ == "__main__":
    unittest.main()
