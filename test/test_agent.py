import unittest

from pymdp.agent import Agent
from pymdp.infer import InferType
from pymdp import utils


class TestAgent(unittest.TestCase):
    def test_fpi(self):
        num_obs = [2, 4]
        num_states = [2, 2]
        num_control = [2, 2]
        A = utils.rand_A_mat(num_obs, num_states)
        B = utils.rand_B_mat(num_states, num_control)

        C = utils.obj_array_rand([num_ob for num_ob in num_obs])

        agent = Agent(A=A, B=B, C=C, control_factors=[1], infer_algo=InferType.FPI)
        obs = [0, 2]
        agent.infer_states(obs)
        agent.infer_policies()
        action = agent.sample_action()

        self.assertEqual(len(action), len(num_control))

    def test_mmp_inference(self):
        num_obs = [2, 4]
        num_states = [2, 2]
        num_control = [2, 2]
        A = utils.rand_A_mat(num_obs, num_states)
        B = utils.rand_B_mat(num_states, num_control)

        C = utils.obj_array_rand([num_ob for num_ob in num_obs])

        agent = Agent(
            A=A, B=B, C=C, control_factors=[1], infer_algo=InferType.MMP, infer_len=1, policy_len=5
        )
        obs = [0, 2]
        beliefs = agent.infer_states(obs)
        agent.infer_policies()

if __name__ == "__main__":
    unittest.main()
