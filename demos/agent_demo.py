import numpy as np

from pymdp import utils, maths, infer
from pymdp.agent import Agent
from pymdp.envs import POMDPEnv


def construct_A(num_states, num_obs):
    A = utils.obj_array_zeros([[o] + num_states for o in num_obs])
    A[0][:, :, 0] = np.ones((num_obs[0], num_states[0])) / num_obs[0]
    A[0][:, :, 1] = np.ones((num_obs[0], num_states[0])) / num_obs[0]
    A[0][:, :, 2] = np.array([[0.8, 0.2], [0.0, 0.0], [0.2, 0.8]])

    A[1][2, :, 0] = np.ones(num_states[0])
    A[1][0:2, :, 1] = maths.softmax(np.eye(num_obs[1] - 1))
    A[1][2, :, 2] = np.ones(num_states[0])

    A[2][0, :, 0] = 1.0
    A[2][1, :, 1] = 1.0
    A[2][2, :, 2] = 1.0
    return A


def construct_B(num_states, num_factors, control_factors=[]):
    B = utils.obj_array(num_factors)
    for f, ns in enumerate(num_states):
        if f in control_factors:
            B[f] = np.eye(ns).reshape(ns, ns, 1)
            B[f] = np.tile(B[f], (1, 1, ns)).transpose(1, 2, 0)
        else:
            B[f] = utils.default_B_mat(ns)
    return B


def construct_C(num_obs):
    C = utils.obj_array_zeros([num_ob for num_ob in num_obs])
    C[1][0] = 1.0
    C[1][1] = -2.0
    return C


if __name__ == "__main__":
    num_steps = 5
    num_obs = [3, 3, 3]
    num_states = [2, 3]
    num_modalities = len(num_obs)
    num_factors = len(num_states)
    control_factors = [1]

    A = construct_A(num_states, num_obs)
    pA = A.copy()
    B = construct_B(num_states, num_factors, control_factors)
    C = construct_C(num_obs)
    agent = Agent(
        A=A, B=B, C=C, pA=pA, control_factors=[1], policy_len=2, infer_algo=infer.InferType.MMP
    )

    init_state = [2, 1]
    env = POMDPEnv(A, B)
    obs = env.reset()

    for t in range(num_steps):
        _, qs = agent.infer_states(obs)
        policy_belief = agent.infer_policies()

        action = agent.sample_action()
        #  agent.infer_A(obs)
        print(f"\nstep {t} obs {obs} action {action}")
    
        obs = env.step(action)
