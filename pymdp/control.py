import itertools
import numpy as np

from pymdp import utils
from pymdp.maths import spm_dot, spm_state_info_gain, softmax, norm_dist, sample


def update_policies(qs, A, B, C, policies, use_utility=True, use_state_info_gain=False, gamma=16.0):
    num_policies = len(policies)
    efe = np.zeros(num_policies)

    for idx, policy in enumerate(policies):
        qs_pi = get_expected_states(qs, B, policy)
        qo_pi = get_expected_obs(qs_pi, A)

        if use_utility:
            efe[idx] = efe[idx] + get_utility(qo_pi, C)

        if use_state_info_gain:
            efe[idx] = efe[idx] + get_state_info_gain(qs_pi, A)

    q_pi = softmax(efe * gamma)
    q_pi = q_pi / q_pi.sum(axis=0)

    return q_pi, efe


def update_policies_mmp(
    pi_qs_seq, A, B, C, policies, use_utility=True, use_state_info_gain=False, gamma=16.0
):
    num_policies = len(policies)
    horizon = len(pi_qs_seq[0])
    efe = np.zeros(num_policies)

    for idx, _ in enumerate(policies):
        qs_seq = pi_qs_seq[idx]

        for t in range(horizon):
            print(qs_seq[t].shape)
            qo_pi = get_expected_obs(qs_seq[t], A)

            if use_utility:
                efe[idx] = efe[idx] + get_utility(qo_pi, C[t])

            if use_state_info_gain:
                efe[idx] = efe[idx] + get_state_info_gain(qs_seq[t], A)

    q_pi = softmax(efe * gamma)
    q_pi = q_pi / q_pi.sum(axis=0)

    return q_pi, efe


def get_expected_states(qs, B, policy):
    num_steps = policy.shape[0]
    num_factors = policy.shape[1]

    qs_pi = [utils.obj_array(num_factors) for _ in range(num_steps)]
    for t in range(num_steps):
        for control_factor, control in enumerate(policy[t, :]):
            qs_pi_t = qs[control_factor] if (t == 0) else qs_pi[t - 1][control_factor]
            qs_pi[t][control_factor] = spm_dot(B[control_factor][:, :, control], qs_pi_t)

    return qs_pi


def get_expected_obs(qs_pi, A):
    num_steps = len(qs_pi)
    num_modalities = len(A)

    qo_pi = [utils.obj_array(num_modalities) for _ in range(num_steps)]
    for t in range(num_steps):
        for modality in range(num_modalities):
            qo_pi[t][modality] = spm_dot(A[modality], qs_pi[t])

    return qo_pi


def get_utility(qo_pi, C):
    num_steps = len(qo_pi)
    num_modalities = len(C)

    utility = 0
    for t in range(num_steps):
        for modality in range(num_modalities):
            lnC = np.log(softmax(C[modality][:, np.newaxis]) + 1e-16)
            utility = utility + spm_dot(qo_pi[t][modality], lnC)

    return utility


def get_state_info_gain(qs_pi, A):
    num_steps = len(qs_pi)
    info_gain = 0
    for t in range(num_steps):
        info_gain = info_gain + spm_state_info_gain(A, qs_pi[t])

    return info_gain


def sample_action(q_pi, policies, num_control):
    num_factors = len(num_control)
    marginals = np.empty(num_factors, dtype=object)
    for ctrl_idx in range(num_factors):
        marginals[ctrl_idx] = np.zeros(num_control[ctrl_idx])

    for p, policy in enumerate(policies):
        for t in range(policy.shape[0]):
            for f, a in enumerate(policy[t, :]):
                marginals[f][a] = marginals[f][a] + q_pi[p]

        marginals = norm_dist(marginals)
        policy = sample(marginals)
        return policy


def construct_policies(num_states, num_control=None, policy_len=1, control_factors=None):
    num_factors = len(num_states)
    if control_factors is None:
        control_factors = list(range(num_factors))

    if num_control is None:
        num_control = []
        for c_idx in range(num_factors):
            num_control.append(num_states[c_idx] if c_idx in control_factors else 1)
        num_control = list(np.array(num_control).astype(int))

    policies = list(itertools.product(*[list(range(i)) for i in num_control * policy_len]))
    for pol_i in range(len(policies)):
        policies[pol_i] = np.array(policies[pol_i]).reshape(policy_len, num_factors)

    return policies, num_control
