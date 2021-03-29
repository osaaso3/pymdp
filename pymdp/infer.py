from enum import IntEnum

import numpy as np

from pymdp import utils
from pymdp.maths import get_joint_likelihood_seq
from pymdp.algos import mmp, fpi


class InferType(IntEnum):
    FPI = 0
    MMP = 1


def infer_states_mmp(A, B, prev_obs, policies, prev_actions=None, prior=None):
    num_obs, num_states, num_modalities, _ = utils.get_model_dimensions(A, B)
    A = utils.to_obj_array(A)
    B = utils.to_obj_array(B)

    prev_obs = utils.process_obs_seq(prev_obs, num_modalities, num_obs)
    if prior is not None:
        prior = utils.to_obj_array(prior)

    ll_seq = get_joint_likelihood_seq(A, prev_obs, num_states)

    pi_qs_seq = utils.obj_array(len(policies))
    pi_vfe = np.zeros(len(policies))

    for p_idx, policy in enumerate(policies):
        pi_qs_seq[p_idx], pi_vfe[p_idx] = mmp(ll_seq, B, policy, prev_actions, prior)

    return pi_qs_seq, pi_vfe


def infer_states(A, B, obs, prior=None):
    num_obs, _, num_modalities, _ = utils.get_model_dimensions(A, B)
    A = utils.to_obj_array(A)
    B = utils.to_obj_array(B)

    obs = utils.process_obs(obs, num_modalities, num_obs)
    if prior is not None:
        prior = utils.to_obj_array(prior)

    qs = fpi(A, B, obs, prior)
    return qs


def average_over_policies(qs_pi, q_pi):
    qs_pi = utils.to_obj_array(qs_pi)
    q_pi = utils.to_obj_array(q_pi)

    num_factors = len(qs_pi[0])
    num_states = [qs_f.shape[0] for qs_f in qs_pi[0]]

    qs_bma = utils.obj_array_zeros(num_states)
    for p_idx, prob_pi in enumerate(q_pi[0]):
        for f in range(num_factors):
            ans = qs_pi[p_idx][f] * prob_pi
            qs_bma[f] = qs_bma[f] + ans
        return qs_bma
