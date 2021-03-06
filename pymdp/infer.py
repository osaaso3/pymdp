from enum import IntEnum
import numpy as np

from pymdp import utils
from pymdp.maths import get_joint_likelihood_seq
from pymdp.algos import mmp

class InferAlgoEnum(IntEnum):
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
