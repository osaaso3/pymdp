import copy
from collections import namedtuple

from pymdp.utils import (
    obj_array,
    to_obj_array,
    rand_A_mat,
    rand_B_mat,
    rand_D_mat,
    get_model_dimensions,
)
from pymdp.maths import norm_dist
from pymdp.learning import update_A_dist, update_B_dist
from pymdp.infer import average_over_policies, infer_states_mmp, infer_states, InferType
from pymdp import control


class Agent:
    def __init__(
        self,
        A=None,
        B=None,
        C=None,
        D=None,
        pA=None,
        pB=None,
        policies=None,
        num_states=None,
        num_obs=None,
        num_control=None,
        control_factors=None,
        infer_algo=InferType.MMP,
        infer_len=1,
        policy_len=1,
        lr=0.01,
    ):

        self.A = rand_A_mat(num_obs, num_states) if A is None else to_obj_array(A)
        self.B = rand_B_mat(num_states, num_control) if B is None else to_obj_array(B)
        self.pA = None if pA is None else to_obj_array(pA)
        self.pB = None if pB is None else to_obj_array(pB)

        dims = get_model_dimensions(self.A, self.B)
        self.num_obs, self.num_states, self.num_modalities, self.num_factors = dims

        self.C = None if C is None else to_obj_array(C)
        self.D = rand_D_mat(self.num_states) if D is None else to_obj_array(D)
        self.set_prior(self.D)

        self.control_factors = (
            list(range(self.num_factors)) if control_factors is None else control_factors
        )

        if policies is None:
            self.policies, num_control = control.construct_policies(
                self.num_states, policy_len=policy_len, control_factors=self.control_factors
            )
        else:
            self.policies = policies
        self.num_control = num_control

        self.infer_algo = infer_algo
        self.infer_len = infer_len
        self.lr = lr

        self.qs = None
        self.qs_avg = None
        self.q_pi = None
        self.obs_seq = []
        self.action_seq = []

    def reset(self):
        self.qs = None
        self.obs_seq = []
        self.set_prior(self.D)

    def infer_states(self, obs):
        if self.infer_algo == InferType.MMP:
            self.obs_seq.append(obs)
            if len(self.obs_seq) > self.infer_len:
                self.obs_seq = self.obs_seq[: self.infer_len]

            res = infer_states_mmp(self.A, self.B, self.obs_seq, self.policies, prior=self.prior)
            self.qs, _ = res

            if self.q_pi is not None:
                self.qs_avg = average_over_policies(self.qs, self.q_pi)

        elif self.infer_algo == InferType.FPI:
            self.qs = infer_states(self.A, self.B, obs, prior=self.prior)

        return self.qs, self.qs_avg

    def infer_policies(self):
        if self.infer_algo == InferType.MMP:
            self.q_pi, _ = control.update_policies_mmp(
                self.qs, self.A, self.B, self.C, self.policies
            )
        elif self.infer_algo == InferType.FPI:
            self.q_pi, _ = control.update_policies(self.qs, self.A, self.B, self.C, self.policies)
        return self.q_pi

    def infer_A(self, obs):
        self.pA = update_A_dist(self.pA, self.A, obs, self.qs, self.lr)
        self.A = norm_dist(self.pA)
        return self.pA

    def infer_B(self, prev_qs):
        self.pB = update_B_dist(self.pB, self.B, self.qs, prev_qs, self.lr)
        self.B = norm_dist(self.pA)
        return self.pB

    def sample_action(self):
        return control.sample_action(self.q_pi, self.policies, self.num_control)

    def set_prior(self, prior=None):
        if prior is None:
            prior = obj_array(len(self.policies))
            for p_idx, _ in enumerate(self.policies):
                prior[p_idx] = copy.deepcopy(self.qs[p_idx][0])
        self.prior = prior
        return self.prior
