import copy

from pymdp.utils import (
    obj_array,
    to_obj_array,
    rand_A_mat,
    rand_B_mat,
    rand_D_mat,
    get_model_dimensions,
)
from pymdp.infer import infer_states_mmp, InferAlgoEnum
from pymdp import control


class Agent:
    def __init__(
        self,
        A=None,
        B=None,
        C=None,
        D=None,
        policies=None,
        num_states=None,
        num_obs=None,
        num_control=None,
        control_factors=None,
        infer_algo=InferAlgoEnum.MMP,
        infer_len=1,
        policy_len=1,
    ):

        self.A = rand_A_mat(num_obs, num_states) if A is None else to_obj_array(A)
        self.B = rand_B_mat(num_states, num_control) if B is None else to_obj_array(B)

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
        self.obs_seq = []
        self.action_seq = []
        self.belief_state = None

    def reset(self):
        self.obs_seq = []
        self.set_prior(self.D)

    def infer_states(self, obs):
        if self.infer_algo == InferAlgoEnum.MMP:
            self.obs_seq.append(obs)
            if len(self.obs_seq) > self.infer_len:
                self.obs_seq = self.obs_seq[: self.infer_len]

            res = infer_states_mmp(self.A, self.B, self.obs_seq, self.policies, prior=self.prior)
            self.state_beliefs, _ = res

        return self.state_beliefs

    def infer_policies(self):
        self.policy_beliefs, _ = control.update_policies(
            self.state_beliefs[0][0], self.A, self.B, self.C, self.policies
        )
        return self.policy_beliefs

    def sample_action(self):
        return control.sample_action(self.policy_beliefs, self.policies, self.num_control)

    def set_prior(self, prior=None):
        if prior is None:
            prior = obj_array(len(self.policies))
            for p_idx, _ in enumerate(self.policies):
                prior[p_idx] = copy.deepcopy(self.state_beliefs[p_idx][0])
        self.prior = prior
        return self.prior
