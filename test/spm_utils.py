import os
from collections import namedtuple
from scipy.io import loadmat

from pymdp import utils

Data = namedtuple(
    "Data", ["A", "B", "prev_obs", "policy", "curr_t", "horizon", "prev_actions", "result"]
)


def load_spm_data(path):
    array_path = os.path.join(path)
    mat = loadmat(file_name=array_path)

    A = mat["A"][0]
    B = mat["B"][0]
    prev_obs = mat["obs_idx"].astype("int64")
    policy = mat["policy"].astype("int64") - 1
    curr_t = mat["t"][0, 0].astype("int64") - 1
    horizon = mat["t_horizon"][0, 0].astype("int64")
    prev_actions = mat["previous_actions"].astype("int64") - 1
    result = mat["qs"][0]

    return Data(A, B, prev_obs, policy, curr_t, horizon, prev_actions, result)



def convert_obs_arr(obs, num_obs):
    """
    Converts SPM observation to pymdp one-hot object arrays
    """

    obs_len = obs.shape[1]
    num_modalities = len(num_obs)

    proc_obs = []
    if num_modalities == 1:
        for t in range(obs_len):
            proc_obs.append(utils.onehot(obs[0, t] - 1, num_obs[0]))
    else:
        for t in range(obs_len):
            obs_obj_arr = utils.obj_array(num_modalities)
            for g in range(num_modalities):
                obs_obj_arr[g] = utils.onehot(obs[g, t] - 1, num_obs[g])
            proc_obs.append(obs_obj_arr)

    return proc_obs

