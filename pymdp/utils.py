import numpy as np

from pymdp import maths


def obj_array(num_arr):
    return np.empty(num_arr, dtype=object)


def obj_array_zeros(shape_list):
    arr = obj_array(len(shape_list))
    for i, shape in enumerate(shape_list):
        arr[i] = np.zeros(shape)
    return arr


def obj_array_uniform(shape_list):
    arr = obj_array(len(shape_list))
    for i, shape in enumerate(shape_list):
        arr[i] = np.ones(shape) / shape
    return arr


def obj_array_rand(shape_list):
    arr = obj_array(len(shape_list))
    for i, shape in enumerate(shape_list):
        arr[i] = np.random.rand(shape)
    return arr


def to_obj_array(arr):
    if is_obj_array(arr):
        return arr
    obj_arr = np.empty(1, dtype=object)
    obj_arr[0] = arr.squeeze()
    return obj_arr


def is_obj_array(arr):
    return arr.dtype == "object"


def onehot(value, num_values):
    arr = np.zeros(num_values)
    arr[value] = 1.0
    return arr


def print_obj_array(obj_array, name="", level=0):
    print(f"[{name}] (level {level}): shape {obj_array.shape}")
    for obj in obj_array:
        if is_obj_array(obj):
            print_obj_array(obj, name=name, level=level + 1)
        else:
            for i, el in enumerate(obj_array):
                obj_array[i] = np.round(el.astype(float), 3)
            print(f"[{name}] (level {level}): values {obj_array}")


def rand_A_mat(num_obs, num_states):
    if type(num_obs) is int:
        num_obs = [num_obs]
    if type(num_states) is int:
        num_states = [num_states]
    num_modalities = len(num_obs)

    A = obj_array(num_modalities)
    for modality, modality_obs in enumerate(num_obs):
        modality_shape = [modality_obs] + num_states
        modality_dist = np.random.rand(*modality_shape)
        A[modality] = maths.norm_dist(modality_dist)
    return A


def rand_B_mat(num_states, num_controls):
    if type(num_states) is int:
        num_states = [num_states]
    if type(num_controls) is int:
        num_controls = [num_controls]
    num_factors = len(num_states)
    assert len(num_controls) == len(num_states)

    B = obj_array(num_factors)
    for factor in range(num_factors):
        factor_shape = (num_states[factor], num_states[factor], num_controls[factor])
        factor_dist = np.random.rand(*factor_shape)
        B[factor] = maths.norm_dist(factor_dist)
    return B


def rand_D_mat(num_states):
    dist = obj_array(len(num_states))
    for i, ns in enumerate(num_states):
        dist[i] = maths.norm_dist(np.ones(ns))
    return dist


def default_B_mat(num_states):
    mat = np.eye(num_states)
    return mat.reshape(num_states, num_states, 1)


def get_model_dimensions(A=None, B=None):
    if A is not None:
        num_obs = [a.shape[0] for a in A] if is_obj_array(A) else [A.shape[0]]
        num_modalities = len(num_obs)
    else:
        num_obs, num_modalities = [0], 0

    if B is not None:
        num_states = [b.shape[0] for b in B] if is_obj_array(B) else [B.shape[0]]
        num_factors = len(num_states)
    else:
        num_states, num_factors = [0], 0

    return num_obs, num_states, num_modalities, num_factors


def process_obs_seq(obs_seq, num_modalities, num_obs):
    """ Helper function for formatting observation sequences """
    proc_obs_seq = obj_array(len(obs_seq))
    for t in range(len(obs_seq)):
        proc_obs_seq[t] = process_obs(obs_seq[t], num_modalities, num_obs)
    return proc_obs_seq


def process_obs(obs, num_modalities, num_obs):
    """ Helper function for formatting observations """
    if isinstance(obs, (int, np.integer)):
        obs = onehot(obs, num_obs[0])

    elif isinstance(obs, tuple) or isinstance(obs, list):
        obs_arr = obj_array(num_modalities)
        for m in range(num_modalities):
            obs_arr[m] = onehot(obs[m], num_obs[m])
        obs = obs_arr

    return obs

