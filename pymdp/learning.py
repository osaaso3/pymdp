import copy
import numpy as np
from pymdp import utils, maths


def update_A_dist(pA, A, obs, qs, lr=1.0, modalities=None):
    pA = utils.to_obj_array(pA)
    A = utils.to_obj_array(A)

    num_obs, _, num_modalities, _ = utils.get_model_dimensions(A=A)
    if modalities is None:
        modalities = range(num_modalities)

    updated_pA = copy.deepcopy(pA)

    if isinstance(obs, (int, np.integer)):
        obs = np.eye(A.shape[0])[obs]

    elif isinstance(obs, tuple):
        obs = np.array(
            [np.eye(num_obs[modality])[obs[modality]] for modality in range(num_obs)],
            dtype=object,
        )

    for modality in modalities:
        # TODO
        dfda = maths.spm_cross(obs[modality], qs)
        dfda = dfda * (A[modality] > 0).astype("float")
        updated_pA[modality] = updated_pA[modality] + (lr * dfda)

    return updated_pA


def update_B_dist(pB, B, actions, qs, qs_prev, lr=1.0, factors=None):
    pB = utils.to_obj_array(pB)
    B = utils.to_obj_array(B)

    _, _, _, num_factors = utils.get_model_dimensions(B=B)
    if factors is None:
        factors = range(num_factors)

    updated_pB = copy.deepcopy(pB)

    for factor, _ in enumerate(num_factors):
        # TODO
        dfdb = maths.spm_cross(qs[factor], qs_prev[factor])
        dfdb = dfdb * (B[factor][:, :, actions[factor]] > 0).astype("float")
        dfdb = updated_pB[factor][:, :, actions[factor]] + (lr * dfdb)
        updated_pB[factor][:, :, actions[factor]] = dfdb

    return updated_pB