import numpy as np

from pymdp import utils

EPS = 1e-16


def spm_dot(X, x, dims_to_omit=None):
    if utils.is_obj_array(x):
        dims = (np.arange(0, len(x)) + X.ndim - len(x)).astype(int)
    else:
        dims = np.array([1], dtype=int)
        x = utils.to_obj_array(x)

    if dims_to_omit is not None:
        dims = np.delete(dims, dims_to_omit)
        x = np.empty([0], dtype=object) if len(x) == 1 else np.delete(x, dims_to_omit)

    for d in range(len(x)):
        s = np.ones(np.ndim(X), dtype=int)
        s[dims[d]] = np.shape(x[d])[0]
        X = X * x[d].reshape(tuple(s))

    Y = np.sum(X, axis=tuple(dims.astype(int))).squeeze()
    if np.prod(Y.shape) <= 1.0:
        Y = np.array([Y.item()]).astype("float64")
    return Y


def spm_cross(x, y=None, *args):
    if len(args) == 0 and y is None:
        if utils.is_obj_array(x):
            z = spm_cross(*list(x))
        elif np.issubdtype(x.dtype, np.number):
            z = x
        else:
            raise ValueError(f"Invalid input to spm_cross ({x})")
        return z

    if utils.is_obj_array(x):
        x = spm_cross(*list(x))

    if y is not None and utils.is_obj_array(y):
        y = spm_cross(*list(y))

    reshape_dims = tuple(list(x.shape) + list(np.ones(y.ndim, dtype=int)))
    A = x.reshape(reshape_dims)

    reshape_dims = tuple(list(np.ones(x.ndim, dtype=int)) + list(y.shape))
    B = y.reshape(reshape_dims)
    z = np.squeeze(A * B)

    for x in args:
        z = spm_cross(z, x)
    return z


def spm_state_info_gain(qs, A):
    _, _, num_modalities, _ = utils.get_model_dimensions(A=A)
    qs = spm_cross(qs)
    indices = list(np.array(np.where(qs > EPS)).T)

    qo, surprise = 0, 0
    for idx in indices:
        prob_obs = np.ones(1)
        for modality in range(num_modalities):
            index_vector = [slice(0, A[modality].shape[0])] + list(idx)
            prob_obs = spm_cross(prob_obs, A[modality][tuple(index_vector)])

        prob_obs = prob_obs.ravel()
        qo = qo + qs[tuple(idx)] * prob_obs
        surprise = surprise + qs[tuple(idx)] * spm_dot(prob_obs, np.log(prob_obs + EPS))

    surprise = surprise - spm_dot(qo, spm_log(qo))
    return surprise


def spm_norm(mat):
    mat = mat + EPS
    normed_mat = np.divide(mat, mat.sum(axis=0))
    return normed_mat


def spm_log(arr):
    return np.log(arr + EPS)


def sample(probs):
    if utils.is_obj_array(probs):
        samples = []
        for _, p in enumerate(probs):
            samples.append(sample(p))
        return samples
    if len(probs.shape) > 1:
        probs = probs.squeeze()
    sample_onehot = np.random.multinomial(1, probs)
    return np.where(sample_onehot == 1)[0][0]


def softmax(dist):
    output = dist - dist.max(axis=0)
    output = np.exp(output)
    output = output / np.sum(output, axis=0)
    return output


def norm_dist(dist):
    if utils.is_obj_array(dist):
        for i, d in enumerate(dist):
            dist[i] = norm_dist(d)
        return dist
    if len(dist.shape) == 3:
        new_dist = np.zeros_like(dist)
        for c in range(dist.shape[2]):
            new_dist[:, :, c] = np.divide(dist[:, :, c], dist[:, :, c].sum(axis=0))
        return new_dist
    else:
        return np.divide(dist, dist.sum(axis=0))


def get_joint_likelihood_seq(A, obs, num_states):
    ll_seq = utils.obj_array(len(obs))
    for t in range(len(obs)):
        ll_seq[t] = get_joint_likelihood(A, obs[t], num_states)
    return ll_seq


def get_joint_likelihood(A, obs, num_states):
    num_states = [num_states] if (type(num_states) is int) else num_states
    A = utils.to_obj_array(A)
    obs = utils.to_obj_array(obs)
    ll = np.ones(tuple(num_states))
    for modality in range(len(A)):
        ll = ll * dot_likelihood(A[modality], obs[modality])
    return ll


def dot_likelihood(A, obs):
    s = np.ones(np.ndim(A), dtype=int)
    s[0] = obs.shape[0]
    X = A * obs.reshape(tuple(s))
    X = np.sum(X, axis=0, keepdims=True)
    ll = np.squeeze(X)

    if np.prod(ll.shape) <= 1.0:
        ll = ll.item()
        ll = np.array([ll]).astype("float64")

    return ll
