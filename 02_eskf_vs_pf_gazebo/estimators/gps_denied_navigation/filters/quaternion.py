"""Scalar-first unit quaternions, active body-to-ENU; vectorized last axis."""
import numpy as np


def normalize(q):
    q = np.asarray(q, dtype=float)
    norms = np.linalg.norm(q, axis=-1, keepdims=True)
    if np.any(norms < 1e-15) or not np.all(np.isfinite(norms)):
        raise FloatingPointError("Invalid quaternion norm")
    return q/norms


def multiply(a, b):
    a, b = np.broadcast_arrays(np.asarray(a), np.asarray(b))
    scalar = a[..., :1]*b[..., :1] - np.sum(a[..., 1:]*b[..., 1:], axis=-1, keepdims=True)
    vector = a[..., :1]*b[..., 1:] + b[..., :1]*a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate((scalar, vector), axis=-1)


def exp_rotation(rotvec):
    v = np.asarray(rotvec, dtype=float)
    angle = np.linalg.norm(v, axis=-1, keepdims=True)
    # np.sinc(x) = sin(pi x)/(pi x), including its continuous limit at zero.
    return np.concatenate((np.cos(angle/2), .5*np.sinc(angle/(2*np.pi))*v), axis=-1)


def rotate(q, vector):
    q = np.asarray(q)
    vector = np.asarray(vector)
    uv = np.cross(q[..., 1:], vector)
    return vector + 2*(q[..., :1]*uv + np.cross(q[..., 1:], uv))


def from_rpy(rpy):
    r, p, y = np.asarray(rpy)/2
    cr, sr, cp, sp, cy, sy = np.cos(r), np.sin(r), np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    return normalize([cr*cp*cy+sr*sp*sy, sr*cp*cy-cr*sp*sy,
                      cr*sp*cy+sr*cp*sy, cr*cp*sy-sr*sp*cy])


def to_rpy(q):
    q = normalize(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    return np.stack((np.arctan2(2*(w*x+y*z), 1-2*(x*x+y*y)),
                     np.arcsin(np.clip(2*(w*y-z*x), -1, 1)),
                     np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))), axis=-1)


def matrix(q):
    return np.column_stack([rotate(q, axis) for axis in np.eye(3)])


def skew(v):
    x, y, z = v
    return np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])


def right_jacobian(rotvec):
    """SO(3) right Jacobian: Exp(a+da) ≈ Exp(a) Exp(Jr(a) da)."""
    angle = np.linalg.norm(rotvec)
    A = skew(rotvec)
    first = .5*np.sinc(angle/(2*np.pi))**2
    second = 1/6-angle**2/120+angle**4/5040 if angle < 1e-4 else (angle-np.sin(angle))/angle**3
    return np.eye(3)-first*A+second*(A @ A)


def weighted_mean(q, weights):
    # Markley mean: sign-invariant dominant eigenvector of sum w_i q_i q_i.T.
    _, vectors = np.linalg.eigh((q.T*weights) @ q)
    result = vectors[:, -1]
    return result if result[0] >= 0 else -result


def conjugate(q):
    q = np.asarray(q, dtype=float)
    return np.concatenate((q[..., :1], -q[..., 1:]), axis=-1)


def log_rotation(q):
    """Rotation vector of unit quaternion(s); inverse of exp_rotation for angles < 2*pi.

    q and -q give the same rotation; the w >= 0 representative is used.
    """
    q = np.asarray(q, dtype=float)
    q = np.where(q[..., :1] < 0, -q, q)
    vector_norm = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    angle = 2*np.arctan2(vector_norm, q[..., :1])
    # angle/sin(angle/2) -> 2 as angle -> 0; use the exact ratio away from zero.
    with np.errstate(invalid="ignore", divide="ignore"):
        scale = np.where(vector_norm > 1e-12, angle/np.where(vector_norm > 1e-12, vector_norm, 1.), 2.)
    return scale*q[..., 1:]
