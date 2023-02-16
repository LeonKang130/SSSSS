import math

import nlopt
import numexpr as ne
import numpy as np

import luisa
from luisa.mathtypes import *

SIGMA_A = np.array([0.01, 0.01, 0.01], dtype=np.float32)
SIGMA_S = np.array([2.0, 2.0, 2.0], dtype=np.float32)
G = np.array([0.0, 0.0, 0.0], dtype=np.float32)
ETA = np.array([1.3, 1.3, 1.3], dtype=np.float32)
WIDTH = 0.03


def calculate_diffuse_mean_free_path(sigma_a: np.ndarray, sigma_s: np.ndarray, g: np.ndarray,
                                     eta: np.ndarray) -> np.ndarray:
    sigma_s *= (1.0 - g)
    sigma_t = sigma_s + sigma_a
    alpha = sigma_s / sigma_t
    fresnel = ne.evaluate('-1.440 / (eta * eta) + 0.710 / eta + 0.668 + 0.0636 * eta')
    A = ne.evaluate('(1.0 + fresnel) / (1.0 - fresnel)')
    albedo = ne.evaluate(
        '0.5 * alpha * (1.0 + exp(-4.0 / 3.0 * A * sqrt(3.0 * (1.0 - alpha)))) * exp(-sqrt(3.0 * (1.0 - alpha)))')
    sigma_tr = ne.evaluate('sqrt(3.0 * (1.0 - alpha)) * sigma_t')
    return ne.evaluate('1.0 / (sigma_tr * (3.5 + 100 * (albedo - 0.33) ** 4))')


print(f"D: {calculate_diffuse_mean_free_path(SIGMA_A, SIGMA_S, G, ETA)}")


def sample_radius(d: np.float32, x: np.ndarray) -> np.ndarray:
    x = ne.evaluate('2 * x - 2')
    y = ne.evaluate('sqrt(x * x + 1)')
    x, y = np.power(y - x, 1 / 3, dtype=np.float32), np.power(y + x, 1 / 3, dtype=np.float32)
    return ne.evaluate('-3 * log(x - y) * d')


GRID_WIDTH = 11
NUM_SAMPLES = 100000000
luisa.init('cuda')
radii = luisa.Buffer.empty(NUM_SAMPLES, dtype=float)
thetas = luisa.Buffer.empty(NUM_SAMPLES, dtype=float)
distribution_buffer = luisa.Buffer.zeros(GRID_WIDTH * GRID_WIDTH, dtype=int)


@luisa.func
def roll_out() -> None:
    radius = radii.read(dispatch_id().x)
    theta = thetas.read(dispatch_id().x)
    x, y = int(radius * cos(theta)), int(radius * sin(theta))
    if x < GRID_WIDTH and y < GRID_WIDTH:
        _ = distribution_buffer.atomic_fetch_add(x * GRID_WIDTH + y, 1)


def f(x: np.ndarray, grad: np.ndarray):
    """
    :param x: array of parameters
    :param grad: the array where the gradient is to be stored
    :return: L2 loss
    """
    arr = np.expand_dims(x, 1)
    diff = arr * arr.T - distribution
    if grad.size > 0:
        grad[:] = 2 * np.matmul(diff + diff.T, x)
    return np.sum(diff ** 2)


if __name__ == "__main__":
    dmfp = calculate_diffuse_mean_free_path(SIGMA_A, SIGMA_S, G, ETA).astype('f4')
    scale = np.amax(dmfp)
    dmfp /= (scale * WIDTH)
    kernel = np.empty((GRID_WIDTH * 2 - 1, 3), dtype=np.float32)
    rng = np.random.default_rng()
    for i in range(3):
        u, v = rng.random(NUM_SAMPLES, dtype=np.float32), rng.random(NUM_SAMPLES, dtype=np.float32)
        radii.copy_from_array(sample_radius(dmfp[i], u))
        thetas.copy_from_array(np.float32(math.pi * 0.5) * v)
        luisa.Buffer.fill_kernel(distribution_buffer, 0, dispatch_size=GRID_WIDTH * GRID_WIDTH)
        roll_out(dispatch_size=NUM_SAMPLES)
        distribution = distribution_buffer.numpy().astype(np.float32).reshape(GRID_WIDTH, GRID_WIDTH)
        distribution /= NUM_SAMPLES
        opt = nlopt.opt(nlopt.LD_LBFGS, GRID_WIDTH)
        opt.set_min_objective(f)
        opt.set_xtol_abs(0.0001)
        opt.set_lower_bounds(np.zeros(GRID_WIDTH, dtype=np.float32))
        xopt = opt.optimize(np.ones(GRID_WIDTH, dtype=np.float32))
        kernel[:, i] = np.concatenate((xopt[::-1], xopt[1:]))
        kernel[:, i] /= kernel[:, i].sum()
    with open("include.glsl", "w") as f:
        f.write(f"#define KERNEL_WIDTH {2 * GRID_WIDTH - 1}\n")
        f.write(f"#define DMFP {scale}\n")
        f.write(f"#define WIDTH {WIDTH}\n")
        content = f"vec4 kernel[KERNEL_WIDTH] = vec4[](\n" + ",\n".join(
            f"    vec4({weight[0]}, {weight[1]}, {weight[2]}, {i - GRID_WIDTH + 1})" for i, weight in
            enumerate(kernel)) + "\n);"
        f.write(content)
