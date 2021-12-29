import os, json
import taichi as ti
import time
ti.init(arch=ti.opengl, print_ir=False,log_level=ti.INFO, allow_nv_shader_extension=False, use_gles=True, ndarray_use_torch=False)
dim = 2
N = 64
n_particles = N * N * 2
n_grid = 128
p_rho = 1
bound = 3
E = 400
@ti.kernel
def substep(x: ti.any_arr(element_dim=1), v: ti.any_arr(element_dim=1), J: ti.any_arr(),
            C: ti.any_arr(element_dim=2), grid_v: ti.any_arr(element_dim=1), grid_m: ti.any_arr()):
    pass
    # dx = 1 / grid_v.shape[0]
    # inv_dx = grid_v.shape[0]
    # p_vol = (dx * 0.5)**2
    # p_mass = p_vol * p_rho
    # dt = min(2.0e-4 / (grid_v.shape[0] / 128), 2.0e-4)
    #for i, j in grid_m:
    #    grid_v[i, j] = [0, 0]
    #    grid_m[i, j] = 0
    #for p in x:
    #    base = (x[p] * inv_dx - 0.5).cast(int)
    #    fx = x[p] * inv_dx - base.cast(float)
    #    w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
    #    stress = -dt * p_vol * (J[p] - 1) * 4 * inv_dx * inv_dx * E
    #    affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
    #    for i, j in ti.static(ti.ndrange(3, 3)):
    #            offset = ti.Vector([i, j])
    #            dpos = (offset - fx) * dx
    #            weight = w[i][0] * w[j][1]
    #            grid_v[base + offset].atomic_add(
    #                    weight * (p_mass * v[p] + affine @ dpos))
    #            grid_m[base + offset].atomic_add(weight * p_mass)


@ti.kernel
def init(x: ti.any_arr(element_dim=1), v: ti.any_arr(element_dim=1), J: ti.any_arr()):
    for i in x:
        x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
        v[i] = [0, -1]
        J[i] = 1
x = ti.Vector.ndarray(dim, ti.f32, n_particles)
v = ti.Vector.ndarray(dim, ti.f32, n_particles)
J = ti.ndarray(ti.f32, n_particles)
C = ti.Matrix.ndarray(dim, dim, ti.f32, n_particles)
grid_v = ti.Vector.ndarray(dim, ti.f32, (n_grid, n_grid))
grid_m = ti.ndarray(ti.f32, (n_grid, n_grid))

@ti.kernel
def test(x: ti.any_arr(element_dim=1), v: ti.any_arr(element_dim=1),):
    pass
def run_steps(n=500):
    for s in range(n):
        substep(x, v, J, C, grid_v, grid_m)


def run():
    init(x, v, J)
    gui = ti.GUI()
    while gui.running:
        run_steps()
        gui.clear(0x112F41)
        gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
        gui.show()
def aot():
    m = ti.aot.Module(ti.opengl)
    m.add_kernel(init, (x, v, J))
    m.add_kernel(substep, (x, v, J, C, grid_v, grid_m))
    dir_name = 'mpm88_ndarray_no_nv_extension'
    m.save(dir_name, '')
    with open(os.path.join(dir_name, 'metadata.json')) as json_file:
        json.load(json_file)

def profile():
    init(x, v, J)
    gui = ti.GUI('MPM88')
    while gui.running and not gui.get_event(gui.ESCAPE):
        start = time.time()
        for _ in range(200):
            run_steps()
            x.to_numpy()
        print(time.time() - start)

# run()
# aot()
profile()
