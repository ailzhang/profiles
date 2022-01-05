# MPM-MLS in 88 lines of Taichi code, originally created by @yuanming-hu
import taichi as ti
import json
import os
import time

ti.init(arch=ti.opengl, log_level=ti.TRACE, allow_nv_shader_extension=False, use_gles=True, ndarray_use_torch=False)

#ti.set_logging_level(ti.TRACE)

N = 64
n_particles = N * N * 2
n_grid = 128
dx = 1 / n_grid
dt = 2e-4

p_rho = 1
p_vol = (dx * 0.5)**2
p_mass = p_vol * p_rho
gravity = 9.8
bound = 3
E = 400

x = ti.Vector.field(2, float, n_particles)
v = ti.Vector.field(2, float, n_particles)
C = ti.Matrix.field(2, 2, float, n_particles)
J = ti.field(float, n_particles)

grid_v = ti.Vector.field(2, float, (n_grid, n_grid))
grid_m = ti.field(float, (n_grid, n_grid))


@ti.kernel
def substep():
    for i, j in grid_m:
        grid_v[i, j] = [0, 0]
        grid_m[i, j] = 0
    for p in x:
        Xp = x[p] / dx
        base = int(Xp - 0.5)
        fx = Xp - base
        w = [0.5 * (1.5 - fx)**2, 0.75 - (fx - 1)**2, 0.5 * (fx - 0.5)**2]
        stress = -dt * 4 * E * p_vol * (J[p] - 1) / dx**2
        affine = ti.Matrix([[stress, 0], [0, stress]]) + p_mass * C[p]
        for i, j in ti.static(ti.ndrange(3, 3)):
            offset = ti.Vector([i, j])
            dpos = (offset - fx) * dx
            weight = w[i].x * w[j].y
            grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
            grid_m[base + offset] += weight * p_mass

@ti.kernel
def init():
    for i in range(n_particles):
        x[i] = [ti.random() * 0.4 + 0.2, ti.random() * 0.4 + 0.2]
        v[i] = [0, -1]
        J[i] = 1


def export():
    m = ti.aot.Module(ti.opengl)
    m.add_field('x', x)
    m.add_field('v', v)
    m.add_field('C', C)
    m.add_field('J', J)
    m.add_field('grid_v', grid_v)
    m.add_field('grid_m', grid_m)
    m.add_kernel(init)
    m.add_kernel(substep)

    filename = 'mpm88'
    tmpdir = 'mpm88_no_nv_extension'
    m.save(tmpdir, filename)
    with open(os.path.join(tmpdir,f'metadata.json')) as json_file:
        json.load(json_file)

def run_steps(n=500):
    for s in range(n):
        substep()

def run():
    init()
    gui = ti.GUI('MPM88')
    while gui.running and not gui.get_event(gui.ESCAPE):
        run_steps()
        gui.clear(0x112F41)
        gui.circles(x.to_numpy(), radius=1.5, color=0x068587)
        gui.show()

def profile():
    init()
    gui = ti.GUI('MPM88')
    while gui.running and not gui.get_event(gui.ESCAPE):
        for _ in range(200):
            run_steps()
#
# run()
# export()
profile()

