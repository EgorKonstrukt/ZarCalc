"""
Damped Harmonic Oscillator - RK4 Solver
========================================
Solves the second-order ODE:
    y'' + 2*zeta*omega*y' + omega^2*y = 0

Uses the 4th-order Runge-Kutta method.
Opens a new chart window showing displacement, velocity, phase portrait,
and compares the numerical result to the analytical solution.
"""

ZETA  = 0.2
OMEGA = 2.0
Y0    = 1.0
DY0   = 0.0
T0    = 0.0
T1    = 20.0
STEPS = 2000

def rk4_step(t, y, dy, h, zeta, omega):
    def accel(y_, dy_):
        return -2.0 * zeta * omega * dy_ - omega**2 * y_
    k1y  = dy
    k1dy = accel(y, dy)
    k2y  = dy  + h/2 * k1dy
    k2dy = accel(y + h/2 * k1y,  dy + h/2 * k1dy)
    k3y  = dy  + h/2 * k2dy
    k3dy = accel(y + h/2 * k2y,  dy + h/2 * k2dy)
    k4y  = dy  + h   * k3dy
    k4dy = accel(y + h   * k3y,  dy + h   * k3dy)
    ny  = y  + h/6 * (k1y  + 2*k2y  + 2*k3y  + k4y)
    ndy = dy + h/6 * (k1dy + 2*k2dy + 2*k3dy + k4dy)
    return ny, ndy


h = (T1 - T0) / STEPS
ts, ys, dys = [T0], [Y0], [DY0]
for _ in range(STEPS):
    ny, ndy = rk4_step(ts[-1], ys[-1], dys[-1], h, ZETA, OMEGA)
    ts.append(ts[-1] + h)
    ys.append(ny)
    dys.append(ndy)

wd = OMEGA * sqrt(max(0.0, 1.0 - ZETA**2))
analytical = [
    exp(-ZETA * OMEGA * t) * (
        Y0 * cos(wd * t) + (DY0 + ZETA * OMEGA * Y0) / max(wd, 1e-12) * sin(wd * t)
    )
    for t in ts
]

win = api.new_window("Damped Oscillator — RK4 vs Analytical")
win.chart.setLabel("bottom", "time t")
win.chart.setLabel("left",   "y(t)")

win.plot(ts, ys,         label="RK4 numerical",  color="#3498db", width=2)
win.plot(ts, analytical, label="analytical",      color="#2ecc71", width=1)
win.plot(ts, dys,        label="velocity y'(t)",  color="#e74c3c", width=1)

win_phase = api.new_window("Phase Portrait  y vs y'")
win_phase.chart.setLabel("bottom", "y")
win_phase.chart.setLabel("left",   "y'")
win_phase.plot(ys, dys, label="orbit", color="#9b59b6", width=1)

envelope_p = [exp(-ZETA * OMEGA * t) for t in ts]
envelope_n = [-v for v in envelope_p]
win.plot(ts, envelope_p, label="+envelope", color="#f39c12", width=1)
win.plot(ts, envelope_n, label="-envelope", color="#f39c12", width=1)

max_err = max(abs(a - b) for a, b in zip(ys, analytical))
api.status(
    f"ODE solved: zeta={ZETA}, omega={OMEGA}, steps={STEPS}  "
    f"max RK4 error = {max_err:.2e}",
    timeout_ms=8000,
)