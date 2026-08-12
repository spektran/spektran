"""Example: simulate a CRDS ring-down decay and recover absorption coefficient.

Demonstrates the CRDS forward model: HITRAN absorption -> ring-down time tau
-> absorption coefficient alpha. Uses the built-in CH4 demo line list.
"""

import numpy as np
from spektran.physics.hitran import demo_ch4_2nu3
from spektran.physics.absorption import absorption_coefficient
from spektran.physics.crds import (
    empty_cavity_tau,
    ring_down_time,
    absorption_from_tau,
)

nu = np.linspace(6046.0, 6048.0, 200)
lines = demo_ch4_2nu3()

alpha = absorption_coefficient(nu, lines, 100e-6, 296.0, 1.0)

cavity_length_cm = 50.0  # 50 cm cavity
mirror_R = 0.99995

tau0 = empty_cavity_tau(cavity_length_cm, mirror_R)
print(f"Empty-cavity ring-down time: {tau0*1e6:.2f} us")

tau_at_peak = ring_down_time(cavity_length_cm, mirror_R, float(np.max(alpha)))
tau_off_line = ring_down_time(cavity_length_cm, mirror_R, float(np.min(alpha)))
print(f"Ring-down time (on-line):  {tau_at_peak*1e6:.2f} us")
print(f"Ring-down time (off-line): {tau_off_line*1e6:.2f} us")

alpha_recovered = absorption_from_tau(tau_at_peak, tau0, cavity_length_cm)
residual = abs(alpha_recovered - float(np.max(alpha)))
print(f"Round-trip residual: {residual:.2e} cm^-1 (should be ~0)")
