"""Parameters of the Kip3 (kinesin-8) traffic problem on the MPD-RDDME lattice.

Built after Leduc et al., PNAS 109, 6100 (2012). They reconstituted Kip3 at 0-15 nM on doubly
stabilised microtubules and saw traffic jams grow from the plus end; their model is the
Parmeggiani-Franosch-Frey exclusion process with Langmuir kinetics on ONE protofilament, stepped
with Delta_t = d/v0 = 0.16 s, so their per-step probabilities kbar are rates in units of the
stepping rate. Three combinations are independent: the Langmuir density
rho_s = kbar_on/(kbar_on + kbar_off), the end-dissociation kbar_end, and the total binding rate
kbar_on N. Their published points are in LEDUC below; jams are bottleneck-induced for
kbar_end <= 0.5 and density-induced for rho_s >= 0.5.

THE DRIVE
    Phi(s) = -alpha s along the track, alpha = 10 kT per site (Delta G_ATP / 2). The barrier of
    the motor's cycle lies below the grid, so the WPE hop gives the stepping rate directly,
    k+ = (D/h^2) B(-alpha) ~ alpha D/h^2, with backward/forward = e^{-alpha} = 4.5e-5: the
    totally asymmetric limit. D is the thermal diffusivity of a BOUND motor on the lattice, not
    a cytosolic coefficient (Bugiel 2015 measured 20 nm^2/s sideways for Kip3; D_AX below is
    56 nm^2/s). The cytosol is the implicit reservoir, as in Leduc's model.

TWO KERNEL FACTS THIS FILE ENCODES (checked in source, neither changed there)
    1. The reaction acceptance is Metropolis on the TOTAL energy change, one-body field included
       (computeFNAcceptance). A birth 0 -> M at arc position s would see the tilt, ~1e4 kT along
       the track. The field is therefore gauged to ZERO AT THE PLUS END, Phi(s) = alpha (L-1-s)
       >= 0, so detachment M -> 0 has Delta Phi <= 0 everywhere and runs at exactly k_off, and
       attachment is a conversion C -> M from an immobile reservoir species C carrying the same
       coupling as M, so its tilt term cancels exactly. C is consumed by attachment and never
       created (that keeps the tip from overflowing); the consumption is recorded in the output.
    2. A lone sigma = h sphere in a voxel carries the local functional's excess free energy
       eta(4-3eta)/(1-eta)^2 = 5.60 kT at eta = pi/6. Hops cancel it; a creating reaction pays
       it. Attachment into an EMPTY site is accepted with e^{-5.60}, uniformly, and into an
       OCCUPIED site with e^{-50} (the kernel clamp). The uniform factor is absorbed into the
       nominal rate in the exclusion arm (F_INS below); the second-motor suppression is what
       makes the Langmuir (1 - rho) emerge rather than being written in.
"""
import math

# --- lattice and track -----------------------------------------------------------------------
H_NM = 10.0          # voxel edge; one voxel = one binding site (tubulin repeat 8 nm)
N_SITES = 1000       # 10 um track, Leduc's N = 1250 x 8 nm
N_LANES = 16         # the 5x5 ring around a 3x3 body; 13 protofilaments on the real tube.
                     # 16 lanes / 10 nm = 1.60 sites/nm against 13 / 8 nm = 1.63: the linear
                     # density of binding sites is reproduced to 2 %.
SIGMA_NM = 9.99      # motor diameter handed to the functional. sigma > h throws; sigma == h is a
                     # cbrt rounding gamble, so stop just short. eta(1 motor) = 0.522.
N_RES = 48           # reservoir copies of C per shell voxel; leaves 16 of 64 slots for motors

# --- motor -----------------------------------------------------------------------------------
ALPHA_KT = 10.0      # tilt per site
V0_NM_S = 56.0       # Leduc, low-density speed, 56 +/- 10 nm/s
K_SIDE_S = 0.30      # sidestep rate per DIRECTION at zero load, Bugiel 2015 (k_L0 0.31, k_R0 0.28)
F_TRACER = 0.015     # tracer fraction of the attachment flux (their Kip3-EGFP at < 0.05 nM)


def bernoulli(u):
    """B(u) = u / (e^u - 1), the WPE hop factor; B(0) = 1."""
    if abs(u) < 1e-9:
        return 1.0
    return u / math.expm1(u)


K_PLUS = V0_NM_S / H_NM                                   # stepping rate, s^-1 (5.6)
D_AX = K_PLUS * (H_NM * 1e-9) ** 2 / bernoulli(-ALPHA_KT)  # m^2/s; k+ = (D/h^2) B(-alpha)
D_LAT = K_SIDE_S * (H_NM * 1e-9) ** 2                       # m^2/s; lateral hops have dU = 0
BACK_OVER_FORWARD = bernoulli(ALPHA_KT) / bernoulli(-ALPHA_KT)


def wb_single_sphere_kT(sigma_nm=SIGMA_NM, h_nm=H_NM):
    """beta F_ex of one hard sphere alone in a voxel, exactly as bFex_wb evaluates it
    (drift_shared_dev.cuh): White Bear / BMCSL from the four weighted densities xi_n = (pi/6V)
    sigma^n, times the voxel volume. Equals the Carnahan-Starling eta(4-3eta)/(1-eta)^2."""
    V = h_nm ** 3
    c = math.pi / (6.0 * V)
    x0, x1, x2, x3 = (c * sigma_nm ** n for n in range(4))
    x3 = min(x3, 1.0 - 1e-9)
    omx = 1.0 - x3
    ln = math.log(omx)
    t1 = -(6.0 / math.pi) * x0 * ln
    t2 = (18.0 / math.pi) * x1 * x2 / omx
    t3 = (6.0 / math.pi) * x2 ** 3 / x3 ** 2 * (x3 / omx ** 2 + ln)
    return V * (t1 + t2 + t3)


F_INS = wb_single_sphere_kT()     # 5.59 kT at sigma 9.99; the uniform attachment penalty


def leduc_rates(kon_N, rho_s, kend_bar, n_sites=N_SITES, k_plus=K_PLUS):
    """Leduc's dimensionless triple -> (k_on per site, k_off, k_end) in s^-1.
    kbar_on = kon_N / N per stepping time; kbar_off from the Langmuir density; kbar_end as given."""
    k_on = kon_N / n_sites * k_plus
    k_off = k_on * (1.0 - rho_s) / rho_s
    k_end = kend_bar * k_plus
    return k_on, k_off, k_end


# Leduc et al. 2012, parameter sets as printed in their legends (kbar_on N, rho_s, kbar_end).
LEDUC = {
    'fig1c':    (0.24, 0.91, 0.2),   # 3.2 nM Kip3, N = 750: the kymograph with the growing jam
    'fig3d_lo': (0.25, 0.91, 0.2),   # ~2 nM, N = 1250: speed-density, low total binding rate
    'fig3d_hi': (1.00, 0.50, 0.2),   # ~8 nM, N = 1250: speed-density, high total binding rate
    'fig4':     (0.30, None, None),  # phase diagram, low binding rate: sweep rho_s x kbar_end
    'fig5':     (1.00, None, None),  # phase diagram, high binding rate
}


def summary():
    return ('h=%.0f nm  L=%d sites  alpha=%.0f kT  v0=%.0f nm/s  k+=%.2f/s  D_ax=%.3g m^2/s '
            '(%.0f nm^2/s)  D_lat=%.3g  back/forward=%.2e  F_ins=%.3f kT (x%.0f on k_on)'
            % (H_NM, N_SITES, ALPHA_KT, V0_NM_S, K_PLUS, D_AX, D_AX * 1e18, D_LAT,
               BACK_OVER_FORWARD, F_INS, math.exp(F_INS)))


if __name__ == '__main__':
    print(summary())
    for k, v in LEDUC.items():
        if v[1] is not None:
            print('  %-9s kon_N=%.2f rho_s=%.2f kend=%.2f -> k_on=%.3g k_off=%.3g k_end=%.3g s^-1'
                  % ((k,) + v + leduc_rates(*v)))
