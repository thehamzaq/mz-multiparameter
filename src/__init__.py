"""SU(2)-Schwinger toolkit for two-parameter Mach-Zehnder quantum metrology."""

from .core import (
    angular_momentum,
    sine_state, noon_state, twin_fock, coherent_spin,
    bs, phase_shift, mzi,
    qfi_pure, variance, expval,
    qfi_reflectivity_analytic, qfi_phase_analytic,
    qfi_two_param_inbetween, compatibility_inbetween,
    block_diagonal_J,
    lossy_density_matrix, lossy_qfi_matrix,
    sld_qfi, _sld_qfi_pair,
    mom_fisher_matrix,
    lossy_block_dim, block_offsets,
)
from .hcrb import hcrb_sdp

__version__ = "0.1.0"
