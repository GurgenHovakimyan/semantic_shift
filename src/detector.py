"""Class-conditional semantic drift detector (public entry point).

The implementation lives in :mod:`src.statistics` (statistics) and
:mod:`src.protocol` (calibration, permutation tests, alarm rule).
"""

from .protocol import CMMDMonitor, permutation_test_cmmd, threshold_from_null
from .statistics import Context, build_context, cmmd, cmmd_per_class

SemanticDriftDetector = CMMDMonitor

__all__ = ['CMMDMonitor', 'SemanticDriftDetector', 'Context', 'build_context', 'cmmd',
           'cmmd_per_class', 'permutation_test_cmmd', 'threshold_from_null']
