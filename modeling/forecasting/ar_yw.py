"""
AR_YW: the former name of `AR` (Yule-Walker estimation), kept as an alias.

Author: Yousef Fekri Dabanloo
Date: 2025-03-05 (rev 2026-10-12)
License: MIT License
"""

from modeling.forecasting.ar import AR

AR_YW = AR

__all__ = ["AR_YW", "AR"]
