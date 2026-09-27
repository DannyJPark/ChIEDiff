"""The five weighted radial terms of the AutoDock Vina scoring function.

Every term is a function of the SURFACE distance

    d = r - (R_i + R_j)

that is, the gap between the two van der Waals surfaces -- NOT the centre-to-centre
distance r.  R_i, R_j are Vina's ``xs_radius`` values (XS_RADIUS_BY_Z in
gated_energy_diffusion/utils/vina_types.py).  Passing r straight in shifts every term
and silently changes the energy.

``VINA_WEIGHT`` holds Vina's published coefficients, in the term order

    [gauss1, gauss2, repulsion, hydrophobic, hbonding]

so a caller stacking the five terms must stack them in exactly that order.

Upstream: KGDiff (MIT License, Copyright (c) 2023 CMACH508).
"""
import torch

VINA_WEIGHT = [-0.0356, -0.00516, 0.840, -0.0351, -0.587]

def gauss1(d):
    return torch.exp(-(2*d)**2)

def gauss2(d):
    return torch.exp(-((d-3.0)/2)**2)

def repulsion(d):
    return torch.where(d < 0, d**2, torch.zeros_like(d))

def hydrophobic(d):
    return torch.clamp(1.5 - d, min=0.0,max=1.0)

def hbonding(d):
    return torch.clamp(d/(-0.7), min=0.0,max=1.0)
