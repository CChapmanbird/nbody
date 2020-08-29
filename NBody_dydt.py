#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:51:12 2020

@author: christian
"""
import numpy as np
import astropy


def dydt(y, mass, dim, dt, use_G=False):
    soften = dt * 10  # Soften the gravitational force dependent on the time-step.
    posit = y[0:len(y) - 1:2]
    veloc = y[1:len(y):2]

    r_matrix = np.zeros((len(mass), len(mass)), dtype=np.object)  # Array of 1/r^2 values between bodies.

    # Calculate 1/r^2 between each mass sequentially - taking advantage of symmetry between masses to halve complexity.
    for i in range(len(mass)):
        for j in range(len(mass)):
            if i > j:
                r_val = posit[i] - posit[j]
                r_matrix[j][i] = r_val / (np.linalg.norm(r_val) ** 3 + np.linalg.norm(r_val) * soften ** 2)
                r_matrix[i][j] = -r_matrix[j][i]

    deriv = np.zeros((len(mass) * 2, dim))
    deriv[0:len(deriv) - 1:2] = veloc  # Velocity at previous time-step corresponds to distance at current time-step.
    deriv_index = np.arange(1, len(mass) * 2, 2)

    mult = np.dot(r_matrix, mass)  # Via N2, we calculate the change in velocity (F/m = p/m = v).
    for i, item in enumerate(mult):
        deriv[deriv_index[i]] += mult[i]

    if use_G:
        deriv[1:len(deriv):2] *= astropy.constants.G.value  # Real-world correction, if units are important.

    return deriv
