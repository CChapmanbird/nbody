#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 13:51:12 2020

@author: christian
"""
import numpy as np
import astropy

def dydt(y,mass,dim):
    posit = y[0:len(y)-1:2]
    veloc = y[1:len(y):2]
    
    r_matrix = np.zeros((len(mass),len(mass)),dtype=np.object)
    for i in range(len(mass)):
        for j in range(len(mass)):
            if i > j:
                r_val = posit[i]-posit[j]
                r_matrix[j][i] = r_val/np.linalg.norm(r_val)**3
                r_matrix[i][j] = -r_matrix[j][i]

    deriv = np.zeros((len(mass)*2,dim))
    deriv[0:len(deriv)-1:2] = veloc
    deriv_index = np.arange(1,len(mass)*2,2)
    #print(r_matrix)
    #    for i, line in enumerate(r_matrix):
#        if i != len(r_matrix):
#            for j, elem in enumerate(line[1:],start=1):
#                if j > i:
#                    for num,part in enumerate(r_matrix[j][j+1:],start=j+1):
##                        if first==True:
#                        print('r'+str(i+1)+str(j+1)+'->'+'r'+str(j+1)+str(num+1)+'->'+'r'+str(num+1)+str(i+1)+'.')
##                        print(deriv)
##                        print(deriv_index)
##                        print(num)
#                        deriv[deriv_index[i]] += mass[j]*r_matrix[i][j] +mass[num]*r_matrix[i][num]
#                        deriv[deriv_index[j]] += mass[num]*r_matrix[j][num] - mass[i]*r_matrix[i][j]
#                        deriv[deriv_index[num]] += -mass[i]*r_matrix[i][num] - mass[j]*r_matrix[j][num]
#    deriv[1:len(deriv):2] *= 1/(len(mass)-2)  #We have solved n-2 3-body problems, so we must normalise our r-val calculations.
#    deriv[1:len(deriv):2] *= (astropy.constants.G).value  #Real-world correction
    
#    print('deriv:',deriv)
#    return deriv

    mult = np.dot(r_matrix,mass)
    for i,item in enumerate(mult):
        deriv[deriv_index[i]] += mult[i]
    deriv[1:len(deriv):2] *= (astropy.constants.G).value  #Real-world correction

    return deriv
