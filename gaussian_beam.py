# -*- coding: utf-8 -*-
"""
Created on Wed May  4 12:42:28 2022

@author: 91939
"""

import numpy as np
import math
import matplotlib.pyplot as plt

def gaussian_beam(x,y,z,w_0,lamb,E_0):
    
    k = 2*np.pi/lamb
    
    z_R = np.pi*w_0**2/lamb 
    
    w_z = w_0*np.sqrt(1+(z/z_R)**2)
    
    if abs(z) < 1e-50:
        R_z = 1e100
    else:
        R_z = z*(1+(z_R/z)**2)
       
    phi_z = np.arctan(z/z_R)
    
    theta = k*z - phi_z + k*(x**2+y**2)/2*R_z

    E_0_xyz = (w_0/w_z)*np.exp(-1*(x**2+y**2)/w_z**2)
    E_0_xyz = E_0_xyz*E_0
    
    E_0_xyz_re = (w_0/w_z)*np.exp(-1*(x**2+y**2)/w_z**2)*np.cos(theta)
    E_0_xyz_re = E_0_xyz_re*E_0
    
    E_0_xyz_im = (w_0/w_z)*np.exp(-1*(x**2+y**2)/w_z**2)*np.sin(theta)
    E_0_xyz_im = E_0_xyz_im*E_0
    
    return [E_0_xyz, E_0_xyz_re, E_0_xyz_im]



e_x = np.array([1,0.,0.])
lamb = 1
w_0 = 1.56*lamb

#plotting along z axis for a particular x, y value
x = 0
y = 0

E_0 = np.array([0,0,0])
re_E_0 = np.array([0,0,0])
im_E_0 = np.array([0,0,0])

z = np.arange(-8*lamb, 8*lamb, 0.01)



for i in z:
    
    E_0_i = gaussian_beam(x,y,i,w_0,lamb,e_x)[0]
    re_E_0_i = gaussian_beam(x,y,i,w_0,lamb,e_x)[1]
    im_E_0_i = gaussian_beam(x,y,i,w_0,lamb,e_x)[2]
    
    E_0 = np.vstack([E_0, E_0_i])
    re_E_0 = np.vstack([re_E_0, re_E_0_i])
    im_E_0 = np.vstack([im_E_0, im_E_0_i])
    
 
E_0 = np.delete(E_0,0,0)
re_E_0 = np.delete(re_E_0,0,0)
im_E_0 = np.delete(im_E_0,0,0)

plt.plot(z,re_E_0[:,0])
plt.xlabel('z/$\lambda$')
plt.ylabel('Re[$E_0$]')
plt.title('Plot at x=0, y=0 with $e_x$ polarization')
# function to show the plot
plt.show()


#Field value at points where atoms are placed i.e. z=0
n=26
N=n**2
a=1

x = np.arange(-(n-1)*a/2, (n)*a/2, a)
y = np.arange(-(n-1)*a/2, (n)*a/2, a)

z=0

for j in y:
    for i in x:
        E_0_i = gaussian_beam(i,j,z,w_0,lamb,e_x)[0]
        re_E_0_i = gaussian_beam(i,j,z,w_0,lamb,e_x)[1]
        im_E_0_i = gaussian_beam(i,j,z,w_0,lamb,e_x)[2]
    
        E_0 = np.vstack([E_0, E_0_i])
        re_E_0 = np.vstack([re_E_0, re_E_0_i])
        im_E_0 = np.vstack([im_E_0, im_E_0_i])
    
E_0 = np.delete(E_0,0,0)
re_E_0 = np.delete(re_E_0,0,0)
im_E_0 = np.delete(im_E_0,0,0)





