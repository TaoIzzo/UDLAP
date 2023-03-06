#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np
import time


# In[15]:


def solver_Fe_simple(I,a,L,dt,F,T):
    t0 = time.process_time()
    Nt = int(round(T/float(dt)))
    t = np.linspace(0,Nt*dt,Nt+1)
    dx = np.sqrt(a*dt/F)
    Nx = int(round(L/dx))
    x = np.linspace(0,L,Nx+1)
#     dx=x[1] - x[0]
#     dt=t[1] - t[0]
    u = np.zeros(Nx+1)
    u_n = np.zeros(Nx+1)
    u_array = np.zeros((Nt+1,Nx+1))

    for i in range(0, Nx+1):
        u_n[i] = I(x[i])
    u_n[0] = 100
    u_n[Nx] = 50

    u_array[0,:] = u_n

    for n in range(0,Nt):
        for i in range(1,Nx):
            u[i] = u_n[i] + F*(u_n[i-1]-2*u_n[i]+u_n[i+1])
        u[0] = 100; u[Nx] = 50

        u_array[n+1,:] = u

        u_n, u=u, u_n
    t1 = time.process_time()
    return u_n,x,t,t1-10,u_array
