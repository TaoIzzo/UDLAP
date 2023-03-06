#!/usr/bin/env python
# coding: utf-8

# In[1]:


from numpy import *
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from lib_funciones_generales import*
import numpy as np
from lib_funciones_particulares2 import*
# Euler's Method

def euler(t0,tn,n,y0):
    h = abs(tn-t0)/n
    t = linspace(0,tn,n+1)
    y = zeros(n+1)
    y[0] = y0
    for k in range(0,n):
        y[k+1] = y[k] + h*f(t[k],y[k]) #Forward Euler
    return y

# Runge Kutta method

def RK4(t0,tn,n,y0):
    h = abs(tn-t0)/n
    t = linspace(t0,tn,n+1)
    y = zeros(n+1)
    y[0] = y0
    for i in range(0,n):
        K1 = f(t[i],y[i])
        K2 = f(t[i]+h/2,y[i]+K1*h/2)
        K3 = f(t[i]+h/2,y[i]+K2*h/2)
        K4 = f(t[i]+h,y[i]+K3*h)
        y[i+1] = y[i] + h*(K1+2*K2+2*K3+K4)/6
    return y
########################################################################### Problem 1
print("-"*30+"Problem 1:"+"-"*30)
#f in the IVP y’ = f(t,y), y(t0)=y0
def f(t,y):
    return (0.04)*y 
fg = 1
t0 = 0
tn = 6 #### final time [years]
y0 = 5000 #[dollars]
h = 1/12
n = int(abs(tn-t0)/h)
t = linspace(t0,tn,n+1)
print('size of t: ', len(t))
ye = euler(t0,tn,n,y0)
print('Using Euler:')
print('ye[12]', ye[12], 'for ', t[12], 'years')
print('ye[24]', ye[24], 'for ', t[24], 'years')
print('ye[36]', ye[36], 'for ', t[36], 'years')
print('ye[48]', ye[48], 'for ', t[48], 'years')
print('ye[60]', ye[60], 'for ', t[60], 'years')
print('ye[72]', ye[72], 'for ', t[72], 'years')
yrk = RK4(t0,tn,n,y0)
print('Using Runge-Kutta:')
print('yrk[12]', yrk[12], 'for ', t[12], 'years')
print('yrk[24]', yrk[24], 'for ', t[24], 'years')
print('yrk[36]', yrk[36], 'for ', t[36], 'years')
print('yrk[48]', yrk[48], 'for ', t[48], 'years')
print('yrk[60]', yrk[60], 'for ', t[60], 'years')
print('yrk[72]', yrk[72], 'for ', t[72], 'years')

########################################################################## Problem 2
print("-"*30+"Problem 2:"+"-"*30)
#f in the IVP dy = f(t,y), y(t0)=y0
fg = 1
t0 = 0
tn = 5 #### final time [rate per years]
y0 = 5000 #[dollars]
h = 1/12
n = int(abs(tn-t0)/h)
t = linspace(t0,tn,n+1)
#################################################  primer elemento euler
def f(t,y):
     return (0.01)*y 
ye = euler(t0,tn,n,y0)
print('Using Euler:')
print('ye[72]', ye[-1], 'for ', t[-1], 'years')
#################################################  segundo elemento euler
def f(t,y):
     return (0.02)*y 
ye = euler(t0,tn,n,y0)
print('ye[72]', ye[-1], 'for ', t[-1], 'years')
#################################################  tercer elemento euler
def f(t,y):
     return (0.03)*y 
ye = euler(t0,tn,n,y0)
print('ye[72]', ye[-1], 'for ', t[-1], 'years')
#################################################  cuarto elemento euler
def f(t,y):
     return (0.04)*y 
ye = euler(t0,tn,n,y0)
print('ye[72]', ye[-1], 'for ', t[-1], 'years')
#################################################  quinto elemento euler
def f(t,y):
     return (0.05)*y 
ye = euler(t0,tn,n,y0)
print('ye[72]', ye[-1], 'for ', t[-1], 'years')
#################################################  sexto elemento euler
def f(t,y):
      return (0.06)*y 
ye = euler(t0,tn,n,y0)
print('ye[72]', ye[-1], 'for ', t[-1], 'years')
#################################################  primero elemento RK4
def f(t,y):
    return (0.01)*y 
t = linspace(t0,tn,n+1)
yrk = RK4(t0,tn,n,y0)
print('Using Runge-Kutta:')
print('yrk[72]', yrk[-1], 'for ', t[-1], 'years')
#################################################  segundo elemento RK4
def f(t,y):
    return (0.02)*y 
yrk = RK4(t0,tn,n,y0)
print('yrk[72]', yrk[-1], 'for ', t[-1], 'years')
#################################################  tercero elemento RK4
def f(t,y):
    return (0.03)*y 
yrk = RK4(t0,tn,n,y0)
print('yrk[72]', yrk[-1], 'for ', t[-1], 'years')
#################################################  cuarto elemento RK4
def f(t,y):
    return (0.04)*y 
yrk = RK4(t0,tn,n,y0)
print('yrk[72]', yrk[-1], 'for ', t[-1], 'years')
#################################################  quinto elemento RK4
def f(t,y):
    return (0.05)*y 
yrk = RK4(t0,tn,n,y0)
print('yrk[72]', yrk[-1], 'for ', t[-1], 'years')
#################################################  sexto elemento RK4
def f(t,y):
    return (0.06)*y 
yrk = RK4(t0,tn,n,y0)
print('yrk[72]', yrk[-1], 'for ', t[-1], 'years')


########################################################################## Problem 3
print("-"*30+"Problem 3:"+"-"*30)
def f(t,y):
    return (0.03)*y 
t0 = 0
tn = 15 #### final time [years]
fg = 1
h = 1/2
y0 = 10000 #[dollars]
n = int(abs(tn-t0)/h)
t = linspace(t0,tn,n+1)
print('size of t: ', len(t))
ye = euler(t0,tn,n,y0)
print('Using Euler:')
print('ye[10]', ye[10], 'for ', t[10], 'years')
print('ye[20]', ye[20], 'for ', t[20], 'years')
print('ye[-1]', ye[-1], 'for ', t[-1], 'years')
yrk = RK4(t0,tn,n,y0)
print('Using Runge-Kutta:')
print('yrk[10]', yrk[10], 'for ', t[10], 'years')
print('yrk[8]', yrk[20], 'for ', t[20], 'years')
print('yrk[-1]', yrk[-1], 'for ', t[-1], 'years')

########################################################################### Problem 4
print("-"*30+"Problem 4:"+"-"*30)
#f in the IVP y’ = f(t,y), y(t0)=y0
def f(t,y):
    return (0.0375)*y 
fg = 1
t0 = 0
tn = 10 #### final time [years]
y0 = 500 #[dollars]
h = 1/4
n = int(abs(tn-t0)/h)
t = linspace(t0,tn,n+1)
print('size of t: ', len(t))
ye = euler(t0,tn,n,y0)
print('Using Euler:')
print('ye[4]', ye[4], 'for ', t[4], 'years')
print('ye[8]', ye[8], 'for ', t[8], 'years')
print('ye[-1]', ye[-1], 'for ', t[-1], 'years')
yrk = RK4(t0,tn,n,y0)
print('Using Runge-Kutta:')
print('yrk[4]', yrk[4], 'for ', t[4], 'years')
print('yrk[8]', yrk[8], 'for ', t[8], 'years')
print('yrk[-1]', yrk[-1], 'for ', t[-1], 'years')
        
##################################################################### Problem 5
print("-"*30+"Problem 5:"+"-"*30)
#f in the IVP y’ = f(t,y), y(t0)=y0
def f(t,y):
    return y**2-y**3
fg = 1
t0 = 0
y0 = 0.01
tn = 2/y0
#h = tn/500
#n = int(abs(tn-t0)/h)
n = 500
t = linspace(t0,tn,n+1)
ye = euler(t0,tn,n,y0)
yrk = RK4(t0,tn,n,y0)
# Script to produce graphs
fg = plt.figure()
plt.plot(t,yrk,'r+',label='Runge-Kutta')
plt.plot(t,ye,'b--',label='Euler')
plt.grid()
fg.suptitle('Flame model. Problem 5', fontsize=12)
#axis([0,tn,-60,40])
plt.legend()
plt.savefig("Problema 5.png")
plt.show()

########################################################################### Problem 6
print("-"*30+"Problem 6:"+"-"*30)
#f in the IVP y’ = f(t,y), y(t0)=y0
def f(t,y):
    return y**2-y**3
fg = 1
t0 = 0
y0 = 1/1000
tn = 2/y0
#h = tn/200
#n = int(abs(tn-t0)/h)
n = 2000
t = linspace(t0,tn,n+1)
ye = euler(t0,tn,n,y0)
yrk = RK4(t0,tn,n,y0)
# Script to produce graphs
fg = plt.figure()
plt.plot(t,yrk,'r+',label='Runge-Kutta')
plt.plot(t,ye,'b--',label='Euler')
plt.grid()
fg.suptitle('Flame model. Problem 6', fontsize=12)
#axis([0,tn,-60,40])
plt.legend()
plt.savefig("Problema 6.png")
plt.show()

########################################################################### Problem 7
print("-"*30+"Problem 7:"+"-"*30)
#f in the IVP y’ = f(t,y), y(t0)=y0
def f(t,y):
    return y

#analytic solution to the IVP y’ = f(t,y), y(t0)=y0
def sol(t,t0,y0):
    C = y0*exp(-t0)
    return exp(t)
fg = 2
t0 = 0
tn = 1
y0 = 1
h = 0.1
n = int(abs(tn-t0)/h)
t = linspace(t0,tn,n+1)
ye = euler(t0,tn,n,y0)
yrk = RK4(t0,tn,n,y0)

# Script to produce graphs
fg = plt.figure()
plt.plot(t,yrk,'r+',label='Runge-Kutta')
plt.plot(t,ye,'bo--',label='Euler')
ysol = sol(t,t0,y0)
plt.plot(t,ysol,color ='cyan',label='Exact')
plt.grid()
fg.suptitle('Problem 7 for h='+str(h), fontsize=12)
#axis([0,tn,-60,40])
plt.legend()
plt.savefig("Problema 7 h=0.1.png")
plt.show()
# Compute absolute and relative errors
print('x_n: ', t[-1])
print('y_n: ', ye[-1], 'using Euler')
print('y_n: ', yrk[-1], 'using Runge-Kutta')
print('Actual value: ', ysol[-1])
print('Abs. error', abs(ysol[-1]-ye[-1]) , 'for Euler approximation.')
print('% Rel. error', abs(ysol[-1]-ye[-1])/ysol[-1]*100, 'for Euler approximation.')
print('Abs. error', abs(ysol[-1]-yrk[-1]) , 'for RK4 approximation.')
print('% Rel. error', abs(ysol[-1]-yrk[-1])/ysol[-1]*100, 'for RK4 approximation.')
h = 0.05
n = int(abs(tn-t0)/h)
t = linspace(t0,tn,n+1)
ye = euler(t0,tn,n,y0)
yrk = RK4(t0,tn,n,y0)
# Script to produce graphs
fg = plt.figure()
plt.plot(t,yrk,'r+',label='Runge-Kutta')
plt.plot(t,ye,'bo--',label='Euler')
ysol = sol(t,t0,y0)
plt.plot(t,ysol,color ='cyan',label='Exact')
plt.grid()
fg.suptitle('Problem 7 for h='+str(h), fontsize=12)
#axis([0,tn,-60,40])
plt.legend()
plt.savefig("Problema 7 h=0.05.png")
plt.show()
# Compute absolute and relative errors
print('x_n: ', t[-1])
print('y_n: ', ye[-1], 'using Euler')
print('y_n: ', yrk[-1], 'using Runge-Kutta')
print('Actual value: ', ysol[-1])
print('Abs. error', abs(ysol[-1]-ye[-1]) , 'for Euler approximation.')
print('% Rel. error', abs(ysol[-1]-ye[-1])/ysol[-1]*100, 'for Euler approximation.')
print('Abs. error', abs(ysol[-1]-yrk[-1]) , 'for RK4 approximation.')
print('% Rel. error', abs(ysol[-1]-yrk[-1])/ysol[-1]*100, 'for RK4 approximation.')


################################################## Problem 8
print("-"*30+"Problem 8:"+"-"*30)
#f in the IVP y’ = f(t,y), y(t0)=y0
def f(t,y):
    return 2*t*y
def sol(t,t0,y0):
    C = y0*np.exp((t0**2)-1)
    return np.exp(t**2-1)
fg = 2
t0 = 1
tn = 1.5
y0 = 1
h = 0.1
n = int(abs(tn-t0)/h)
t = linspace(t0,tn,n+1)
ye = euler(t0,tn,n,y0)
yrk = RK4(t0,tn,n,y0)

# Script to produce graphs
fg = plt.figure()
plt.plot(t,yrk,'r+',label='Runge-Kutta')
plt.plot(t,ye,'bo--',label='Euler')
ysol = sol(t,t0,y0)
plt.plot(t,ysol,color ='cyan',label='Exact')
plt.grid()
fg.suptitle('Problem 8 for h='+str(h), fontsize=12)
#axis([0,tn,-60,40])
plt.legend()
plt.savefig("Problema 8 h=0.1.png")
plt.show()

# Compute absolute and relative errors
print('x_n: ', t[-1])
print('y_n: ', ye[-1], 'using Euler')
print('y_n: ', yrk[-1], 'using Runge-Kutta')
print('Actual value: ', ysol[-1])
print('Abs. error', abs(ysol[-1]-ye[-1]) , 'for Euler approximation.')
print('% Rel. error', abs(ysol[-1]-ye[-1])/ysol[-1]*100, 'for Euler approximation.')
print('Abs. error', abs(ysol[-1]-yrk[-1]) , 'for RK4 approximation.')
print('% Rel. error', abs(ysol[-1]-yrk[-1])/ysol[-1]*100, 'for RK4 approximation.')
h = 0.05
n = int(abs(tn-t0)/h)
t = linspace(t0,tn,n+1)
ye = euler(t0,tn,n,y0)
yrk = RK4(t0,tn,n,y0)

# Script to produce graphs
fg = plt.figure()
plt.plot(t,yrk,'r+',label='Runge-Kutta')
plt.plot(t,ye,'bo--',label='Euler')
ysol = sol(t,t0,y0)
plt.plot(t,ysol,color ='cyan',label='Exact')
plt.grid()
fg.suptitle('Problem 8 for h='+str(h), fontsize=12)
#axis([0,tn,-60,40])
plt.legend()
plt.savefig("Problema 8 h=0.05.png")
plt.show()
# Compute absolute and relative errors
print('x_n: ', t[-1])
print('y_n: ', ye[-1], 'using Euler')
print('y_n: ', yrk[-1], 'using Runge-Kutta')
print('Actual value: ', ysol[-1])
print('Abs. error', abs(ysol[-1]-ye[-1]) , 'for Euler approximation.')
print('% Rel. error', abs(ysol[-1]-ye[-1])/ysol[-1]*100, 'for Euler approximation.')
print('Abs. error', abs(ysol[-1]-yrk[-1]) , 'for RK4 approximation.')
print('% Rel. error', abs(ysol[-1]-yrk[-1])/ysol[-1]*100, 'for RK4 approximation.')

########################################################################### Problem 9
print("-"*30+"Problem 9:"+"-"*30)
#f in the IVP y’ = f(t,y), y(t0)=y0
def f(t,y):
    return 2*y*cos(t)
fg =1
t0 = 0
tn = 1.5
y0 = 1
h = 0.25
n = int(abs(tn-t0)/h)
t = linspace(t0,tn,n+1)
ye = euler(t0,tn,n,y0)
yrk = RK4(t0,tn,n,y0)
# Script to produce graphs
fg = plt.figure()
plt.plot(t,yrk,'r+',label='Runge-Kutta')
plt.plot(t,ye,'bo--',label='Euler')
plt.grid()
fg.suptitle('Problem 9 for h='+str(h), fontsize=12)
#axis([0,tn,-60,40])
plt.legend()
plt.savefig("Problema 9 0.25.png")
plt.show()
h = 0.1
n = int(abs(tn-t0)/h)
t = linspace(t0,tn,n+1)
ye = euler(t0,tn,n,y0)
yrk = RK4(t0,tn,n,y0)

# Script to produce graphs
fg = plt.figure()
plt.plot(t,yrk,'r+',label='Runge-Kutta')
plt.plot(t,ye,'bo--',label='Euler')
plt.grid()
fg.suptitle('Problem 9 for h='+str(h), fontsize=12)
#axis([0,tn,-60,40])
plt.legend()
plt.savefig("Problema 9 0.1.png")
plt.show()
h = 0.05
n = int(abs(tn-t0)/h)
t = linspace(t0,tn,n+1)
ye = euler(t0,tn,n,y0)
yrk = RK4(t0,tn,n,y0)

# Script to produce graphs
fg = plt.figure()
plt.plot(t,yrk,'r+',label='Runge-Kutta')
plt.plot(t,ye,'bo--',label='Euler')
plt.grid()
fg.suptitle('Problem 9 for h='+str(h), fontsize=12)
#axis([0,tn,-60,40])
plt.legend()
plt.savefig("Problema 9 h=0.05.png")
plt.show()
################################### Problem 10
print("-"*30+"Problem 10:"+"-"*30)
def f(t,y):
    return y*(10-2*y)
fg =1
t0 = 0
tn = 1.5
y0 = 1
h = 0.25
n = int(abs(tn-t0)/h)
t = linspace(t0,tn,n+1)
ye = euler(t0,tn,n,y0)
yrk = RK4(t0,tn,n,y0)

# Script to produce graphs
fg = plt.figure()
plt.plot(t,yrk,'r+',label='Runge-Kutta')
plt.plot(t,ye,'bo--',label='Euler')
plt.grid()
fg.suptitle('Problem 10 for h='+str(h), fontsize=12)
#axis([0,tn,-60,40])
plt.legend()
plt.savefig("Problema 10 h=0.25.png")
plt.show()
h = 0.1
n = int(abs(tn-t0)/h)
t = linspace(t0,tn,n+1)
ye = euler(t0,tn,n,y0)
yrk = RK4(t0,tn,n,y0)
# Script to produce graphs
fg = plt.figure()
plt.plot(t,yrk,'r+',label='Runge-Kutta')
plt.plot(t,ye,'bo--',label='Euler')
plt.grid()
fg.suptitle('Problem 10 for h='+str(h), fontsize=12)
#axis([0,tn,-60,40])
plt.legend()
plt.savefig("Problema 10 h=0.1.png")
plt.show()
h = 0.05
n = int(abs(tn-t0)/h)
t = linspace(t0,tn,n+1)
ye = euler(t0,tn,n,y0)
yrk = RK4(t0,tn,n,y0)
# Script to produce graphs
fg = plt.figure()
plt.plot(t,yrk,'r+',label='Runge-Kutta')
plt.plot(t,ye,'bo--',label='Euler')
plt.grid()
fg.suptitle('Problem 10 for h='+str(h), fontsize=12)
#axis([0,tn,-60,40])
plt.legend()
plt.savefig("Problema 10 h=0.05.png")
plt.show()
#################################################################
print("-"*30+"Problem 11:"+"-"*30)
def I(x):
    z=np.sin(((np.pi)/2)*x)
    return z
def sol(t,t0,y0):
    return np.exp(-(((np.pi)**2)/4)*t0)*np.sin(((np.pi)/2)*y0)
K = 1
L = 2
Nx = 4  
dx = L/Nx ##### h
T = 0.1
Nt = 2
dt = T/Nt ####### k
F = dt*K/dx**2
print('F', F)
ans = solver_BE_simple(I, K, L, dt, F, T, 0, 0)
u = ans[0]
x = ans[1]
t = ans[2]
cpu = ans[3]
u_array = ans[4]
#print('Table:', u_array)
#print('t:', t)
#print('u(x,t)', u_array[1000])
nt = Nt
timet = t
y_time = u_array
x_sec_min = 0
x_sec_max = L
nx = Nx
xlab = 'Longitud [cm]'
ylab = 'Temperatura [°C]'
fig_title = 'Problema 11, F: '+str(F)
legend = False
figure1 = XYplot_profiles(nt, timet, y_time, x_sec_min, x_sec_max, nx, xlab, ylab, fig_title, legend)
################################################## problem 2a
print("-"*30+"Problem 12 a:"+"-"*30)
def I(x):
    z = np.sin(2*(np.pi)*x)
    return z
def sol(t,t0,y0):
    return np.exp(-4*((np.pi)**2)*t0)*np.sin(2*np.pi*y0)
K = 1
L = 2
dx = 0.4 ##### h
Nx = int(L/dx)
T = 0.5
dt = 0.01 ####### k
Nt = int(T/dt)
F = dt*K/dx**2 
print('F', F)
ans = solver_FE_simple(I, K, L, dt, F, T, 0, 0)
u = ans[0]
x = ans[1]
t = ans[2]
cpu = ans[3]
u_array = ans[4]
nt = Nt 
time = t
y_time = u_array 
x_sec_min = 0
x_sec_max = L
nx = Nx 
xlab = 'Longitud [cm]'
ylab = 'Temperatura [°C]'
fig_title = 'Problem 12a, F: '+str(F)
legend = False
figure1 = XYplot_profiles(nt, time, y_time, x_sec_min, x_sec_max, nx, xlab, ylab, fig_title, legend)
for i in range(0, Nt+1):
    #print('i', i)
    if t[i]==0.5:
        print('u(x,t) ', u_array[i], 'for t= ', t[i])
        fig_title = 'Solución para t= '+str(t[i])
        y0 = np.linspace(0, L, Nx+1)
        usol = sol(t,0.5,y0)
        plt.plot(y0,usol,color ='cyan',label='Exact')
        plt.savefig("Problema 12a.jpg")
        plt.show()
############################################################### 
print("-"*30+"Problem 12b:"+"-"*30)
def I(x):
    z = np.sin(2*(np.pi)*x)
    return z
#analytic solution to the IVP y’ = f(t,y), y(t0)=y0
def sol(t,t0,y0):
    return np.exp(-4*((np.pi)**2)*t0)*np.sin(2*np.pi*y0)
L = 2
dx = 0.4 ##### h
Nx = int(L/dx)
T = 0.5
dt = 0.5 ####### k
Nt = int(T/dt)
F = dt*K/dx**2 
print('F', F)
ans = solver_FE_simple(I, K, L, dt, F, T, 0, 0)
u = ans[0]
x = ans[1]
t = ans[2]
cpu = ans[3]
u_array = ans[4]
#print('Table:', u_array)
#print('t:', t)
#print('u(x,t)', u_array[1000])
nt = Nt 
time = t
y_time = u_array 
x_sec_min = 0
x_sec_max = L
nx = Nx 
xlab = 'Longitud [cm]'
ylab = 'Temperatura [°C]'
fig_title = 'Problem 12b, F: '+str(F)
legend = False
figure1 = XYplot_profiles(nt, time, y_time, x_sec_min, x_sec_max, nx, xlab, ylab, fig_title, legend)
for i in range(0, Nt+1):
    #print('i', i)
    if t[i]==0.5:
        print('u(x,t) ', u_array[i], 'for t= ', t[i])
        fig_title = 'Solución para t= '+str(t[i])
        y0 = np.linspace(0, L, Nx+1)
        usol = sol(t,0.5,y0)
        plt.plot(y0,usol,color ='cyan',label='Exact')
        plt.savefig("Problema 12b.png")
        plt.show()
############################################################### problem 3
print("-"*30+"Problem 13:"+"-"*30)
def I(x):
    z = np.sin(x)
    return z
#analytic solution to the IVP y’ = f(t,y), y(t0)=y0
def sol(t,t0,y0):
    return np.exp(-t0)*np.sin(y0)
K = 1
L = np.pi 
dx = np.pi/10 ##### h
Nx = int(L/dx)
T = 0.5
dt = 0.05 ####### k
Nt = int(T/dt)
F = dt*K/dx**2 
print('F', F)
ans = solver_FE_simple(I, K, L, dt, F, T, 0, 0)
u = ans[0]
x = ans[1]
t = ans[2]
cpu = ans[3]
u_array = ans[4]
nt = Nt 
time = t
y_time = u_array 
x_sec_min = 0
x_sec_max = L
nx = Nx 
xlab = 'Longitud [cm]'
ylab = 'Temperatura [°C]'
fig_title = 'Problema 13, F: '+str(F)
legend = False
figure1 = XYplot_profiles(nt, time, y_time, x_sec_min, x_sec_max, nx, xlab, ylab, fig_title, legend)
for i in range(0, Nt+1):
    #print('i', i)
    if t[i]==0.5:
        print('u(x,t) ', u_array[i], 'for t= ', t[i])
        fig_title = 'Solución para t= '+str(t[i])
        y0 = np.linspace(0, L, Nx+1)
        usol = sol(t,0.5,y0)
        plt.plot(y0,usol,color ='cyan',label='Exact')
        plt.savefig("Problema 13.png")
        plt.show()

