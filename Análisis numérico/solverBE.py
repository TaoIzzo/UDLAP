import numpy as np 
from lib_funciones_particulares2 import *
from lib_funciones_generales import *

def I(x):
	if x==0:
		return 100
	elif x==L:
		return 50
	else:
		return 0

def I2(x):
    if x < 1/2 and x>=0:
        return 2*x
    elif x <= 1 and x>=1/2:
        return 2 -2*x



#################################### Ejemplo ilustrativo 1a (excel)

K = 0.8348  #### K=k/(rho*C)
L = 10 

Nx = 5
dx = L/Nx
T = 2
Nt = 20
dt =T/Nt

F = dt*K/dx**2  

ans = solver_BE_simple(I, K, L, dt, F, T, 100, 50)

u = ans[0]
x = ans[1]
t = ans[2]
cpu = ans[3]
u_array = ans[4]
#print('Table:', u_array)
#print('t:', t)


nt = Nt 
time = t
y_time = u_array 
x_sec_min = 0
x_sec_max = L
nx = Nx 
xlab = 'Longitud [cm]'
ylab = 'Temperatura [°C]'
fig_title = 'Ejemplo ilustrativo 3a, F: '+str(F)
legend = False
figure1 = XYplot_profiles(nt, time, y_time, x_sec_min, x_sec_max, nx, xlab, ylab, fig_title, legend)

#output_namefile = 'AHR'
#plt.savefig('figure1a'+output_namefile+'.png')


#################################### Ejemplo ilustrativo 1b


#K = 0.8348  #### a
#L = 10 

Nx = 10
dx = L/Nx
#T = 2
#Nt = 20
#dt =T/Nt

F = dt*K/dx**2  

ans = solver_BE_simple(I, K, L, dt, F, T, 100, 50)

u = ans[0]
x = ans[1]
t = ans[2]
cpu = ans[3]
u_array = ans[4]
#print('Table:', u_array)
#print('t:', t)

 
ylab = 'Temperatura [°C]'
fig_title = 'Ejemplo ilustrativo 3b, F: '+str(F)
legend = False
figure2 = XYplot_profiles(Nt, t, u_array, 0, L, Nx, 'Longitud [cm]', ylab, fig_title, legend)

#output_namefile = 'AHR'
#plt.savefig('figure1b'+output_namefile+'.png')


#################################### Ejemplo ilustrativo 3a

K = 1  #### K=k/(rho*C)
L = 1

Nx = 20
dx = L/Nx
T = 0.13
dt = 0.0013

Nt = int(T/dt)
print('Nt',Nt)


F = dt*K/dx**2  

ans = solver_BE_simple(I2, K, L, dt, F, T, 0, 0)

u = ans[0]
x = ans[1]
t = ans[2]
cpu = ans[3]
u_array = ans[4]
#print('Table:', u_array)
#print('t:', t)


nt = Nt 
time = t
y_time = u_array 
x_sec_min = 0
x_sec_max = L
nx = Nx 
xlab = 'Longitud [cm]'
ylab = 'Temperatura [°C]'
fig_title = 'Ejemplo ilustrativo 4a, F: '+str(F)
legend = False
figure1 = XYplot_profiles(nt, time, y_time, x_sec_min, x_sec_max, nx, xlab, ylab, fig_title, legend)

fig_title = 'Solución para t= '+str(t[0])
XYplot(x, u_array[0], x_sec_min, x_sec_max, 0, 1, xlab, ylab, fig_title, 'blue', False)

fig_title = 'Solución para t= '+str(t[10])
XYplot(x, u_array[10], x_sec_min, x_sec_max, 0, 1, xlab, ylab, fig_title, 'blue', False)

fig_title = 'Solución para t= '+str(t[30])
XYplot(x, u_array[30], x_sec_min, x_sec_max, 0, 1, xlab, ylab, fig_title, 'blue', False)

fig_title = 'Solución para t= '+str(t[50])
XYplot(x, u_array[50], x_sec_min, x_sec_max, 0, 1, xlab, ylab, fig_title, 'blue', False)

#output_namefile = 'AHR'
#plt.savefig('figure2a'+output_namefile+'.png')

#################################### Ejemplo ilustrativo 3b

K = 1  #### K=k/(rho*C)
L = 1

Nx = 20
dx = L/Nx
T = 1.25
dt = 0.0125

Nt = int(T/dt)
print('Nt',Nt)


F = dt*K/dx**2  

ans = solver_BE_simple(I2, K, L, dt, F, T, 0, 0)

u = ans[0]
x = ans[1]
t = ans[2]
cpu = ans[3]
u_array = ans[4]
#print('Table:', u_array)
#print('t:', t)

nt = Nt 
time = t
y_time = u_array 
x_sec_min = 0
x_sec_max = L
nx = Nx 
xlab = 'Longitud [cm]'
ylab = 'Temperatura [°C]'
fig_title = 'Ejemplo ilustrativo 4b, F: '+str(F)
legend = False
figure1 = XYplot_profiles(nt, time, y_time, x_sec_min, x_sec_max, nx, xlab, ylab, fig_title, legend)

fig_title = 'Solución para t= '+str(t[1])
XYplot(x,u_array[1], x_sec_min, x_sec_max, 0, 1, xlab, ylab, fig_title, 'blue', False)

fig_title = 'Solución para t= '+str(t[10])
XYplot(x,u_array[10], x_sec_min, x_sec_max, 0, 1, xlab, ylab, fig_title, 'blue', False)

fig_title = 'Solución para t= '+str(t[30])
XYplot(x,u_array[30], x_sec_min, x_sec_max, 0, 1, xlab, ylab, fig_title, 'blue', False)

fig_title = 'Solución para t= '+str(t[50])
XYplot(x,u_array[50], x_sec_min, x_sec_max, 0, 1, xlab, ylab, fig_title, 'blue', False)

#output_namefile = 'AHR'
#plt.savefig('figure2a'+output_namefile+'.png')


output_namefile='IOM_Actividad7'
multipage('figuresBE_'+ output_namefile +'.pdf') 
