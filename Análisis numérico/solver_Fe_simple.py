import numpy as np
from main import *
from lib_funciones_generales import *


def I(x):
    z = 0
    return z


def I1(x):
    if x == 0:
        return 100
    elif x == L:
        return 50
    else:
        return 0


def I2(x):
    if 1 / 2 >= x >= 0:
        return 2 * x
    elif 1 >= x >= 1 / 2:
        return 2 - 2 * x


# EJEMPLO ILUSTRATIVO 1 EXCEL

K = 0.8348
L = 10
Nx = 5
dx = L / Nx
T = 2
Nt = 20
dt = T / Nt
F = dt * K / dx ** 2
ans = solver_Fe_simple(I1, K, L, dt, F, T, 100, 50)
u = ans[0]
x = ans[1]
t = ans[2]
cpu = ans[3]
u_array = ans[4]

# print('Table: \n', u_array)
# print('t:', t)

nt = Nt
time = t
y_time = u_array
x_sec_min = 0
x_sec_max = L
nx = Nx
xlab = 'Longitud[cm]'
ylab = 'Temperatura[C]'
fig_title = 'Ejemplo Ilustrativo 1a, F: ' + str(F)
legend = False
figure1 = XYplot_profiles(nt, time, y_time, x_sec_min, x_sec_max, nx, xlab, ylab, fig_title, legend)
output_namefile = 'IOM_F' + str(F)
plt.savefig('fig_1' + output_namefile + '.png')

# EJEMPLO ILUSTRATIVO 1B

Nx = 10
dx = L / Nx
F = dt * K / dx ** 2

ans = solver_Fe_simple(I1, K, L, dt, F, T, 100, 50)

u = ans[0]
x = ans[1]
t = ans[2]
cpu = ans[3]
u_array = ans[4]

# print('Table: \n', u_array)
# print(t)

xlab = 'Longitud[cm]'
ylab = 'Temperatura[C]'
fig_title = 'Ejemplo Ilustrativo 1B, F: ' + str(F)
legend = False

figure2 = XYplot_profiles(Nt, t, u_array, 0, L, Nx, xlab, ylab, fig_title, legend)
output_namefile = 'IOM_F' + str(F)
plt.savefig('fig1B' + output_namefile + '.png')

# EJEMPLO ILUSTRATIVO 2

K = 1
L = 1
Nx = 20
dx = L / Nx
T = 2
dt = 0.0012
Nt = int(T / dt)
print('Nt:', Nt)
F = dt * K / dx ** 2

ans = solver_Fe_simple(I2, K, L, dt, F, T, 0, 0)

u = ans[0]
x = ans[1]
t = ans[2]
cpu = ans[3]
u_array = ans[4]

# print('Table: \n', u_array)
# print(t)

xlab = 'Longitud[cm]'
ylab = 'Temperatura[C]'
fig_title = 'Ejemplo Ilustrativo 2a, F: ' + str(F)
legend = False

figure2 = XYplot_profiles(Nt, t, u_array, 0, L, Nx, xlab, ylab, fig_title, legend)
output_namefile = 'IOM_F' + str(F)
plt.savefig('fig_2a' + output_namefile + '.png')

# EJEMPLO ILUSTRATIVO 2B

K = 1
L = 1
Nx = 20
dx = L / Nx
T = 2
dt = 0.0013
Nt = int(T / dt)
print('Nt:', Nt)
F = dt * K / dx ** 2

ans = solver_Fe_simple(I2, K, L, dt, F, T, 0, 0)

u = ans[0]
x = ans[1]
t = ans[2]
cpu = ans[3]
u_array = ans[4]

# print('Table: \n', u_array)
# print(t)

xlab = 'Longitud[cm]'
ylab = 'Temperatura[C]'
fig_title = 'Ejemplo Ilustrativo 2b, F: ' + str(F)
legend = False

figure2 = XYplot_profiles(Nt, t, u_array, 0, L, Nx, xlab, ylab, fig_title, legend)
output_namefile = 'IOM_F' + str(F)
plt.savefig('fig_2b' + output_namefile + '.png')

output_namefile = 'IOM_Actividad7'
multipage('figuresFE' + output_namefile + '.pdf')
