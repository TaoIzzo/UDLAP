import time

import numpy as np


def solver_BE_simple(I, a, L, dt, F, T, val1, val2):

   t0 = time.process_time() # for measuring the CPU time

   Nt = int(round(T/float(dt)))

   print("constant", Nt*dt)

   t = np.linspace(0, Nt*dt, Nt+1)  # Mesh points in time

   dx = np.sqrt(a*dt/F)

   Nx = int(round(L/dx))

   x = np.linspace(0, L, Nx+1)      # Mesh points in space

   F = dt*a/(dx*2)

   print('Value of F:', F)



   u  = np.zeros(Nx+1)

   u_n = np.zeros(Nx+1)

   u_array = np.zeros((Nt+1, Nx+1))



   # Data structures for the linear system

   A = np.zeros((Nx+1, Nx+1))

   b = np.zeros(Nx+1)



   for i in range(1, Nx):

      A[i,i-1] = -F

      A[i,i] = 1 + 2*F

      A[i,i+1] = - F



   A[0,0] = A[Nx,Nx] = 1

   A[0,1] = A[Nx,Nx-1] = 0

   print('Imprimiendo a la matriz A:', A)





   # Set initial condition u(x,0) = I(x)

   print("Time step=", 0, ', time=', t[0], ',', (t[0]/T)*100.0, ' %')

   for i in range(0, Nx+1):

      u_n[i] = I(x[i])

      u_array[0,i]=u_n[i]



   for j in range(0, Nt):

      print("Time step= ", j+1, ', time=', t[j+1], ',', (t[j+1]/T)*100.0, ' %')



       # Compute b and solve linear system

      for i in range(1, Nx):

         b[i] = u_n[i]

      b[0] = val1

      b[Nx] = val2

      u[:] = np.linalg.solve(A, b)

      u_array[j+1,:]=u



      # Update u_n before next step

      u_n, u = u, u_n



   t1 = time.process_time()

   return u, x, t, t1-t0, u_array