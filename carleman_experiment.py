
import numpy as np
from matplotlib import pyplot as plt
from cycler import cycler

u_0 = 0.5
t_0 = 0
t_f = 5
Dt = t_f - t_0
num = 100000
dt = Dt / num
t = np.linspace(t_0, t_f, num)

plot_Carleman = True
plot_error = True
use_absolute_error = False

# ODE u=au+bu^2
a = -1
b = 1
# exact solution
u = a/((a/u_0+b)*np.exp(-a*t)-b)
plt.figure(0)
plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))
plt.plot(t,u,'k', label='exact')

Ns = list(range(3,6))
er = np.empty((len(Ns),num))
er_index = 0

for N in Ns:
    #carleman linearization (truncation at order N) with forward Euler Method
    y_0 = np.zeros((N,1))

    for j in range(N):
        y_0[j] = np.pow(u_0, j+1)

    y = np.zeros((N,num))

    y[:, 0] = y_0.flatten()  # Set the initial condition

    

    A = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i==j:
                A[i,j] = 1+(i+1)*a*dt
            elif j==(i+1):
                A[i,j]=(i+1)*b*dt
    #print(str(A)+"\n")
    for i in range(1,num):
        y[:,i] = A.dot(y[:,i-1])
    
    if plot_error:
        # relative error calculation
        for k in range(0,num):
            if use_absolute_error:
                er[er_index,k]=np.abs((u[k]-y[0,k]))
            else:
                er[er_index,k]=np.abs((u[k]-y[0,k])/u[k])
        er_index = er_index+1

    if plot_Carleman:
        plt.plot(t,y[0,:], '--',label='N='+str(N)+'-Carleman')

plt.legend()
plt.xlabel("$t$")
plt.ylabel("$x(t)$")
plt.savefig('Figure_1.png', dpi=300)



if plot_error:
    plt.figure(1)
    plt.rc('axes', prop_cycle=(cycler('color', ['r', 'g', 'b'])))

    i=0
    for N in Ns:
        plt.plot(t, er[i,:], label='N='+str(N)+'-Carleman')
        i = i+1
    plt.legend()
    plt.xlabel("$t$")
    y_label_string = "$e_r(t)$"
    if use_absolute_error:
        y_label_string = y_label_string.replace("_r", "")

    plt.ylabel(y_label_string)
    plt.savefig('Figure_2.png', dpi=300)
