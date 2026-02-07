# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 16:50:30 2026

@author: jjohnson16
"""

# -*- coding: utf-8 -*-
"""
This code is to numerically
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import time



# This is the piecewise function for the marginal benefit function
def MB(dkp,kpmax,kQ,Qf,a,ai,lam):
  # Number of companies
  N = len(a)
  # Average Adverising
  adavg = np.sum(a)/N


  if hasattr(ai, "__len__"):
    # Marginal Benefit Value
    val = np.zeros(len(ai))
    # It is only nonzero when it is close to the average
    val = np.where(np.abs(ai -adavg) > lam, val, dkp/(8*lam)*(N-1)/N*(-kQ**2 + Qf**2/((dkp/(2*lam))*(adavg-ai) + (dkp/2 +kpmax) )**2))
  else:
    val = 0
    if np.abs(ai -adavg) < lam:
      val = dkp/(8*lam)*(N-1)/N*(-kQ**2 + Qf**2/((dkp/(2*lam))*(adavg-ai) + (dkp/2 +kpmax) )**2)


  return val

def dadt(t,a,dkp,kpmax,kQ,Qf,lam,ka):
  dadt = np.zeros(len(a))
  #for i in range(len(a)):
  #  dadt[i] = MB(dkp,kpmax,kQ,Qf,a,a[i],lam) -ka
  dadt =  MB(dkp,kpmax,kQ,Qf,a,a,lam) -ka
  # Number of companies
  N = len(a)
  # Average Adverising
  adavg = np.sum(a)/N
  #print
  dadt = np.where(np.logical_and(a<=0, dadt<= 0), 0, dadt)
  #dadt = [0 if (a[i] <= 0 and dadt[i] <= 0) else dadt[i]  for i  in range(len(dadt)) ]
  #dadt = np.where(np.logical_and(a<0, dadt < 0), ka, dadt)

  return dadt


def R23(t0,T,h0,y0,tol,dkp,kpmax,Qf,lam,ka):
  # Time array
  t = np.array([t0])
  # Y value array
  y = np.array([y0])
  # Step size
  h = h0
  # Run until time  hits T
  while  t[-1] < T:
    # If the time step would hop past desired time adjust
    # time step to exactly hit T
    #print(t[-1] + h )
    if t[-1] + h > T:
      h = T - t[-1]
    # First slope
    f1 = dadt(t[-1],y[-1],dkp,kpmax,kQ,Qf,lam,ka)
    # Second slope
    f2 = dadt(t[-1]+h,y[-1]+h*f1,dkp,kpmax,kQ,Qf,lam,ka)
    # Third slope
    f3 = dadt(t[-1]+h/2,y[-1]+h*(f1+f2)/4,dkp,kpmax,kQ,Qf,lam,ka)
    # RK2 step
    y1 = y[-1] + h *(f1+f2)/2
    # RK3 Step
    y2 = y[-1] + h *(f1+f2+4*f3)/6

    y1 = np.where(y1 < 0, 0, y1)
    y2 = np.where(y2 < 0, 0, y2)
    
    # Compare RK2 and RK3
   #if np.abs(np.sum(y1-y2)**2) < tol:
    if np.linalg.norm((y1-y2),ord=np.inf)< tol: 
      # increment time by h if below tolerance
      t = np.append(t,t0+h)

      # accept RK3 as value
      y = np.vstack((y,y2))

      #
      t0 =t[-1]
      h = 2*h
    else:
      # Adjust step size (larger if under tol and smaller if over tol)
      h = h/2
  #return time and RK3 values
  return [t,y]


def R23adapt(t0,T,h0,y0,tol,dkp,kpmax,Qf,lam,ka):
  # Time array
  t = np.array([t0])
  # Y value array
  y = np.array([y0])
  # Step size
  h = h0
  # Run until time  hits T
  while  t[-1] < T:
    # If the time step would hop past desired time adjust
    # time step to exactly hit T
    #print(t[-1] + h )
    if t[-1] + h > T:
      h = T - t[-1]
    # First slope
    f1 = dadt(t[-1],y[-1],dkp,kpmax,kQ,Qf,lam,ka)
    # Second slope
    f2 = dadt(t[-1]+h,y[-1]+h*f1,dkp,kpmax,kQ,Qf,lam,ka)
    # Third slope
    f3 = dadt(t[-1]+h/2,y[-1]+h*(f1+f2)/4,dkp,kpmax,kQ,Qf,lam,ka)
    # RK2 step
    y1 = y[-1] + h *(f1+f2)/2
    # RK3 Step
    y2 = y[-1] + h *(f1+f2+4*f3)/6

    y1 = np.where(y1 < 0, 0, y1)
    y2 = np.where(y2 < 0, 0, y2)
    
    lte = y2-y1
    t = np.append(t,t0+h)
    y = np.vstack((y,y2))
    t0 =t[-1]
    h = h*abs(tol/lte)**(1/3)
    
  return [t,y]



# =============================================================================
# maxka = -dkp*(kpmax**2*kQ**2-Qfs**2)/(8*kpmax**2*lam)*(m-1)/m
# midka = -dkp*((dkp+2*kpmax)**2*kQ**2-4*Qfs**2)/(8*lam)/(dkp+2*kpmax)**2*(m-1)/m
# kas = np.linspace(1,max(maxka),n)
# =============================================================================

#y0 = np.append(0.1*np.ones(round(m/2)),2*np.ones(round(m/2)))
# =============================================================================
# y0 = 1*np.random.random(m)+3
# t0 = 0
# T = 10
# h0 = 0.01
# tol = 0.0001
# 
# start = time.time()
# for i in range(len(Qfs)):
#   Qf = Qfs[i]
# 
#   for j in range(len(kas)):
# 
#     ka = kas[j]
#     [t1,y1] = R23(t0,T,h0,y0,tol,dkp,kpmax,Qf,lam,ka)
#     maxad[i,j] = max(y1[-1,:])
#     fracx[i,j] = np.sum(y1[-1,:]<=np.mean(y1[-1,:]))/len(y1[-1,:])
#     print((i)*n+j+1)
# 
# end = time.time()
# print(end-start)
# =============================================================================





xs = np.arange(0.01,1,0.01)
n = len(xs)

dkp = 1
kpmax =1
lam =1
kQ =2
m =100
Qfs =10
maxka = -dkp*(kpmax**2*kQ**2-Qfs**2)/(8*kpmax**2*lam)*(m-1)/m
midka = -dkp*((dkp+2*kpmax)**2*kQ**2-4*Qfs**2)/(8*lam)/(dkp+2*kpmax)**2*(m-1)/m
minka = -dkp*((dkp+kpmax)**2*kQ**2-Qfs**2)/(8*lam)/(dkp+kpmax)**2*(m-1)/m
l = 50
kas = np.linspace(1,midka*1.1,l)
endx  = np.zeros([n,l])
endxtrue = np.zeros([n,l])

#y0 = np.append(0.1*np.ones(round(m/2)),2*np.ones(round(m/2)))
#y0 = 1*np.random.random(m)+3

t0 = 0
T = 15
h0 = 0.01
tol = 0.01

start = time.time()
for i in range(n):
    y0 = np.append(0.1*np.ones(round(m*xs[i])),(lam/xs[i])*np.ones(round(m*(1-xs[i]))))+0.001*np.random.random(m)
    for j in range(l):
        ka = kas[j]
        [t1,y1] = R23(t0,T,h0,y0,tol,dkp,kpmax,Qfs,lam,ka)
        endx[i,j] = np.sum(y1[-1,:]<=np.mean(y1[-1,:]))/len(y1[-1,:])
        if np.round(endx[i,j] -xs[i],2) == 0:
            endxtrue[i,j] = 1
        print(i*l+j+1)


end = time.time()
print(end-start)

plt.figure(1)
plt.imshow(endx, extent=[min(kas), max(kas), min(xs), max(xs)],origin='lower', interpolation='nearest', aspect='auto')
plt.xlabel('Initial Fractionation $x$')
plt.ylabel('Marginal Advertising Cost $k_a$')
plt.colorbar()
plt.show()



plt.figure(2)
plt.imshow(endxtrue.T, extent=[min(xs), max(xs), min(kas), max(kas)],origin='lower', interpolation='nearest', aspect='auto')
plt.xlabel('Initial Fractionation $x$')
plt.ylabel('Marginal Advertising Cost $k_a$')
plt.colorbar()
plt.show()

plt.figure(3)
plt.imshow(endxtrue, extent=[min(kas), max(kas), min(xs), max(xs)],origin='lower', interpolation='nearest', aspect='auto')
plt.plot(np.array([midka, midka]), np.array([xs[0], xs[-1]]),'r--')
plt.ylabel('Initial Fractionation $x$')
plt.xlabel('Marginal Advertising Cost $k_a$')
plt.show()


plt.figure(4)
plt.imshow(endxtrue, extent=[min(kas), max(kas), min(xs), max(xs)],origin='lower', interpolation='nearest', aspect='auto')
plt.plot(np.array([midka, midka]), np.array([xs[0], xs[-1]]),'r--')
plt.plot(np.array([minka, minka]), np.array([xs[0], xs[-1]]),'b--')
plt.ylabel('Initial Fractionation $x$')
plt.xlabel('Marginal Advertising Cost $k_a$')
plt.show()
