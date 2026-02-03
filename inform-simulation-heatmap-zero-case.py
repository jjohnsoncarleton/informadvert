# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 15:24:39 2026

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
    if np.abs(np.sum(y1-y2)**2) < tol:
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



n = 100
Qfs = np.linspace(5,10,n)


maxad = np.zeros([n,n])
dkp = 1
kpmax =1
lam =1
kQ =2
m =100

maxka = -dkp*(kpmax**2*kQ**2-Qfs**2)/(8*kpmax**2*lam)*(m-1)/m
midka = -dkp*((dkp+2*kpmax)**2*kQ**2-4*Qfs**2)/(8*lam)/(dkp+2*kpmax)**2*(m-1)/m

kas = np.linspace(1,max(midka)*1.1,n)



y0 = 0.001*np.random.random(m)+0
t0 = 0
T = 10
h0 = 0.01
tol = 0.001

start = time.time()
for i in range(len(Qfs)):
  Qf = Qfs[i]

  for j in range(len(kas)):

    ka = kas[j]
    [t1,y1] = R23(t0,T,h0,y0,tol,dkp,kpmax,Qf,lam,ka)
    maxad[i,j] = max(y1[-1,:])
    print((i)*n+j+1)

end = time.time()
print(end-start)

plt.figure(1)
plt.imshow(maxad.T, extent=[min(Qfs), max(Qfs), min(kas), max(kas)],origin='lower')
plt.plot(Qfs,midka,'r--')
plt.xlabel('Zero Price Demand  $Q_{free}$')
plt.ylabel('Marginal Advertising Cost $k_a$')
plt.show()