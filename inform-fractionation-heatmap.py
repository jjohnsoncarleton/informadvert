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
#from scipy.integrate import solve_ivp
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



# Fraction/Proportion of Generic Companies (at the start)
xs = np.arange(0.01,1,0.01)
n = len(xs)



######
# Parameters of the simulaiton



dkp = 1
kpmax =1
lam =1
kQ =2
# Number of firms
m =100
Qfs =10
l = 100


# Threshold values of ka



# Determines when we can have stable differentiated state with more than
# 50 percent generic
minka = -dkp*((dkp+kpmax)**2*kQ**2-Qfs**2)/(8*lam)/(dkp+kpmax)**2*(m-1)/m




# Determines when the undifferentiated state is stable 
midka = -dkp*((dkp+2*kpmax)**2*kQ**2-4*Qfs**2)/(8*lam)/(dkp+2*kpmax)**2*(m-1)/m

# Determines when differentiated state is possible!
#maxka = -dkp*(kpmax**2*kQ**2-Qfs**2)/(8*kpmax**2*lam)*(m-1)/m

# Set ka so that we see the important transitions
kas = np.linspace(minka*0.9,midka*1.1,l)


# Important parameter groupng
K= kQ**2*dkp+8*kas*lam

# Threshold for maximum stable fraction of generic firms
xthresh = dkp/2/((dkp/K)**(1/2)*Qfs-kpmax)
minthreshx= np.ones(len(kas))*0.5
# Set to 0.99 for visual purposes (can't set x = 1)
maxthreshx= np.ones(len(kas))*max(xs)
truethreshx = np.maximum(np.minimum(maxthreshx,xthresh),minthreshx)

# Create arrays tracking the final fraction of generic firms
# AND where or not the final fraction agreed with the initial fraction
endx  = np.zeros([n,l])
endxtrue = np.zeros([n,l])


# Initial time
t0 = 0
# Final time
T = 15

# Initial time step (RK23 adapt its timestep for accuaracy)
h0 = 0.01

# Tolerance for the infinity norm in RK23
tol = 0.01

# For timing purposes, start the timer
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

# End the timer
end = time.time()
# Print run time
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


plotendx = plt.figure(4)
plt.imshow(endxtrue, extent=[min(kas), max(kas), min(xs), max(xs)],origin='lower', interpolation='nearest', aspect='auto')
plt.plot(kas,truethreshx,'r',linewidth=3)
plt.plot(np.array([midka, midka]), np.array([xs[0], xs[-1]]),'b--.',linewidth=3)
plt.plot(np.array([minka, minka]), np.array([xs[0], xs[-1]]),'m:',linewidth=3)
plt.ylabel('Initial Fractionation $x$')
plt.xlabel('Marginal Advertising Cost $k_a$')
plt.show()


plotendx.savefig("End_Fractionation_Heatmap.pdf")
plotendx.savefig("End_Fractionation_Heatmap.png")

np.save("endfractionation.npy")


