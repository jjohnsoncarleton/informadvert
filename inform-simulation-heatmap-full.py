# -*- coding: utf-8 -*-
"""
Created on Mon Feb 16 09:30:26 2026

@author: jjohnson16
"""

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






def R45(t0,T,h0,y0,tol,dkp,kpmax,Qf,lam,ka):
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
    f2 = dadt(t[-1]+h/4,y[-1]+h/4*f1,dkp,kpmax,kQ,Qf,lam,ka)
    # Third slope
    f3 = dadt(t[-1]+3*h/8,y[-1]+3*h/32*(f1+3*f2),dkp,kpmax,kQ,Qf,lam,ka)
    # Fourth slope
    f4 = dadt(t[-1]+12*h/13,y[-1]+h/2197*(1932*f1-7200*f2+7296*f3),dkp,kpmax,kQ,Qf,lam,ka)
    # Fifth slope
    f5 = dadt(t[-1]+h,y[-1]+h*((439/216)*f1-8*f2+(3680/513)*f3-(845/4104)*f4),dkp,kpmax,kQ,Qf,lam,ka)
    # Fifth slope
    f6 = dadt(t[-1]+h/2,y[-1]+h*((-8/27)*f1+2*f2+(-3544/2565)*f3+(1859/4104)*f4-(11/40)*f5),dkp,kpmax,kQ,Qf,lam,ka)
    # RK4 step
    y1 = y[-1] + h *((25/216)*f1+(1408/2565)*f3+(2197/4104)*f4-1/5*f5)
    # RK5 Step
    y2 = y[-1] + h *((16/135)*f1+(6656/12825)*f3+(28561/56430)*f4-9*f5/50+2*f6/55)

    y1 = np.where(y1 < 0, 0, y1)
    y2 = np.where(y2 < 0, 0, y2)
   # epses= np.finfo(float).eps
    # Compare RK4 and RK5
    #if np.abs(np.sum(y1-y2)**2) < tol:
    if np.linalg.norm((y1-y2),ord=np.inf)< tol:
    #if np.linalg.norm((y1-y2)/len(y1))< tol:    
      # increment time by h if below tolerance
      t = np.append(t,t0+h)

      # accept RK3 as value
      y = np.vstack((y,y2))

      #
      t0 =t[-1]
      h = 2*h
      if h> 1/16:
          h=1/16
    else:
      # Adjust step size (larger if under tol and smaller if over tol)
      h = h/2
    
  return [t,y]





######
# Parameters of the simulaiton

n = 20
Qfs = np.linspace(5,10,n)

dkp = 1
kpmax =1
lam =1
kQ =2
# Number of firms
m = 100
#Qfs =10
l = 20

# Threshold values of ka



# Determines when we can have stable differentiated state with more than
# 50 percent generic
minka = -dkp*((dkp+kpmax)**2*kQ**2-Qfs**2)/(8*lam)/(dkp+kpmax)**2*(m-1)/m




# Determines when the undifferentiated state is stable 
midka = -dkp*((dkp+2*kpmax)**2*kQ**2-4*Qfs**2)/(8*lam)/(dkp+2*kpmax)**2*(m-1)/m

# Determines when differentiated state is possible!
maxka = -dkp*(kpmax**2*kQ**2-Qfs**2)/(8*kpmax**2*lam)*(m-1)/m

# Set ka so that we see the important transitions
maxkas = np.linspace(np.min(maxka),np.max(maxka),l)

# Set ka so that we see the important transitions
midkas = np.linspace(np.min(midka),np.max(midka),l)


# Important parameter groupng
#K= kQ**2*dkp+8*kas*lam


# Initial time
t0 = 0
# Final time
T = 10

# Initial time step (RK23 adapt its timestep for accuaracy)
h0 = 0.01

# Tolerance for the infinity norm in RK23
tol = 0.001

# For timing purposes, start the timer
start = time.time()

#
maxadundiff = np.zeros([n,l])
#
maxaddiff = np.zeros([n,l])


y01 = 0.001*np.random.random(m)+0
#y02 = 3*np.random.random(m)+5
y02 = 1*np.random.random(m)+3

for i in range(n):
    
    

    Qf = Qfs[i]
    for j in range(l):
        ka = midkas[j]
        ka2 =maxkas[j]
        # The Zero case
        [t1,y1] = R45(t0,T,h0,y01,tol,dkp,kpmax,Qf,lam,ka)
        [t2,y2] = R45(t0,T,h0,y02,tol,dkp,kpmax,Qf,lam,ka2)
        maxadundiff[i,j] = max(y1[-1,:])
        maxaddiff[i,j] = max(y2[-1,:])
        print(i*l+j+1)

# End the timer
end = time.time()
# Print run time
print(end-start)


plt.figure(1)
plt.imshow(maxadundiff.T, extent=[min(Qfs), max(Qfs), min(midkas), max(midkas)],origin='lower', interpolation='nearest', aspect='auto')
plt.plot(Qfs,midka,'r--')
plt.xlabel('Zero Price Demand  $Q_{free}$')
plt.ylabel('Marginal Advertising Cost $k_a$')
plt.title('Maximum Advertising Value')
plt.colorbar()
plt.show()

plt.figure(2)
plt.imshow(maxaddiff.T, extent=[min(Qfs), max(Qfs), min(maxkas), max(maxkas)],origin='lower', interpolation='nearest', aspect='auto')
plt.plot(Qfs,maxka,'r--')
plt.xlabel('Zero Price Demand  $Q_{free}$')
plt.ylabel('Marginal Advertising Cost $k_a$')
plt.title('Maximum Advertising Value')
plt.colorbar()
plt.show()



#plotendx.savefig("End_Fractionation_Heatmap.pdf")
#plotendx.savefig("End_Fractionation_Heatmap.png")

#np.save("endfractionation.npy")



fig = plt.figure(3)
plt.subplot(211)
plt.imshow(maxadundiff.T, extent=[min(Qfs), max(Qfs), min(midkas), max(midkas)],origin='lower', interpolation='nearest', aspect='auto')
plt.plot(Qfs,midka,'r--')

plt.title('Maximum Advertising Value')
plt.colorbar()
ax = plt.subplot(212)

im=ax.imshow(maxaddiff.T, extent=[min(Qfs), max(Qfs), min(maxkas), max(maxkas)],origin='lower', interpolation='nearest', aspect='auto')
ax.plot(Qfs,maxka,'r--')
ax.set_xlabel('Zero Price Demand  $Q_{free}$')
ax.set_ylabel('Marginal Advertising Cost $k_a$')
ax.yaxis.label.set_position((0.5, 1))  # (x, y) coordinates
fig.colorbar(im,ax=ax)
plt.show()
