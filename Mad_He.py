import numpy as np
from math import log, exp
from numba import jit

def compute_mad_age(time, temperature, method='AHe', beta=2, dt=1, grain_radius=60, default_age=100, default_age_temperature=800):
    
    if (type(default_age)==np.ndarray or type(default_age)==list or
        type(default_age_temperature)==np.ndarray or type(default_age_temperature)==list):
        if type(default_age) != type(default_age_temperature):
            raise ValueError('default age and temperature must be the same type')
        elif len(default_age) != len(default_age_temperature):
            raise ValueError('default age and temperature must have the same length')
        else:
            max_d_age = max(default_age)
    elif type(default_age) == float or type(default_age) == int:
        max_d_age = default_age
        
    else:
        raise ValueError('default age has wrong type')
    
    if time.max()<max_d_age:
        time = np.append(time, default_age)
        temperature = np.append(temperature, default_age_temperature)
 
    ind_sort = time.argsort()[::-1]
    time = time[ind_sort]
    temperature = temperature[ind_sort]
    
    return mad_age(time, temperature, method=method, beta=beta, dt=dt, grain_radius=grain_radius)
    
# @jit    
def mad_age(time, temperature, method='AHe', beta=2, dt=1, grain_radius=60):
    """
    Production-diffusion-cooling algorithm (Mad_He.f90) as documented in
    Braun et al., 2006 (Quantitative Thermochronology. Cambridge Uni. Press)

    INPUT PARAMETERS:
    time                  vector of times before present [Ma]
    temperature           vector of corresponding temperatures [C]
    dt                    time step [Ma]
    beta                  geometrical factor (1=cylindrical, 2=spherical)

    OUTPUT PARAMETERS:
    out                   vector of apparent ages [Ma]
    """

    if method == 'AHe':
        # D0 = .0032
        D0 = 50e-4
        a = grain_radius*1e-6 # diffusion radius (m)
        D0a2 = D0/a**2
        Ea = 138e3
    elif method == 'ZHe':
        # D0a2 = 4.6e3
        D0 = 0.46e-4
        a = grain_radius*1e-6
        D0a2 = D0/a**2
        Ea = 169e3
    elif method == 'KsAr':
        # D0a2 = 5.6
        D0a2 = (9.8e-3 * 1e-4) / (10e-6)**2
        # Ea = 120e3
        Ea = 183e3
    elif method == 'PlAr':
        D0a2 = 125.7
        Ea = 168.4e3
    elif method == 'BiAr':
        # D0a2 = 160.
        D0a2 = (7.5e-2 * 1e-4) / (750e-6)**2
        # Ea = 211e3
        Ea = 197e3
    elif method == 'MuAr':
        # D0a2 = 13.2
        D0a2 = (4e-4 * 1e-4) / (750e-6)**2
        # Ea = 183e3
        Ea = 180e3
    elif method == 'HbAr':
        # D0a2 = 24
        D0a2 = (6e-2 * 1e-4) / (500e-6)**2
        # Ea = 276e3
        Ea = 268e3
    else:
        raise ValueError('method not implemented')
        
    realmin = np.nextafter(0, 1)
    Ma = 365.25*24.*3600.*1e6
    R = 8.3144621

    n=50        # number of spatial bins
    ntime=len(time)   # number of temporal bins
    alpha=0.5            # time marching parameter (explicit-implicit scheme)

    age=np.zeros(n)
    D=np.zeros(n)
    L=np.zeros(n)
    U=np.zeros(n)
    xf=np.zeros(n)

    for itime in range(ntime-1):
        nstep=int(max(1, np.floor((time[itime]-time[itime+1]+realmin)/dt)))
        dt=(time[itime]-time[itime+1])/nstep
        dr=1/(n-1)
        temps=temperature[itime]+273.15
        tempf=temperature[itime+1]+273.15
        da2now=D0a2*Ma*exp(-Ea/R/temps)

        for istep in range(nstep):
            # calculate current tempreature and diffusion parameter
            fstep=(istep+1)/nstep
            temp=temps+(tempf-temps)*fstep
            da2then=da2now
            da2now=D0a2*Ma*exp(-Ea/R/temp)

            f1=alpha*dt*da2now/dr**2
            f2=(1-alpha)*dt*da2then/dr**2

            # compose the tridiagonal matrix and send it to decomposition
            D[0]=1
            U[0]=-1
            xf[0]=0

            for i in range(1, n-1):
                D[i]=1.+2.*f1
                U[i]=-f1*(1+beta/i/2)
                L[i]=-f1*(1-beta/i/2)
                xf[i]=age[i]+f2*((age[i+1]-2*age[i]+age[i-1]) + 
                    beta*(age[i+1]-age[i-1])/i/2)+dt

            D[-1]=1
            L[-1]=0
            xf[-1]=0
            age = tridag(L,D,U,xf,n)

            # average solution over the grain's volume using trapezoidal rule
            agei=0
            for i in range(n):                    
                fact=1
                if (i==0) or (i==n-1):
                    fact=0.5

                agei+=age[i]*fact*dr**3*i**2

            agei=3.*agei
            # out[itime+1]=agei
    
    return agei, age

@jit
def tridag(a, b, c, r, n):
    # Solution of a tri-diagonal linear system, as described in
    # Press, 2007. Numerical Recipes. p. 56-57
    
    u = np.zeros(n)
    gam=np.zeros(n)
    bet=b[0]
    # try:
    u[0]=r[0]/bet
    # except ZeroDivisionError:
    #     raise
    
    for j in range(1, n):
        gam[j]=c[j-1]/bet
        bet=b[j]-a[j]*gam[j]
        # try:
        u[j]=(r[j]-a[j]*u[j-1])/bet
        # except ZeroDivisionError:
        #     raise

    for j in range(n-2, -1, -1):
        u[j]-=gam[j+1]*u[j+1]

    return u