import numpy as np
from math import log, exp
from numba import jit

# @jit
def compute_tt(geo_time, uplift, elevation=np.nan, kappa=25, dt=.1, plot=False,
               sealevel_t=20, base_t=520, model_thick=20, surface_t=np.nan):

    """
    predict tt history using uplift and elevation history
    elevation is in meter
    uplift rate is in km/Ma
    time is in Ma
    """

    if np.isnan(elevation):
        elevation = np.zeros(len(geo_time))
    elif type(elevation)==float or type(elevation)==int:
        elevation = np.ones(len(geo_time)) * elevation

    if len(elevation)!=len(geo_time) or len(geo_time)!=len(uplift)+1:
        raise ValueError('lengths of input elevation, time, or uplift (+1) are not consistent')

    elevation = elevation / 1000.
    if np.isnan(surface_t):
        surface_t = sealevel_t - elevation * 4

    #model set up
    thick = model_thick  #model depth
    ny = int(thick)
    
    start = geo_time[0]
    nt = int(start/dt)  #number of time step
    TtV = np.zeros(nt)
    ly = np.zeros(nt)
    Tb_top = np.zeros(nt)
    TtV[0] = uplift[0]
    Tb_top[0] = surface_t[0]
    ly[0] = thick + elevation[0]
    t = geo_time[0]
    for i in range(nt):
        for j in range(len(geo_time)-1):
            if t<=geo_time[j] and t>geo_time[j+1]:
                TtV[i] = uplift[j]
                mm = (geo_time[j] - t) / (geo_time[j] - geo_time[j+1])
                Tb_top[i] = (1 - mm) * surface_t[j] + mm * surface_t[j+1]
                ly[i] = thick + (1 - mm) * elevation[j] + mm * elevation[j+1]
                break

        t = t - dt

#location of the data points to track history for
    Z = ly[-1]
    time = 0
    TtZ = np.zeros(nt)
    geo_time_out = np.zeros(nt)

    for i in range(nt-1, -1, -1):
        geo_time_out[i] = time
        TtZ[i] = Z
        Z = Z - TtV[i] * dt
        time = time + dt

#initial condition
    dy = ly[0] / (ny - 1)
    profileZ = np.zeros(ny)
    profileT = np.zeros(ny)
    profile_z0 = np.zeros(ny)
    for i in range(ny):
        profileZ[i] = i * dy
        profileT[i] = base_t + (Tb_top[0] - base_t) * profileZ[i]/ly[0]

#diffusion
    temperature = np.zeros(nt)
    for it in range(nt):
        profile_z0[:] = profileZ[:]
        dy = ly[it] / (ny - 1)
        for i in range(ny):
            profileZ[i] = i * dy
    
        profileT = np.interp(profileZ, profile_z0, profileT)
        profileT[-1] = Tb_top[it]
        profileT = diffusion1d(profileT, dy, dt, kappa)
        profileT = advect1d(profileT, dy, dt, TtV[it])
        

#plot(profile.T, profile.Z); hold on;
#drawnow

#save Tt history
        yd = TtZ[it]
        iy = int(yd/dy)-1
        if iy<=0:
            temperature[it] = base_t
        elif iy<ny-1:
            u = (yd - profileZ[iy]) / (profileZ[iy+1] - profileZ[iy])
            temperature[it] = (1-u)*profileT[iy] + u*profileT[iy+1]
        else:
            temperature[it] = Tb_top[it]

    return geo_time_out, temperature

@jit
def tridiag( a, b, c, f ):

#  Solve the  n x n  tridiagonal system for y:
#
#  [ a(1)  c(1)                                  ] [  y(1)  ]   [  f(1)  ]
#  [ b(2)  a(2)  c(2)                            ] [  y(2)  ]   [  f(2)  ]
#  [       b(3)  a(3)  c(3)                      ] [        ]   [        ]
#  [            ...   ...   ...                  ] [  ...   ] = [  ...   ]
#  [                    ...    ...    ...        ] [        ]   [        ]
#  [                        b(n-1) a(n-1) c(n-1) ] [ y(n-1) ]   [ f(n-1) ]
#  [                                 b(n)  a(n)  ] [  y(n)  ]   [  f(n)  ]
#
#  f must be a vector (row or column) of length n
#  a, b, c must be vectors of length n (note that b(1) and c(n) are not used)

# some additional information is at the end of the file

    n = len(f)
    v = np.zeros(n)
    y = np.zeros(n)
    w = a[0]
    y[0] = f[0]/w
    
    for i in range(1, n):
        v[i-1] = c[i-1]/w
        w = a[i] - b[i]*v[i-1]
        y[i] = (f[i]-b[i]*y[i-1])/w

    for j in range(n-2, -1, -1):
        y[j] = y[j] - v[j]*y[j+1]

    return y

@jit
def diffusion1d (h, dx, dt, kd):

    nx = len(h)
    tint = h
    diag = np.zeros(nx)
    sup = np.zeros(nx)
    inf = np.zeros(nx)
    f = np.zeros(nx)

    for i in range(1, nx-1):
        factx = kd * dt / dx**2
        diag[i] = 1. + 2. * factx
        sup[i] = -factx
        inf[i] = -factx
        f[i] = tint[i]

    # for fixed boundary at i=0
    diag[0] = 1.
    sup[0] = 0.
    f[0] = tint[0]

    # for fixed boundary at i=nx
    diag[-1] = 1.
    inf[-1] = 0.
    f[-1] = tint[-1]
    res = tridiag(diag, inf, sup, f)
    
    return res

@jit
def advect1d (h, dx, dt, vx):

    nx = len(h)
    tint = h
    diag = np.zeros(nx)
    sup = np.zeros(nx)
    inf = np.zeros(nx)
    f = np.zeros(nx)

    for i in range(1, nx-1):
        diag[i] = 1. + vx * dt / dx
        sup[i] = 0.
        inf[i] = -vx * dt / dx
        f[i] = tint[i]

    diag[0] = 1.
    sup[0] = 0.
    f[0] = tint[0]
    diag[-1] = 1.
    inf[-1] = 0.
    f[-1] = tint[-1]
    res = tridiag (diag, inf, sup, f)

    return res