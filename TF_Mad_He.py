import numpy as np
import tensorflow as tf
# from math import log, exp
# from numba import jit

DTYPE='float32'

# @jit
# @tf.function
# def mad_age(time, temperature, method='AHe', beta=2, grain_radius=60):
#     """
#     Production-diffusion-cooling algorithm (Mad_He.f90) as documented in
#     Braun et al., 2006 (Quantitative Thermochronology. Cambridge Uni. Press)

#     INPUT PARAMETERS:
#     time                  vector of times before present [Ma]
#     temperature           vector of corresponding temperatures [C]
#     dt                    time step [Ma]
#     beta                  geometrical factor (1=cylindrical, 2=spherical)

#     OUTPUT PARAMETERS:
#     out                   vector of apparent ages [Ma]
#     """        
#     n=50
#     D = tf.Variable(np.ones(n), shape=(n,), dtype=DTYPE)
#     L = tf.Variable(np.zeros(n), shape=(n,), dtype=DTYPE)
#     U = tf.Variable(np.zeros(n), shape=(n,), dtype=DTYPE)
#     xf = tf.Variable(np.zeros(n), shape=(n,), dtype=DTYPE)
    
#     return solve_age(time, temperature, D, L, U, xf, method=method, beta=beta, grain_radius=grain_radius)
    
@tf.function
def solve_age(time, temperature, D, L, U, xf, method='AHe', beta=2, grain_radius=60):
    # D0a2, Ea = get_D0a2(method, grain_radius=grain_radius)
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
        
    realmin = tf.math.nextafter(0, 1)
    Ma = 365.25*24.*3600.*1e6
    R = 8.3144621

    # n=50        # number of spatial bins
    n = D.shape[0]
    ntime=len(time)   # number of temporal bins
    alpha=0.5            # time marching parameter (explicit-implicit scheme)
    dr = 1/(n-1)
        
    # D = tf.Variable(np.ones(n), shape=(n,), dtype=DTYPE)
    # L = tf.Variable(np.zeros(n), shape=(n,), dtype=DTYPE)
    # U = tf.Variable(np.zeros(n), shape=(n,), dtype=DTYPE)
    # xf = tf.Variable(np.zeros(n), shape=(n,), dtype=DTYPE)
    age = tf.constant(0., shape=(n,), dtype=DTYPE)
    ii = np.arange(1, n-1)

    temp = temperature[0]+273.15
    da2now = D0a2*Ma*tf.math.exp(-Ea/R/temp)
    for itime in range(1, ntime):
        age_i = age[1:n-1]
        age_iplus1 = age[2:]
        age_iminus1 = age[:-2]
        dt = time[itime-1]-time[itime]
        da2then = da2now
        temp = temperature[itime]+273.15
        da2now = D0a2*Ma*tf.math.exp(-Ea/R/temp)
        
        f1 = alpha*dt*da2now/dr**2
        f2 = (1-alpha)*dt*da2then/dr**2

            # compose the tridiagonal matrix and send it to decomposition
            
        D[1:n-1].assign(tf.ones(n-2) * (1 + 2 * f1))
        L[1:n-1].assign(tf.ones(n-2) * (-f1*(1-beta/ii/2)))
        U[1:n-1].assign(tf.ones(n-2) * (-f1*(1+beta/ii/2)))
        xf[1:n-1].assign(age_i + dt +
                        f2 * (age_iplus1 - 2 * age_i + age_iminus1 +
                               beta * (age_iplus1 - age_iminus1)/ii/2))

        diagonals = tf.stack([U, D, L])
        age = tf.linalg.tridiagonal_solve(diagonals, xf)

    # average solution over the grain's volume using trapezoidal rule
    fact = np.ones(n)
    fact[0] = .5
    fact[-1] = .5
    
    iii = np.arange(0, n)
    agei = 3 * tf.reduce_sum(age * fact * dr**3 * iii**2)
        
    return agei, age