import numpy as np
import tensorflow as tf

DTYPE='float32'

@tf.function
def solve_ft_age(model_time, model_temperature, dpar=2.):
    geotime = model_time[-1] - model_time
    temperature = model_temperature + 273.15
    return model_age(geotime, temperature, dpar=dpar)

def model_age(model_time, model_temperature, dpar=2.):
    cstd_length_reduction = 0.893
    reduced_lengths = length_reduction(model_time, model_temperature , dpar=dpar)
    secinmyr = 365.25 * 24. * 3600. * 1e6
    # numTTNodes = model_time.shape[0]
    model_time = model_time * secinmyr

    midLengths = (reduced_lengths[:-1] + reduced_lengths[1:]) / 2.
    obs_bias = correct_observational_bias(midLengths)
    model_dt = model_time[:-2] - model_time[1:-1]
    ft_model_age = tf.reduce_sum(obs_bias * model_dt)
    ft_model_age += correct_observational_bias(reduced_lengths[-1]) * (model_time[-2] - model_time[-1])
    ft_model_age /= cstd_length_reduction * secinmyr

    return ft_model_age

def length_reduction(time, temperature, dpar=2):
    secinmyr = tf.constant(365.25 * 24. * 3600. * 1e6, dtype=DTYPE)
    time = time * secinmyr;
    crmr0 = convert_Dpar_to_rmr0(dpar)
    numTTnodes = time.shape[0]
    reduced_lengths = []

    c0 = tf.constant(0.39528, dtype=DTYPE)
    c1 = tf.constant(0.01073, dtype=DTYPE)
    c2 = tf.constant(-65.12969, dtype=DTYPE)
    c3 = tf.constant(-7.91715, dtype=DTYPE)
    a = tf.constant(0.04672, dtype=DTYPE)
    k = 1.04 - crmr0

    MIN_OBS_RCMOD = 0.13
    equivTotAnnLen = MIN_OBS_RCMOD**(1.0 / k) * (1.0 - crmr0) + crmr0
    equivTime = tf.constant(0., dtype=DTYPE)
    tempCalc = tf.math.log(1.0 / ((temperature[-2] +  temperature[-1]) / 2.0))

    for node in range(numTTnodes-2, -1, -1):
        timeInt = time[node] - time[node + 1] + equivTime
        x1 = (tf.math.log(timeInt) - c2) / (tempCalc - c3)
        x2 = (c0 + c1 * x1) ** (1.0 / a) + 1.0
        rl = 1.0 / x2
            
        rl = tf.where(rl < equivTotAnnLen, 0., rl)

        if rl < 0.999 and rl > 0.:
            tempCalc = tf.math.log(1.0 / ((temperature[node-1] + temperature[node]) / 2.0))
            equivTime = tf.math.pow(1.0 / rl - 1.0, a)
            equivTime = (equivTime - c0) / c1
            equivTime = tf.math.exp(equivTime * (tempCalc - c3) + c2)
            
        reduced_lengths = tf.concat([reduced_lengths, [rl]], axis=0)
        
    reduced_lengths = reduced_lengths[::-1]
    reduced_lengths = tf.where(reduced_lengths < crmr0, 0.,
                               tf.math.pow(tf.math.abs(reduced_lengths - crmr0) / (1.0 - crmr0), k))

    return reduced_lengths

def correct_observational_bias(rcmod):
    MIN_OBS_RCMOD = 0.13
    return tf.reduce_sum([tf.where(rcmod >= 0.765, 1.6 * rcmod - 0.6, 0.),
                          tf.where((rcmod >= MIN_OBS_RCMOD) & (rcmod < 0.765),
                                   9.205 * rcmod * rcmod - 9.157 * rcmod + 2.269, 0.)
                         ], axis=0)

def convert_Dpar_to_rmr0(dpar):
    return tf.reduce_sum([tf.where(dpar <= 1.75, 0.84, 0.),
                          tf.where((dpar > 1.75) & (dpar <= 4.58), 0.84 * ((4.58 - dpar) / 2.98) ** 0.21, 0.)
                         ], axis=0)