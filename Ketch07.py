import numpy as np
from math import log, exp

def model_age(model_time, model_temperature, dpar=2.):
    cstd_length_reduction = 0.893
    reduced_lengths, first_node = length_reduction(model_time, model_temperature , dpar=dpar)
    secinmyr = 365.25 * 24. * 3600. * 1e6
    numTTNodes = model_time.shape[0]
    model_time = model_time * secinmyr

    ft_model_age = 0.
    for node in range(numTTNodes - 2):
        midLength = (reduced_lengths[node] + reduced_lengths[node+1]) / 2.
        ft_model_age += correct_observational_bias(midLength) * (model_time[node] - model_time[node+1])

    ft_model_age += correct_observational_bias(reduced_lengths[numTTNodes-2]) * (model_time[node] - model_time[node+1])
    ft_model_age /= cstd_length_reduction * secinmyr

    return ft_model_age

def correct_observational_bias(rcmod):
    MIN_OBS_RCMOD = 0.13
    if rcmod >= 0.765:
        correction =  1.600 * rcmod - 0.600
    elif rcmod >= MIN_OBS_RCMOD:
        correction = 9.205 * rcmod * rcmod - 9.157 * rcmod + 2.269
    else:
        correction =  0.0

    return correction

def length_reduction(time, temperature, dpar=2):
    secinmyr = 365.25 * 24. * 3600. * 1e6
    time = time * secinmyr;
    crmr0 = convert_Dpar_to_rmr0(dpar)
    numTTnodes = time.shape[0]
    reduced_lengths = np.zeros(numTTnodes - 1)

    c0 = 0.39528
    c1 = 0.01073
    c2 = -65.12969
    c3 = -7.91715
    a = 0.04672
    b = 0
    k = 1.04 - crmr0

    MIN_OBS_RCMOD = 0.13
    equivTotAnnLen = MIN_OBS_RCMOD**(1.0 / k) * (1.0 - crmr0) + crmr0
    equivTime = 0.
    tempCalc = log(1.0 / ((temperature[numTTnodes - 2] +  temperature[numTTnodes-1]) / 2.0))

    for node in range(numTTnodes-2, -1, -1):
        timeInt = time[node] - time[node + 1] + equivTime
        x1 = (log(timeInt) - c2) / (tempCalc - c3)
        x2 = (c0 + c1 * x1) ** (1.0 / a) + 1.0
        reduced_lengths[node] = 1.0 / x2

        if reduced_lengths[node] < equivTotAnnLen:
            reduced_lengths[node] = 0.

        if reduced_lengths[node] == 0. or node == 1:
            if node > 0:
                node += 1
            first_node = node
            
            for nodeB in range(first_node, numTTnodes - 1):
                if reduced_lengths[nodeB] < crmr0:
                    reduced_lengths[nodeB] = 0.0
                    first_node = nodeB
                else:
                    reduced_lengths[nodeB] = pow((reduced_lengths[nodeB] - crmr0) / (1.0 - crmr0), k)

            break

        if reduced_lengths[node] < 0.999:
            tempCalc = log(1.0 / ((temperature[node-1] + temperature[node]) / 2.0))
            equivTime = pow(1.0 / reduced_lengths[node] - 1.0, a)
            equivTime = (equivTime - c0) / c1
            equivTime = exp(equivTime * (tempCalc - c3) + c2)

    return reduced_lengths, first_node

def convert_Dpar_to_rmr0(dpar):
    if dpar <= 1.75:
        rmr0 = 0.84
    elif dpar >= 4.58:
        rmr0 = 0
    else:
        rmr0 = 0.84 * ((4.58 - dpar) / 2.98) ** 0.21
    
    return rmr0