import numpy as np
### ------ Specification ------ ###
# Version 3, predefined 8 ECMs + 1 new (8.10)
# Use CPE, no inductor; warburg using normalized impedance formula
#
# CM1 : R0 + (R1 || CPE1)
# CM2 : R0 + (R1 || CPE1) + (R2 || CPE2)
# CM3 : R0 + (R1 || CPE1) + (R2 || CPE2) + (R3 || CPE3)
# CM4 : R0 + (R1 || CPE1) + (R2 || CPE2) + (R3 || CPE3) + (R4 || CPE4)

# CM5 : R0 + ( (R1 + AW) || CPE1)
# CM6 : R0 + (R1 || CPE1) + ((R2 + AW) || CPE2)
# CM7 : R0 + (R1 || CPE1) + (R2 || CPE2) + ((R3 + AW) || CPE3)
# CM8 : R0 + (R1 || CPE1) + (R2 || CPE2) + (R3 || CPE3) + ((R4 + AW) || CPE4)
# CM9 : R0 + (R1 || CPE1) + (R2 || CPE2) + (R3 || CPE3) + Aw



def compute_v3CM1_impedance(params, angular_freq):
    R0_val, R1_val, C1_val, n1_val = params
    
    ZR0 = R0_val
    ZR1 = R1_val
    ZC1 =  1/(C1_val * (angular_freq*1j)**n1_val)
    
    ZRC1 = 1 / (1/ZR1 + 1/ZC1)
    Zsum = ZR0 + ZRC1
    return Zsum

def compute_v3CM2_impedance(params, angular_freq):
    R0_val, R1_val, R2_val, C1_val, n1_val, C2_val, n2_val = params
    
    ZR0 = R0_val
    ZR1 = R1_val
    ZR2 = R2_val
    ZC1 =  1/(C1_val * (angular_freq*1j)**n1_val)
    ZC2 =  1/(C2_val * (angular_freq*1j)**n2_val)

    ZRC1 = 1 / (1/ZR1 + 1/ZC1)
    ZRC2 = 1 / (1/ZR2 + 1/ZC2)

    Zsum = ZR0 + ZRC1 + ZRC2
    return Zsum

def compute_v3CM3_impedance(params, angular_freq):
    R0_val, R1_val, R2_val, R3_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val = params
    
    ZR0 = R0_val
    ZR1 = R1_val
    ZR2 = R2_val
    ZR3 = R3_val
    ZC1 =  1/(C1_val * (angular_freq*1j)**n1_val)
    ZC2 =  1/(C2_val * (angular_freq*1j)**n2_val)
    ZC3 =  1/(C3_val * (angular_freq*1j)**n3_val)
    
    ZRC1 = 1 / (1/ZR1 + 1/ZC1)
    ZRC2 = 1 / (1/ZR2 + 1/ZC2)
    ZRC3 = 1 / (1/ZR3 + 1/ZC3)

    Zsum = ZR0 + ZRC1 + ZRC2 + ZRC3
    return Zsum

def compute_v3CM4_impedance(params, angular_freq):
    R0_val, R1_val, R2_val, R3_val, R4_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val, C4_val, n4_val = params
    
    ZR0 = R0_val
    ZR1 = R1_val
    ZR2 = R2_val
    ZR3 = R3_val
    ZR4 = R4_val

    ZC1 =  1/(C1_val * (angular_freq*1j)**n1_val)
    ZC2 =  1/(C2_val * (angular_freq*1j)**n2_val)
    ZC3 =  1/(C3_val * (angular_freq*1j)**n3_val)
    ZC4 =  1/(C4_val * (angular_freq*1j)**n4_val)

    ZRC1 = 1 / (1/ZR1 + 1/ZC1)
    ZRC2 = 1 / (1/ZR2 + 1/ZC2)
    ZRC3 = 1 / (1/ZR3 + 1/ZC3)
    ZRC4 = 1 / (1/ZR4 + 1/ZC4)

    Zsum = ZR0 + ZRC1 + ZRC2 + ZRC3 + ZRC4
    return Zsum


def compute_v3CM5_impedance(params, angular_freq):
    R0_val, R1_val, C1_val, n1_val, sigma_val = params
    
    ZR0 = R0_val
    ZR1 = R1_val
    ZC1 =  1/(C1_val * (angular_freq*1j)**n1_val)
    Zw = ( sigma_val * np.sqrt(2) ) / np.sqrt(1j*angular_freq)
    
    ZRCAW1 = 1 / (1/(ZR1 + Zw) + 1/ZC1)
    Zsum = ZR0 + ZRCAW1
    return Zsum

def compute_v3CM6_impedance(params, angular_freq):
    R0_val, R1_val, R2_val, C1_val, n1_val, C2_val, n2_val, sigma_val = params
    
    ZR0 = R0_val
    ZR1 = R1_val
    ZR2 = R2_val
    ZC1 =  1/(C1_val * (angular_freq*1j)**n1_val)
    ZC2 =  1/(C2_val * (angular_freq*1j)**n2_val)
    Zw = ( sigma_val * np.sqrt(2) ) / np.sqrt(1j*angular_freq)

    ZRC1 = 1 / (1/ZR1 + 1/ZC1)
    ZRCAW2 = 1 / (1/(ZR2 + Zw) + 1/ZC2)

    Zsum = ZR0 + ZRC1 + ZRCAW2
    return Zsum


def compute_v3CM7_impedance(params, angular_freq):
    R0_val, R1_val, R2_val, R3_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val, sigma_val = params
    
    ZR0 = R0_val
    ZR1 = R1_val
    ZR2 = R2_val
    ZR3 = R3_val
    ZC1 =  1/(C1_val * (angular_freq*1j)**n1_val)
    ZC2 =  1/(C2_val * (angular_freq*1j)**n2_val)
    ZC3 =  1/(C3_val * (angular_freq*1j)**n3_val)
    Zw = ( sigma_val * np.sqrt(2) ) / np.sqrt(1j*angular_freq)

    
    ZRC1 = 1 / (1/ZR1 + 1/ZC1)
    ZRC2 = 1 / (1/ZR2 + 1/ZC2)
    ZRCAW3 = 1 / (1/(ZR3 + Zw) + 1/ZC3)

    Zsum = ZR0 + ZRC1 + ZRC2 + ZRCAW3
    return Zsum


def compute_v3CM8_impedance(params, angular_freq):
    R0_val, R1_val, R2_val, R3_val, R4_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val, C4_val, n4_val, sigma_val = params
    
    ZR0 = R0_val
    ZR1 = R1_val
    ZR2 = R2_val
    ZR3 = R3_val
    ZR4 = R4_val

    ZC1 =  1/(C1_val * (angular_freq*1j)**n1_val)
    ZC2 =  1/(C2_val * (angular_freq*1j)**n2_val)
    ZC3 =  1/(C3_val * (angular_freq*1j)**n3_val)
    ZC4 =  1/(C4_val * (angular_freq*1j)**n4_val)

    Zw = ( sigma_val * np.sqrt(2) ) / np.sqrt(1j*angular_freq)

    ZRC1 = 1 / (1/ZR1 + 1/ZC1)
    ZRC2 = 1 / (1/ZR2 + 1/ZC2)
    ZRC3 = 1 / (1/ZR3 + 1/ZC3)
    ZRCAW4 = 1 / (1/(ZR4 + Zw) + 1/ZC4)

    Zsum = ZR0 + ZRC1 + ZRC2 + ZRC3 + ZRCAW4
    return Zsum


def compute_v3CM9_impedance(params, angular_freq):
    R0_val, R1_val, R2_val, R3_val, C1_val, n1_val, C2_val, n2_val, C3_val, n3_val, sigma_val = params
    
    ZR0 = R0_val
    ZR1 = R1_val
    ZR2 = R2_val
    ZR3 = R3_val

    ZC1 =  1/(C1_val * (angular_freq*1j)**n1_val)
    ZC2 =  1/(C2_val * (angular_freq*1j)**n2_val)
    ZC3 =  1/(C3_val * (angular_freq*1j)**n3_val)

    Zw = ( sigma_val * np.sqrt(2) ) / np.sqrt(1j*angular_freq)

    ZRC1 = 1 / (1/ZR1 + 1/ZC1)
    ZRC2 = 1 / (1/ZR2 + 1/ZC2)
    ZRC3 = 1 / (1/ZR3 + 1/ZC3)
   
    Zsum = ZR0 + ZRC1 + ZRC2 + ZRC3 + Zw
    return Zsum



# ReZ = np.real(Zsum)
# neg_ImZ = -np.imag(Zsum)

