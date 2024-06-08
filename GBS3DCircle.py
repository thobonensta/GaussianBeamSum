import numpy as np
from scipy.special import j1
import matplotlib.pyplot as plt
from GaussianBeamSummation3D import GSM_3D

def calculate_wavenumbers(freq0,theta_inc,n1,n2,c0):

    k_1 = 2.0*np.pi*freq0/c0 * n1    # wavenumber in medium 1
    k_2 = (2.0*np.pi*freq0/c0) * n2   # wavenumber in medium 2
    k_iz = - k_1*np.cos(theta_inc)  # incident wavenumber along z
    k_tx = k_1*np.sin(theta_inc)   # incident/transmitted wavenumber along x (k_tx = k_ix)
    k_tz = np.sqrt(k_2**2-k_tx**2) # transmitted wavenumber along z. sqrt with positive imag part
    if np.imag(k_tz)<0:
        k_tz = - k_tz
    return k_1,k_2,k_iz,k_tz

def Fresnel_coef(n1,n2,k_iz,k_tz,polarisation):
    if polarisation == 1:
        R_fresnel = ( n2**2*k_iz - n1**2*k_tz ) / ( n2**2*k_iz + n1**2*k_tz )
    elif polarisation == 2:
        R_fresnel = ( k_iz - k_tz ) / ( k_iz + k_tz )
    return R_fresnel

def u_circle(f,r,c,theta,rcircle):
    ''' Function that computes the SER of a rectangular plate of dim axb with phi=0'''
    lambda_sim = c/f
    k = 2*np.pi/lambda_sim
    i=0
    u = np.zeros(len(theta),dtype='complex')
    for theta_i in theta:
        if theta_i == 0:
            tmp = np.exp(-1j*k*r)/r *np.pi*rcircle**2/lambda_sim
        else:
            tmp = np.exp(-1j*k*r)/r *np.pi*rcircle**2/lambda_sim*2*j1(2*k*rcircle*np.sin(theta_i))/(2*k*rcircle*np.sin(theta_i))*np.cos(theta_i)
        u[i] = tmp
        i+=1
    return u

if __name__ == '__main__':
    # Initialisation of the simulation
    C0 = 299792458
    f = 1e9
    omega = 2*np.pi*f
    lambda_sim = C0/f
    Z0 = 377
    mu = 4 * np.pi * 1e-7
    k = 2*np.pi/lambda_sim

    r = 100 # Distance of the observation point

    # Initialisation of the PEC plate
    xc = 0
    yc = 0
    rcircle = lambda_sim*5

    # Initialisation of the GSM method
    w0 = np.sqrt(2 * C0 / omega * r) # optimal choice for the half beam width



    # 3D with rectangular plate -- phi = 0 -- in test -> computation of F to check by comparing with ana or OP (seems ok up to a constant)
    N_phi = 361
    N_sum = 400
    step = np.pi  /4/ (N_sum - 1)
    phi_s = np.linspace(0, np.pi/4 , N_sum) # until pi/4 due to paraxial approximation
    theta_i = np.linspace(-np.pi / 2, np.pi / 2, N_phi)

    # Compute the propagation with the GBS method
    u_tot_3D = np.zeros(N_phi, dtype='complex')
    i = 0
    for theta in theta_i:
        if theta == 0:
            F = 4*np.pi*np.pi*rcircle**2/lambda_sim
        else:
            F = 4*np.pi*np.pi*rcircle**2/lambda_sim*2*j1(2*k*rcircle*np.sin(theta))/(2*k*rcircle*np.sin(theta))*np.cos(theta)
        u_tmp = GSM_3D(f, r, C0, w0, phi_s , step)
        u_tot_3D[i] = F*u_tmp
        i += 1

    u_GB_dB_3D = 20*np.log10(np.abs(u_tot_3D)+1e-15)
    u_ex_rect = u_circle(f, r, C0, theta_i, rcircle) # PO estimation of the RCS of a rectangular PEC target

    plt.figure()
    plt.grid()
    v_max = np.max(20 * np.log10(np.abs(u_ex_rect) + 1e-15)) + 1
    v_min = v_max - 100
    plt.ylabel(r'$u$ [dB]')
    plt.plot(theta_i*180/np.pi,u_GB_dB_3D ,label=r'$u^{GSM}$|$\omega_0=5\lambda$')
    plt.plot(theta_i * 180 / np.pi, 20 * np.log10(np.abs(u_ex_rect) + 1e-15),'--',label=r'PO solution')
    plt.legend()
    plt.ylim([v_min,v_max])

    RCS = 4 * np.pi * r ** 2 * np.abs(u_tot_3D) ** 2
    RCS_true = 4 * np.pi * r ** 2 * np.abs(u_ex_rect) ** 2

    RCS_dB = 10*np.log10(RCS)
    RCS_true_dB = 10*np.log10(RCS_true)

    with open("SERMetCircle5lambda.dat", 'r') as f:
        f.readline()
        f.readline()
        angle = []
        SER_FEKO_dB = []
        for line in f:
            data = line.split()
            ang = float(data[0])
            ser = float(data[1])
            angle.append(ang)
            SER_FEKO_dB.append(ser)

    plt.figure()
    plt.grid()
    v_max = np.max(RCS_true_dB) + 1
    v_min = v_max - 100
    plt.ylabel(r'$u$ [dB]')
    plt.plot(theta_i*180/np.pi,RCS_dB ,linewidth=2,label=r'$u^{GSM}$|$\omega_0=5\lambda$')
    plt.plot(angle, SER_FEKO_dB, label=r'$FEKO$')
    plt.plot(theta_i * 180 / np.pi, RCS_true_dB,'--',label=r'PO solution')
    plt.legend()
    plt.ylim([v_min,v_max])



    plt.show()




