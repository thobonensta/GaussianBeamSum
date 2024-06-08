import numpy as np
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

def u_rect(f,r,c,theta,a,b):
    ''' Function that computes the SER of a rectangular plate of dim axb with phi=0'''
    Z0 = 377
    mu = 4*np.pi*1e-7
    omega = 2*np.pi*f
    lambda_sim = c/f
    k = 2*np.pi/lambda_sim

    return -1j*omega*mu/(2*np.pi*Z0)*np.exp(-1j*k*r)/r * a * b * np.sinc(a*k*np.sin(theta)/np.pi)*np.cos(theta)

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
    a = lambda_sim*5
    b = lambda_sim*5

    # Initialisation of the GSM method
    w0 = np.sqrt(2 * C0 / omega * r) # optimal choice for the half beam width



    # 3D with rectangular plate -- phi = 0 -- in test -> computation of F to check by comparing with ana or OP (seems ok up to a constant)
    N_phi = 361
    N_sum = 100
    step = np.pi  /4/ (N_sum - 1)
    phi_s = np.linspace(0, np.pi/4 , N_sum) # until pi/4 due to paraxial approximation
    theta_i = np.linspace(-np.pi / 2, np.pi / 2, N_phi)

    # Compute the propagation with the GBS method
    u_tot_3D = np.zeros(N_phi, dtype='complex')
    i = 0
    for theta in theta_i:
        F = -1j*2*omega*mu/Z0*a*b*np.sinc(a*k*np.sin(theta)/np.pi)*np.cos(theta)
        u_tmp = GSM_3D(f, r, C0, w0, phi_s , step)
        u_tot_3D[i] = F*u_tmp
        i += 1

    u_GB_dB_3D = 20*np.log10(np.abs(u_tot_3D)+1e-15)
    u_ex_rect = u_rect(f, r, C0, theta_i, a, b) # PO estimation of the RCS of a rectangular PEC target

    plt.figure()
    plt.grid()
    v_max = np.max(u_GB_dB_3D) + 1
    v_min = v_max - 100
    plt.ylabel(r'$u$ [dB]')
    plt.plot(theta_i*180/np.pi,u_GB_dB_3D ,label=r'$u^{GSM}$')
    plt.plot(theta_i * 180 / np.pi, 20 * np.log10(np.abs(u_ex_rect) + 1e-15),'--',label=r'PO solution')
    plt.legend()
    plt.ylim([v_min,v_max])

    RCS = 4 * np.pi * r ** 2 * np.abs(u_tot_3D) ** 2
    RCS_true = 4 * np.pi * r ** 2 * np.abs(u_ex_rect) ** 2

    RCS_dB = 10*np.log10(RCS)
    RCS_true_dB = 10*np.log10(RCS_true)

    # Retrieve the results obtained with FEKO
    with open("SERMetTarget5lambda.dat", 'r') as f:
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
    v_max = np.max(RCS_dB) + 1
    v_min = v_max - 60
    plt.ylabel(r'$RCS$ [dBm$^{2}$]')
    plt.xlabel(r'$\theta$ [$^{\circ}$]')
    plt.plot(theta_i*180/np.pi,RCS_dB ,'-',linewidth=1.5,label=r'$u^{GSM}$')
    plt.plot(angle, SER_FEKO_dB,':',linewidth=2,label=r'$FEKO$')
    plt.plot(theta_i * 180 / np.pi, RCS_true_dB,':',linewidth=2,label=r'PO solution')
    plt.legend()
    plt.ylim([v_min,v_max])

    # Case of dielectric target (rectangular)
    # Free space permittivity and thickness
    f = 1e9
    epsilon0 = 1 / (C0 ** 2 * mu)
    n1 = 1.0
    eps_2 = 20*epsilon0 - 1j * 0.02 / (2*np.pi*f)
    n2 = np.sqrt(eps_2)
    d = 0.04*lambda_sim

    # Compute the propagation using the GBS method and comparison with FEKO
    u_tot_3D_dielec = np.zeros(N_phi, dtype='complex')
    i=0
    for theta in theta_i:
        _,_,k_iz,k_tz = calculate_wavenumbers(f,theta,n1,n2,C0)
        # R12 = (n1*np.cos(theta)-n2*np.sqrt(1-(n1/n2)**2*np.sin(theta)**2))/(n1*np.cos(theta)+n2*np.sqrt(1-(n1/n2)**2*np.sin(theta)**2))
        R12 = Fresnel_coef(n1,n2,k_iz,k_tz,2)
        theta_t = np.arcsin(n1/n2*np.sin(theta))
        # print(R12)
        # _,_,k_iz,k_tz = calculate_wavenumbers(f,theta_t,n2,n1,C0)
        # print(Fresnel_coef(n2,n1,k_iz,k_tz,2))
        R21 = 1 #-R12 #1 #Fresnel_coef(n2,n1,k_iz,k_tz,2)
        beta = 2*np.pi*n2*d*np.sqrt(1-(n1/n2)**2*np.sin(theta)**2)/lambda_sim
        Req = (R12+R21*np.exp(-2*1j*beta))/(1+R12*R21*np.exp(-2*1j*beta))
        F = -1j*2*omega*(1+R12)/2*mu/Z0*a*b*np.sinc(a*k*np.sin(theta)/np.pi)*np.cos(theta)
        u_tmp = GSM_3D(f, r, C0, w0, phi_s , step)
        u_tot_3D_dielec[i] = F*u_tmp
        i += 1

    u_GB_dB_3D_dielec = 20*np.log10(np.abs(u_tot_3D_dielec)+1e-15)






    # RCS
    RCS_dielec = 4*np.pi*r**2*np.abs(u_tot_3D_dielec)**2

    RCS_dielec_dB = 10*np.log10(RCS_dielec)

    with open("SERDielecSheet5lambda.dat", 'r') as f:
        f.readline()
        f.readline()
        angle = []
        SER_FEKO_dielec_dB = []
        for line in f:
            data = line.split()
            ang = float(data[0])
            ser = float(data[1])
            angle.append(ang)
            SER_FEKO_dielec_dB.append(ser)


    plt.figure()
    plt.grid()
    v_max = np.max(RCS_dB) + 1
    v_min = v_max - 100
    plt.ylabel(r'$u$ [dB]')
    plt.plot(angle, SER_FEKO_dielec_dB,label=r'$FEKO|dielec$')
    plt.plot(theta_i*180/np.pi,RCS_dielec_dB ,'g:',label=r'$u^{GSM}$|dielec')
    plt.legend()
    plt.ylim([v_min,v_max])


    plt.show()




