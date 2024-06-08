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

def u_polygon(f,vertices,r,c,theta,phi,nsides):
    theta_points = len(theta)
    z_0 = 376.7343
    mu = 4 * np.pi * 1e-7
    omega = 2*np.pi*f
    wavenumber = omega/c

    alpha_n = np.zeros((nsides, 3))
    for n in range(nsides):
        if n == nsides - 1:
            alpha_n[nsides - 1] = vertices[0] - vertices[nsides - 1]
        else:
            alpha_n[n] = vertices[n + 1] - vertices[n]
        alpha_n[n] /= np.linalg.norm(alpha_n[n])

    normal_vect = np.array([0, 0, 1])

    es = np.zeros(theta_points, dtype=np.complex)
    for i_theta in range(theta_points):
        theta_vect = np.array([
            np.cos(theta[i_theta]) * np.cos(phi),
            np.cos(theta[i_theta]) * np.sin(phi),
            -np.sin(theta[i_theta])
        ])
        phi_vect = np.array([
            -np.sin(phi),
            np.cos(phi),
            0
        ])
        w_vect = np.array([
            2 * wavenumber * np.sin(theta[i_theta]) * np.cos(phi),
            2 * wavenumber * np.sin(theta[i_theta]) * np.sin(phi),
            0
        ])
        s_term = 0.0
        for n in range(nsides):
            expterm = np.exp(1j * np.dot(w_vect, vertices[n]))
            if n == 0:
                num = np.dot(np.cross(normal_vect, alpha_n[n]), alpha_n[nsides - 1])
                denom = np.dot(w_vect, alpha_n[n]) * np.dot(w_vect, alpha_n[nsides - 1])
            else:
                num = np.dot(np.cross(normal_vect, alpha_n[n]), alpha_n[n - 1])
                denom = np.dot(w_vect, alpha_n[n]) * np.dot(w_vect, alpha_n[n - 1])
            s_term += num * expterm / denom
        vect_term = np.dot(theta_vect, np.cross(phi_vect, normal_vect))
        es[i_theta] = -1j * omega * mu / (2 * np.pi * z_0) * vect_term * s_term * np.exp(-1j * wavenumber * r) / r

    return es

def sigma_ana_iso(f,c,thetai,phi,a,b):
    lambda0 = c / f
    k = 2 * np.pi / lambda0

    alpha = k*a*np.sin(thetai)*np.cos(phi)
    beta = k*b*np.sin(thetai)*np.sin(phi)
    A = a*b/2
    if phi == 0:
        sigma0 = np.sin(alpha)**4/alpha**4+(np.sin(2*alpha)-2*alpha)**2/(4*alpha**4)
    elif phi==np.pi/2:
        sigma0 = np.sin(beta/2)**4/(beta/2)**4
    else:
        sigma01 = 0.25*(np.sin(phi)**2)*(2*a/b*np.cos(phi)*np.sin(beta)-np.sin(phi)*np.sin(2*alpha))**2
        sigma0 = ((np.sin(alpha)**2-np.sin(beta/2)**2)**2+sigma01)/(alpha**2-(beta/2)**2)

    return 4*np.pi*A**2/lambda_sim**2*np.cos(thetai)**2*sigma0



if __name__ == '__main__':
    # Initialisation of the simulation
    C0 = 299795645
    f = 1e9
    omega = 2*np.pi*f
    lambda_sim = C0/f
    Z0 = 377
    mu = 4 * np.pi * 1e-7
    k = 2*np.pi/lambda_sim

    r = 100 # Distance of the observation point

    nsides = 3
    vert = np.zeros((3,3))
    a = 5*lambda_sim
    b = 7*lambda_sim
    vert[0,:] = np.array((0,-b/2,0))
    vert[1,:] = np.array((a,0.0,0.0))
    vert[2,:] = np.array((0,b/2,0.0))
    #
    vert[0,:] = np.array((0,0,0))
    vert[1,:] = np.array((1.0,0.5,0.0))
    vert[2,:] = np.array((1.5,0.0,0.0))


    N_phi = 180
    thetai = np.linspace(-np.pi / 2, np.pi / 2, N_phi)
    phi =  0 #np.pi/2 #0.0001*np.pi/180

    u = u_polygon(f, vert, r, C0, thetai, phi, nsides)

    # Compute the propagation with the GBS method
    N_sum = 400
    step = np.pi / 4 / (N_sum - 1)
    phi_s = np.linspace(0, np.pi / 4, N_sum)

    w0 = np.sqrt(2 * C0 / omega * r)  # optimal choice for the half beam width

    u_tot_3D = np.zeros(N_phi, dtype='complex')
    i = 0
    for theta in thetai:
        u_tmp = GSM_3D(f, r, C0, w0, phi_s , step)
        u_tot_3D[i] = u_tmp
        i += 1
    F = u_polygon(f, vert, r, C0, thetai, phi, nsides)
    u_tot_3D = 4 * np.pi * r * np.exp(1j * k * r) * F*u_tot_3D

    RCS = 4 * np.pi * r ** 2 * np.abs(u_tot_3D) ** 2

    rcs = 4*np.pi*r**2*np.abs(u)**2

    with open("SERMetPolyDirect.dat", 'r') as f:
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
    # print(u)

    # rcs_ana = sigma_ana_iso(f,C0,thetai,0,a,b)

    rcs_dB = 10*np.log10(rcs+1e-15)
    # rcs_ana_dB = 20*np.log10(rcs_ana+1e-15)

    plt.figure()
    plt.grid()
    v_max = np.max(rcs_dB) + 1
    v_min = v_max - 80
    plt.plot(thetai * 180 / np.pi, 10*np.log10(RCS+1e-15), '-', label=r'GBS')
    # plt.plot(thetai * 180 / np.pi, rcs_dB, ':', label=r'PO solution')
    plt.plot(angle, SER_FEKO_dB,':',linewidth=2, label=r'$FEKO$')
    plt.ylabel(r'$RCS$ [dBm$^{2}$]')
    plt.xlabel(r'$\theta$ [$^{\circ}$]')

    plt.legend()
    plt.ylim([v_min, v_max])
    plt.show()
