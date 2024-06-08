import numpy as np
import matplotlib.pyplot as plt
import time


def gaussian_beam(phi,f,r,c,w0):
    ''' Function that computes the field from one gaussian beam of take-off angle phi
    The code is mostly based on the article of Kaissar Abboud et al. "Computing the Radar
    Cross-Section of Dielectric Targets Using the Gaussian Beam Summation Method", Remote
    Sensing, 15, 2023, and on the articles of Popov and Cerveny.
    The gaussian beam corresponds to a paraxial solution of the Helmholtz equation, the idea
    of this function is to compute the propagation of a one in its relative coordinates (s,n)
    Inputs :
    - phi : float (rad) - take off angle
    - f : float - simulation frequency
    - r : float or array - distance of the point of computation M
    - c : float - celerity of the wave
    - w0 : float - half beam width
    Output :
    - u : complex or C-array (depending on r) - field computation at r for phi
    '''
    # Compute omega (angular frequency)
    omega = 2*np.pi*f
    # Distance s
    s = r*np.cos(phi)
    # Distance n
    n = r*np.sin(phi)
    # Q0 (parameter of the gaussian beam)
    Q0 = -1j*omega*w0**2/(2*c)
    # Compute the amplitude of one gaussian beam
    ampli = np.sqrt(c/(Q0+s))
    # Compute the phase of a gaussian beam
    phase =-omega*(-1j/c*s-1j/(2*c)*n**2/(Q0+s))
    return ampli*np.exp(phase)

def GSM_2D(f,r,c,w0,phi_s,step):
    ''' Functions that compute the gaussian beam summation method in 2D, mostly based on the article of
    Kaissar Abboud et al., Popov and Cerveny. The idea is to sum all the contribution (quadrature - here using
    rectangles) of the Gaussian beams for different take off angle. This leads to a decomposition (the integral),
    then the function F is calculated using the stationary phase theorem and matched with an analytical or
    numerical results. The advantage of the GSM method is that with gaussian beam the caustic problem is avoided.
    Inputs :
    - f : float - simulation frequency
    - r : float or array - distance of the point of computation M
    - c : float - celerity of the wave
    - w0 : float - half beam width
    - phi_s : array of float (rad) - angle for the quadrature
    - step : float (rad) - step for the quadrature
    Output :
    - u : array - field at the position r computed with GSM
    '''
    # omega (angular frequency)
    omega = 2 * np.pi * f
    # Q0 (parameter of the beam)
    Q0 = -1j * omega * w0 ** 2 / (2 * c)
    # Function phi
    Phi = -1j / (4 * np.pi) * np.sqrt(Q0 / c)

    # Sum over the gaussian beam (quadrature of the integral)
    u_GBS = gaussian_beam(phi_s[0], f, r, c, w0)
    for i in range(1,len(phi_s)):
        phi = phi_s[i]
        u_GB = gaussian_beam(phi,f,r,c,w0)
        u_GBS += u_GB
    u_GBS*=Phi*step

    return u_GBS

def GSM_2D_simpson(f,r,c,w0,phi_s,step):
    ''' Functions that compute the gaussian beam summation method in 2D, mostly based on the article of
    Kaissar Abboud et al., Popov and Cerveny. The idea is to sum all the contribution (quadrature - here using
    Simpson method) of the Gaussian beams for different take off angle. This leads to a decomposition (the integral),
    then the function F is calculated using the stationary phase theorem and matched with an analytical or
    numerical results. The advantage of the GSM method is that with gaussian beam the caustic problem is avoided.
    Inputs :
    - f : float - simulation frequency
    - r : float or array - distance of the point of computation M
    - c : float - celerity of the wave
    - w0 : float - half beam width
    - phi_s : array of float (rad) - angle for the quadrature
    - step : float (rad) - step for the quadrature
    Output :
    - u : array - field at the position r computed with GSM
    '''
    # omega (angular frequency)
    omega = 2 * np.pi * f
    # Q0 (parameter of the beam)
    Q0 = -1j * omega * w0 ** 2 / (2 * c)
    # Function phi
    Phi = -1j / (4 * np.pi) * np.sqrt(Q0 / c)

    # Integration using Simpson rule
    u_GBS = gaussian_beam(phi_s[0], f, r, c, w0)
    for i in range(1,len(phi_s)):
        phi = phi_s[i]
        u_GB = gaussian_beam(phi, f, r, c, w0)
        if i == len(phi_s) - 1:
            u_GBS +=  u_GB
        else:
            if i % 2 == 0:
                u_GBS +=   2 * u_GB
            else:
                u_GBS +=   4 * u_GB

    u_GBS *=  step / 3*Phi
    return u_GBS

def u_ex(f,r,c):
    ''' Function that computes the exact propagation for a half dipole (test from Popov, Cerveny, and
    Kaissar-Abboud et al. articles)
    Inputs : frequency (f, float), radius (r, vector [Nx]) and celerity (c, float)
    Outputs : field according to ray theory (u, vector [Nx])
    '''
    omega = 2*np.pi*f
    return -1/4*np.sqrt(2*c/(np.pi*omega*np.abs(r)))*np.exp(-1j*omega*r/c+1j*np.pi/4)

def u_rect(f,r,c,theta,a):
    ''' Function that computes the SER of a rectangular plate of dim axb with phi=0'''
    Z0 = 377
    mu = 4*np.pi*1e-7
    omega = 2*np.pi*f
    lambda_sim = c/f
    k = 2*np.pi/lambda_sim

    return -omega*mu/(2*Z0)*np.exp(-1j*k*r)*np.sqrt(2/(np.pi*k*r)) * a  * np.sinc(a*k*np.sin(theta)/np.pi)*np.cos(theta)*np.exp(1j*np.pi/4)

if __name__ == '__main__':
    C0 = 3*1e8

    N_r = 101
    r = np.linspace(-100,100,N_r)
    f=1*1e9
    lambda_sim = C0/f
    kw = 2*np.pi/lambda_sim
    u_ray = u_ex(f,r,C0)


    # Test of convergence with w0 (the half beam width)
    N_sum = 400
    step = np.pi/2/(N_sum-1)
    phi_s = np.linspace(-np.pi/4,np.pi/4,N_sum)

    w0 = 5 * lambda_sim
    t_start = time.time()
    u_test = GSM_2D(f,r,C0,w0,phi_s,step)
    print('Time GSM numba = ', time.time()-t_start)
    w0 = 8 * lambda_sim
    t_start = time.time()
    u_test_2 = GSM_2D(f,r,C0,w0,phi_s,step)
    print('Time GSM numba = ', time.time()-t_start)

    t_start = time.time()
    u_test_3 = 0
    omega = 2 * np.pi * f
    w0 = 15 * lambda_sim
    for phi in phi_s:
        Q0 = -1j * omega * w0 ** 2 / (2 * C0)
        Phi = -1j/(4*np.pi)*np.sqrt(Q0/C0)
        u_GB = gaussian_beam(phi,f,r,C0,w0)
        u_test_3 = u_test_3 + Phi*u_GB*step
    print('Time GSM = ', time.time()-t_start)

    t_start = time.time()
    u_test_4 = GSM_2D_simpson(f,r,C0,w0,phi_s,step)
    print('Time GSM numba Simpson = ', time.time()-t_start)


    u_ray_dB = 20*np.log10(np.abs(u_ray)+1e-15)
    u_GB_dB = 20*np.log10(np.abs(u_test)+1e-15)
    u_GB_dB_2 = 20*np.log10(np.abs(u_test_2)+1e-15)
    u_GB_dB_3 = 20*np.log10(np.abs(u_test_3)+1e-15)
    u_GB_dB_4 = 20*np.log10(np.abs(u_test_4)+1e-15)

    # plt.figure()
    # plt.plot(r, u_ray_dB, label=r'$u_{ex}$')
    # plt.grid()
    # plt.xlabel('r [m]')
    # plt.ylabel(r'$u$ [dB]')
    # plt.plot(r,u_GB_dB,label=r'$u^{GSM}$|$\omega_0=5\lambda$')
    # plt.plot(r,u_GB_dB_2,':',label=r'$u^{GSM}$|$\omega_0=8\lambda$')
    # plt.plot(r,u_GB_dB_3,'--',label=r'$u^{GSM}$|$\omega_0=15\lambda$')
    # plt.plot(r,u_GB_dB_4,'--',label=r'$u^{GSM}$|$\omega_0=15\lambda$')
    # plt.legend()
    # plt.show()

    # Test for various incident angle (monostatic case) -- phi = 0
    r = 1000
    N_sum = 200
    step = np.pi  / (N_sum - 1)
    phi_s = np.linspace(-np.pi / 2, np.pi / 2, N_sum)
    N_phi = 181
    theta_i = np.linspace(-np.pi/2,np.pi/2,N_phi)

    w0 = np.sqrt(2*C0/omega*r) # optimal choice for a given distance for the half beam-width

    u_tot = np.zeros(N_phi,dtype='complex')
    u_ray = np.zeros(N_phi,dtype='complex')
    i = 0
    for theta in theta_i:
        u_tmp = GSM_2D(f, r, C0, w0, phi_s-theta, step)
        u_tot[i] = u_tmp
        u_ray[i] = u_ex(f,r,C0)
        i+=1

    plt.figure()
    plt.plot(theta_i*180/np.pi,20*np.log10(np.abs(u_tot)+1e-15),label='GSM')
    plt.plot(theta_i*180/np.pi,20*np.log10(np.abs(u_ray)+1e-15),label='ray method')
    plt.legend()
    plt.show()

    L = 10*lambda_sim
    print(L)

    N_semiana = 48
    L_mesh = np.linspace(0,L,N_semiana)
    u_ex_semi = np.zeros(N_phi, dtype='complex')

    # w0 = 3 * lambda_sim

    u_test = u_rect(f,r,C0,theta_i,L/2)
    i = 0
    for theta in theta_i:
        u_tmp = 0
        for k in range(N_semiana-1):
                Lx = L_mesh[k+1]-L_mesh[k]
                tx = Lx/2+L_mesh[k]
                r1 = np.sqrt((r-tx*np.sin(theta))**2)
                theta1 = np.arccos((r * np.cos(theta)) / r1)
                if np.abs(theta1) <= 1e-1:
                    theta1 = theta
                tmp = GSM_2D(f, r1, C0, w0, phi_s - theta1, step)
                Z0 = 377
                mu = 4 * np.pi * 1e-7
                u_ex_tmp1 = tmp *np.cos(theta1)*omega*mu/(2*Z0)*Lx*np.sinc(kw*Lx*np.sin(theta1)/np.pi)*2
                u_tmp += u_ex_tmp1
        u_ex_semi[i] = u_tmp
        i+=1


    u_op_dB = 20 * np.log10(np.abs(u_test) + 1e-15)
    u_gsm_dB = 20 * np.log10(np.abs(u_ex_semi) + 1e-15)
    plt.figure()
    plt.plot(theta_i * 180 / np.pi,u_op_dB ,'--',label='OP')
    plt.plot(theta_i * 180 / np.pi,u_gsm_dB,label='semi GSM')
    plt.legend()
    plt.show()