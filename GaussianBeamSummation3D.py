import numpy as np
import matplotlib.pyplot as plt

def ray_propagation(f,r,c):
    '''Function that computes the field in the ray theory approximation
    given by the Green function exp(jwr/c)/(4pir)

    Inputs : frequency (f, float), radius (r, vector [Nx]) and celerity (c, float)
    Outputs : field according to ray theory (u, vector [Nx])
    '''
    return np.exp(1j*2*np.pi*f*r/c)/(4*np.pi*r)

def gaussian_beam_3D(f,r,c,w0,phi):
    ''' Function that computes the field from one 3D Gaussian beam of take-off angle phi
    In this case the amplitude is already precomputed so as to have an equal value with
    ray propagation using the stationary phase theorem.
    Inputs :
    - f : float - simulation frequency
    - r : float or array - distance of the point of computation M
    - c : float - celerity of the wave
    - w0 : float - half beam width
    - phi : float (rad) - take off angle
    Output :
    - u : complex or C-array (depending on r) - field computation at r for phi
    '''
    # Compute omega (angular frequency)
    omega = 2 * np.pi * f
    # Distance s
    s = r * np.cos(phi)
    # Distance n
    n = r * np.sin(phi)
    # Q0 (parameter of the beam)
    Q0 = -1j * omega * w0 ** 2 / (2 * c)
    # ampli
    ampli = 1/(Q0+s)*(1j*omega*Q0)/(8*np.pi**2*c)
    # phase
    phase = -omega*(-1j/c*s-1j/(2*c)*n**2/(Q0+s))
    return ampli*np.exp(phase)

def GSM_3D(f,r,c,w0,phi_s,step):
    ''' Functions that compute the gaussian beam summation method in 3D, mostly based on the article of
    Kaissar Abboud et al.. The idea is to sum all the contribution (quadrature - here using
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

    # first term of the sum
    u_GBS = gaussian_beam_3D(f, r, c, w0, phi_s[0])*np.sin(phi_s[0])
    for i in range(1,len(phi_s)):
        phi = phi_s[i]
        u_GB = gaussian_beam_3D(f, r, c, w0, phi)
        u_GBS +=  u_GB* np.sin(phi)
    u_GBS*=step*2*np.pi
    return u_GBS

if __name__ == '__main__':
    # Initialisation of the simulation for the validation test
    C0 = 3*1e8
    f = 1e9
    omega = 2*np.pi*f
    lambda_sim = C0/f

    # Validation test for a given distance with w0 (the half beam width)
    N_sum = 400
    step = np.pi /2/ (N_sum-1)
    phi_s = np.linspace(0, np.pi/2, N_sum)

    N_r = 101
    r = np.linspace(-100, 100, N_r)
    w0 = 5 * lambda_sim
    u_test_3D = GSM_3D(f, r, C0, w0, phi_s, step)

    u_GB_dB_3D = 20*np.log10(np.abs(u_test_3D)+1e-15)

    u_ray = ray_propagation(f, r, C0)
    u_ray_dB = 20*np.log10(np.abs(u_ray)+1e-15)


    plt.figure()
    plt.plot(r, u_ray_dB, label=r'$u_{ex}$')
    plt.grid()
    plt.xlabel('r [m]')
    plt.ylabel(r'$u$ [dB]')
    plt.plot(r,u_GB_dB_3D ,label=r'$u^{GSM}$|$\omega_0=5\lambda$')
    plt.legend()
    plt.show()

    # Test for various incident angle (monostatic case) -- phi = 0
    r = 100
    N_phi = 180
    N_sum = 400
    step = np.pi /2/ (N_sum -1)
    phi_s = np.linspace(0, np.pi/2, N_sum) # pi/2 can be reduced to pi/4 (paraxial approximation)
    theta_i = np.linspace(-np.pi / 2, np.pi / 2, N_phi)
    u_tot_3D = np.zeros(N_phi, dtype='complex')
    u_ray_3D = np.zeros(N_phi, dtype='complex')

    w0 = np.sqrt(2 * C0 / omega * r)

    i = 0
    for theta in theta_i:
        u_tmp = GSM_3D(f, r, C0, w0, phi_s, step)
        u_tot_3D[i] =  u_tmp
        u_ray_3D[i] = ray_propagation(f, r, C0)
        i += 1

    u_GB_dB_3D = 20 * np.log10(np.abs(u_tot_3D) + 1e-15)
    u_ray_dB = 20 * np.log10(np.abs(u_ray_3D) + 1e-15)

    plt.figure()
    plt.plot(theta_i*180/np.pi, u_ray_dB, '--', label=r'$u_{ex}$')
    plt.grid()
    v_max = np.max(u_GB_dB_3D) + 1
    v_min = v_max - 100
    plt.ylabel(r'$u$ [dB]')
    plt.plot(theta_i * 180 / np.pi, u_GB_dB_3D, label=r'$u^{GSM}$|$\omega_0=5\lambda$')
    plt.legend()
    plt.ylim([v_min, v_max])
    plt.show()
