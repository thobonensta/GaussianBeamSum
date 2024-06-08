import numpy as np
import matplotlib.pyplot as plt

def u_rect(f,r,c,theta,phi,a,b):
    ''' Function that computes the SER of a rectangular plate of dim axb with phi=0'''
    lambda_sim = c/f
    k = 2*np.pi/lambda_sim

    return -1j/lambda_sim*np.exp(-1j*k*r)/r * a * b * np.sinc(a*k*np.sin(theta)*np.cos(phi)/np.pi)* np.sinc(b*k*np.sin(theta)*np.sin(phi)/np.pi)*np.cos(theta)*np.cos(phi)


if __name__ == '__main__':
    # Initialisation of the simulation
    C0 = 3*1e8
    f = 3e9
    omega = 2*np.pi*f
    lambda_sim = C0/f
    Z0 = 377
    mu = 4 * np.pi * 1e-7
    k = 2*np.pi/lambda_sim

    r = 100*lambda_sim # Distance of the observation point

    # Initialisation of the PEC plate
    a = lambda_sim*5
    b = lambda_sim*5



    # 3D with rectangular plate -- phi = 0 -- in test -> computation of F to check by comparing with ana or OP (seems ok up to a constant)
    N_phi = int(361)
    theta_i_ana = np.linspace(-np.pi / 2, np.pi / 2, N_phi)
    theta_i = np.linspace(-np.pi / 2, np.pi / 2, N_phi)

    # Compute the propagation with the PO
    phi = 0*np.pi/180

    u_ex_semi = np.zeros(N_phi, dtype='complex')
    i=0

    N_semi = 48
    x_disc = np.linspace(-a/2,a/2,N_semi)
    #print(x_disc)
    y_disc = np.linspace(-b/2,b/2,N_semi)
    xv,yv = np.meshgrid(x_disc,y_disc)
    Lx = x_disc[1]-x_disc[0]
    print(Lx)
    Ly = y_disc[1]-y_disc[0]
    print(Ly)
    u_ex_rect = u_rect(f, r, C0, theta_i_ana, phi,a/2, b*2) # PO estimation of the RCS of a rectangular PEC target

    for theta in theta_i:
        u_tmp = 0
        for k in range(len(x_disc)-1):
            for p in range(len(y_disc)-1):
                tx = (x_disc[k+1]+x_disc[k])/2
                Lx = x_disc[k+1]-x_disc[k]
                # print(tx)
                ty = (y_disc[p+1]+y_disc[p])/2
                Ly = y_disc[p+1]-y_disc[p]
                # print(ty)
                r1 = np.sqrt((r*np.sin(theta)*np.cos(phi)-tx)**2+(r*np.sin(theta)*np.sin(phi)-ty)**2+(r*np.cos(theta))**2)
                theta1 = np.arccos((r * np.cos(theta)) / r1)
                phi1 = np.arctan((r*np.sin(theta)*np.sin(phi)-ty)/(r*np.sin(theta)*np.cos(phi)-tx))
                if np.abs(theta1) <= 1e-1:
                    theta1 = theta
                    phi1 = phi
                u_ex_tmp1 = u_rect(f, r1, C0, theta1,phi1, Lx, Ly) # PO estimation of the RCS of a rectangular PEC target
                u_tmp += u_ex_tmp1
        u_ex_semi[i] = u_tmp
        i+=1

    # test =  u_rect(f, r, C0, theta_i,phi, Lx, b) # PO estimation of the RCS of a rectangular PEC target


    plt.figure()
    plt.grid()
    v_max = np.max(20 * np.log10(np.abs(u_ex_rect) + 1e-15)) + 1
    v_min = v_max - 100
    plt.ylabel(r'$u$ [dB]')
    plt.plot(theta_i_ana * 180 / np.pi, 20 * np.log10(np.abs(u_ex_rect) + 1e-15),'--',label=r'PO solution')
    plt.plot(theta_i * 180 / np.pi, 20 * np.log10(np.abs(u_ex_semi) + 1e-15),'--',label=r'semi-analytic PO')
    # plt.plot(theta_i * 180 / np.pi, 20 * np.log10(np.abs(test) + 1e-15),'--',label=r'test')

    plt.legend()
    plt.ylim([v_min,v_max])
    plt.show()