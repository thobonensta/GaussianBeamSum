import numpy as np
import matplotlib.pyplot as plt

frequency = 1.0e9
c = 299795645.0
w = 2 * np.pi * frequency
r = 100.0
wavenumber = w / c
mu = 4.0 * np.pi * 1.0e-7
z_0 = 376.7343
nsides = 3
vertices = np.zeros((nsides, 3))
vertices[0] = [0.0, 0.0, 0.0]
vertices[1] = [1.0, 0.5, 0.0]
vertices[2] = [1.5, 0.0, 0.0]

alpha_n = np.zeros((nsides, 3))
for n in range(nsides):
    if n == nsides - 1:
        alpha_n[nsides - 1] = vertices[0] - vertices[nsides - 1]
    else:
        alpha_n[n] = vertices[n + 1] - vertices[n]
    alpha_n[n] /= np.linalg.norm(alpha_n[n])

normal_vect = np.array([0, 0, 1])
theta_points = 180
theta = np.linspace(-0.5 * np.pi, 0.5 * np.pi, theta_points)
phi = 0

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
    es[i_theta] = -1j * w * mu / (2 * np.pi * z_0) * vect_term * s_term * np.exp(-1j * wavenumber * r) / r

rcs = 20 * np.log10(np.sqrt(4 * np.pi) * r * np.abs(es))
plt.plot(180 * theta / np.pi, rcs)
plt.axis([-90, 90, -120, 20])
plt.show()