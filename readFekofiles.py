import matplotlib.pyplot as plt
import numpy as np

with open("SERMetTarget5lambda.dat",'r') as f:
    f.readline()
    f.readline()
    angle = []
    SER_dB = []
    for line in f:
        data = line.split()
        ang = float(data[0])
        ser = float(data[1])
        angle.append(ang)
        SER_dB.append(ser)

    plt.figure()
    plt.plot(angle,SER_dB)
    v_max = np.max(SER_dB) + 1
    v_min = v_max - 100
    plt.ylim([v_min,v_max])
    plt.show()
