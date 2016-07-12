import numpy as np
import matplotlib.pyplot as plt
import math
import random

def gen_input(n):
    ang = np.random.rand(n,1)*math.pi*2
    r = np.absolute(np.random.normal(0.25, 0.05, (n,1)))
    p = np.concatenate([r*np.cos(ang), r*np.sin(ang)], axis=1)

    ang_ = np.random.rand(n,1)*math.pi
    r_ = np.absolute(np.random.normal(0.5, 0.05, (n,1)))
    p_ = np.concatenate([r_*np.cos(ang), r_*np.sin(ang)], axis=1)

    P = np.concatenate([p,p_])
    l = np.concatenate([np.ones(n, dtype=np.int8), np.zeros(n, dtype=np.int8)])

    # shuffle    
    a=range(n*2)
    random.shuffle(a)
    P = P[a,:]
    l = l[a]

    # save
    np.save('pts', P)
    np.save('labels', l)
    print '%d pts generated.' % n

    # plot
    plt.scatter(P[l==0,0], P[l==0,1], s=100, facecolor=[1,0,0])
    plt.scatter(P[l==1,0], P[l==1,1], s=100, facecolor=[0,1,0])
    plt.show()
    

if __name__ == "__main__":
    # p = gen_input2()
    # show_input(p)
    # plt.show()

    gen_input(50)




