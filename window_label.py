# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 14:55:27 2021
Function:
    1) Label the window, which is cutted from image. The window size
        is related to the beamwidth and distance.
@author: wrhsd
"""
import matplotlib.pyplot as plt

def window_label(window):
    img = window
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(img)
    plt.autoscale(tight=True)
    plt.show()
    k = input('The label is foliage(f) or gap(g) or unclear(n)?')
    return k