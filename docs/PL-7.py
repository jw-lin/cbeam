import matplotlib.pyplot as plt
import numpy as np

u0 = [1,0,0,0,0,0] # launch field, LP01
zs,us,uf = PLprop.propagate(u0)

PLprop.plot_mode_powers(zs,us)