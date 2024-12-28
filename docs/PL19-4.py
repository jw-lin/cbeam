import matplotlib.pyplot as plt
plt.plot(prop.zs,prop.neffs[:,18]-prop.neffs[:,19])
plt.axhline(y=0,color='k',ls='dashed')
plt.xlabel("z")
plt.title("difference in effective index, modes 18 & 19")
plt.show()

prop.plot_neff_diffs()