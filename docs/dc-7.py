# we will run 100 dicoupler simulations with different lenghts
stretch_amounts = np.linspace(0,10000,100)

u0 = [1,0]
pwrs = []

for i,stretch in enumerate(stretch_amounts):
    zs = np.copy(dc_prop.zs)
    zs[np.argmax(zs>=dicoupler.z_ex/2):] += stretch # stretch out the z array

    dc_prop.make_interp_funcs(zs) # remake the interpolation functions
    zs,us,uf = dc_prop.propagate(u0,zs[0],zs[-1]) # rerun the propagator

    pwr = np.power(np.abs(uf),2)
    pwrs.append(pwr)

pwrs = np.array(pwrs)

pred_period = 4735 ## predicted oscillation period, see next section for the formula ##

zmax = stretch_amounts[np.argmax(pwrs[:,0])] # translating the sinusoid to match - not trying to match absolute phase (see next section)

# plot predicted cos^2 dependence
plt.plot(stretch_amounts,np.power(np.cos(np.pi/pred_period*(stretch_amounts-zmax)),2),color='k',ls='dashed',label="predicted")

plt.plot(stretch_amounts,pwrs.T[0],label="channel 1")
plt.plot(stretch_amounts,pwrs.T[1],label="channel 2")
plt.legend(loc='best',frameon=False)
plt.xlabel("dicoupler length")
plt.ylabel("power")
plt.title("output of dicoupler channels vs. of coupling length")
plt.show()