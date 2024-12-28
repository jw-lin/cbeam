import matplotlib.pyplot as plt
fig,axs = plt.subplots(1,2)

dicoupler.plot_mesh(z=0,ax=axs[0])
dicoupler.plot_mesh(z=dicoupler.z_ex/2,ax=axs[1])
plt.show()