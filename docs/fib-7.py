fig,axs = plt.subplots(1,2)
rect_fiber.plot_mesh(z=0,ax=axs[0])
rect_fiber.plot_mesh(z=length,ax=axs[1])
axs[0].set_title("z=0")
axs[0].set_title("z="+str(length))
plt.show()