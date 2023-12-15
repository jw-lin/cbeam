from cbeam import waveguide
rcore = 10
rclad = 30
ncore = 1.445
nclad = 1.44

fiber = waveguide.CircularStepIndexFiber(rcore,rclad,ncore,nclad)

fiber.plot_mesh()