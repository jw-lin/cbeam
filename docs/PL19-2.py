from cbeam.waveguide import get_19port_positions
core_pos = get_19port_positions(rclad/2.5)

# mesh params #
core_res = 16
clad_res = 60
jack_res = 30

from cbeam.waveguide import PhotonicLantern
PL19 = PhotonicLantern(core_pos,rcores,rclad,rjack,ncores,nclad,njack,z_ex,taper_factor,core_res,clad_res,jack_res)