from cbeam.propagator import Propagator

prop = Propagator(wl,PL19,Nmax=21)

# comment/uncomment as necessary
# this take around 10 minutes on my laptop
# prop.compute_neffs()
prop.load(tag="19port_neffs")