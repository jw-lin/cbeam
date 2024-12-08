# note that this function uses the phased mode amplitudes uf, not the unphased amplitudes in u.
amps = PLprop.to_channel_basis(uf)

print(np.power(np.abs(amps),2))