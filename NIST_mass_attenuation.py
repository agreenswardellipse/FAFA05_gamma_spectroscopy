import numpy as np
import matplotlib.pyplot as plt
import NIST_mass_attenuation_data as NIST
import importlib
importlib.reload(NIST)

#class MassAttenData:
COL_MU = 1
COL_MUEN = 2

def log_spaced_array(start, end, points):
	arr = pow(np.logspace(np.log10(start), np.log10(end), points), 10.0)
	return arr
	
def log_interp(x, xp, yp):
	logx = np.log10(x)
	logxp = np.log10(xp)
	logyp = np.log10(yp)
	return np.power(10.0, np.interp(logx, logxp, logyp))
	
def get_list_of_elements():
	Z_list = list(zip(NIST.data["z"].values(), NIST.data["z"]))
	Z_list_str = "List of elements included in data: \n"
	Z_list_str += "----------------------------------\n"
	for i in range(1,len(Z_list), 2):
		Z_list_str += str(Z_list[i][0]) + ": " + Z_list[i][1].capitalize() + " (" + Z_list[i-1][1] + ")\n"
	print(Z_list_str)
	
	
def get_Z_idx(Z):
	if type(Z) is str:
		try:
			Z_idx = NIST.data["z"][Z.lower()]
		except:
			raise ValueError("Element " + Z + " not found in database.")
	elif type(Z) is int:	
		Z_idx = Z
	else:
		raise TypeError("Only ints and strings allowed for element selection.")
		
	return Z_idx
	
def interp_data(E, Z, col):
	Z_idx = get_Z_idx(Z)
	mu_data = NIST.data[Z_idx]["data"]	
	E_data = mu_data[:,0]
	y_data = mu_data[:,col]
	
	return log_interp(E, E_data, y_data)
	
	
def get_mu_rho(E, Z):
	return interp_data(E,Z,COL_MU)
	
def get_muen_rho(E, Z):
	return interp_data(E,Z,COL_MUEN)
	
def get_mu(E, Z):
	rho = NIST.data[get_Z_idx(Z)]["density"]
	return interp_data(E,Z,COL_MU) * rho
	
def get_muen(E, Z):
	rho = NIST.data[get_Z_idx(Z)]["density"]
	return interp_data(E,Z,COL_MUEN) * rho
	
def plot_spectrum(Z_list, same_plot=False):
	if type(Z_list) is not list:
		Z_list = [Z_list]
	for Z in Z_list:
		Z_idx = get_Z_idx(Z)
		data = NIST.data[Z_idx]["data"]
		E = data[:,0]
		mu = data[:,1]
		muen = data[:,2]
		
		plt.figure()
		plt.plot(E, mu, '-')
		plt.plot(E, muen, '--')
		plt.title(NIST.data[Z_idx]["name"])
		plt.xlabel("E (MeV)")
		plt.ylabel(r'$\mu/\rho \ or \ \mu_{en}/\rho \ (cm^2g^{-1})$')
		plt.legend([r'$\mu/\rho$', r'$\mu_{en}/\rho$'])
		plt.yscale('log')
		plt.xscale('log')
		plt.show()
	
