import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import gamma_spectra_data as gsd
import importlib

importlib.reload(gsd)

MCA_CHANNELS = 2048
CH_E_FAC = 1.12

calibration_data = {
	'cs137': {
		'ch': [4.98373328e+01, 896.1729677], 
		'E': [32, 661.67]
	}
}
	
def calibrate_spectra_internal(channel_arr, ch_arr, E_arr):
	p = np.polyfit(ch_arr, E_arr, 1)
	return np.polyval(p, channel_arr)	
	
def calibrate_spectra(channel_arr, iso):
	if iso in calibration_data:
		E = calibration_data[iso]['E']
		ch = calibration_data[iso]['ch']
		return calibrate_spectra_internal(channel_arr, ch, E)
	else:
		print("No calibration for " + iso)
	
def evaluate_spectrum(iso, range_list=[]):
	#get spectrum
	ch_arr, counts_arr = get_spectra(iso, per_s=False)
	
	#fitting
	def gauss(x,a,x0,sigma):
		return a*np.exp(-(x-x0)**2/(2*sigma**2))
	
	plt.figure()
	plt.plot(ch_arr, counts_arr)
	ylims = plt.ylim()
	y0 = ylims[0]
	y1 = ylims[1]
	for row in range_list:
		x0 = row[0]
		x1 = row[1]
		plt.plot((x0, x0), (y0, y1), 'k')
		plt.plot((x1, x1), (y0, y1), 'r')
		#fitting
		selection_idx = (ch_arr > x0) & (ch_arr < x1)
		x = ch_arr[selection_idx]
		y = counts_arr[selection_idx]
		mean = sum(x * y) / sum(y)
		sigma = np.sqrt(sum(y * (x - mean)**2) / sum(y))
		
		popt, pcov = curve_fit(gauss, x, y, p0=[y.max(),mean,sigma])
		plt.plot(x, gauss(x, *popt), 'g')
		print(row, "fit: ", popt)
		
	plt.show()
	
def get_spectra(iso, num_ch=MCA_CHANNELS, per_s=True, calibrate=False):
	divisor = 1 if per_s is False else gsd.data["dt"]	
	iso_counts = gsd.data[iso.lower()]	
	iso_num_ch = len(iso_counts)
	iso_ch_arr = np.array(range(len(iso_counts)))
	out_ch_arr = np.array(range(num_ch))
	conv_ch_arr = np.linspace(0, num_ch * CH_E_FAC, 1025)
	out_counts = np.interp(out_ch_arr, conv_ch_arr, iso_counts) / divisor
	if calibrate:
		out_ch_arr = calibrate_spectra(out_ch_arr, iso)
	
	return out_ch_arr, out_counts