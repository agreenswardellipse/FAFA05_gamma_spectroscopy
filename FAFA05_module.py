import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

import NIST_mass_attenuation as NISTX
import gamma_spectra_generator as gsg
import importlib
importlib.reload(gsg)
importlib.reload(NISTX)

class FAFA05_module:
	def __init__(self, attenuator='Al', src='cs137', E=0.662, N=1000, unit="m", student_ambition_factor=0.01):
		self.current_attenuator = attenuator
		self.current_source = src
		self.current_E = E		
		self.current_N = N
		self.unit_str = unit
		self.student_ambition_factor = student_ambition_factor
		self.result_history_list = []
		
		self.set_properties()

			

	def set_properties(self, attenuator=None, src=None, E=None, N=None, unit=None, student_ambition_factor=None):
		if attenuator: self.current_attenuator = attenuator
		if src: self.current_source = src
		if E: self.current_E = E
		if N: self.current_N = N
		if unit: self.unit_str = unit
		if student_ambition_factor: self.student_ambition_factor = student_ambition_factor
		convert_unit = 100.0 #1/cm to 1/m default		
		if self.unit_str.lower() == "cm": 
			convert_unit = 1.0
		
		self.current_mu = NISTX.get_mu(self.current_E, self.current_attenuator) * convert_unit # 1/cm to 1/m
		self.current_channels, self.current_spectrum = gsg.get_spectra(self.current_source, calibrate=True)
		self.current_spectrum_norm = self.current_spectrum / np.sum(self.current_spectrum)
		self.student_ambition_factor = min(max(self.student_ambition_factor, 0.01), 0.99)

	
	def randomize_gammas(self, I):
		randomize = np.random.rand(*self.current_channels.shape) #* self.rand_factor
		rand_I = I * self.current_N 
		rand_I += randomize * rand_I.max() / 20
		# rand_I *= np.random.normal(1, 0.2, 1)
		rand_I *= np.random.uniform(self.student_ambition_factor, 1.0 + (1.0-self.student_ambition_factor))
		return rand_I

	def simulate_attenuation(self, thickness=0.0, measurement_time=10, material=None, animate=True, randomize=True, y_log=False):
		"""
		Here we simulate a gamma spectrum measurement for a given thickness during a given measuremnt time/duration
		"""
		total_counts = np.zeros(self.current_channels.shape) + 1e-5
		#update material if needed
		if material is not None and material is not self.current_attenuator:
			self.set_properties(attenuator=material)
		if animate:
			self.fig, self.ax = plt.subplots()
			self.ax.set_ylim([1, 1000])
			self.ax.set_xlabel("E (KeV)")
			self.ax.set_ylabel("Counts")
			if y_log:
				self.ax.set_yscale('log')
			self.barplot = self.ax.bar(self.current_channels, total_counts, width=1.0)
		
		I = self.current_spectrum_norm * np.exp(-self.current_mu*thickness)
		# print(t)
		# measurement time is used as number of simulated instances of gamma, with N gammas each second approximately
		total_time = time.perf_counter()
		for t in range(np.int(measurement_time)):		
			t0 = time.perf_counter()
			if randomize:
				total_counts += self.randomize_gammas(I)  
			else:
				total_counts += I * self.current_N
			if animate:
				for i, bar in enumerate(self.barplot):
					bar.set_height(np.round(total_counts[i]))
					if total_counts[i] > self.ax.get_ylim()[1]:
						self.ax.set_ylim([self.ax.get_ylim()[0], total_counts[i] * 1.1])
				self.fig.canvas.draw()				
				time_passed = "{:.2f}".format(time.perf_counter() - total_time)
				self.ax.set_title("Measuring " + self.current_attenuator + ", " + "thickness=" + "{:.3f}".format(thickness) + "m, " + str(time_passed) + "s")
				time_to_wait = np.maximum(1 - (time.perf_counter() - t0), 0.0) + 1e-8
				plt.pause(time_to_wait)
				
		total_counts = np.round(total_counts) 
		return total_counts
		
	def simulate_attenuation_series(self, thickness=[0.0], measurement_time=10, animate=True, randomize=True, y_log=True):
		res = []
		for th in thickness:
			counts = self.simulate_attenuation(thickness=th, measurement_time=measurement_time, animate=animate, randomize=randomize, y_log=y_log)
			res.append([th, counts])
		return res
		
	def show_spectrum(self, lim_left=0, lim_right=0, show_selection=True, y_log=False):
		"""
		This function presents a sample spectrum for the decaying isotope selected in the module
		"""
		lim_left = np.argmin(abs(self.current_channels-lim_left))
		lim_right = np.argmin(abs(self.current_channels-lim_right))
		color_arr = ['b']*len(self.current_channels)
		if show_selection and lim_left < lim_right:
			color_arr = ['b']*lim_left + ['r']*(lim_right-lim_left) + ['b']*(len(self.current_channels)-lim_right)
		plt.figure()
		plt.title("Spectrum of " + self.current_source.capitalize())
		plt.xlabel("E (KeV)")
		plt.ylabel("Counts")
		plt.bar(self.current_channels, self.current_spectrum, width=1.0, color=color_arr)
		plt.show()
		
	def get_available_attenuators(self):
		return NISTX.get_list_of_elements()
		
	def get_channels(self):
		return self.current_channels
		
	def get_x_array(self):
		return self.current_channels
		
	def append_result(self, thickness, selection_sum):
		self.result_history_list.append([self.current_attenuator, thickness, selection_sum])
		
	def clear_results(self):
		self.result_history_list = []
		
	def get_results(self):
		result_history_string = "Result history:\n-------------------\n"
		for i, res in enumerate(self.result_history_list):
			material = res[0]
			thickness = str(res[1])
			count = str(res[2])
			result_history_string += "Experiment " + str(i) + "\n"
			result_history_string += "Material: " + material + "\n"
			result_history_string += "Thickness: " + thickness + " " + self.unit_str +"\n"
			result_history_string += "Peak sum: " + count +"\n"			
			result_history_string += "-------------------\n" 
		
		return result_history_string
		
	def print_results(self):
		print(self.get_results())
		
	def get_peak_counts(self, channel_counts, thickness, lim_left, lim_right, show_selection=False, plot_title=""):
		lim_left = np.argmin(abs(self.current_channels-lim_left))
		lim_right = np.argmin(abs(self.current_channels-lim_right))
		lim_extension = 20
		y_left = np.mean(channel_counts[lim_left-lim_extension:lim_left])
		y_right = np.mean(channel_counts[lim_right:lim_right+lim_extension])
		
		bg = (lim_right - lim_left) * (y_left + y_right) * 0.5 #trapz rule
		all_counts = np.sum(channel_counts)
		selection_counts = np.sum(channel_counts[lim_left:lim_right])
		selection_counts_no_bg = selection_counts - bg
		
		if show_selection:
			color_arr = ['b']*lim_left + ['r']*(lim_right-lim_left) + ['b']*(len(channel_counts)-lim_right)
			
			fig, ax = plt.subplots()
			plt.title(plot_title)
			# plt.xlabel("Thickness" + " / " + self.unit_str)
			plt.xlabel("E (KeV)")
			plt.ylabel("Counts")
			ax.bar(self.current_channels, channel_counts, width=1.0, color=color_arr)
			
			#show background
			bg_heights = np.zeros(channel_counts.shape)
			for i, b in enumerate(bg_heights):
				if i >= lim_left and i <= lim_right:
					bg_heights[i] = (i - lim_left) * (y_right - y_left) / (lim_right - lim_left) + y_left
			ax.bar(self.current_channels, bg_heights, width=1.0, color='g')
			legend_elements = [
                   Patch(facecolor='b', edgecolor='b', label='Counts: ' + str(int(all_counts))),
                   Patch(facecolor='r', edgecolor='r', label='Selection: ' + str(int(selection_counts_no_bg))),
                   Patch(facecolor='g', edgecolor='g', label='Background: ' + str(int(bg)))
				   ]
			ax.legend(handles=legend_elements, loc='upper right')
			plt.show()	
			
		self.append_result(thickness, selection_counts)

		return selection_counts