import numpy as np
import pandas as pd
import os
import math as m
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import itertools

class PropellerAnalysis:
    """Class to perform propeller performance analysis using Blade Element Momentum Theory.

    Attributes:
        density (float): Fluid density [kg/m^3].
        blade_num (int): Number of blades.
        radius (float): Propeller radius [m].
        mass (float): Mass of boat or system [kg].
        drag_coeff (float): Base drag coefficient.
        base_damping_factor (float): Default damping factor.
        naca (str): NACA 4-digit airfoil designation.
        gear_ratio (float): Gearbox ratio.
        input_rpm (float): Input RPM from engine.
        engine_power (float): Engine power [W].
        ... (other attributes like twist distribution, fi array, chord params)
    """

    def __init__(self, density=1000, blade_num=1, radius=1, mass=83, drag_coeff=0.5,
                 base_damping_factor=0.1, naca='2412', gear_ratio=1, input_rpm=600,
                 engine_power=1000):
        """Initialize the PropellerAnalysis instance with physical and geometric parameters."""
        # Engine and propeller speed
        self.input_rpm = input_rpm
        self.gear_ratio = gear_ratio
        self.rot_speed = self.input_rpm / self.gear_ratio
        self.engine_power = engine_power

        # Propeller and fluid properties
        self.density = density
        self.blade_num = blade_num
        self.radius = radius
        self.mass = mass
        self.drag_coeff = drag_coeff
        self.base_damping_factor = base_damping_factor
        self.v_boat = 0.05
        self.prop_diameter_m = self.radius * 2
        self.epsilon = 1e-6
        self.total_points = 100
        self.area = 2  # wetted area
        self.naca = naca

        # Chord and twist parameters
        self.c_twist = 0.1
        self.root = 0.5
        self.tip = 0.4
        self.cmax = 0.8

        # Initialize arrays
        self.fi_store = np.zeros(self.total_points)
        self.twists = np.linspace(0.205, 0.225, self.total_points) * 0.5

        # Polynomial fits for aerodynamic coefficients
        self.cl_fit = None
        self.cd_fit = None
        self.alpha_min = None
        self.alpha_max = None

    def twist(self, r):
        """Compute twist angle distribution along the radius.

        Args:
            r (float or ndarray): Radial position(s) [m].

        Returns:
            float or ndarray: Twist angle(s) [rad].
        """
        fi_avg = np.average(self.fi_store)
        scale = 1 / 4
        offset = 0.2 + 0.2 * np.tanh(fi_avg * scale)
        return 0.1 * r + offset

    def chord_distribution(self, r):
        """Compute chord length distribution along the radius.

        Args:
            r (float or ndarray): Radial position(s) [m].

        Returns:
            float or ndarray: Chord length(s) [m].
        """
        d_mid = self.cmax - self.root
        d_tip = self.tip - self.root
        A = 2 * (d_tip - 2 * d_mid) / (self.radius ** 2)
        B = (4 * d_mid - d_tip) / self.radius
        return A * r ** 2 + B * r + self.root

    def adaptive_brent(self, residual, fi_min, fi_max, step=0.001, max_attempts=50):
        """Root finding using Brent's method with adaptive interval adjustment.

        Args:
            residual (callable): Residual function to find root for.
            fi_min (float): Initial lower bound.
            fi_max (float): Initial upper bound.
            step (float): Interval adjustment step.
            max_attempts (int): Maximum number of interval expansions.

        Returns:
            float: Root of the residual function.

        Raises:
            ValueError: If no root is found within max_attempts.
        """
        attempts = 0
        while attempts < max_attempts:
            try:
                if residual(fi_min) * residual(fi_max) < 0:
                    return brentq(residual, fi_min, fi_max, xtol=1e-6)
                fi_min -= step
                fi_max += step
                attempts += 1
            except Exception:
                fi_min -= step
                fi_max += step
                attempts += 1
        raise ValueError(f"Failed to find a valid range for the root after {max_attempts} attempts.")

    def fit_eq(self, filename, degree=6):
        """Fit polynomial curves to Cl and Cd data from CSV file.

        Args:
            filename (str): CSV file with 'Alpha', 'Cl', 'Cd' columns.
            degree (int): Polynomial degree for fitting.
        """
        file_path = os.path.join(os.getcwd(), filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {filename}")

        df = pd.read_csv(file_path)
        required_columns = {'Alpha', 'Cl', 'Cd'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"File must contain columns {required_columns}")

        alpha = df['Alpha']
        cl = df['Cl']
        cd = df['Cd']
        self.cl_fit = np.poly1d(np.polyfit(alpha, cl, deg=degree))
        self.cd_fit = np.poly1d(np.polyfit(alpha, cd, deg=degree))
        self.alpha_min = alpha.min()
        self.alpha_max = alpha.max()

    def _residual(self, fi, r_index, radius_points, twist_angles, sigma):
        """Residual function for BEM root-finding at a given blade element.

        Args:
            fi (float): Angle of attack guess [rad].
            r_index (int): Index of the blade element.
            radius_points (ndarray): Radial positions.
            twist_angles (ndarray): Twist distribution.
            sigma (ndarray): Local solidity distribution.

        Returns:
            float: Residual value.
        """
        temp_alpha = twist_angles[r_index] - fi
        f = self.blade_num / 2 * ((self.radius - radius_points[r_index]) / radius_points[r_index] * np.abs(np.sin(fi)))
        F = (2 / np.pi) * np.arccos(np.exp(-f))
        Cl = self.cl_fit(temp_alpha) * 0.7
        Cd = self.cd_fit(temp_alpha) * 0.7
        Cn = Cl * np.cos(fi) - Cd * np.sin(fi)
        Ct = Cl * np.sin(fi) + Cd * np.cos(fi)
        axial = 1 / (((4 * F * np.sin(fi)**2) / (sigma[r_index] * Cn)) + 1 + self.epsilon)
        tangential = 1 / (((4 * F * np.sin(fi) * np.cos(fi)) / (sigma[r_index] * Ct)) - 1 + self.epsilon)
        return (np.sin(fi) / (1 + axial)) - ((self.v_boat) / ((self.rot_speed * np.pi / 30) * radius_points[r_index])) * (np.cos(fi) / (1 - tangential))

    def bem(self):
        """Blade Element Momentum theory computation for thrust, torque, and efficiency.

        Returns:
            tuple: total_thrust (N), total_torque (Nm), omega (rad/s), efficiency, avg_alpha (rad), dL (N)
        """
        tol = 0.1
        r_min = 0.2 * self.radius
        r_max = self.radius
        radius_points = np.linspace(r_min, r_max, self.total_points)

        twist_angles = self.twists if self.twists is not None else self.twist(radius_points)
        sigma = (self.blade_num * self.chord_distribution(radius_points)) / (2 * np.pi * radius_points)
        fi_values = np.zeros_like(radius_points)

        for i in range(len(radius_points)):
            try:
                fi = self.adaptive_brent(lambda fi, idx=i: self._residual(fi, idx, radius_points, twist_angles, sigma), 0, np.pi / 2)
                fi_values[i] = fi
            except ValueError:
                fi_values[i] = fi_values[i - 1] if i > 0 else 0

        self.fi_store = fi_values
        alpha_array = twist_angles - fi_values
        Cl_full = self.cl_fit(alpha_array) * 0.7
        Cd_full = self.cd_fit(alpha_array) * 0.7

        radius_used = radius_points[1:]
        Cl = np.abs(Cl_full[1:])
        Cd = Cd_full[1:]
        fi_used = fi_values[1:]
        sigma_used = sigma[1:]

        f_used = (self.blade_num / 2) * ((self.radius - radius_used) / (radius_used * (np.abs(np.sin(fi_used)) + self.epsilon)))
        F_used = (2 / np.pi) * np.arccos(np.exp(-f_used))
        Cn = np.abs(Cl * np.cos(fi_used) - Cd * np.sin(fi_used))
        Ct = Cl * np.sin(fi_used) + Cd * np.cos(fi_used)
        axial = 1 / (((4 * F_used * np.sin(fi_used)**2) / (sigma_used * Cn)) + 1 + self.epsilon)
        tangential = 1 / (((4 * F_used * np.sin(fi_used) * np.cos(fi_used)) / (sigma_used * Ct)) - 1 + self.epsilon)
        w = np.sqrt((self.v_boat * (1 + axial))**2 + ((self.rot_speed * np.pi / 30) * radius_used * (1 - tangential))**2)

        thrust = self.blade_num * Cn * 0.5 * self.density * w**2 * self.chord_distribution(radius_used)
        torque = self.blade_num * Ct * 0.5 * self.density * w**2 * self.chord_distribution(radius_used) * radius_used
        total_thrust = np.trapz(thrust, radius_used) * 0.9
        total_torque = np.trapz(torque, radius_used)
        omega = (self.rot_speed * 2 * np.pi) / 60

        n = self.rot_speed / 60
        co_thrust = total_thrust / (self.density * n**2 * self.prop_diameter_m**4)
        co_torque = total_torque / (self.density * n**2 * self.prop_diameter_m**5)
        J = self.v_boat / (n * self.prop_diameter_m)
        eff = (J * co_thrust) / (2 * np.pi * co_torque)
        avg_alpha = np.nanmean(alpha_array[1:])
        dL = 0.5 * self.density * self.v_boat**2 * np.average(Cl) * self.radius

        return total_thrust, total_torque, omega, eff, avg_alpha, dL

    def dynamic_simulation(self, csv_file, damping_factor=None, initial_time_step=0.001,
                           min_time_step=0.001, max_time_step=0.1, acc_threshold=0.5):
        """Simulate propeller-boat system dynamics over time.

        Args:
            csv_file (str): NACA airfoil data file.
            damping_factor (float): Optional damping factor.
            initial_time_step (float): Initial integration step.
            min_time_step (float): Minimum allowed time step.
            max_time_step (float): Maximum allowed time step.
            acc_threshold (float): Acceleration threshold for adaptive timestep.

        Returns:
            tuple: final_state dict, time_series list
        """
        self.fit_eq(csv_file)
        dt = initial_time_step
        damping_factor = damping_factor if damping_factor is not None else self.base_damping_factor
        time = 0
        time_series = []

        while time < 20:
            thrust, torque, omega, efficiency, avg_alpha, lift = self.bem()
            omega_prop = self.rot_speed * 2 * np.pi / 60
            P_prop = torque * omega_prop
            scale_factor = min(1, self.engine_power / P_prop) if P_prop > 0 else 1
            thrust_eff = scale_factor * thrust
            torque_eff = scale_factor * torque

            # Drag computation
            RE = (self.v_boat * 3) / 1e-6
            self.drag_coeff = 0.075 / ((np.log10(RE) - 2) ** 2)
            Fr = self.v_boat / (np.sqrt(9.81 * 3))
            if Fr <= 0.2:
                Cr = 0.001
            elif 0.2 <= Fr < 0.4:
                Cr = 0.002
            else:
                Cr = 0.003
            self.drag_coeff += Cr
            drag_force = 0.5 * self.density * self.v_boat**2 * self.drag_coeff * self.area

            net_force = thrust_eff - drag_force
            acceleration = net_force / self.mass
            self.v_boat += acceleration * dt
            time += dt

            time_series.append({
                'time': time,
                'Boat Velocity': self.v_boat,
                'Thrust': thrust_eff,
                'Torque': torque_eff,
                'Drag Force': drag_force,
                'Acceleration': acceleration,
                'Efficiency': efficiency,
                'Alpha': avg_alpha,
                'lift': lift,
                'dt': dt,
                'Drag coefficient': self.drag_coeff
            })

            # Adaptive timestep
            if abs(acceleration) > acc_threshold:
                dt = max(min_time_step, dt / 2)
            else:
                dt = min(max_time_step, dt * 1.1)

        useful_power = thrust_eff * self.v_boat
        if useful_power > self.engine_power:
            print(f"Warning: Useful power ({useful_power:.2f} W) exceeds engine power ({self.engine_power:.2f} W)!")

        final_state = {
            'Time': time,
            'Boat Velocity': self.v_boat,
            'Thrust': thrust_eff,
            'Torque': torque_eff,
            'Drag Force': drag_force,
            'Acceleration': acceleration,
            'Efficiency': efficiency,
            'Alpha': avg_alpha,
            'Output RPM': self.rot_speed,
            'Input RPM': self.input_rpm,
            'Useful Power (W)': useful_power
        }

        return final_state, time_series


# Main Execution: Parameter Sweep
if __name__ == "__main__":
    # Parameter ranges
    density_range    = [1025]
    blade_num_range  = [1]
    radius_range     = [0.25]
    mass_range       = [83]
    drag_coeff_range = [0.012]
    damping_range    = [0.02]
    naca_range       = ['2412']
    input_rpm_range  = [90]
    gear_ratio_range = [(1/5)]
    engine_power_range = [240]
    root_range      = [0.04]
    tip_range       = [0.001]
    cmax_range      = [0.06]
    m_twist_range   = [0.0808]
    c_twist_range   = [0.2048]

    results = []
    time_series_results = []
    iteration_counter = 0

    # Sweep over all parameter combinations
    for (density, blade_num, radius, mass, drag_coeff, damping, naca,
         input_rpm, gear_ratio, engine_power) in itertools.product(
            density_range, blade_num_range, radius_range, mass_range,
            drag_coeff_range, damping_range, naca_range,
            input_rpm_range, gear_ratio_range, engine_power_range):
        for (root, tip, cmax, m_twist, c_twist) in itertools.product(
                root_range, tip_range, cmax_range, m_twist_range, c_twist_range):

            iteration_counter += 1
            prop_analysis = PropellerAnalysis(
                density=density, blade_num=blade_num, radius=radius, mass=mass,
                drag_coeff=drag_coeff, base_damping_factor=damping, naca=naca,
                gear_ratio=gear_ratio, input_rpm=input_rpm, engine_power=engine_power
            )
            prop_analysis.root = root
            prop_analysis.tip = tip
            prop_analysis.cmax = cmax
            prop_analysis.m_twist = m_twist
            prop_analysis.c_twist = c_twist

            try:
                final_result, time_series = prop_analysis.dynamic_simulation("NACA 4412.csv")
            except Exception as e:
                print(f"Error during simulation: {e}")
                continue

            # Store results
            results.append({**final_result, **{
                'Iteration': iteration_counter,
                'density': density,
                'blade_num': blade_num,
                'radius': radius,
                'mass': mass,
                'drag_coeff': drag_coeff,
                'base_damping_factor': damping,
                'naca': naca,
                'root': root,
                'tip': tip,
                'cmax': cmax,
                'm_twist': m_twist,
                'c_twist': c_twist
            }})

            for record in time_series:
                record['Iteration'] = iteration_counter
                time_series_results.append(record)

    # Save overall results
    results_df = pd.DataFrame(results)
    results_df.to_csv("Main_results.csv", index=False)
    print("Data saved to 'Main_results.csv'.")

    ts_df = pd.DataFrame(time_series_results)
    ts_df.to_csv("Scale_factor_0.5.csv", index=False)
    print("Time series data saved.")