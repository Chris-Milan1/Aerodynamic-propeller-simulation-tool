import numpy as np
import pandas as pd
import os
import math as m
from scipy.optimize import brentq
import matplotlib.pyplot as plt
import itertools

# ====================================
# PropellerAnalysis Class Definition
# ====================================
class PropellerAnalysis:
    def __init__(self, density=1000, blade_num=1, radius=1, mass=83,drag_coeff=0.5, base_damping_factor=0.1, naca='2412',
                 gear_ratio=1, input_rpm=600, engine_power=1000):        
        self.input_rpm = input_rpm
        self.gear_ratio = gear_ratio
        self.rot_speed = self.input_rpm / self.gear_ratio
        self.engine_power = engine_power

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
        self.area = 2 # wetted area
        self.naca = naca

        # Chord parms
        self.c_twist = 0.1   # Tip twist angle (radians)
        self.root = 0.5      # Chord at root (m)
        self.tip = 0.4       # Chord at tip (m)
        self.cmax = 0.8      # Maximum chord (m)

        # Initialize a structure for fi array to be stored
        self.fi_store = np.zeros(self.total_points)
        self.twists = np.array([
    0.205, 0.20520202, 0.20540404, 0.20560606, 0.20580808, 0.2060101,
    0.20621212, 0.20641414, 0.20661616, 0.20681818, 0.2070202, 0.20722222,
    0.20742424, 0.20762626, 0.20782828, 0.2080303, 0.20823232, 0.20843434,
    0.20863636, 0.20883838, 0.2090404, 0.20924242, 0.20944444, 0.20964646,
    0.20984848, 0.21005051, 0.21025253, 0.21045455, 0.21065657, 0.21085859,
    0.21106061, 0.21126263, 0.21146465, 0.21166667, 0.21186869, 0.21207071,
    0.21227273, 0.21247475, 0.21267677, 0.21287879, 0.21308081, 0.21328283,
    0.21348485, 0.21368687, 0.21388889, 0.21409091, 0.21429293, 0.21449495,
    0.21469697, 0.21489899, 0.21510101, 0.21530303, 0.21550505, 0.21570707,
    0.21590909, 0.21611111, 0.21631313, 0.21651515, 0.21671717, 0.21691919,
    0.21712121, 0.21732323, 0.21752525, 0.21772727, 0.21792929, 0.21813131,
    0.21833333, 0.21853535, 0.21873737, 0.21893939, 0.21914141, 0.21934343,
    0.21954545, 0.21974747, 0.21994949, 0.22015152, 0.22035354, 0.22055556,
    0.22075758, 0.2209596, 0.22116162, 0.22136364, 0.22156566, 0.22176768,
    0.2219697, 0.22217172, 0.22237374, 0.22257576, 0.22277778, 0.2229798,
    0.22318182, 0.22338384, 0.22358586, 0.22378788, 0.2239899, 0.22419192,
    0.22439394, 0.22459596, 0.22479798, 0.225
])* 0.5

        
        self.cl_fit = None
        self.cd_fit = None
        self.alpha_min = None
        self.alpha_max = None

    def twist(self, r):
        fi_avg = np.average(self.fi_store)
        scale = 1/4
        offset = 0.2 + 0.2 * np.tanh(fi_avg * scale)
        return 0.1 * r + offset

    def chord_distribution(self, r):
        d_mid = self.cmax - self.root
        d_tip = self.tip - self.root
        A = 2 * (d_tip - 2 * d_mid) / (self.radius ** 2)
        B = (4 * d_mid - d_tip) / self.radius
        return A * r**2 + B * r + self.root

    def adaptive_brent(self, residual, fi_min, fi_max, step=0.001, max_attempts=50):
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
        file_path = os.path.join(os.getcwd(), filename)
        if not os.path.exists(file_path):
            raise FileNotFoundError("Error: File not found. Please check the filename.")
        df = pd.read_csv(file_path)
        required_columns = {'Alpha', 'Cl', 'Cd'}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"Error: The file must contain columns {required_columns}")
        alpha = df['Alpha']
        cl = df['Cl']
        cd = df['Cd']
        self.cl_fit = np.poly1d(np.polyfit(alpha, cl, deg=degree))
        self.cd_fit = np.poly1d(np.polyfit(alpha, cd, deg=degree))
        self.alpha_min = alpha.min()
        self.alpha_max = alpha.max()

    def _residual(self, fi, r_index, radius_points, twist_angles, sigma):
        temp_alpha = (twist_angles[r_index] - fi)
        f = self.blade_num / 2 * ((self.radius - radius_points[r_index]) /
                                  radius_points[r_index] * np.abs(np.sin(fi)))
        F = (2 / np.pi) * np.arccos(np.exp(-f))
        Cl = self.cl_fit(temp_alpha) * 0.7
        Cd = self.cd_fit(temp_alpha) * 0.7
        Cn = Cl * np.cos(fi) - Cd * np.sin(fi)
        Ct = Cl * np.sin(fi) + Cd * np.cos(fi)
        axial = 1 / (((4 * F * np.sin(fi)**2) / (sigma[r_index] * Cn)) + 1 + self.epsilon)
        tangential = 1 / (((4 * F * np.sin(fi) * np.cos(fi)) / (sigma[r_index] * Ct)) - 1 + self.epsilon)
        return (np.sin(fi) / (1 + axial)) - ((self.v_boat) / ((self.rot_speed * np.pi / 30) *
                radius_points[r_index])) * (np.cos(fi) / (1 - tangential))

    def bem(self):
        tol = 0.1
        r_min = 0.2 * self.radius
        r_max = self.radius
        radius_points = np.linspace(r_min, r_max, self.total_points)
        
        if self.twists is None:
            twist_angles = self.twist(radius_points)
            self.twists = twist_angles
        else:
            twist_angles = self.twists 
        sigma = (self.blade_num * self.chord_distribution(radius_points)) / (2 * np.pi * radius_points)
        fi_values = np.zeros_like(radius_points)
    
        for i in range(len(radius_points)):
            try:
                fi = self.adaptive_brent(
                    lambda fi, idx=i: self._residual(fi, idx, radius_points, twist_angles, sigma),
                    0, np.pi / 2
                )
                current_residual = np.abs(self._residual(fi, i, radius_points, twist_angles, sigma))
                fi_values[i] = fi if current_residual < tol else (fi_values[i - 1] if i > 0 else 0)
            except ValueError:
                fi_values[i] = fi_values[i - 1] if i > 0 else 0
    
        fi_array = np.array(fi_values)
        twist_array = np.array(twist_angles)
        alpha_array = (twist_array - fi_array)
        
        # Update the stored fi values for future iterations in twist()
        self.fi_store = fi_array
    
        # Compute the full aerodynamic coefficients.
        Cl_full = self.cl_fit(alpha_array) * 0.7
        Cd_full = self.cd_fit(alpha_array) * 0.7
    
        Cl = np.abs(Cl_full[1:])
        Cd = Cd_full[1:]
        fi_used = fi_array[1:]
        alpha_used = alpha_array[1:]
        sigma_used = sigma[1:]
        radius_used = radius_points[1:]
    
        # Recalculate aerodynamic terms for the reduced arrays.
        f_used = (self.blade_num / 2) * ((self.radius - radius_used) /
                                         (radius_used * (np.abs(np.sin(fi_used)) + self.epsilon)))
        F_used = (2 / np.pi) * np.arccos(np.exp(-f_used))
        Cn = np.abs(Cl * np.cos(fi_used) - Cd * np.sin(fi_used))
        Ct = Cl * np.sin(fi_used) + Cd * np.cos(fi_used)
        axial = 1 / (((4 * F_used * np.sin(fi_used)**2) / (sigma_used * Cn)) + 1 + self.epsilon)
        tangential = 1 / (((4 * F_used * np.sin(fi_used) * np.cos(fi_used)) / (sigma_used * Ct)) - 1 + self.epsilon)
        w = np.sqrt((self.v_boat * (1 + axial))**2 +
                    ((self.rot_speed * np.pi / 30) * radius_used * (1 - tangential))**2)
    
        # Integration to compute lift, thrust, and torque over the reduced blade span.
        cl_blade = np.average(Cl)#
        dL = 0.5 * self.density * self.v_boat**2 * cl_blade * self.radius
        thrust = self.blade_num * Cn * 0.5 * self.density * w**2 * self.chord_distribution(radius_used)
        torque = self.blade_num * Ct * 0.5 * self.density * w**2 * self.chord_distribution(radius_used) * radius_used
        total_thrust = np.trapz(thrust, radius_used) * 0.9
        total_torque = np.trapz(torque, radius_used)
        omega = (self.rot_speed * 2 * np.pi) / 60
    
        # Compute non-dimensional coefficients and efficiency.
        n = self.rot_speed / 60
        co_thrust = total_thrust / (self.density * n**2 * self.prop_diameter_m**4)
        co_torque = total_torque / (self.density * n**2 * self.prop_diameter_m**5)
        J = self.v_boat / (n * self.prop_diameter_m)
        eff = (J * co_thrust) / (2 * np.pi * co_torque)
        avg_alpha = np.nanmean(alpha_used)
    
        return total_thrust, total_torque, omega, eff, avg_alpha, dL


    def dynamic_simulation(self, csv_file, damping_factor=None, initial_time_step=0.001,
                           min_time_step=0.001, max_time_step=0.1, acc_threshold=0.5):
        self.fit_eq(csv_file)
        dt = initial_time_step
        damping_factor = damping_factor if damping_factor is not None else self.base_damping_factor
        time = 0
        time_series = []  # List to record state at each time step

        while time < 20:
            thrust, torque, omega, efficiency, avg_alpha, lift = self.bem()

            # Compute propeller mechanical power (W)
            omega_prop = self.rot_speed * 2 * np.pi / 60
            P_prop = torque * omega_prop

            # Scale the propeller load so that propeller power does not exceed engine power.
            scale_factor = min(1, self.engine_power / P_prop) if P_prop > 0 else 1
            thrust_eff = scale_factor * thrust
            torque_eff = scale_factor * torque

            # Reynolds number-based drag coefficient update.
            RE = (self.v_boat * 3) / (1e-6)
            self.drag_coeff = 0.075 / ((np.log10(RE) - 2)**2)
            Fr = self.v_boat / (np.sqrt(9.81 * 3))
            if Fr <= 0.2:
                Cr = 0.001
            elif 0.2 <= Fr < 0.4:
                Cr = 0.002
            else:  # Fr >= 0.4
                Cr = 0.003
            self.drag_coeff = self.drag_coeff + Cr

            drag_force = 0.5 * self.density * self.v_boat**2 * self.drag_coeff * self.area
            net_force = thrust_eff - drag_force
            acceleration = net_force / self.mass

            # Update boat velocity and time.
            self.v_boat += acceleration * dt
            time += dt

            # Record the simulation state at this time step.
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

            # Adjust the time step adaptively.
            if abs(acceleration) > acc_threshold:
                dt = max(min_time_step, dt / 2)
            else:
                dt = min(max_time_step, dt * 1.1)

        useful_power = thrust_eff * self.v_boat
        if useful_power > self.engine_power:
            print("Warning: The final useful power ({:.2f} W) exceeds the engine input power ({:.2f} W)!"
                  .format(useful_power, self.engine_power))

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

# ====================================
# SolidWorks Class Definition (Updated)
# ====================================
class SolidWorks:
    def __init__(self, prop_analysis):
        self.pa = prop_analysis

    def chord_distribution_custom(self, r, radius, chord_root, chord_max, chord_tip):
        A = 2 * (chord_tip - chord_root - 2 * (chord_max - chord_root)) / (radius**2)
        B = (4 * (chord_max - chord_root) - (chord_tip - chord_root)) / radius
        return A * r**2 + B * r + chord_root

    def parse_naca(self, naca_str):
        if len(naca_str) != 4 or not naca_str.isdigit():
            raise ValueError("NACA code must be a 4-digit string, e.g. '2412'.")
        m_val = int(naca_str[0]) / 100.0
        p = int(naca_str[1]) / 10.0
        t = int(naca_str[2:]) / 100.0
        return m_val, p, t

    def close_curve(self, curve, tol=1e-6):
        if len(curve) < 2:
            return curve
        if m.hypot(curve[0][0] - curve[-1][0], curve[0][1] - curve[-1][1]) > tol:
            curve.append(curve[0])
        return curve

    def generate_naca_airfoil(self, chord, naca, num_points=50):
        m_val, p, t = self.parse_naca(naca)
        beta = np.linspace(0, m.pi, num_points)
        x = 0.5 * (1 - np.cos(beta))  # Cosine spacing: x goes from 0 to 1
        camber = []
        dydx = []
        for xi in x:
            if xi < p:
                yc = (m_val/(p**2)) * (2*p*xi - xi**2)
                dydx_val = 2*m_val/(p**2) * (p - xi)
            else:
                yc = (m_val/((1-p)**2)) * ((1-2*p) + 2*p*xi - xi**2)
                dydx_val = 2*m_val/((1-p)**2) * (p - xi)
            camber.append(yc)
            dydx.append(dydx_val)
        yt = 5*t*(0.2969*np.sqrt(x) - 0.1260*x - 0.3516*x**2 +
                  0.2843*x**3 - 0.1015*x**4)
        upper = []
        lower = []
        for xi, yc, dxi, thickness in zip(x, camber, dydx, yt):
            theta = m.atan(dxi)
            x_upper = xi * chord - thickness * chord * m.sin(theta)
            y_upper = yc * chord + thickness * chord * m.cos(theta)
            x_lower = xi * chord + thickness * chord * m.sin(theta)
            y_lower = yc * chord - thickness * chord * m.cos(theta)
            upper.append((x_upper, y_upper, 0.0))
            lower.append((x_lower, y_lower, 0.0))
        airfoil = upper[::-1] + lower[1:]
        return self.close_curve(airfoil)

    def recenter_airfoil(self, airfoil, chord):
        recentered = []
        for (x, y, z) in airfoil:
            recentered.append((x - chord/2, y, z))
        return recentered

    def rotate_about_z(self, points, angle_rad):
        cos_a = m.cos(angle_rad)
        sin_a = m.sin(angle_rad)
        rotated = []
        for (x, y, z) in points:
            x_new = x * cos_a - y * sin_a
            y_new = x * sin_a + y * cos_a
            rotated.append((x_new, y_new, z))
        return rotated

    def generate_section(self, r, num_airfoil_points=50):
        chord = self.chord_distribution_custom(r, self.pa.radius,
                                                 self.pa.root, self.pa.cmax, self.pa.tip)
        chord = max(chord, 1e-4)
        # 1) Generate the airfoil with chord running from x=0 to x=chord.
        airfoil = self.generate_naca_airfoil(chord, self.pa.naca, num_points=num_airfoil_points)
        # 2) Recenter the airfoil about its midpoint (approx. at x=chord/2, y=0).
        recentered = self.recenter_airfoil(airfoil, chord)
        # 3) Apply twist by rotating about the Z-axis by twist(r)
        twist_angle = self.pa.twist(r)
        twisted = self.rotate_about_z(recentered, twist_angle)
        # 4) Translate vertically so that the section's center is at z = r.
        #    (The X and Y coordinates remain unchanged.)
        section = [(x, y, r) for (x, y, z) in twisted]
        return self.close_curve(section)

    def write_section_file(self, filename, points):
        with open(filename, "w") as f:
            for (x, y, z) in points:
                f.write(f"{x:.6f}, {y:.6f}, {z:.6f}\n")

    def main(self):
        num_sections = 10
        num_airfoil_points = 50
        for i in range(num_sections):
            # Here, r is taken as the vertical (stacking) coordinate.
            # All airfoil midpoints lie at (0,0) in XY and at z = r.
            r = self.pa.radius * i / (num_sections - 1)
            section_points = self.generate_section(r, num_airfoil_points)
            filename = f"section_{i+1:02d}.txt"
            self.write_section_file(filename, section_points)
            print(f"Generated {filename}")

# ====================================
# Main Execution Block with Parameter Sweep
# ====================================
if __name__ == "__main__":
    # Define parameter ranges.
    density_range    = [1025]
    blade_num_range  = [1]
    radius_range     = [0.25]
    mass_range       = [83]
    drag_coeff_range = [0.012]
    damping_range    = [0.02]
    naca_range       = ['2412']

    # Engine and gearbox parameters.
    input_rpm_range    = [90]
    gear_ratio_range   = [(1/5)]  
    engine_power_range = [240]

    # Twist and chord parameter ranges.
    root_range     = [0.04]
    tip_range      = [0.001]
    cmax_range     = [0.06]
    m_twist_range  = [0.0808]
    c_twist_range  = [0.2048]

    results = []
    time_series_results = []  # To store time step data for every iteration.
    iteration_counter = 0

    # Loop over all parameter combinations.
    for (density, blade_num, radius, mass, drag_coeff, damping, naca,
         input_rpm, gear_ratio, engine_power) in itertools.product(
            density_range, blade_num_range, radius_range, mass_range,
            drag_coeff_range, damping_range, naca_range,
            input_rpm_range, gear_ratio_range, engine_power_range):
        for (root, tip, cmax, m_twist, c_twist) in itertools.product(
                root_range, tip_range, cmax_range, m_twist_range, c_twist_range):

            iteration_counter += 1
            # Create the propeller analysis instance.
            prop_analysis = PropellerAnalysis(
                density=density,
                blade_num=blade_num,
                radius=radius,
                mass=mass,
                drag_coeff=drag_coeff,
                base_damping_factor=damping,
                naca=naca,
                gear_ratio=gear_ratio,
                input_rpm=input_rpm,
                engine_power=engine_power
            )
            prop_analysis.root    = root
            prop_analysis.tip     = tip
            prop_analysis.cmax    = cmax
            prop_analysis.m_twist = m_twist
            prop_analysis.c_twist = c_twist

            try:
                final_result, time_series = prop_analysis.dynamic_simulation("NACA 4412.csv")
            except Exception as e:
                print(f"Error during simulation: {e}")
                continue

            results.append({
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
                'c_twist': c_twist,
                'Input RPM': final_result['Input RPM'],
                'Output RPM': final_result['Output RPM'],
                'Boat Velocity': final_result['Boat Velocity'],
                'Thrust': final_result['Thrust'],
                'Torque': final_result['Torque'],
                'Efficiency': final_result['Efficiency'],
                'Useful Power (W)': final_result['Useful Power (W)'],
                'Time': final_result['Time']
            })
            print(f"Iteration {iteration_counter}: Completed simulation for input_rpm={input_rpm}, gear_ratio={gear_ratio}")

            # Append iteration number to each time step record and add to global list.
            for record in time_series:
                record['Iteration'] = iteration_counter
                time_series_results.append(record)

    # Save overall simulation results.
    results_df = pd.DataFrame(results)
    results_df.to_csv("Main_results.csv", index=False)
    print("Data saved to 'Main_results.csv'.")
    print("\nPreview of Simulation Results:")
    print(results_df.head())

    # Save time step results from every simulation.
    ts_df = pd.DataFrame(time_series_results)
    ts_df.to_csv("Scale factor 0.5.csv", index=False)
    print("Time series data saved to 'naca 2412_results'.")

    # Plot: Boat Velocity vs. Tip Twist Angle (c_twist).
    plt.figure(figsize=(8, 5))
    plt.plot(results_df['radius'], results_df['Boat Velocity'], marker='o', linestyle='-')
    plt.xlabel("radius (m)")
    plt.ylabel("Boat Velocity (m/s)")
    plt.title("Boat Velocity vs. radius")
    plt.grid(True)
    plt.show()
    
    engine_power_val = 240  
    
    plt.figure(figsize=(8, 5))
    # If there are multiple iterations, plot each on the same graph.
    for iter_num in ts_df['Iteration'].unique():
        iter_data = ts_df[ts_df['Iteration'] == iter_num]
        # Calculate useful power at each time step (in Watts)
        useful_power = iter_data['Boat Velocity'] * iter_data['Thrust']
        mechanical_loss = engine_power_val - useful_power
        plt.plot(iter_data['time'], mechanical_loss, marker='o', linestyle='-', label=f"Iteration {iter_num}")
    


    # Create an instance of PropellerAnalysis
    pa = PropellerAnalysis()
    
    if pa.cl_fit is None or pa.cd_fit is None:
        pa.alpha_min = 0.0
        pa.alpha_max = np.pi / 4  # for example, 45 degrees in radians
        pa.cl_min = 0.0
        pa.cl_max = 1.0
        pa.cd_min = 0.0
        pa.cd_max = 0.1
        pa.cl_fit = np.poly1d([1.0, 0.0])
        pa.cd_fit = np.poly1d([0.1, 0.0])
    
    # Set up the radial discretisation parameters
    r_min = 0.2 * pa.radius
    r_max = pa.radius
    radius_points = np.linspace(r_min, r_max, pa.total_points)
    twist_angles = pa.twist(radius_points)
    sigma = (pa.blade_num * pa.chord_distribution(radius_points)) / (2 * np.pi * radius_points)
    
    # Choose a blade element index (for example, index 10)
    r_index = 10
    
    # Define a range of fi values (in radians) to evaluate the residual function
    fi_values = np.linspace(0.05, np.pi/2, 200)
    residual_values = [pa._residual(fi, r_index, radius_points, twist_angles, sigma) for fi in fi_values]
    
    # Plotting the residual function
    plt.figure(figsize=(8,6))
    plt.plot(fi_values, residual_values, label=f"Residual at blade element index {r_index}")
    plt.xlabel("fi (radians)")
    plt.ylabel("Residual")
    plt.title("Residual Function vs fi")
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.xlabel("Time (s)")
    plt.ylabel("Mechanical Loss (W)")
    plt.title("Mechanical Losses Over Time")
    plt.grid(True)
    plt.show()
    # Ask whether to generate SolidWorks coordinates.
    generate_coords = input("Do you want to generate SolidWorks coordinates? (y/n): ").strip().lower()
    if generate_coords == 'y':
        solidworks = SolidWorks(prop_analysis)
        solidworks.main()
    else:
        print("Skipping SolidWorks coordinate generation.")
