import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import colors as mcolors
from scipy.linalg import solve_banded
from scipy import interpolate
from scipy.fft import fft, fftfreq  # For FFT in vortex detection
from scipy.interpolate import griddata  # For particle velocity interp
import copy

class Particle:
    def __init__(self, pos, vel, size=0.05, density=1.5):  # Density >1 for sedimentation
        self.pos = np.array(pos, dtype=float)  # [x,y]
        self.vel = np.array(vel, dtype=float)
        self.size = size
        self.density = density
        self.stuck = False
        self.exited = False  # NEW: Flag for exited domain

class ChannelFlowSimulator:
    def __init__(self, nx=400, ny=101, Lx=4.0, Ly=1.0, nu=0.001, G=0.1, dt_fixed=0.0005, nt=200000,
                 constriction_width=0.4, constriction_strength=0.1, constriction_shape='gaussian', min_height=0.1,
                 roughness_amp=0.01, high_res_n=100, save_interval_steps=1000, plot_interval_time=0.1, ideal_case=False, c_cfl=5.0,
                 dynamic_constriction=False, osc_amp=0.0, osc_freq=0.0,
                 n_particles=100, particle_size=0.05, gravity=0.1, stick_prob=0.2, clog_threshold=0.5):
        self.nx = nx
        self.ny = ny
        self.Lx = Lx
        self.Ly = Ly
        self.nu = nu
        self.G = G
        self.dt_fixed = dt_fixed
        self.t_max = nt * dt_fixed  # Target total simulation time
        self.dt_max = 10.001  # Upper bound for dt
        self.dt = dt_fixed  # Initial dt
        self.safety_factor = 0.8  # For stability conditions
        self.c_v = 4.0  # Viscous stability constant (2D)
        self.c_cfl = c_cfl  # CFL constant, increased for larger dt
        self.dx = Lx / nx
        self.dy = Ly / (ny - 1)
        self.min_dxdy = min(self.dx, self.dy)
        self.min_dxdy2 = self.min_dxdy**2
        self.constriction_width = constriction_width
        self.constriction_strength = 0.0 if ideal_case else constriction_strength
        self.constriction_shape = constriction_shape.lower()
        self.min_height = min_height
        self.roughness_amp = 0.0 if ideal_case else roughness_amp
        self.high_res_n = high_res_n
        self.save_interval_time = save_interval_steps * dt_fixed
        self.plot_interval_time = plot_interval_time  
        self.dynamic_constriction = dynamic_constriction
        self.osc_amp = osc_amp
        self.osc_freq = osc_freq
        self.n_particles = n_particles

        # Compute Re
        self.Re = self.G * self.Ly**3 / (16 * self.nu**2)

        # Theoretical Poiseuille center velocity
        self.U_theory = self.G * self.Ly**2 / (8 * self.nu)

        # Grid
        self.x = np.linspace(0, self.Lx, self.nx)
        self.y = np.linspace(0, self.Ly, self.ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        # Center y index
        self.center_j = np.argmin(np.abs(self.y - self.Ly / 2))

        # Generate geometry
        self._generate_geometry()

        # Fields
        self.u = np.zeros((self.ny, self.nx))
        self.v = np.zeros((self.ny, self.nx))
        self.p = np.zeros((self.ny, self.nx))  # Pressure field

        # Precompute for Poisson
        self.kx = 2 * np.pi * np.fft.fftfreq(self.nx, d=self.dx)
        self.a = np.ones(self.ny) / self.dy**2
        self.b_base = np.ones(self.ny) * (-2 / self.dy**2)
        self.c = np.ones(self.ny) / self.dy**2
        self.a[0] = 0
        self.c[0] = 2 / self.dy**2
        self.a[-1] = 2 / self.dy**2
        self.c[-1] = 0

        # For animation and plotting
        self.u_list = []
        self.v_list = []
        self.p_list = []
        self.particles_list = []
        self.save_times = []
        self.times = []
        self.avg_center_vels = []
        self.delta_p_list = []
        self.shedding_freq_list = []
        self.diss_list = []
        self.particle_trajectories = [[] for _ in range(n_particles)] if n_particles > 0 else []
        self.fig = None
        self.axs = None
        self.pc1 = None
        self.pc2 = None
        self.pc3 = None
        self.cbar1 = None
        self.cbar2 = None
        self.cbar3 = None
        self.current_t = 0.0
        self.save_next_t = 0.0
        self.plot_next_t = 0.0
        self.reached_steady = False
        self.vort_time_series = []
        self.probe_x = self.Lx / 2 + 0.5 * self.constriction_width
        self.probe_i = np.argmin(np.abs(self.x - self.probe_x))
        self.probe_j = self.center_j

        # Particles
        self.particles = []
        self.gravity = gravity
        self.stick_prob = stick_prob
        self.clog_threshold = clog_threshold
        if n_particles > 0:
            for _ in range(n_particles):
                inlet_x = 0.1 * np.random.rand()
                closest_i = np.argmin(np.abs(self.x - inlet_x))
                local_lower = self.y_lower[closest_i]
                local_upper = self.y_upper[closest_i]
                inlet_y = local_lower + (local_upper - local_lower) * np.random.rand()
                inlet_pos = [inlet_x, inlet_y]
                self.particles.append(Particle(inlet_pos, [0.0, 0.0], particle_size))

    def _generate_geometry(self):
        high_res_x = np.linspace(0, self.Lx, self.high_res_n)
        np.random.seed(42)
        roughness = self.roughness_amp * np.random.randn(self.high_res_n)

        self.y_lower = np.zeros(self.nx)
        self.y_upper = self.Ly * np.ones(self.nx)

        constriction_center = self.Lx / 2
        for i in range(self.nx):
            x_pos = self.x[i]
            if self.constriction_strength > 0:
                if self.constriction_shape == 'gaussian':
                    constriction = self.constriction_strength * np.exp(-((x_pos - constriction_center) / self.constriction_width)**2)
                elif self.constriction_shape == 'step':
                    half_width = self.constriction_width / 2
                    constriction = self.constriction_strength if abs(x_pos - constriction_center) <= half_width else 0
                elif self.constriction_shape == 'triangular':
                    half_width = self.constriction_width / 2
                    dist = abs(x_pos - constriction_center)
                    constriction = self.constriction_strength * (1 - dist / half_width) if dist <= half_width else 0
                elif self.constriction_shape == 'sinusoidal':
                    constriction = self.constriction_strength * (1 + np.sin(2 * np.pi * (x_pos - constriction_center + self.constriction_width / 2) / self.constriction_width)) / 2 if abs(x_pos - constriction_center) <= self.constriction_width / 2 else 0
                else:
                    raise ValueError(f"Unknown constriction shape: {self.constriction_shape}. Options: 'gaussian', 'step', 'triangular', 'sinusoidal'")
                self.y_lower[i] += constriction
                self.y_upper[i] -= constriction

        if self.roughness_amp > 0:
            rough_interp = np.interp(self.x, high_res_x, roughness)
            self.y_lower += rough_interp
            self.y_upper -= rough_interp

        self.y_lower = np.clip(self.y_lower, 0, self.Ly - self.min_height)
        self.y_upper = np.clip(self.y_upper, self.min_height, self.Ly)

        self.obstacle = (self.Y < self.y_lower[None, :]) | (self.Y > self.y_upper[None, :])

    def der_x(self, f):
        return (np.roll(f, -1, axis=1) - np.roll(f, 1, axis=1)) / (2 * self.dx)

    def der_y(self, f):
        dfdy = np.zeros_like(f)
        dfdy[1:-1, :] = (f[2:, :] - f[:-2, :]) / (2 * self.dy)
        return dfdy

    def lap(self, f):
        l = np.zeros_like(f)
        l[:, :] = (np.roll(f, -1, axis=1) + np.roll(f, 1, axis=1) - 2 * f) / self.dx**2
        l[1:-1, :] += (f[2:, :] + f[:-2, :] - 2 * f[1:-1, :]) / self.dy**2
        return l

    def solve_poisson(self, source):
        source_hat = np.fft.fft(source, axis=1)
        phi_hat = np.zeros((self.ny, self.nx), dtype=complex)
        for ik in range(self.nx):
            kk = self.kx[ik]**2
            bb = self.b_base - kk
            c_local = self.c.copy()
            if abs(self.kx[ik]) < 1e-10:
                bb[0] = 1
                c_local[0] = 0
                rhs = source_hat[:, ik].copy()
                rhs[0] = 0
            else:
                rhs = source_hat[:, ik]
            ab = np.zeros((3, self.ny))
            ab[0, 1:] = c_local[:-1]
            ab[1, :] = bb
            ab[2, :-1] = self.a[1:]
            phi_k = solve_banded((1, 1), ab, rhs)
            phi_hat[:, ik] = phi_k
        phi = np.fft.ifft(phi_hat, axis=1).real
        return phi

    def advect(self, field):
        back_x = np.clip(self.X - self.dt * self.u, 0, self.Lx)
        back_y = np.clip(self.Y - self.dt * self.v, 0, self.Ly)
        # Interpolate
        xi = np.stack((back_y.ravel(), back_x.ravel()), axis=-1)
        adv_field = interpolate.interpn((self.y, self.x), field, xi, method='linear', bounds_error=False, fill_value=0).reshape(self.ny, self.nx)
        return adv_field

    def step(self):
        # Compute dynamic dt based on current fields
        max_u = np.max(np.abs(self.u)) + 1e-10
        max_v = np.max(np.abs(self.v)) + 1e-10
        max_vel = max(max_u, max_v)
        dt_cfl = self.min_dxdy / (self.c_cfl * max_vel)
        dt_visc = self.min_dxdy2 / (self.c_v * self.nu)
        self.dt = min(dt_cfl, dt_visc, self.dt_max) * self.safety_factor

        if self.dynamic_constriction:
            constriction_mod = self.osc_amp * np.sin(2 * np.pi * self.osc_freq * self.current_t)
            self.constriction_strength += constriction_mod  # Temporary modulation
            self._generate_geometry()
            self.constriction_strength -= constriction_mod  # Reset base
            self.obstacle = (self.Y < self.y_lower[None, :]) | (self.Y > self.y_upper[None, :])

        # Semi-Lagrangian advection
        u_adv = self.advect(self.u)
        v_adv = self.advect(self.v)

        u_diff = u_adv + self.dt * self.nu * self.lap(u_adv)
        v_diff = v_adv + self.dt * self.nu * self.lap(v_adv)

        u_star = u_diff + self.dt * self.G
        v_star = v_diff
        
        u_star[self.obstacle] = 0
        v_star[self.obstacle] = 0
        
        u_star[0, :] = 0
        u_star[-1, :] = 0
        v_star[0, :] = 0
        v_star[-1, :] = 0
        
        div_star = self.der_x(u_star) + self.der_y(v_star)
        # Solve for phi
        phi = self.solve_poisson(div_star / self.dt)
        self.p = phi / self.dt
        # Project
        self.u = u_star - self.dt * self.der_x(phi)
        self.v = v_star - self.dt * self.der_y(phi)
        # Set to zero in obstacle
        self.u[self.obstacle] = 0
        self.v[self.obstacle] = 0
        # Enforce wall BC
        self.u[0, :] = 0
        self.u[-1, :] = 0
        self.v[0, :] = 0
        self.v[-1, :] = 0

        # Clip any NaNs/infs for robustness
        self.u = np.nan_to_num(self.u, nan=0.0, posinf=0.0, neginf=0.0)
        self.v = np.nan_to_num(self.v, nan=0.0, posinf=0.0, neginf=0.0)

        # Particle update (only if particles exist)
        if self.n_particles > 0:
            active_particles = [p for p in self.particles if not p.stuck and not p.exited]
            if active_particles:
                pos = np.array([p.pos for p in active_particles])
                xi = np.stack((pos[:,1], pos[:,0]), axis=-1)  # (y, x)
                fluid_u = interpolate.interpn((self.y, self.x), self.u, xi, method='linear', bounds_error=False, fill_value=0)
                fluid_v = interpolate.interpn((self.y, self.x), self.v, xi, method='linear', bounds_error=False, fill_value=0)
                for idx, part in enumerate(active_particles):
                    tau = (part.size**2 * (part.density - 1)) / (18 * self.nu) if (part.density - 1) != 0 else 1e-6
                    accel_x = (fluid_u[idx] - part.vel[0]) / tau
                    accel_y = (fluid_v[idx] - part.vel[1]) / tau - self.gravity
                    part.vel += np.array([accel_x, accel_y]) * self.dt
                    part.pos += part.vel * self.dt
                    part_idx = self.particles.index(part)
                    self.particle_trajectories[part_idx].append(part.pos.copy())
                    # Boundary check
                    if part.pos[0] > self.Lx or part.pos[0] < 0:
                        part.exited = True
                        # Reset to inlet for continuous injection
                        inlet_x = 0.1 * np.random.rand()
                        closest_i = np.argmin(np.abs(self.x - inlet_x))
                        local_lower = self.y_lower[closest_i]
                        local_upper = self.y_upper[closest_i]
                        inlet_y = local_lower + (local_upper - local_lower) * np.random.rand()
                        part.pos = np.array([inlet_x, inlet_y])
                        part.vel = np.array([0.0, 0.0])
                        part.exited = False
                        part.stuck = False
                        self.particle_trajectories[part_idx] = []
                        continue
                    closest_i = np.argmin(np.abs(self.x - part.pos[0]))
                    if part.pos[1] < self.y_lower[closest_i] or part.pos[1] > self.y_upper[closest_i]:
                        if np.random.rand() < self.stick_prob:
                            part.stuck = True
                        else:
                            part.vel[1] *= -0.8
                            part.pos[1] = np.clip(part.pos[1], self.y_lower[closest_i], self.y_upper[closest_i])

            # Clogging
            constr_center_i = self.nx // 2
            local_parts = [p for p in self.particles if abs(p.pos[0] - self.Lx/2) < self.constriction_width and p.stuck]
            density = len(local_parts) / (self.constriction_width * self.Ly) if self.constriction_width > 0 else 0  
            if density > self.clog_threshold:
                narrow = density * 0.1 * self.Ly
                self.y_lower[constr_center_i-10:constr_center_i+10] += narrow / 2
                self.y_upper[constr_center_i-10:constr_center_i+10] -= narrow / 2
                self._generate_geometry()

        vort_at_probe = (self.der_x(self.v) - self.der_y(self.u))[self.probe_j, self.probe_i]
        self.vort_time_series.append(vort_at_probe)

        self.current_t += self.dt

    def compute_delta_p(self):
        inlet_p = np.mean(self.p[:, 0])
        outlet_p = np.mean(self.p[:, -1])
        return inlet_p - outlet_p

    def compute_flow_rate(self):
        return np.trapz(self.u[:, 0], self.y)

    def detect_shedding_freq(self):
        if len(self.vort_time_series) < 200:
            return 0.0
        signal = np.array(self.vort_time_series)
        fft_vals = fft(signal)
        freqs = fftfreq(len(signal), self.dt)
        peak_freq = freqs[np.argmax(np.abs(fft_vals[1:])) + 1]
        return abs(peak_freq)

    def compute_clog_fraction(self):
        if self.n_particles == 0:
            return 0
        stuck = sum(p.stuck for p in self.particles)
        return stuck / len(self.particles) if self.particles else 0

    def compute_energy_dissipation(self):
        der_u_x = self.der_x(self.u)
        der_v_y = self.der_y(self.v)
        der_u_y = self.der_y(self.u)
        der_v_x = self.der_x(self.v)
        Sxx = der_u_x
        Syy = der_v_y
        Sxy = 0.5 * (der_u_y + der_v_x)
        phi = 2 * self.nu * (Sxx**2 + Syy**2 + 2 * Sxy**2)
        fluid_mask = ~self.obstacle & (self.Y >= self.y_lower[None, :] + 1e-6) & (self.Y <= self.y_upper[None, :] - 1e-6)
        diss = np.sum(phi[fluid_mask]) * self.dx * self.dy
        return diss

    def run(self, interactive=True):
        if interactive:
            plt.ion()
        self.u_list = []
        self.v_list = []
        self.p_list = []
        self.particles_list = []
        self.save_times = []
        self.times = []
        self.avg_center_vels = []
        self.delta_p_list = []  
        self.shedding_freq_list = [] 
        self.diss_list = []
        self._setup_plot()

        # Append initial state
        self.u_list.append(self.u.copy())
        self.v_list.append(self.v.copy())
        self.p_list.append(self.p.copy())
        if self.n_particles > 0:
            self.particles_list.append(copy.deepcopy(self.particles))
        self.save_times.append(self.current_t)
        avg_vel = np.mean(self.u[self.center_j, ~self.obstacle[self.center_j]]) if np.any(~self.obstacle[self.center_j]) else 0
        self.times.append(self.current_t)
        self.avg_center_vels.append(avg_vel)
        delta_p = self.compute_delta_p()
        self.delta_p_list.append(delta_p)
        shedding_freq = self.detect_shedding_freq()
        self.shedding_freq_list.append(shedding_freq)
        diss = self.compute_energy_dissipation()
        self.diss_list.append(diss)
        if interactive:
            self._update_plot(self.current_t)

        self.save_next_t = self.save_interval_time
        self.plot_next_t = self.plot_interval_time

        n = 0
        while self.current_t < self.t_max:
            self.step()
            if self.current_t >= self.save_next_t:
                self.u_list.append(self.u.copy())
                self.v_list.append(self.v.copy())
                self.p_list.append(self.p.copy())
                if self.n_particles > 0:
                    self.particles_list.append(copy.deepcopy(self.particles))
                self.save_times.append(self.current_t)
                self.save_next_t += self.save_interval_time
                delta_p = self.compute_delta_p()
                self.delta_p_list.append(delta_p)
                shedding_freq = self.detect_shedding_freq()
                self.shedding_freq_list.append(shedding_freq)
                diss = self.compute_energy_dissipation()
                self.diss_list.append(diss)
            if self.current_t >= self.plot_next_t:
                avg_vel = np.mean(self.u[self.center_j, ~self.obstacle[self.center_j]]) if np.any(~self.obstacle[self.center_j]) else 0
                self.times.append(self.current_t)
                self.avg_center_vels.append(avg_vel)
                if not self.reached_steady and avg_vel >= 0.99 * self.U_theory:
                    print(f"Reached 99% of Poiseuille velocity at t = {self.current_t:.2f} s")
                    self.reached_steady = True
                if interactive:
                    self._update_plot(self.current_t)
                self.plot_next_t += self.plot_interval_time
            if n % 1000 == 0:
                print(f"Step {n} for Re = {self.Re:.0f}, t={self.current_t:.4f}, dt={self.dt:.6f}")
            n += 1

        if interactive:
            plt.ioff()

    def _setup_plot(self):
        # Use 3 subplots for n_particles == 0 (velocity, vorticity, streamlines)
        # Use 4 subplots for n_particles > 0 (velocity, vorticity, pressure, streamlines & particles)
        n_subplots = 4 if self.n_particles > 0 else 3
        figsize = (24, 5) if self.n_particles > 0 else (18, 5)
        self.fig, self.axs = plt.subplots(1, n_subplots, figsize=figsize)
        self.fig.suptitle('Fluid Flow Simulation at t = %.2f (s), Re = %.0f' % (0, self.Re))

        masked_dummy_vel = np.ma.masked_where(self.obstacle, np.zeros((self.ny, self.nx)))
        masked_dummy_vort = np.ma.masked_where(self.obstacle, np.zeros((self.ny, self.nx)))
        masked_dummy_p = np.ma.masked_where(self.obstacle, np.zeros((self.ny, self.nx)))

        # Velocity magnitude plot
        self.pc1 = self.axs[0].pcolormesh(self.X, self.Y, masked_dummy_vel, vmin=0, vmax=1, shading='auto')
        self.axs[0].plot(self.x, self.y_lower, 'k-', lw=2)
        self.axs[0].plot(self.x, self.y_upper, 'k-', lw=2)
        self.axs[0].set_title('Velocity Magnitude')
        self.cbar1 = self.fig.colorbar(self.pc1, ax=self.axs[0], fraction=0.046, pad=0.04)

        # Vorticity plot
        self.pc2 = self.axs[1].pcolormesh(self.X, self.Y, masked_dummy_vort, vmin=-1, vmax=1, shading='auto')
        self.axs[1].plot(self.x, self.y_lower, 'k-', lw=2)
        self.axs[1].plot(self.x, self.y_upper, 'k-', lw=2)
        self.axs[1].set_title('Vorticity')
        self.cbar2 = self.fig.colorbar(self.pc2, ax=self.axs[1], fraction=0.046, pad=0.04)

        if self.n_particles > 0:
            # Pressure plot
            self.pc3 = self.axs[2].pcolormesh(self.X, self.Y, masked_dummy_p, shading='auto')
            self.axs[2].plot(self.x, self.y_lower, 'k-', lw=2)
            self.axs[2].plot(self.x, self.y_upper, 'k-', lw=2)
            self.axs[2].set_title('Pressure')
            self.cbar3 = self.fig.colorbar(self.pc3, ax=self.axs[2], fraction=0.046, pad=0.04)

            # Streamlines and particles plot
            self.axs[3].plot(self.x, self.y_lower, 'k-', lw=2)
            self.axs[3].plot(self.x, self.y_upper, 'k-', lw=2)
            self.axs[3].set_title('Streamlines & Particles')
            self.axs[3].streamplot(self.x, self.y, np.zeros((self.ny, self.nx)), np.zeros((self.ny, self.nx)), density=1)
            self.axs[3].set_xlim(0, self.Lx)
            self.axs[3].set_ylim(0, self.Ly)
        else:
            # Streamlines plot (no particles)
            self.axs[2].plot(self.x, self.y_lower, 'k-', lw=2)
            self.axs[2].plot(self.x, self.y_upper, 'k-', lw=2)
            self.axs[2].set_title('Streamlines')
            self.axs[2].streamplot(self.x, self.y, np.zeros((self.ny, self.nx)), np.zeros((self.ny, self.nx)), density=1)
            self.axs[2].set_xlim(0, self.Lx)
            self.axs[2].set_ylim(0, self.Ly)

    def _update_plot(self, t):
        vel_mag = np.sqrt(self.u**2 + self.v**2)
        vort = self.der_x(self.v) - self.der_y(self.u)
        masked_vel = np.ma.masked_where(self.obstacle, vel_mag)
        masked_vort = np.ma.masked_where(self.obstacle, vort)
        masked_p = np.ma.masked_where(self.obstacle, self.p)

        # Update velocity magnitude
        self.pc1.set_array(masked_vel.ravel())
        max_vel = max(np.nanmax(masked_vel), 1e-6)
        self.pc1.set_clim(0, max_vel)

        # Update vorticity
        self.pc2.set_array(masked_vort.ravel())
        min_vort = np.nanmin(masked_vort)
        max_vort = np.nanmax(masked_vort)
        bound_vort = max(abs(min_vort), abs(max_vort), 1e-6)
        self.pc2.set_clim(-bound_vort, bound_vort)

        if self.n_particles > 0:
            # Update pressure
            self.pc3.set_array(masked_p.ravel())
            min_p = np.nanmin(masked_p)
            max_p = np.nanmax(masked_p)
            self.pc3.set_clim(min_p, max_p)

            # Update streamlines and particles
            self.axs[3].clear()
            self.axs[3].plot(self.x, self.y_lower, 'k-', lw=2)
            self.axs[3].plot(self.x, self.y_upper, 'k-', lw=2)
            self.axs[3].set_title('Streamlines & Particles')
            self.axs[3].streamplot(self.x, self.y, self.u, self.v, density=1)
            free_x = [p.pos[0] for p in self.particles if not p.stuck and not p.exited]
            free_y = [p.pos[1] for p in self.particles if not p.stuck and not p.exited]
            stuck_x = [p.pos[0] for p in self.particles if p.stuck]
            stuck_y = [p.pos[1] for p in self.particles if p.stuck]
            self.axs[3].scatter(free_x, free_y, c='red', s=10, marker='o', label='Free Particles')
            self.axs[3].scatter(stuck_x, stuck_y, c='cyan', s=30, marker='s', label='Stuck Particles')
            self.axs[3].set_xlim(0, self.Lx)
            self.axs[3].set_ylim(0, self.Ly)
            self.axs[3].legend()
        else:
            # Update streamlines only
            self.axs[2].clear()
            self.axs[2].plot(self.x, self.y_lower, 'k-', lw=2)
            self.axs[2].plot(self.x, self.y_upper, 'k-', lw=2)
            self.axs[2].set_title('Streamlines')
            self.axs[2].streamplot(self.x, self.y, self.u, self.v, density=1)
            self.axs[2].set_xlim(0, self.Lx)
            self.axs[2].set_ylim(0, self.Ly)

        self.fig.suptitle('Fluid Flow Simulation at t = %.2f (s), Re = %.0f' % (t, self.Re))
        plt.draw()
        plt.pause(0.01)

    def create_animation(self, filename='simulation.gif'):
        if len(self.u_list) == 0:
            print("No frames to animate.")
            return

        def update(frame):
            t = self.save_times[frame]
            u_frame = self.u_list[frame]
            v_frame = self.v_list[frame]
            p_frame = self.p_list[frame]
            particles_frame = self.particles_list[frame] if self.n_particles > 0 else []
            vel_mag = np.sqrt(u_frame**2 + v_frame**2)
            vort = self.der_x(v_frame) - self.der_y(u_frame)
            masked_vel = np.ma.masked_where(self.obstacle, vel_mag)
            masked_vort = np.ma.masked_where(self.obstacle, vort)
            masked_p = np.ma.masked_where(self.obstacle, p_frame)

            # Update velocity magnitude
            self.pc1.set_array(masked_vel.ravel())
            max_vel = max(np.nanmax(masked_vel), 1e-6)
            self.pc1.set_clim(0, max_vel)

            # Update vorticity
            self.pc2.set_array(masked_vort.ravel())
            min_vort = np.nanmin(masked_vort)
            max_vort = np.nanmax(masked_vort)
            bound_vort = max(abs(min_vort), abs(max_vort), 1e-6)
            self.pc2.set_clim(-bound_vort, bound_vort)

            if self.n_particles > 0:
                # Update pressure
                self.pc3.set_array(masked_p.ravel())
                min_p = np.nanmin(masked_p)
                max_p = np.nanmax(masked_p)
                self.pc3.set_clim(min_p, max_p)

                # Update streamlines and particles
                self.axs[3].clear()
                self.axs[3].plot(self.x, self.y_lower, 'k-', lw=2)
                self.axs[3].plot(self.x, self.y_upper, 'k-', lw=2)
                self.axs[3].set_title('Streamlines & Particles')
                self.axs[3].streamplot(self.x, self.y, u_frame, v_frame, density=1)
                free_x = [p.pos[0] for p in particles_frame if not p.stuck and not p.exited]
                free_y = [p.pos[1] for p in particles_frame if not p.stuck and not p.exited]
                stuck_x = [p.pos[0] for p in particles_frame if p.stuck]
                stuck_y = [p.pos[1] for p in particles_frame if p.stuck]
                self.axs[3].scatter(free_x, free_y, c='red', s=10, marker='o', label='Free Particles')
                self.axs[3].scatter(stuck_x, stuck_y, c='cyan', s=30, marker='s', label='Stuck Particles')
                self.axs[3].set_xlim(0, self.Lx)
                self.axs[3].set_ylim(0, self.Ly)
                self.axs[3].legend()
            else:
                # Update streamlines only
                self.axs[2].clear()
                self.axs[2].plot(self.x, self.y_lower, 'k-', lw=2)
                self.axs[2].plot(self.x, self.y_upper, 'k-', lw=2)
                self.axs[2].set_title('Streamlines')
                self.axs[2].streamplot(self.x, self.y, u_frame, v_frame, density=1)
                self.axs[2].set_xlim(0, self.Lx)
                self.axs[2].set_ylim(0, self.Ly)

            self.fig.suptitle('Fluid Flow Simulation at t = %.2f (s), Re = %.0f' % (t, self.Re))
            return self.axs

        anim = FuncAnimation(self.fig, update, frames=len(self.u_list), interval=200, blit=False)
        anim.save(filename, writer='pillow')
        plt.show()

    def plot_center_velocity(self, filename='center_vel.png'):
        fig_vel, ax_vel = plt.subplots()
        ax_vel.plot(self.times, self.avg_center_vels, label='Simulation Avg Center Velocity')
        ax_vel.axhline(self.U_theory, linestyle='--', color='r', label='Poiseuille Law')
        ax_vel.set_xlabel('Time (s)')
        ax_vel.set_ylabel('Average Center Velocity')
        ax_vel.set_title(f'Average Center Velocity vs Time, Re = {self.Re:.0f}')
        ax_vel.legend()
        fig_vel.savefig(filename)
        plt.close(fig_vel)

    def plot_dissipation(self, filename='dissipation.png'):
        fig, ax = plt.subplots()
        ax.plot(self.save_times, self.diss_list)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Energy Dissipation Rate')
        ax.set_title(f'Energy Dissipation vs Time, Re = {self.Re:.0f}')
        fig.savefig(filename)
        plt.close(fig)

    def plot_particles(self, filename='particles.png'):
        if self.n_particles == 0:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.x, self.y_lower, 'k-', lw=2, label='Lower Wall')
        ax.plot(self.x, self.y_upper, 'k-', lw=2, label='Upper Wall')
        free_x = [p.pos[0] for p in self.particles if not p.stuck and not p.exited and 0 <= p.pos[0] <= self.Lx and 0 <= p.pos[1] <= self.Ly]
        free_y = [p.pos[1] for p in self.particles if not p.stuck and not p.exited and 0 <= p.pos[0] <= self.Lx and 0 <= p.pos[1] <= self.Ly]
        stuck_x = [p.pos[0] for p in self.particles if p.stuck and 0 <= p.pos[0] <= self.Lx and 0 <= p.pos[1] <= self.Ly]
        stuck_y = [p.pos[1] for p in self.particles if p.stuck and 0 <= p.pos[0] <= self.Lx and 0 <= p.pos[1] <= self.Ly]
        ax.scatter(free_x, free_y, c='red', s=10, marker='o', label='Free Particles')
        ax.scatter(stuck_x, stuck_y, c='cyan', s=30, marker='s', label='Stuck Particles')
        ax.set_title('Final Particle Positions: Stuck vs Free')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, self.Lx)
        ax.set_ylim(0, self.Ly)
        ax.legend()
        fig.savefig(filename)
        plt.close(fig)

    def plot_trajectories(self, filename='trajectories.png'):
        if self.n_particles == 0:
            return  
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(self.x, self.y_lower, 'k-', lw=2, label='Lower Wall')
        ax.plot(self.x, self.y_upper, 'k-', lw=2, label='Upper Wall')
        for traj in self.particle_trajectories:
            if traj:
                traj = np.array(traj)
                mask = (traj[:, 0] >= 0) & (traj[:, 0] <= self.Lx) & (traj[:, 1] >= 0) & (traj[:, 1] <= self.Ly)
                traj = traj[mask]
                if len(traj) > 1:
                    ax.plot(traj[:, 0], traj[:, 1], alpha=0.5)
        stuck_x = [p.pos[0] for p in self.particles if p.stuck and 0 <= p.pos[0] <= self.Lx and 0 <= p.pos[1] <= self.Ly]
        stuck_y = [p.pos[1] for p in self.particles if p.stuck and 0 <= p.pos[0] <= self.Lx and 0 <= p.pos[1] <= self.Ly]
        ax.scatter(stuck_x, stuck_y, c='cyan', s=30, marker='s', label='Stuck Particles')
        ax.set_title('Particle Trajectories')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, self.Lx)
        ax.set_ylim(0, self.Ly)
        ax.legend()
        fig.savefig(filename)
        plt.close(fig)

    def plot_density_map(self, filename='density.png'):
        if self.n_particles == 0:
            return
        stuck_pos = [p.pos for p in self.particles if p.stuck and 0 <= p.pos[0] <= self.Lx and 0 <= p.pos[1] <= self.Ly]
        if stuck_pos:
            stuck_pos = np.array(stuck_pos)
            fig, ax = plt.subplots(figsize=(10, 5))
            h = ax.hist2d(stuck_pos[:, 0], stuck_pos[:, 1], bins=50, range=[[0, self.Lx], [0, self.Ly]])
            ax.scatter(stuck_pos[:, 0], stuck_pos[:, 1], c='cyan', s=30, marker='s', label='Stuck Particles')
            fig.colorbar(h[3], ax=ax)
            ax.set_title('Stuck Particle Density Map')
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.legend()
            fig.savefig(filename)
            plt.close(fig)

# To perform sweep over different Re
def re_sweep(re_list=[1000, 3000, 5000, 7000]):
    G = 0.1
    Ly = 1.0
    for re in re_list:
        # Compute nu for given Re
        nu = np.sqrt(G * Ly**3 / (16 * re))
        sim = ChannelFlowSimulator(nu=nu, ideal_case=True, n_particles=0)
        sim.run(interactive=False)  # Run non-interactively for sweep
        sim.create_animation(filename=f'simulation_Re_{re:.0f}.gif')
        sim.plot_center_velocity(filename=f'center_vel_Re_{re:.0f}.png')
        sim.plot_dissipation(filename=f'diss_Re_{re:.0f}.png')

def constriction_sweep(shapes=['gaussian', 'step'], strengths=[0.05, 0.35], re_list=[1000, 4000, 7000]):
    results = {}  # Dict to store R, St, Diss per case
    for shape in shapes:
        for strength in strengths:
            for re in re_list:
                nu = np.sqrt(0.1 * 1.0**3 / (16 * re))  # Adjust for G=0.1, Ly=1
                sim = ChannelFlowSimulator(constriction_shape=shape, constriction_strength=strength, nu=nu, n_particles=0)
                sim.run(interactive=False)
                avg_delta_p = np.mean(sim.delta_p_list[-10:])  # Steady-state avg
                avg_Q = sim.compute_flow_rate()  # Final state
                R = avg_delta_p / avg_Q if avg_Q != 0 else np.inf
                avg_St = np.mean(sim.shedding_freq_list[-10:]) * sim.constriction_width / sim.U_theory
                avg_diss = np.mean(sim.diss_list[-10:])
                results[(shape, strength, re)] = {'R': R, 'St': avg_St, 'Diss': avg_diss}
                sim.create_animation(f'simulation_{shape}_str{strength}_Re{re}.gif')
                sim.plot_dissipation(f'diss_{shape}_str{strength}_Re{re}.png')
    # Save results to CSV
    import pandas as pd
    df = pd.DataFrame([(k[0], k[1], k[2], v['R'], v['St'], v['Diss']) for k, v in results.items()],
                      columns=['Shape', 'Strength', 'Re', 'R', 'St', 'Diss'])
    df.to_csv('constriction_results.csv', index=False)
    return results

def particle_sweep(re_list=[1000, 7000], strengths=[0.05, 0.35], n_parts=[1000]):
    results = {}
    for re in re_list:
        for strength in strengths:
            for num_parts in n_parts:
                nu = np.sqrt(0.1 * 1.0**3 / (16 * re))
                sim = ChannelFlowSimulator(constriction_strength=strength, nu=nu, n_particles=num_parts)
                sim.run(interactive=False)
                clog_frac = sim.compute_clog_fraction()
                results[(re, strength, num_parts)] = clog_frac
                sim.create_animation(f'simulation_particle_Re{re}_str{strength}_n{num_parts}.gif')
                sim.plot_center_velocity(f'center_vel_particle_Re{re}_str{strength}_n{num_parts}.png')
                sim.plot_dissipation(f'diss_particle_Re{re}_str{strength}_n{num_parts}.png')
                sim.plot_particles(f'particles_Re{re}_str{strength}_n{num_parts}.png')
                sim.plot_trajectories(f'traj_particle_Re{re}_str{strength}_n{num_parts}.png')
                sim.plot_density_map(f'density_particle_Re{re}_str{strength}_n{num_parts}.png')
    return results

if __name__ == "__main__":
    re_sweep()
    constriction_results = constriction_sweep()
    print(constriction_results)
    particle_results = particle_sweep()
    print(particle_results)