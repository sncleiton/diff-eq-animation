import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp

# Constante gravitacional
G = 1.0

# Configuração inicial do sistema
mass_star = 50.0  # Massa da estrela central
planets = [
    {"mass": 1.0, "position": [10.0, 0.0], "velocity": [0.0, 1.5]},  # Planeta 1
    {"mass": 0.5, "position": [7.0, 0.0], "velocity": [0.0, 2.0]},  # Planeta 2
    {"mass": 2.0, "position": [15.0, 0.0], "velocity": [0.0, 1.2]},  # Planeta 3
]
simulation_time = 100  # Tempo de simulação

# Resolução temporal
time_span = (0, simulation_time)
time_eval = np.linspace(*time_span, 2000)

# Função para calcular forças gravitacionais
def compute_acceleration(positions, masses):
    n = len(positions)
    accelerations = np.zeros_like(positions)
    for i in range(n):
        for j in range(n):
            if i != j:
                r_vec = positions[j] - positions[i]
                r = np.linalg.norm(r_vec)
                accelerations[i] += G * masses[j] * r_vec / r**3
    return accelerations

# Função para resolver o sistema de equações diferenciais
def n_body_orbits(t, state, masses):
    n = len(masses)
    positions = state[:2*n].reshape((n, 2))
    velocities = state[2*n:].reshape((n, 2))
    accelerations = compute_acceleration(positions, masses)
    derivatives = np.concatenate([velocities.flatten(), accelerations.flatten()])
    return derivatives

# Preparar condições iniciais
def prepare_initial_conditions(planets):
    positions = np.array([p["position"] for p in planets])
    velocities = np.array([p["velocity"] for p in planets])
    masses = np.array([p["mass"] for p in planets])
    initial_conditions = np.concatenate([positions.flatten(), velocities.flatten()])
    return masses, initial_conditions

masses, initial_conditions = prepare_initial_conditions(planets)

# Resolver o sistema
def solve_system(mass_star, planets):
    global masses, initial_conditions
    star_position = np.array([0.0, 0.0])  # Estrela fixa no centro
    star_mass = np.array([mass_star])
    all_masses = np.concatenate([star_mass, masses])
    all_positions = np.vstack([star_position, initial_conditions[:2*len(planets)].reshape((len(planets), 2))])
    all_velocities = np.vstack([np.array([0.0, 0.0]), initial_conditions[2*len(planets):].reshape((len(planets), 2))])
    all_initial_conditions = np.concatenate([all_positions.flatten(), all_velocities.flatten()])
    sol = solve_ivp(
        n_body_orbits, time_span, all_initial_conditions, t_eval=time_eval, args=(all_masses,)
    )
    return sol.t, sol.y

time, solution = solve_system(mass_star, planets)
n_bodies = len(planets) + 1  # Inclui a estrela
positions = solution[:2*n_bodies].reshape((n_bodies, 2, -1))

# Configuração do gráfico
fig, ax = plt.subplots(figsize=(8, 8))
plt.subplots_adjust(bottom=0.3)

def setup_plot():
    ax.clear()
    ax.set_xlim(-20, 20)
    ax.set_ylim(-20, 20)
    ax.set_title("Órbitas de Planetas")
    ax.set_xlabel("x (posição)")
    ax.set_ylabel("y (posição)")
    ax.plot(0, 0, 'yo', markersize=12, label="Estrela (fixa)")  # Estrela no centro
    return ax

setup_plot()

# Adicionar trajetórias e planetas
colors = ["b", "g", "r", "c", "m", "y", "k"]
planet_lines = []
planet_points = []
for i in range(1, n_bodies):  # Começa no 1 porque a estrela é fixa
    line, = ax.plot([], [], colors[i % len(colors)] + "-", label=f"Planeta {i}")
    point, = ax.plot([], [], colors[i % len(colors)] + "o", markersize=8)
    planet_lines.append(line)
    planet_points.append(point)

ax.legend()

# Parâmetro para o comprimento da trilha
trail_length = 200  # Número de pontos mais recentes a serem exibidos

# Atualização da animação com trilha limitada
def update_orbits(frame):
    for i in range(1, n_bodies):
        start = max(0, frame - trail_length)  # Mostra apenas os últimos 'trail_length' pontos
        planet_lines[i-1].set_data(positions[i, 0, start:frame], positions[i, 1, start:frame])
        planet_points[i-1].set_data(positions[i, 0, frame], positions[i, 1, frame])
    return planet_lines + planet_points

ani = FuncAnimation(fig, update_orbits, frames=len(time), interval=30, blit=True)

# Sliders para controle
ax_star = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor="lightgoldenrodyellow")
ax_planet = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor="lightgoldenrodyellow")
ax_distance = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow")
ax_velocity = plt.axes([0.15, 0.00, 0.65, 0.03], facecolor="lightgoldenrodyellow")

star_slider = Slider(ax_star, "Massa Estrela", 1.0, 100.0, valinit=mass_star, valstep=1.0)
planet_slider = Slider(ax_planet, "Massa Planetas", 0.1, 5.0, valinit=planets[0]["mass"], valstep=0.1)
distance_slider = Slider(ax_distance, "Distância Inicial", 5.0, 20.0, valinit=planets[0]["position"][0], valstep=0.5)
velocity_slider = Slider(ax_velocity, "Velocidade Inicial", 0.5, 5.0, valinit=planets[0]["velocity"][1], valstep=0.1)

def update_params(val):
    ani.event_source.stop()  # Para a animação
    ani.event_source.start()  # Reinicia a animação desde o início
    ani.frame_seq = ani.new_frame_seq()  # Reseta a sequência de frames
    ani.event_source.start()  # Reinicia a animação

    """Atualiza os parâmetros interativos e reinicia a simulação."""
    global planets, mass_star
    mass_star = star_slider.val
    for p in planets:
        p["mass"] = planet_slider.val
        p["position"][0] = distance_slider.val
        p["velocity"][1] = velocity_slider.val
    reset_simulation()

def reset_simulation():
    """Reinicia a simulação com novos parâmetros."""
    global time, solution, positions, ani
    time, solution = solve_system(mass_star, planets)
    positions = solution[:2*n_bodies].reshape((n_bodies, 2, -1))
    ani.event_source.stop()
    setup_plot()
    ani.event_source.start()

star_slider.on_changed(update_params)
planet_slider.on_changed(update_params)
distance_slider.on_changed(update_params)
velocity_slider.on_changed(update_params)

plt.show()
