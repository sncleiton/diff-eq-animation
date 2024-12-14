import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# Parâmetros iniciais do modelo SIR
beta = 0.3  # Taxa de transmissão
gamma = 0.1  # Taxa de recuperação
population = 1000  # População total
initial_infected = 1  # Número inicial de infectados
initial_recovered = 0  # Número inicial de recuperados
initial_susceptible = population - initial_infected - initial_recovered

animation_type = "curves"  # Alterna entre "curves" e "points"
is_paused = False  # Variável para controlar o estado de pausa

# Intervalo de tempo para a simulação
t_span = (0, 200)
t_eval = np.linspace(*t_span, 500)

# Resolver o modelo SIR
def solve_sir(beta, gamma):
    def sir(t, populations):
        S, I, R = populations
        dS_dt = -beta * S * I / population
        dI_dt = beta * S * I / population - gamma * I
        dR_dt = gamma * I
        return [dS_dt, dI_dt, dR_dt]

    initial_populations = [initial_susceptible, initial_infected, initial_recovered]
    solution = solve_ivp(sir, t_span, initial_populations, t_eval=t_eval)
    return solution.t, solution.y[0], solution.y[1], solution.y[2]

# Dados iniciais
time, susceptible, infected, recovered = solve_sir(beta, gamma)

# Configuração da figura
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)
scat = None

def setup_plot():
    """Configura o gráfico inicial."""
    global s_line, i_line, r_line

    ax.clear()
    ax.set_xlim(0, t_span[1])
    ax.set_ylim(0, population)
    ax.set_title("Simulação do modelo SIR")
    ax.set_xlabel("Tempo")
    ax.set_ylabel("População")
    s_line, = ax.plot([], [], label="Susceptíveis", color="blue", lw=2)
    i_line, = ax.plot([], [], label="Infectados", color="red", lw=2)
    r_line, = ax.plot([], [], label="Recuperados", color="green", lw=2)
    ax.legend(loc='upper right')

    return ax

def update_curves(frame):
    """Atualiza o gráfico de curvas."""
    ax.legend(loc='upper right')

    s_line.set_data(time[:frame], susceptible[:frame])
    i_line.set_data(time[:frame], infected[:frame])
    r_line.set_data(time[:frame], recovered[:frame])
    return s_line, i_line, r_line

def update_points(frame):
    """Atualiza a animação de pontos."""
    ax.clear()
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_title("Animação de pontos: SIR")
    
    # Distribuir pontos proporcionalmente às populações
    np.random.seed(1)
    susceptible_positions = np.random.uniform(-50, 50, (int(susceptible[frame]), 2))
    infected_positions = np.random.uniform(-50, 50, (int(infected[frame]), 2))
    recovered_positions = np.random.uniform(-50, 50, (int(recovered[frame]), 2))
    
    ax.scatter(susceptible_positions[:, 0], susceptible_positions[:, 1], color="blue", alpha=0.6, label="Susceptíveis")
    ax.scatter(infected_positions[:, 0], infected_positions[:, 1], color="red", alpha=0.6, label="Infectados")
    ax.scatter(recovered_positions[:, 0], recovered_positions[:, 1], color="green", alpha=0.6, label="Recuperados")
    ax.legend(loc='upper right')
    return ax

def on_key(event):
    """Controla pausa e alternância de animação."""
    global is_paused, animation_type
    if event.key == " ":
        is_paused = not is_paused
        if is_paused:
            ani.event_source.stop()
        else:
            ani.event_source.start()
    elif event.key == "t":
        animation_type = "points" if animation_type == "curves" else "curves"
        reset_animation()

def on_click(event):
    """Exibe informações das populações no ponto clicado."""
    if 0 <= event.xdata <= t_span[1]:
        time_idx = np.abs(time - event.xdata).argmin()
        print(f"Tempo: {event.xdata:.1f}, Susceptíveis: {susceptible[time_idx]:.1f}, "
              f"Infectados: {infected[time_idx]:.1f}, Recuperados: {recovered[time_idx]:.1f}")

def reset_animation():
    """Reinicia a animação com o tipo atual."""
    global ani
    ani.event_source.stop()
    if animation_type == "curves":
        setup_plot()
        ani = FuncAnimation(fig, update_curves, frames=len(time), interval=30, blit=True)
    elif animation_type == "points":
        ani = FuncAnimation(fig, update_points, frames=len(time), interval=150, blit=False)
    plt.draw()

def update_params(val):
    """Atualiza os parâmetros e reinicia a animação."""
    global beta, gamma, time, susceptible, infected, recovered
    beta = beta_slider.val
    gamma = gamma_slider.val
    time, susceptible, infected, recovered = solve_sir(beta, gamma)
    reset_animation()

# Configuração inicial do gráfico
global s_line, i_line, r_line
setup_plot()

ani = FuncAnimation(fig, update_curves, frames=len(time), interval=30, blit=True)

# Adiciona sliders para ajustar parâmetros
ax_beta = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor="lightgoldenrodyellow")
ax_gamma = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor="lightgoldenrodyellow")

beta_slider = Slider(ax_beta, "Beta", 0.1, 1.0, valinit=beta, valstep=0.01)
gamma_slider = Slider(ax_gamma, "Gamma", 0.01, 0.5, valinit=gamma, valstep=0.01)

beta_slider.on_changed(update_params)
gamma_slider.on_changed(update_params)

# Eventos de teclado e clique
fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("button_press_event", on_click)

plt.show()
