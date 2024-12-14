import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider

# Parâmetros iniciais do modelo Lotka-Volterra
alpha = 0.1  # Taxa de crescimento das presas
beta = 0.02  # Taxa de caça (interação presa-predador)
delta = 0.01  # Taxa de crescimento dos predadores ao se alimentar
gamma = 0.1  # Taxa de morte natural dos predadores
animation_type = "curves"  # Alterna entre "curves" (gráfico) e "points" (animação de pontos)

# Populações iniciais
initial_populations = [40, 9]  # Presas e predadores
t_span = (0, 200)  # Intervalo de tempo
t_eval = np.linspace(*t_span, 500)  # Pontos de avaliação

# Variável de pausa
is_paused = False

# Função para resolver o modelo Lotka-Volterra
def solve_model(alpha, beta, delta, gamma):
    def lotka_volterra(t, populations):
        P, C = populations  # Presas (P) e predadores (C)
        dP_dt = alpha * P - beta * P * C
        dC_dt = delta * P * C - gamma * C
        return [dP_dt, dC_dt]

    solution = solve_ivp(lotka_volterra, t_span, initial_populations, t_eval=t_eval)
    return solution.t, solution.y[0], solution.y[1]

# Dados iniciais
time, preys, predators = solve_model(alpha, beta, delta, gamma)

# Criação da figura
fig, ax = plt.subplots(figsize=(10, 6))
plt.subplots_adjust(bottom=0.25)  # Ajusta espaço para os sliders
scat = None  # Placeholder para animação de pontos

def setup_plot():
    """Configura o gráfico inicial."""
    global predator_line, prey_line
    ax.clear()
    ax.set_xlim(0, t_span[1])
    ax.set_ylim(0, max(max(preys), max(predators)) * 1.1)
    ax.set_title("Animação: Simulação de Lotka-Volterra")
    ax.set_xlabel("Tempo")
    ax.set_ylabel("População")
    prey_line, = ax.plot([], [], label="Presas (Coelhos)", color="blue", lw=2)
    predator_line, = ax.plot([], [], label="Predadores (Lobos)", color="red", lw=2)
    ax.legend(loc='upper right')

    return ax

def update_curves(frame):
    """Atualiza o gráfico de curvas."""
    prey_line.set_data(time[:frame], preys[:frame])
    predator_line.set_data(time[:frame], predators[:frame])
    return prey_line, predator_line

def update_points(frame):
    """Atualiza a animação de pontos."""
    ax.clear()
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    ax.set_title("Animação de pontos: Presas (Azul) e Predadores (Vermelho)")
    np.random.seed(1)
    prey_positions = np.random.uniform(-50, 50, (int(preys[frame]), 2))
    predator_positions = np.random.uniform(-50, 50, (int(predators[frame]), 2))
    ax.scatter(prey_positions[:, 0], prey_positions[:, 1], color="blue", alpha=0.6, label="Presas")
    ax.scatter(predator_positions[:, 0], predator_positions[:, 1], color="red", alpha=0.6, label="Predadores")
    ax.legend(loc='upper right')
    return ax

def on_key(event):
    """Controla pausa/continuação e alternância de animação."""
    global is_paused, animation_type
    if event.key == " ":
        is_paused = not is_paused  # Alterna entre pausado e rodando
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
        prey_count = preys[time_idx]
        predator_count = predators[time_idx]
        print(f"Tempo: {event.xdata:.1f}, Presas: {prey_count:.1f}, Predadores: {predator_count:.1f}")

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
    """Atualiza os parâmetros do modelo e reinicia a animação."""
    global alpha, beta, delta, gamma, time, preys, predators
    alpha = alpha_slider.val
    beta = beta_slider.val
    delta = delta_slider.val
    gamma = gamma_slider.val
    time, preys, predators = solve_model(alpha, beta, delta, gamma)
    reset_animation()

# Configuração inicial da animação
global predator_line, prey_line
setup_plot()

ani = FuncAnimation(fig, update_curves, frames=len(time), interval=30, blit=True)

# Eventos de teclado e clique
fig.canvas.mpl_connect("key_press_event", on_key)
fig.canvas.mpl_connect("button_press_event", on_click)

# Adiciona sliders para ajustar os parâmetros
ax_alpha = plt.axes([0.15, 0.15, 0.65, 0.03], facecolor="lightgoldenrodyellow")
ax_beta = plt.axes([0.15, 0.10, 0.65, 0.03], facecolor="lightgoldenrodyellow")
ax_delta = plt.axes([0.15, 0.05, 0.65, 0.03], facecolor="lightgoldenrodyellow")
ax_gamma = plt.axes([0.15, 0.00, 0.65, 0.03], facecolor="lightgoldenrodyellow")

alpha_slider = Slider(ax_alpha, "Alpha", 0.01, 0.5, valinit=alpha, valstep=0.01)
beta_slider = Slider(ax_beta, "Beta", 0.01, 0.5, valinit=beta, valstep=0.01)
delta_slider = Slider(ax_delta, "Delta", 0.01, 0.5, valinit=delta, valstep=0.01)
gamma_slider = Slider(ax_gamma, "Gamma", 0.01, 0.5, valinit=gamma, valstep=0.01)

alpha_slider.on_changed(update_params)
beta_slider.on_changed(update_params)
delta_slider.on_changed(update_params)
gamma_slider.on_changed(update_params)

plt.show()
