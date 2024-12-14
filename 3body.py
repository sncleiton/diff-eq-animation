import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import solve_ivp
from matplotlib.widgets import Slider

# Função que define as equações diferenciais do problema dos três corpos
def three_body_equations(t, state, masses):
    # Extrai as posições e velocidades do estado atual
    x1, y1, x2, y2, x3, y3, vx1, vy1, vx2, vy2, vx3, vy3 = state
    m1, m2, m3 = masses

    # Calcula as distâncias entre os corpos
    r12 = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    r13 = np.sqrt((x3 - x1)**2 + (y3 - y1)**2)
    r23 = np.sqrt((x3 - x2)**2 + (y3 - y2)**2)

    # Forças gravitacionais
    F12x = m2 * (x2 - x1) / r12**3
    F12y = m2 * (y2 - y1) / r12**3
    F13x = m3 * (x3 - x1) / r13**3
    F13y = m3 * (y3 - y1) / r13**3
    F21x = m1 * (x1 - x2) / r12**3
    F21y = m1 * (y1 - y2) / r12**3
    F23x = m3 * (x3 - x2) / r23**3
    F23y = m3 * (y3 - y2) / r23**3
    F31x = m1 * (x1 - x3) / r13**3
    F31y = m1 * (y1 - y3) / r13**3
    F32x = m2 * (x2 - x3) / r23**3
    F32y = m2 * (y2 - y3) / r23**3

    # Acelerações resultantes
    ax1 = F12x + F13x
    ay1 = F12y + F13y
    ax2 = F21x + F23x
    ay2 = F21y + F23y
    ax3 = F31x + F32x
    ay3 = F31y + F32y

    # Retorna as derivadas (velocidades e acelerações)
    return [vx1, vy1, vx2, vy2, vx3, vy3, ax1, ay1, ax2, ay2, ax3, ay3]

# Função para resolver o problema dos três corpos
def solve_three_body(state0, masses, t_span, t_eval):
    solution = solve_ivp(
        three_body_equations,
        t_span,
        state0,
        t_eval=t_eval,
        args=(masses,),
        method='RK45'
    )
    return solution

# Parâmetros iniciais
def default_parameters():
    masses = [1.0, 1.0, 3.0]  # Massas dos corpos
    state0 = [
        -1.0, 0.0,  # Posição inicial do corpo 1
        1.0, 0.0,   # Posição inicial do corpo 2
        0.0, 1.0,   # Posição inicial do corpo 3
        0.0, -0.5,  # Velocidade inicial do corpo 1
        0.0, 0.5,   # Velocidade inicial do corpo 2
        0.5, 0.0    # Velocidade inicial do corpo 3
    ]
    t_span = (0, 20)  # Intervalo de tempo
    t_eval = np.linspace(*t_span, 1000)  # Pontos de tempo para avaliação
    return masses, state0, t_span, t_eval

# Função de animação interativa
def interactive_animation():
    masses, state0, t_span, t_eval = default_parameters()

    # Resolve o problema dos três corpos
    solution = solve_three_body(state0, masses, t_span, t_eval)

    # Extrai as soluções
    x1, y1, x2, y2, x3, y3 = solution.y[0], solution.y[1], solution.y[2], solution.y[3], solution.y[4], solution.y[5]

    # Configura a figura e os eixos
    fig, ax = plt.subplots()
    plt.subplots_adjust(bottom=0.25)
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_aspect('equal')

    # Linhas e pontos para os corpos
    body1, = ax.plot([], [], 'o', label="Body 1", markersize=8, color="blue")
    body2, = ax.plot([], [], 'o', label="Body 2", markersize=8, color="green")
    body3, = ax.plot([], [], 'o', label="Body 3", markersize=8, color="red")
    trail1, = ax.plot([], [], '-', color="blue", lw=1)
    trail2, = ax.plot([], [], '-', color="green", lw=1)
    trail3, = ax.plot([], [], '-', color="red", lw=1)

    # Sliders para ajustar massas
    ax_mass1 = plt.axes([0.2, 0.1, 0.65, 0.03])
    ax_mass2 = plt.axes([0.2, 0.06, 0.65, 0.03])
    ax_mass3 = plt.axes([0.2, 0.02, 0.65, 0.03])

    slider_mass1 = Slider(ax_mass1, 'Mass 1', 0.1, 10.0, valinit=masses[0])
    slider_mass2 = Slider(ax_mass2, 'Mass 2', 0.1, 10.0, valinit=masses[1])
    slider_mass3 = Slider(ax_mass3, 'Mass 3', 0.1, 10.0, valinit=masses[2])

    def update(frame):
        # Atualiza as posições dos corpos
        body1.set_data(x1[frame], y1[frame])
        body2.set_data(x2[frame], y2[frame])
        body3.set_data(x3[frame], y3[frame])
        
        # Atualiza os trails (limite os pontos visíveis do trail)
        trail1.set_data(x1[max(0, frame-100):frame], y1[max(0, frame-100):frame])
        trail2.set_data(x2[max(0, frame-100):frame], y2[max(0, frame-100):frame])
        trail3.set_data(x3[max(0, frame-100):frame], y3[max(0, frame-100):frame])

        return body1, body2, body3, trail1, trail2, trail3

    def on_change(val):
        anim.event_source.stop()  # Para a animação
        anim.event_source.start()  # Reinicia a animação desde o início
        anim.frame_seq = anim.new_frame_seq()  # Reseta a sequência de frames
        anim.event_source.start()  # Reinicia a animação

        nonlocal masses, state0, solution, x1, y1, x2, y2, x3, y3
        masses = [slider_mass1.val, slider_mass2.val, slider_mass3.val]
        solution = solve_three_body(state0, masses, t_span, t_eval)
        x1, y1, x2, y2, x3, y3 = solution.y[0], solution.y[1], solution.y[2], solution.y[3], solution.y[4], solution.y[5]

    slider_mass1.on_changed(on_change)
    slider_mass2.on_changed(on_change)
    slider_mass3.on_changed(on_change)
    
    anim = FuncAnimation(fig, update, frames=len(t_eval), interval=30, blit=True)
    
    # Exibe a legenda
    ax.legend()

    plt.show()

# Executa a animação interativa
interactive_animation()
