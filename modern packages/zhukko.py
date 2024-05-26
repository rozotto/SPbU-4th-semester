import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def zhukko(side_length, speed, proportional_speed=False):
    initial_positions = np.array([
        [0, 0],
        [side_length, 0],
        [side_length, side_length],
        [0, side_length]
    ])

    def system(t, y):
        positions = y.reshape(4, 2)
        derivatives = []
        for i in range(4):
            next_i = (i + 1) % 4
            dx = positions[next_i, 0] - positions[i, 0]
            dy = positions[next_i, 1] - positions[i, 1]
            distance = np.sqrt(dx ** 2 + dy ** 2)
            current_speed = speed * (
                        distance / np.linalg.norm([side_length, side_length])) if proportional_speed else speed
            vx = current_speed * dx / distance
            vy = current_speed * dy / distance
            derivatives.append(vx)
            derivatives.append(vy)
        return derivatives

    y0 = initial_positions.flatten()
    t_span = [0, side_length / speed]

    sol = solve_ivp(system, t_span, y0, method='RK45', t_eval=np.linspace(0, side_length / speed, 1000))

    positions = sol.y.T.reshape(-1, 4, 2)

    plt.figure(figsize=(8, 8))
    for i in range(4):
        plt.plot(positions[:, i, 0], positions[:, i, 1], label=f'Таракан {i + 1}')

    plt.xlim(-0.1 * side_length, 1.1 * side_length)
    plt.ylim(-0.1 * side_length, 1.1 * side_length)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title("Траектории движения тараканов")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.grid(True)
    plt.show()


zhukko(123, 1, proportional_speed=True)
