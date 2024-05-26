import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp


def madcoc(x0, y0, z0, a, fi, V):
    initial_position = np.array([x0, y0, z0])

    def system(t, y):
        x, y, z = y
        speed_xy = V * np.cos(fi)
        speed_z = V * np.sin(fi)

        norm_xy = np.sqrt(x ** 2 + y ** 2)
        dxdt = speed_xy * x / norm_xy
        dydt = speed_xy * y / norm_xy
        dzdt = speed_z

        return [dxdt, dydt, dzdt]

    t_span = [0, 10]
    y0 = initial_position

    sol = solve_ivp(system, t_span, y0, method='RK45', t_eval=np.linspace(0, 10, 1000))

    x = sol.y[0]
    y = sol.y[1]
    z = sol.y[2]

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x, y, z)

    X = np.linspace(-np.max(x), np.max(x), 100)
    Y = np.linspace(-np.max(y), np.max(y), 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.sqrt(X ** 2 + Y ** 2) / a
    ax.plot_surface(X, Y, Z, alpha=0.5, color='gray')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Траектория движения таракана по конусу')
    ax.legend()
    plt.show()


madcoc(10, 20, 30, 1, np.pi / 4, 10)
