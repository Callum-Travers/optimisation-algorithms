import numpy as np
import random
import math
import matplotlib.pyplot as plt

def f(x, y):
    return x**2 + y**2 + 3*math.sin(2*x) + 4*math.sin(3*y)

n_particles = 10
n_iterations = 50
w = 0.7
c1 = 1.4
c2 = 1.4

xmin, xmax = -10, 10
ymin, ymax = -10, 10

positions = np.column_stack((
    np.random.uniform(xmin, xmax, n_particles),
    np.random.uniform(ymin, ymax, n_particles)
))

velocities = np.zeros((n_particles, 2))

pbest = positions.copy()
pbest_values = np.array([f(x, y) for x, y in positions])

gbest_index = np.argmin(pbest_values)
gbest = pbest[gbest_index].copy()
gbest_value = pbest_values[gbest_index]

convergence = []

for it in range(n_iterations):

    for i in range(n_particles):

        r1, r2 = random.random(), random.random()

        velocities[i] = (
            w * velocities[i]
            + c1 * r1 * (pbest[i] - positions[i])
            + c2 * r2 * (gbest - positions[i])
        )

        positions[i] += velocities[i]

        positions[i, 0] = np.clip(positions[i, 0], xmin, xmax)
        positions[i, 1] = np.clip(positions[i, 1], ymin, ymax)

        value = f(positions[i, 0], positions[i, 1])

        if value < pbest_values[i]:
            pbest[i] = positions[i].copy()
            pbest_values[i] = value

    gbest_index = np.argmin(pbest_values)
    if pbest_values[gbest_index] < gbest_value:
        gbest = pbest[gbest_index].copy()
        gbest_value = pbest_values[gbest_index]

    convergence.append(gbest_value)
    print(f"Iteration {it+1}: best value = {gbest_value:.6f}")

print("\nFinal best position:", gbest)
print("Final best f(x,y) =", gbest_value)

plt.figure(figsize=(10, 5))
plt.plot(convergence, marker='o', linewidth=1.5)
plt.title("PSO Convergence Curve")
plt.xlabel("Iteration")
plt.ylabel("Best Value Found")
plt.grid(True)
plt.tight_layout()
plt.show()
