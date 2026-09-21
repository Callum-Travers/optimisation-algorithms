import numpy as np
import random


dist = np.array([
    [0, 3, 10, 11, 7],
    [3, 0, 8, 4, 6],
    [10, 8, 0, 9, 3],
    [11, 4, 9, 0, 5],
    [7, 6, 3, 5, 0]
], dtype=float)

n_cities = dist.shape[0]

# -----------------------------
# ACO hyperparameters
# -----------------------------
n_ants = 10
n_iterations = 10
pheinf = 1.0        # pheromone influence
vis = 5.0         # visibility
evapr = 0.5          # evaporation rate
phefact = 100            # pheromone factor


pheromone = np.ones((n_cities, n_cities))


visibility = 1 / (dist + np.eye(n_cities)*1e9)


def tour_length(tour):
    return sum(dist[tour[i], tour[(i+1) % n_cities]] for i in range(n_cities))


best_tour = None
best_length = float("inf")
city_labels = ["A", "B", "C", "D", "E"]

for iteration in range(n_iterations):

    all_tours = []
    all_lengths = []

    for ant in range(n_ants):

        
        start = random.randint(0, n_cities - 1)
        tour = [start]
        unvisited = set(range(n_cities)) - {start}

        
        while unvisited:
            current = tour[-1]

            pher = pheromone[current, list(unvisited)]
            vis = visibility[current, list(unvisited)]
            probs = (pher ** pheinf) * (vis ** vis)
            probs = probs / probs.sum()

            next_city = random.choices(list(unvisited), weights=probs)[0]
            tour.append(next_city)
            unvisited.remove(next_city)

        all_tours.append(tour)
        L = tour_length(tour)
        all_lengths.append(L)

        if L < best_length:
            best_length = L
            best_tour = tour

   
    print(f"\nIteration {iteration+1}")
    for i, (tour, L) in enumerate(zip(all_tours, all_lengths)):
        labelled = [city_labels[x] for x in tour]
        print(f"  Ant {i+1}: {labelled}  |  Cost = {L}")

    iter_best_index = all_lengths.index(min(all_lengths))
    iter_best_tour = all_tours[iter_best_index]
    iter_best_labelled = [city_labels[x] for x in iter_best_tour]
    print(f"  Best this iteration: {iter_best_labelled}  |  Cost = {min(all_lengths)}")

    
    global_best_labelled = [city_labels[x] for x in best_tour]
    print(f"  Global best so far: {global_best_labelled}  |  Cost = {best_length}")

    
    pheromone *= (1 - evapr)

    for tour, L in zip(all_tours, all_lengths):
        deposit = phefact / L
        for i in range(n_cities):
            a = tour[i]
            b = tour[(i+1) % n_cities]
            pheromone[a, b] += deposit
            pheromone[b, a] += deposit



city_labels = ["A", "B", "C", "D", "E"]
best_tour_labels = [city_labels[i] for i in best_tour]

print("\nBest tour found:", best_tour_labels)
print("Best length:", best_length)
