"""
Genetic Algorithm for the Traveling Salesman Problem (TSP)
Usage example:
python genetic.py --input large.csv --population 50 --generations 200 \
                 --tournament 5 --mutation 0.02 --elitism 0.1
"""

import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

# Fitness function: Calculates the total distance of a tour
# The lower the distance, the better the solution
# tour: list of city indices representing the visiting order
# cities: numpy array of city coordinates

def calculate_total_distance(tour, cities):
    """
    Calculates the total distance of a tour.
    tour: List of city indices representing the visiting order
    cities: Numpy array of city coordinates
    """
    distance = 0.0
    # Iterate through each city in the tour
    for i in range(len(tour)):
        city_a = cities[tour[i]]
        city_b = cities[tour[(i + 1) % len(tour)]]  # Return to the starting city at the end
        distance += np.linalg.norm(city_a - city_b)  # Euclidean distance
    return distance

# Initializes the population with random permutations
# Each individual is a random tour (permutation of city indices)
def initialize_population(population_size, num_cities):
    """
    Creates the initial population as random permutations of city indices.
    population_size: Number of individuals in the population
    num_cities: Number of cities (length of each tour)
    """
    population = []
    for _ in range(population_size):
        individual = list(np.random.permutation(num_cities))  # Random permutation
        population.append(individual)
    return population

# Tournament selection: Selects the best individual among a random subset
def tournament_selection(population, fitnesses, tournament_size):
    """
    Selects the best individual among a random subset (tournament) of the population.
    population: List of individuals (tours)
    fitnesses: List of fitness values (total distances)
    tournament_size: Number of individuals in each tournament
    """
    selected = random.sample(list(zip(population, fitnesses)), tournament_size)  # Randomly select tournament_size individuals
    selected.sort(key=lambda x: x[1])  # Sort by fitness (lower is better)
    return selected[0][0]  # Return the best individual's tour

# Ordered Crossover (OX):
# Offsprings are created by selecting a random segment from parent1 and filling the rest of the segment with cities from parent2 in order.
def ordered_crossover(parent1, parent2):
    """
    Performs Ordered Crossover (OX) between two parents to produce a child.
    parent1, parent2: Parent tours (lists of city indices)
    Returns: A new child tour
    """
    size = len(parent1)
    child = [None] * size
    # Select two random crossover points
    a, b = sorted(random.sample(range(size), 2))
    # Copy the segment from parent1
    child[a:b+1] = parent1[a:b+1]
    # Fill the remaining positions with cities from parent2 in order, skipping those already in the segment
    fill_pos = (b + 1) % size
    for city in parent2:
        if city not in child:
            child[fill_pos] = city
            fill_pos = (fill_pos + 1) % size
    return child

# Swap mutation: Swaps two cities in the tour with a given probability
def swap_mutation(individual, mutation_rate):
    """
    Performs swap mutation on an individual with a given probability.
    individual: The tour to mutate
    mutation_rate: Probability of mutation
    Returns: Mutated individual (new list)
    """
    individual = individual.copy()
    if random.random() < mutation_rate:
        a, b = random.sample(range(len(individual)), 2)  # Select two random positions
        individual[a], individual[b] = individual[b], individual[a]  # Swap them
    return individual

# Selects the top elite individuals to carry over to the next generation
def get_elites(population, fitnesses, elitism_rate):
    """
    Selects the top elite individuals based on fitness.
    population: List of individuals
    fitnesses: List of fitness values
    elitism_rate: Fraction of population to keep as elites
    Returns: List of elite individuals
    """
    num_elites = max(1, int(len(population) * elitism_rate))  # At least one elite
    # Sort population by fitness (ascending)
    sorted_pop = [x for _, x in sorted(zip(fitnesses, population), key=lambda x: x[0])]
    return sorted_pop[:num_elites]

# Main Genetic Algorithm loop
def genetic_algorithm(cities, population_size, generations, tournament_size, mutation_rate, elitism_rate):
    """
    Main loop of the Genetic Algorithm for TSP.
    cities: Numpy array of city coordinates
    population_size: Number of individuals in the population
    generations: Number of generations to run
    tournament_size: Number of individuals in tournament selection
    mutation_rate: Probability of mutation
    elitism_rate: Fraction of population to keep as elites
    Returns: Best tour, its distance, and the list of best distances per generation
    """
    num_cities = len(cities)
    population = initialize_population(population_size, num_cities)  # Initial population
    best_distances = []  # Track best distance in each generation
    best_tour = None
    best_distance = float('inf')

    for gen in range(generations):
        # Calculate fitness for each individual (lower is better)
        fitnesses = [calculate_total_distance(ind, cities) for ind in population]
        # Find the best individual in the current generation
        min_idx = np.argmin(fitnesses)
        if fitnesses[min_idx] < best_distance:
            best_distance = fitnesses[min_idx]
            best_tour = population[min_idx]
        best_distances.append(best_distance)
        print(f"Generation {gen+1}: Best distance = {best_distance:.4f}")

        # Elitism: Carry over the best individuals to the next generation
        new_population = get_elites(population, fitnesses, elitism_rate)
        # Fill the rest of the new population
        while len(new_population) < population_size:
            # Select parents using tournament selection
            parent1 = tournament_selection(population, fitnesses, tournament_size)
            parent2 = tournament_selection(population, fitnesses, tournament_size)
            # Create a child using crossover
            child = ordered_crossover(parent1, parent2)
            # Mutate the child
            child = swap_mutation(child, mutation_rate)
            new_population.append(child)
        # Update the population for the next generation
        population = new_population[:population_size]

    return best_tour, best_distance, best_distances

# Visualization: Plots the tour and optionally labels each city with its order
def plot_tour(cities, tour, title="En İyi Tur", show_labels=True):
    """
    Plots the cities and the tour using matplotlib.
    - The tour is shown as a blue line.
    - The first city (start) is marked with a green circle.
    - The last city (just before returning to start) is marked with a red X.
    - If show_labels=True, each city is labeled with its order in the tour.
    cities: Numpy array of city coordinates
    tour: List of city indices (order)
    title: Plot title
    show_labels: Whether to show city order labels (default: True)
    """
    tour_cities = cities[tour + [tour[0]]]  # Add the starting city at the end to close the tour
    plt.figure(figsize=(8, 6))
    # Draw the tour path
    plt.plot(tour_cities[:, 0], tour_cities[:, 1], 'o-', color='blue', markersize=6, label='Tour')
    # Mark the first city (start)
    plt.scatter(tour_cities[0, 0], tour_cities[0, 1], color='green', s=100, marker='o', label='Start (First City)')
    # Mark the last city (before returning to start)
    plt.scatter(tour_cities[-2, 0], tour_cities[-2, 1], color='red', s=100, marker='X', label='End (Last City)')
    # Optionally label each city with its order in the tour
    if show_labels:
        for idx, (x, y) in enumerate(tour_cities[:-1]):  # Exclude the last point (duplicate of start)
            plt.text(x, y, str(idx+1), fontsize=8, color='black', ha='center', va='center',
                     bbox=dict(facecolor='white', alpha=0.5, edgecolor='none', boxstyle='round,pad=0.2'))
    plt.title(title)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Genetic Algorithm for TSP")
    parser.add_argument('--input', type=str, required=True, help='CSV file path (x,y coordinates)')
    parser.add_argument('--population', type=int, default=50, help='Population size')
    parser.add_argument('--generations', type=int, default=200, help='Number of generations')
    parser.add_argument('--tournament', type=int, default=5, help='Tournament size')
    parser.add_argument('--mutation', type=float, default=0.02, help='Mutation rate')
    parser.add_argument('--elitism', type=float, default=0.1, help='Elitism rate')
    args = parser.parse_args()

    # Read city coordinates from CSV file
    df = pd.read_csv(args.input)
    cities = df.values  # (N, 2) numpy array

    # Run the genetic algorithm
    best_tour, best_distance, best_distances = genetic_algorithm(
        cities,
        population_size=args.population,
        generations=args.generations,
        tournament_size=args.tournament,
        mutation_rate=args.mutation,
        elitism_rate=args.elitism
    )

    # Print the best tour and its total distance
    print("\nBest tour order:", best_tour)
    print(f"Total distance: {best_distance:.4f}")

    # Plot the best tour (with city order labels by default)
    plot_tour(cities, best_tour, title=f"Best Tour (Distance: {best_distance:.2f})", show_labels=True)
