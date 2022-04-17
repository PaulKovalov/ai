import random
from typing import List, Tuple

OBSTACLE = '#'
FREE = '.'
PLANE_DIMENSION = 100
EPOCHS = 200
BEST_THRESHOLD = 24
MUTATION_COUNT = 5


def generate_plane(filename):
    with open(filename, 'w') as file:
        for _ in range(PLANE_DIMENSION):
            for _ in range(PLANE_DIMENSION):
                cell = random.randint(0, 2)
                file.write('#' if cell == 1 else '.')
            file.write('\n')


def load_plane():
    with open('plane.txt', 'r') as file:
        rows = file.readlines()
        rows = [row.strip() for row in rows]
        return rows


# Cost of the route, the less cost - the better fit
def fitness_function(path: List[Tuple[int, int]]) -> int:
    cost = len(path)
    for i in range(1, len(path)):
        cost += abs(path[i][1] - path[i - 1][1])
    return cost


def generate_random_path(plane, i: int) -> Tuple[List[Tuple[int, int]], bool]:
    path = []
    for j in range(PLANE_DIMENSION):
        # If the next cell is free, advance
        if plane[j][i] == FREE:
            path.append((j, i))
        # The next cell is obstacle, move back, and find path left or right
        else:
            # Choose randomly direction where to go - left/right
            direction = random.randint(0, 1)
            if direction == 0:
                # Go left
                while i >= 0:
                    if plane[j][i] == FREE:
                        break
                    i -= 1
                if i < 0:
                    return path, False
            if direction == 1:
                # Go right
                while i < PLANE_DIMENSION:
                    if plane[j][i] == FREE:
                        break
                    i += 1
                if i >= PLANE_DIMENSION:
                    return path, False
            path.append((j, i))
    return path, True


def cross(path1, path2, plane):
    coords = list(range(PLANE_DIMENSION))
    random.shuffle(coords)
    successful_crosses = []
    for i in coords:
        y1 = min(path1[i][1], path2[i][1])
        y2 = max(path1[i][1], path2[i][1])
        if all(cell == FREE for cell in plane[i][y1:y2]):
            successful_crosses.append([*path1[:i], *path2[i:]])
    return successful_crosses


def print_path(path, plane):
    char_plane = []
    for row in plane:
        r = []
        for char in row:
            r.append(char)
        char_plane.append(r)
    for p in path:
        char_plane[p[0]][p[1]] = 'P'
    with open('generated-path.txt', 'w') as file:
        for line in char_plane:
            for char in line:
                file.write(char)
            file.write('\n')
        file.flush()


def run_simulation(population, plane):
    # Start algorithm
    for epoch in range(EPOCHS):
        if epoch % 100 == 0:
            print(f'Generation {epoch} best is {population[0][1]}')
        population = list(map(lambda p: p[0], population))
        # Step 1: Crossing. Choose best from the current population and use them for crossing and mutations
        crossed = []
        for i in range(BEST_THRESHOLD):
            j = random.randint(0, BEST_THRESHOLD - 1)
            while j == i:
                j = random.randint(0, BEST_THRESHOLD - 1)
            crossed.extend(cross(population[i], population[j], plane))
        # Self check
        for path in crossed:
            if not all(plane[x][y] == FREE for x, y in path):
                print('WTF')

        # Step 2: Mutation. Add a certain number of random paths, and remove the same number of random paths.
        for _ in range(MUTATION_COUNT):
            random_path, generated = generate_random_path(plane, random.randint(0, PLANE_DIMENSION - 1))
            if generated:
                crossed.append(random_path)

        # Step 3: Calculate target function for all generations
        population = [(p, fitness_function(p)) for p in crossed]
        # print(f'Generated population of size {len(population)}')
        # Step 4: Select best from the new generation
        population = sorted(population, key=lambda tup: tup[1])
        population = population[:BEST_THRESHOLD]

    # End of algorithm. Select the best path and print it on the board
    print(population[0][0])
    print(population[0][1])
    print_path(population[0][0], plane)


def main():
    plane = load_plane()
    # Generate initial population
    population = []
    for i in range(PLANE_DIMENSION):
        p, valid = generate_random_path(plane, i)
        if valid:
            population.append((p, fitness_function(p)))
    # Launch genetic algorithm
    print(f'Generated {len(population)} elements of the initial population')
    run_simulation(population, plane)


if __name__ == "__main__":
    main()
