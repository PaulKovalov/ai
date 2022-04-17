import random
from typing import List, Tuple
from os.path import exists


EPOCHS = 5000
BEST_THRESHOLD = 24
MUTATION_COUNT = 8
GEN_REASSEMBLY_COUNT = 5
PLANE_DIMENSION = 50
OBSTACLE = '#'
FREE = '.'


class Plane:
    @staticmethod
    def _generate_plane(filename):
        with open(filename, 'w') as file:
            for _ in range(PLANE_DIMENSION):
                for _ in range(PLANE_DIMENSION):
                    cell = random.randint(0, 3)
                    file.write(OBSTACLE if cell == 1 else FREE)
                file.write('\n')

    @staticmethod
    def _load_plane(filename):
        with open(filename, 'r') as file:
            rows = file.readlines()
            rows = [row.strip() for row in rows]
            return rows

    def __init__(self, plane_file):
        if not exists(plane_file):
            print(f'File {plane_file} not found, generating one...')
            self._generate_plane(filename=plane_file)
        self.plane = self._load_plane(filename=plane_file)
        if len(self.plane) != PLANE_DIMENSION:
            raise ValueError('PLANE_DIMENSION is different from loaded from file')

        self.free_segments = []
        for i in range(PLANE_DIMENSION):
            row_free_segments = []
            segment_start = 0
            segment_end = -1
            for j in range(PLANE_DIMENSION):
                if self.plane[i][j] == FREE:
                    segment_end = j
                else:
                    if segment_start <= segment_end:
                        row_free_segments.append((segment_start, segment_end))
                    segment_start = j + 1
            self.free_segments.append(row_free_segments)

    @staticmethod
    def _intersects(segment1, segment2):
        if segment1[0] <= segment2[1] and segment2[0] <= segment1[1]:
            return True
        return False

    def _dfs(self, segment):
        stack = [(segment, 0, [])]
        while stack:
            curr_segment, curr_row, path = stack.pop()
            if curr_row == PLANE_DIMENSION:
                return path, True
            for segment in self.free_segments[curr_row]:
                if self._intersects(curr_segment, segment):
                    next_point = (curr_row, random.randint(segment[0], segment[1]))
                    stack.append((segment, curr_row + 1, path + [next_point]))
        return [], False

    def generate_random_path(self, i):
        # From every i try to build random path to the end of the plane
        # Initial segment which intersects with i:
        return self._dfs((i, i))

    def print_path(self, path):
        char_plane = []
        for row in self.plane:
            r = []
            for char in row:
                r.append(char)
            char_plane.append(r)
        for p in path[0]:
            char_plane[p[0]][p[1]] = 'P'
        with open(f'generated-path-{EPOCHS}.txt', 'w') as file:
            file.write(f'Path length is {path[1]}\n')
            for line in char_plane:
                for char in line:
                    file.write(char)
                file.write('\n')
            file.flush()


# Cost of the route, the less cost - the better fit
def fitness_function(path: List[Tuple[int, int]]) -> int:
    cost = len(path)
    for i in range(1, len(path)):
        cost += abs(path[i][1] - path[i - 1][1])
    return cost


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


def mutate(path, plane):
    genes_to_change = random.sample(range(PLANE_DIMENSION), GEN_REASSEMBLY_COUNT)
    for gen in genes_to_change:
        point = path[gen][1]
        p_segment = (point, point)
        whole_segment = None
        for s in plane.free_segments[gen]:
            if plane._intersects(s, p_segment):
                whole_segment = s
                break
        new_point = random.randint(whole_segment[0], whole_segment[1])
        path[gen] = (gen, new_point)


def run_simulation(population, plane):
    print(f'Starting from {len(population)} elements of the initial population')
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
            crossed.extend(cross(population[i], population[j], plane.plane))
        # Self check
        for path in crossed:
            if not all(plane.plane[x][y] == FREE for x, y in path):
                print('WTF')

        # Step 2: Mutation. Change a certain number of paths randomly.
        for path in random.sample(crossed, MUTATION_COUNT):
            mutate(path, plane)

        # Step 3: Calculate target function for all generations
        population = [(p, fitness_function(p)) for p in crossed]
        # print(f'Generated population of size {len(population)}')
        # Step 4: Select best from the new generation
        population = sorted(population, key=lambda tup: tup[1])
        population = population[:BEST_THRESHOLD]

    # End of algorithm. Select the best path and print it on the board
    plane.print_path(population[0])


def main():
    random.seed()
    plane = Plane('plane.txt')
    # Generate initial population
    population = []
    for i in range(PLANE_DIMENSION):
        p, valid = plane.generate_random_path(i)
        if valid:
            population.append((p, fitness_function(p)))
    # Launch genetic algorithm
    run_simulation(population, plane)


if __name__ == "__main__":
    main()
