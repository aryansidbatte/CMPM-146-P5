import copy
import heapq
import metrics
import multiprocessing.pool as mpool
import os
import random
import shutil
import time
import math

width = 200
height = 16

options = [
    "-",  # an empty space
    "X",  # a solid wall
    "?",  # a question mark block with a coin
    "M",  # a question mark block with a mushroom
    "B",  # a breakable block
    "o",  # a coin
    "|",  # a pipe segment
    "T",  # a pipe top
    "E",  # an enemy
    #"f",  # a flag, do not generate
    #"v",  # a flagpole, do not generate
    #"m"  # mario's start position, do not generate
]

# The level as a grid of tiles


class Individual_Grid(object):
    __slots__ = ["genome", "_fitness"]

    def __init__(self, genome):
        self.genome = copy.deepcopy(genome)
        self._fitness = None

    # Update this individual's estimate of its fitness.
    # This can be expensive so we do it once and then cache the result.
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Print out the possible measurements or look at the implementation of metrics.py for other keys:
        # print(measurements.keys())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Modify this, and possibly add more metrics.  You can replace this with whatever code you like.
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients))
        return self

    # Return the cached fitness value or calculate it as needed.
    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    # Mutate a genome into a new genome.  Note that this is a _genome_, not an individual!
    def mutate(self, genome):
        # STUDENT implement a mutation operator, also consider not mutating this individual
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc

        max_height = len(genome)
        max_width = len(genome[0]) if genome else 0

        mutation_rate = 0.2  #20% chance to mutate each tile
        left = 1
        right = width - 1
        for y in range(height):
            for x in range(left, right):
                if random.random() < mutation_rate:
                
                    if (y + 1 == max_height):
                        # Ensure ground only mutates into block, pipe, or empty space
                        new_tile = random.choices(['X', '|', '-'], weights=[3, 1, 2])[0]
                        genome[y][x] = new_tile
                    
                    # Prevent the first and last 5 columns from mutating
                    elif x in range(0, 5) or x in range(max_width - 5, max_width):
                        continue
                        
                    else:
                        new_tile = random.choices(options, weights=[4, 3, 3, 2, 4, 5, 3, 3, 1])[0]
                        genome[y][x] = new_tile
        return genome


    def apply_constraints(genome):
        max_height = len(genome)
        max_width = len(genome[0]) if genome else 0
            
        for y in range(max_height):
            for x in range(max_width):

                if y in range(0, 2):
                    # Keep top of stage a bit clean
                        genome[y][x] =  "-" # Replace with empty space if not valid

                if x in range(0, 5):
                    # Allow breathing room at the start of the stage
                    if genome[y][x] not in ("m", "X"):
                        genome[y][x] =  "-" # Replace with empty space if not valid

                elif x in range(max_width - 5, max_width):
                    # Ensure flag and flagpole at the end do not get overwritten and leave space before
                    if genome[y][x] not in ("f", "v", "X"):
                        genome[y][x] = "-"  # Replace with empty space if not valid

                elif genome[y][x] in ("X", "?", "M", "B"):
                    # Ensure blocks are placed on rows that are multiples of 4 from the top of the map
                    if (max_height - (y + 1)) % 4 != 0 or y == max_height - 2:
                        genome[y][x] = "-"  # Replace with empty space if not on valid row
                    else:
                        # Additional checks can go here if needed
                        pass   

                elif genome[y][x] in ("|", "T"):
                    if genome[y - 1][x] in ("|", "T"):
                        # There is a pipe piece above here, change to | pipe
                        genome[y][x] = "|"                    
                    elif genome[y - 2][x] == "-" and genome[y - 2] == "-":
                        #There is empty space above, change to T pipe
                        genome[y][x] = "T"
                    else:
                        # There is a piece blocking the pipe, change the pipe to an empty space
                        genome[y][x] = "-"


                    if genome[y][x] != "-":
                        # Check to see if piece is too tall and also touches the ground
                        ground_found = False
                        for depth in range(1, 3):
                            if y + depth >= max_height:
                                ground_found = True #Depth is okay as piece touches ground
                                break 

                            if genome[y + depth][x] in ("X", "|"):
                                # If piece is connected to a straight pipe below continue, otherwise break as piece touches ground
                                if genome[y + depth][x] == "X":
                                    ground_found = True
                                    break
                            else:
                                # If tile comes across anything other than the ground or straight piece, break
                                break
                        if not ground_found:
                            genome[y][x] = "-"  # Replace with empty space if not valid


                elif genome[y][x] == "E":  # Enemy
                    # Ensure enemy is positioned on ground or a block
                    if y + 1 >= max_height or genome[y + 1][x] not in ("X", "?", "M", "B"):
                        genome[y][x] = "-"  # Replace with empty space if not valid

        return genome

    # Create zero or more children from self and other
    def generate_children(self, other):
        new_genome = copy.deepcopy(self.genome)
        # Leaving first and last columns alone...
        # do crossover with other
        left = 1
        right = width - 1
        for y in range(height):
            for x in range(left, right):
                # STUDENT Which one should you take?  Self, or other?  Why?
                # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
                if random.choice([True, False]):
                    new_genome[y][x] = self.genome[y][x]
                else:
                    new_genome[y][x] = other.genome[y][x]
        # do mutation; note we're returning a one-element tuple here
        new_genome = self.mutate(new_genome)
        new_genome = Individual_Grid.apply_constraints(new_genome)
        return (Individual_Grid(new_genome),)

    # Turn the genome into a level string (easy for this genome)
    def to_level(self):
        """Convert the genome to a level string representation with extra columns of air at the end."""
        # Create a deep copy of the genome to avoid modifying the original
        extended_genome = copy.deepcopy(self.genome)
        extra_columns = 3
        # Add extra columns at the end
        for row in extended_genome:
            row.extend(['-'] * extra_columns)

        # Ensure the floor level is solid throughout
        extended_genome[height - 1] = ['X'] * (width + extra_columns)

        return extended_genome

    # These both start with every floor tile filled with Xs
    # STUDENT Feel free to change these
    @classmethod
    def empty_individual(cls):
        g = [["-" for col in range(width)] for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        for col in range(8, 14):
            g[col][-1] = "f"
        for col in range(14, 16):
            g[col][-1] = "X"
        return cls(g)

    @classmethod
    def random_individual(cls):
        # STUDENT consider putting more constraints on this to prevent pipes in the air, etc
        # STUDENT also consider weighting the different tile types so it's not uniformly random
        g = [random.choices(options, k=width) for row in range(height)]
        g[15][:] = ["X"] * width
        g[14][0] = "m"
        g[7][-1] = "v"
        g[8:14][-1] = ["f"] * 6
        g[14:16][-1] = ["X", "X"]
        return cls(g)


def offset_by_upto(val, variance, min=None, max=None):
    val += random.normalvariate(0, variance**0.5)
    if min is not None and val < min:
        val = min
    if max is not None and val > max:
        val = max
    return int(val)


def clip(lo, val, hi):
    if val < lo:
        return lo
    if val > hi:
        return hi
    return val

# Inspired by https://www.researchgate.net/profile/Philippe_Pasquier/publication/220867545_Towards_a_Generic_Framework_for_Automated_Video_Game_Level_Creation/links/0912f510ac2bed57d1000000.pdf


class Individual_DE(object):
    # Calculating the level isn't cheap either so we cache it too.
    __slots__ = ["genome", "_fitness", "_level"]

    # Genome is a heapq of design elements sorted by X, then type, then other parameters
    def __init__(self, genome):
        self.genome = list(genome)
        heapq.heapify(self.genome)
        self._fitness = None
        self._level = None

    # Calculate and cache fitness
    def calculate_fitness(self):
        measurements = metrics.metrics(self.to_level())
        # Default fitness function: Just some arbitrary combination of a few criteria.  Is it good?  Who knows?
        # STUDENT Add more metrics?
        # STUDENT Improve this with any code you like
        coefficients = dict(
            meaningfulJumpVariance=0.5,
            negativeSpace=0.6,
            pathPercentage=0.5,
            emptyPercentage=0.6,
            linearity=-0.5,
            solvability=2.0
        )
        penalties = 0
        # STUDENT For example, too many stairs are unaesthetic.  Let's penalize that
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) > 5:
            penalties -= 10
        if len(list(filter(lambda de: de[1] == "6_stairs", self.genome))) < 2:
            penalties -= 10     

        if len(list(filter(lambda de: de[1] == "7_pipe", self.genome))) > 5:
            penalties -= 10                        
        if len(list(filter(lambda de: de[1] == "7_pipe", self.genome))) < 2:
            penalties -= 10   

        if len(list(filter(lambda de: de[1] == "0_hole", self.genome))) > 7:
            penalties -= 10                        
        if len(list(filter(lambda de: de[1] == "0_hole", self.genome))) < 4:
            penalties -= 10               

        if len(list(filter(lambda de: de[1] == "2_enemy", self.genome))) > 10:
            penalties -= 20                        
        if len(list(filter(lambda de: de[1] == "2_enemy", self.genome))) < 5:
            penalties -= 20  

        if len(list(filter(lambda de: de[1] == "3_coin", self.genome))) > 30:
            penalties -= 30                        
        if len(list(filter(lambda de: de[1] == "3_coin", self.genome))) < 15:
            penalties -= 30 

        if len(list(filter(lambda de: de[1] == "1_platform", self.genome))) > 15:
            penalties -= 30                        
        if len(list(filter(lambda de: de[1] == "1_platform", self.genome))) < 10:
            penalties -= 30                          

        # STUDENT If you go for the FI-2POP extra credit, you can put constraint calculation in here too and cache it in a new entry in __slots__.
        self._fitness = sum(map(lambda m: coefficients[m] * measurements[m],
                                coefficients)) + penalties
        return self

    def fitness(self):
        if self._fitness is None:
            self.calculate_fitness()
        return self._fitness

    def mutate(self, new_genome):
        # STUDENT How does this work?  Explain it in your writeup.
        # STUDENT consider putting more constraints on this, to prevent generating weird things
        if random.random() < 0.4 and len(new_genome) > 0:
            to_change = random.randint(0, len(new_genome) - 1)
            de = new_genome[to_change]
            new_de = de
            x = de[0]
            de_type = de[1]
            choice = random.random()
            if de_type == "4_block":
                y = de[2]
                breakable = de[3]
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    breakable = not de[3]
                new_de = (x, de_type, y, breakable)
            elif de_type == "5_qblock":
                y = de[2]
                has_powerup = de[3]  # boolean
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                else:
                    has_powerup = not de[3]
                new_de = (x, de_type, y, has_powerup)
            elif de_type == "3_coin":
                y = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    y = offset_by_upto(y, height / 2, min=0, max=height - 1)
                new_de = (x, de_type, y)
            elif de_type == "7_pipe":
                h = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                else:
                    h = offset_by_upto(h, 2, min=2, max=height - 4)
                new_de = (x, de_type, h)
            elif de_type == "0_hole":
                w = de[2]
                if choice < 0.5:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 4)
                else:
                    w = offset_by_upto(w, 4, min=1, max=width - 2)
                new_de = (x, de_type, w)
            elif de_type == "6_stairs":
                h = de[2]
                dx = de[3]  # -1 or 1
                if choice < 0.33:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.66:
                    if dx == 1:
                        h = offset_by_upto(h, 8, min=1, max=6)
                    else:
                        h = offset_by_upto(h, 8, min=1, max=4)
                else:
                    dx = -dx
                new_de = (x, de_type, h, dx)
            elif de_type == "1_platform":
                w = de[2]
                y = de[3]
                madeof = de[4]  # from "?", "X", "B"
                if choice < 0.25:
                    x = offset_by_upto(x, width / 8, min=1, max=width - 2)
                elif choice < 0.5:
                    w = offset_by_upto(w, 8, min=1, max=width - 2)
                elif choice < 0.75:
                    y = offset_by_upto(y, height, min=0, max=height - 1)
                else:
                    madeof = random.choice(["?", "X", "B"])
                new_de = (x, de_type, w, y, madeof)
            elif de_type == "2_enemy":
                pass
            new_genome.pop(to_change)
            heapq.heappush(new_genome, new_de)
        return new_genome

    def generate_children(self, other):        
        if not self.genome or not other.genome:
            # If either parent has an empty genome, return empty
            return []

        # Choose how many genomes will be copied over to children for both parents
        pa = random.randint(0, len(self.genome) - 1)
        pb = random.randint(0, len(other.genome) - 1)

        # Splice parents genomes up to the number from pa/pb or return empty if no genomes
        a_part = self.genome[:pa] if len(self.genome) > 0 else []
        b_part = other.genome[pb:] if len(other.genome) > 0 else []

        # Combine the two parts into one
        ga = a_part + b_part
        
        b_part = other.genome[:pb] if len(other.genome) > 0 else []
        a_part = self.genome[pa:] if len(self.genome) > 0 else []
        gb = b_part + a_part
        
        # Do mutation
        return Individual_DE(self.mutate(ga)), Individual_DE(self.mutate(gb))


    # Apply the DEs to a base level.
    def to_level(self):
        if self._level is None:
            base = Individual_Grid.empty_individual().to_level()
            for de in sorted(self.genome, key=lambda de: (de[1], de[0], de)):
                # de: x, type, ...
                x = de[0]
                de_type = de[1]
                if de_type == "4_block":
                    y = de[2]
                    breakable = de[3]
                    base[y][x] = "B" if breakable else "X"
                elif de_type == "5_qblock":
                    y = de[2]
                    has_powerup = de[3]  # boolean
                    base[y][x] = "M" if has_powerup else "?"
                elif de_type == "3_coin":
                    y = de[2]
                    base[y][x] = "o"
                elif de_type == "7_pipe":
                    h = de[2]
                    base[height - h - 1][x] = "T"
                    for y in range(height - h, height):
                        base[y][x] = "|"
                elif de_type == "0_hole":
                    w = de[2]
                    for x2 in range(w):
                        base[height - 1][clip(1, x + x2, width - 2)] = "-"
                elif de_type == "6_stairs":
                    h = de[2]
                    dx = de[3]  # -1 or 1
                    for x2 in range(1, h + 1):
                        for y in range(x2 if dx == 1 else h - x2):
                            base[clip(0, height - y - 1, height - 1)][clip(1, x + x2, width - 2)] = "X"
                elif de_type == "1_platform":
                    w = de[2]
                    h = de[3]
                    madeof = de[4]  # from "?", "X", "B"
                    for x2 in range(w):
                        base[clip(0, height - h - 1, height - 1)][clip(1, x + x2, width - 2)] = madeof
                elif de_type == "2_enemy":
                    base[height - 2][x] = "E"
            self._level = base
        return self._level

    @classmethod
    def empty_individual(_cls):
        # STUDENT Maybe enhance this
        g = []
        return Individual_DE(g)

    @classmethod
    def random_individual(_cls):
        # STUDENT Maybe enhance this
        elt_count = random.randint(8, 128)
        g = [random.choice([
            (random.randint(1, width - 2), "0_hole", random.randint(1, 6)),
            (random.randint(1, width - 2), "1_platform", random.randint(1, 8), random.randint(0, height - 1), random.choice(["?", "X", "B"])),
            (random.randint(1, width - 2), "2_enemy"),
            (random.randint(1, width - 2), "3_coin", random.randint(0, height - 1)),
            (random.randint(1, width - 2), "4_block", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "5_qblock", random.randint(0, height - 1), random.choice([True, False])),
            (random.randint(1, width - 2), "6_stairs", random.randint(1, 5), random.choice([-1, 1])),
            (random.randint(1, width - 2), "7_pipe", random.randint(2, 4))
        ]) for i in range(elt_count)]
        return Individual_DE(g)


Individual = Individual_DE

def generate_successors(population):
    results = []
    # STUDENT Design and implement this
    # Hint: Call generate_children() on some individuals and fill up results.
    
    # Sorting population by fitness
    population.sort(key=Individual.fitness, reverse=True)
    elite_count = max(1, len(population) // 10)
    results.extend(population[:elite_count])

    # Generate children from the top-performing individuals
    while len(results) < len(population):
        parent1, parent2 = random.sample(population[:elite_count], 2)
        children = parent1.generate_children(parent2)
        
        for child in children:
            # Calculate fitness and check solvability
            child.calculate_fitness()
            metrics_results = metrics.metrics(child.to_level())
            
            if metrics_results['solvability'] > 0:
                results.append(child)
            else:
                # Create a new individual to replace the unsolvable one
                new_individual = Individual.random_individual()
                new_individual.calculate_fitness()
                results.append(new_individual)

    return results[:len(population)]

def ga():
    # STUDENT Feel free to play with this parameter
    pop_limit = 480
    # Code to parallelize some computations
    batches = os.cpu_count()
    if pop_limit % batches != 0:
        print("It's ideal if pop_limit divides evenly into " + str(batches) + " batches.")
    batch_size = int(math.ceil(pop_limit / batches))
    with mpool.Pool(processes=os.cpu_count()) as pool:
        init_time = time.time()
        # STUDENT (Optional) change population initialization
        population = [Individual.random_individual() if random.random() < 0.9
                      else Individual.empty_individual()
                      for _g in range(pop_limit)]
        # But leave this line alone; we have to reassign to population because we get a new population that has more cached stuff in it.
        population = pool.map(Individual.calculate_fitness,
                              population,
                              batch_size)
        init_done = time.time()
        print("Created and calculated initial population statistics in:", init_done - init_time, "seconds")
        generation = 0
        start = time.time()
        now = start
        print("Use ctrl-c to terminate this loop manually.")
        try:
            while True:
                now = time.time()
                # Print out statistics
                if generation > 0:
                    best = max(population, key=Individual.fitness)
                    print("Generation:", str(generation))
                    print("Max fitness:", str(best.fitness()))
                    print("Average generation time:", (now - start) / generation)
                    print("Net time:", now - start)
                    with open("levels/last.txt", 'w') as f:
                        for row in best.to_level():
                            f.write("".join(row) + "\n")
                generation += 1

                # Implementing the stopping condition based on elapsed time
                stop_condition = now - start > 400  # Set to a higher value for a longer run

                if stop_condition:
                    print(f"Stopping condition met: Elapsed time {now - start:.2f} seconds.")
                    break

                gentime = time.time()
                next_population = generate_successors(population)
                gendone = time.time()
                print("Generated successors in:", gendone - gentime, "seconds")
                # Calculate fitness in batches in parallel
                next_population = pool.map(Individual.calculate_fitness,
                                           next_population,
                                           batch_size)
                popdone = time.time()
                print("Calculated fitnesses in:", popdone - gendone, "seconds")
                population = next_population
        except KeyboardInterrupt:
            pass
    return population



if __name__ == "__main__":
    final_gen = sorted(ga(), key=Individual.fitness, reverse=True)
    best = final_gen[0]
    print("Best fitness: " + str(best.fitness()))
    now = time.strftime("%m_%d_%H_%M_%S")
    # STUDENT You can change this if you want to blast out the whole generation, or ten random samples, or...
    for k in range(0, 10):
        with open("levels/" + now + "_" + str(k) + ".txt", 'w') as f:
            for row in final_gen[k].to_level():
                f.write("".join(row) + "\n")
