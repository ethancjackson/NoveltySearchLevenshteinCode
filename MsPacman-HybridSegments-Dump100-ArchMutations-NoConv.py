from Individual import Individual, from_JSON
from Mutator import mutate
from NetworkMapper import map_to_network, mutate_conv_module
from numpy.random import randint, RandomState
from joblib import Parallel, delayed
import numpy as np
import NoConvEvaluator
from numpy import fmax
import KerasConvModules
import gc
import Levenshtein
from itertools import chain
import sys

"""
Let's try splitting each episode's actions into smaller segments.
For example if an episode is 2500 frames long, we can split this into 5 segments of upto 500 actions each.
We can then assess behaviour by computing the Levenshtein distance between sequences of 500 instead of up to 2500.
"""

num_cores = 32
exp_name = 'NoConv-Dump100-AllArchOps'

def generate_individual(in_shape, out_shape, init_connect_rate, init_seed):
    return Individual(in_shape=in_shape, out_shape=out_shape, init_seed=init_seed, init_connect_rate=init_connect_rate)


def generate_init_population(in_shape, out_shape, num_individuals=10, init_connect_rate=0.5):
    population = []
    for i in range(num_individuals):
        # init_seed = randint(low=0, high=4294967295)
        ind = generate_individual(in_shape, out_shape, init_connect_rate, i)
        population.append(ind)
    return population


def levenshtein_distance(seq1, seq2):
    return Levenshtein.distance(seq1, seq2)

def play_atari(individual, game_name, game_iterations, env_seeds, vis=False, sleep=0.05):
    """
    Given an indivudual, play an Atari game and return the score.
    :param individual:
    :return: sequence of actions as numpy array
    """

    from cv2 import cvtColor, COLOR_RGB2GRAY, resize
    import keras as K
    import gym.spaces

    # Initialize conv layers
    # conv_module = KerasConvModules.gen_DQN_architecture(input_shape=(84, 84, 4), seed=individual.init_seed)
    # Apply mutations to conv layers according to generation seeds
    # mutate_conv_module(individual, conv_module, mut_power=0.008)
    # Initialize logic layer (mutable when complexification rate > 0)
    logic_module = map_to_network(individual, mut_power=0.008, complexification_rate=0.5)
    # Initialize modular evaluator
    me = NoConvEvaluator.NoConvEvaluator(logic_module)
    me.create_output_readers()

    # Integer RNG for initial game actions
    rnd_int = RandomState(seed=individual.init_seed).randint

    total_reward = 0
    episode_actions_performed = []
    env = gym.make(game_name)
    action_space_size = logic_module.out_size

    for env_seed in env_seeds:

        actions_performed = []
        env.seed(env_seed)
        x = resize(env.reset(), (84, 84))
        x_lum = cvtColor(x, COLOR_RGB2GRAY).reshape(84, 84, 1)
        x = np.concatenate((x, x_lum), axis=2)
        x = x.reshape(1, 84, 84, 4)
        prev_frame = np.zeros_like(x)

        for _ in range(30):
            action = rnd_int(low=0, high=action_space_size)
            env.step(action)
            actions_performed.append(action)

        for i in range(game_iterations):

            # Evaluate network
            step = me.eval(fmax(x, prev_frame))

            # Store frame for reuse
            prev_frame = x

            # Compute one-hot boolean action - THIS WILL DEPEND ON THE GAME
            y_ = step.flatten()
            a = y_.argmax()
            actions_performed.append(a)
            observation, reward, done, info = env.step(a)

            if done:
                # Ensure all action sequences have the same length
                extra_actions = game_iterations - i - 1
                for _ in range(extra_actions):
                    actions_performed.append('x')
                break
            total_reward += reward

            # Use previous frame information to avoid flickering
            x = resize(observation, (84, 84))
            x_lum = cvtColor(x, COLOR_RGB2GRAY).reshape(84, 84, 1)
            x = np.concatenate((x, x_lum), axis=2)
            x = x.reshape(1, 84, 84, 4)

            # sys.stdout.write("Game iteration:{}\tAction:{}\r".format(i, a))
            # sys.stdout.flush()
        episode_actions_performed.append(actions_performed)
    env.close()
    K.backend.clear_session()
    del me # Think this might be needed to cut memory use, we'll see
    del logic_module # this too...

    sys.stdout.write('{}{}\n'.format(individual, total_reward))
    sys.stdout.flush()
    return (individual, total_reward, episode_actions_performed)

def partition(archive, num_partitions=num_cores):
    archive_partitions = []
    items_per_partition = len(archive) // num_partitions
    for i in range(num_partitions):
        archive_partitions.append(archive[i*items_per_partition:(i+1)*items_per_partition])
    return archive_partitions

def compute_distances_segmented(seq, archive, segment_size):
    num_segments = len(seq) // segment_size + 1
    segment_distances = []
    for i in range(num_segments):
        segment_distances.append([levenshtein_distance(seq[i*segment_size:(i+1)*segment_size], x[i*segment_size:(i+1)*segment_size]) for x in archive])
    distances = np.array(segment_distances)
    sum_distances = np.sum(distances, axis=0)
    return sum_distances.tolist()

def compute_novelty(individual, game_score, actions_performed, archive, segment_size=500, n_neighbours=25):

    num_episodes = len(actions_performed)

    # PARALLEL
    novelty_scores = []
    for episode in range(num_episodes):
        # convert actions_performed to a string
        episode_actions_performed = ''.join(str(i) for i in actions_performed[episode])
        archive_partitions = partition(archive[episode])
        distance_lists = Parallel(n_jobs=num_cores)(
            delayed(compute_distances_segmented)(
                episode_actions_performed, part, segment_size
            ) for part in archive_partitions
        )

        # compute distance between actions performed and sequences in archive
        distances = list(chain(*distance_lists))

        # find distances to n_neighbours
        nearest_neighbours = sorted(distances)[:n_neighbours]

        del distance_lists
        del distances

        novelty_scores.append(np.mean(nearest_neighbours))
    # print(novelty_scores)
    novelty_score = np.mean(novelty_scores)
    # return (individual, game_score, actions_performed, mean distance to nearest neighbours)
    sys.stdout.write('{} {} {}\n'.format(individual, game_score, novelty_score))
    sys.stdout.flush()
    return (individual, game_score, actions_performed, novelty_score)

def save_population(population, generation, expname):
    import pickle, gzip
    population_json = [ind.to_JSON() for ind in population]
    file = gzip.open('./{}/population-{}-{}.pkl'.format(exp_name, expname, generation), 'wb')
    pickle.dump(population_json, file)
    file.flush()
    file.close()

def load_population(filename):
    import pickle, gzip
    with gzip.open(filename, 'rb') as file:
        population_json = pickle.load(file)
        population = [from_JSON(ind) for ind in population_json]
        generations = len(population[len(population)-1].generation_seeds)
        file.close()
        return population, generations

def save_archive(archive, gen, expname):
    import pickle, gzip
    file = gzip.open('./{}/archive-{}-{}.pkl'.format(exp_name, expname, gen), 'wb')
    pickle.dump(archive, file)
    file.flush()
    file.close()

def load_archive(filename):
    import pickle, gzip
    with gzip.open(filename, 'rb') as file:
        archive = pickle.load(file)
        file.close()
        return archive

elite_g = None


def evolve_solution(game_name, action_space_size, archive_p=0.1):
    global elite_g
    in_shape = (28224,)
    out_shape = (action_space_size,)
    init_connect_rate = 1.0
    pop_size = 500 + 1
    generations = 1000
    starting_generation = 0
    game_iterations = 5000
    population = generate_init_population(in_shape, out_shape, pop_size, init_connect_rate)
    # population, starting_generation = load_population('./{}/hybrid_exp-499.pkl'.format(exp_name))
    starting_generation = 0

    env_seeds = [0, 1]
    num_episodes = len(env_seeds)
    rnd_int = np.random.randint
    T = 100
    elite = None
    archive = [[] for _ in range(num_episodes)]
    # archive = load_archive('./{}/archive-hybrid_exp-499.pkl'.format(exp_name))
    mean_game_scores = []
    best_game_scores = []
    rand = np.random.RandomState(seed=1).uniform

    for gen in range(starting_generation, generations):

        print('\nGeneration', gen)
        save_population(population, gen, exp_name)

        # Clear archive after every 100 generations
        if gen % 100 == 0:
            archive = [[] for _ in range(num_episodes)]

        # PARALLEL
        results = Parallel(n_jobs=num_cores)(
            delayed(play_atari)(
                ind, game_name, game_iterations, env_seeds) for ind in population
        )

        for _, _, actions_performed in results:
            if rand() < archive_p:
                for episode in range(num_episodes):
                    episode_actions_performed = actions_performed[episode]
                    archive[episode].append(''.join(str(i) for i in episode_actions_performed))

        save_archive(archive, gen, exp_name)

        # Computing Novelty Scores
        print('Computing novelty scores.')
        # print('Archive Size: {}'.format(len(archive)))
        results_novelty = []
        # archive partitions

        for ind, game_score, actions_performed in results:
            results_novelty.append(compute_novelty(ind, game_score, actions_performed, archive))

        game_scores = [x[1] for x in results_novelty]

        # Compute mean game score
        mean_game_score = np.mean(game_scores)
        mean_game_scores.append(mean_game_score)

        # Find Most Novel Individual
        results_novelty = sorted(results_novelty, key=lambda x: x[3])
        n_elite_ind, n_elite_game_score, n_elite_actions_performed, n_elite_novelty_score = results_novelty[-1]

        # Find Best Scoring Individual
        g_elite_ind, g_elite_game_score, g_elite_actions_performed, g_elite_novelty_score = \
            sorted(results_novelty, key=lambda x: x[1])[-1]

        best_game_scores.append(g_elite_game_score)

        elite = n_elite_ind.copy()

        file = open('./{}/{}-elite_game-{}-score-{}.txt'.format(exp_name, game_name, gen, g_elite_game_score), 'w')
        file.write(g_elite_ind.to_JSON())
        file.flush()
        file.close()
        file = open('./{}/{}-elite_novelty-{}-score-{}.txt'.format(exp_name, game_name, gen, n_elite_game_score), 'w')
        file.write(n_elite_ind.to_JSON())
        file.flush()
        file.close()

        print('Best Score (Novelty): {} {} {}'.format(n_elite_ind, n_elite_game_score, n_elite_novelty_score))
        print('Best Score (Game Score): {} {} {}'.format(g_elite_ind, g_elite_game_score, g_elite_novelty_score))
        print('Previous Mean Game Scores: {}'.format(mean_game_scores))
        print('Current Mean Game Score: {}'.format(mean_game_score))
        print('Previous Best Game Scores: {}'.format(best_game_scores))
        print('Current Best Game Score: {}'.format(best_game_scores[-1]))

        run_file = open('./{}/run_info.txt'.format(exp_name), 'w')
        run_file.write('Mean Game Scores\n{}\n'.format(mean_game_scores))
        run_file.write('Best Game Scores\n{}\n'.format(best_game_scores))
        run_file.flush()
        run_file.close()

        # Truncate based on novelty score
        pop_trunc = [result for result in results_novelty[-T:]]
        # Then select half with highest game score
        pop_trunc = sorted(pop_trunc, key=lambda x: x[1])[-T // 2:]

        if gen == generations - 1:
            return elite
        new_pop = [elite]
        for _ in range(pop_size - 1):
            offspring, _, _, _ = pop_trunc[rnd_int(low=0, high=len(pop_trunc))]
            offspring = offspring.copy()
            mutate(offspring)
            new_pop.append(offspring)

        population = new_pop

        gc.collect()


### Testing
import gym.spaces

game_name = 'MsPacman-v0'
env = gym.make(game_name)
action_space_size = env.action_space.n
agent = evolve_solution(game_name, action_space_size)