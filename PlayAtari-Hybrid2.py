from Individual import Individual, from_JSON
from Mutator import mutate
from numpy.random import randint, RandomState
from NetworkMapper import mutate_conv_module
from joblib import Parallel, delayed
import numpy as np
import KerasEvaluator
from numpy import fmax
import KerasConvModules
import gc
import Levenshtein
from itertools import chain
import sys
from itertools import product

"""
Novelty. Select top T novelty scores from each generation.
Non-hybrid. Single form of selection pressure.  
Select top-scoring individual in validation among top T novel as elite.
"""

num_cores = 8
game_name = 'MsPacman-v0'
exp_name = 'Hybrid2'

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

    # Initialize Keras model
    conv_module = KerasConvModules.gen_DQN_architecture(input_shape=individual.in_shape,
                                                        output_size=individual.out_size,
                                                        seed=individual.init_seed)

    mutate_conv_module(individual, conv_module, mut_power=0.002)
    me = KerasEvaluator.KerasEvaluator(conv_module)

    # Integer RNG for initial game actions
    rnd_int = RandomState(seed=individual.init_seed).randint

    total_reward = 0
    episode_actions_performed = []
    env = gym.make(game_name)
    action_space_size = env.action_space.n

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
    del me

    sys.stdout.write('{} {} {}\n'.format(individual, individual.init_seed, total_reward))
    sys.stdout.flush()
    return (individual, total_reward, episode_actions_performed)

def partition(episode_archive):
    num_partitions = len(episode_archive) if len(episode_archive) < num_cores else num_cores
    archive_partitions = []
    items_per_partition = len(episode_archive) // num_partitions
    for i in range(num_partitions):
        archive_partitions.append(episode_archive[i * items_per_partition:(i + 1) * items_per_partition])
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
        nearest_neighbours = [d for d in nearest_neighbours if not np.isnan(d)]

        del distance_lists
        del distances

        novelty_scores.append(np.mean(nearest_neighbours))

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


def evolve_solution(game_name, action_space_size, pop_size=1000+1, archive_p=0.01, archive_dump=None, n_generations=1000,
                    training_frames=20000, training_episodes=(0,), starting_generation=0, T=40, archive=None, population=None,
                    validation_frames=10000, validation_episodes=(1,)):

    in_shape = (84,84,4)
    out_shape = (action_space_size,)
    init_connect_rate = 1.0

    num_episodes = len(training_episodes)

    if population is None:
        population = generate_init_population(in_shape, out_shape, pop_size, init_connect_rate)


    if archive is None:
        archive = [[] for _ in range(num_episodes)]

    rnd_int = np.random.RandomState(seed=1).randint
    rnd_uniform = np.random.RandomState(seed=1).uniform

    mean_game_scores = []
    best_game_scores = []
    mean_validation_scores = []

    for gen in range(starting_generation, n_generations):

        print('\nGeneration', gen)
        save_population(population, gen, exp_name)

        # Clear archive, maybe
        if not archive_dump is None:
            if gen % archive_dump == 0:
                archive = [[] for _ in range(num_episodes)]

        # PARALLEL
        results = Parallel(n_jobs=num_cores)(
            delayed(play_atari)(
                ind, game_name, training_frames, training_episodes) for ind in population
        )

        # Randomly select archive_p*pop_size agent behaviours for archiving
        archive_indices = np.random.choice(list(range(len(results))), size=int(archive_p*pop_size))
        for i in archive_indices:
            _, _, actions_performed = results[i]
            for episode in range(num_episodes):
                episode_actions_performed = actions_performed[episode]
                archive[episode].append(''.join(str(i) for i in episode_actions_performed))

        save_archive(archive, gen, exp_name)

        # Computing novelty scores
        print('Computing novelty scores.')
        results_novelty = []
        for ind, game_score, actions_performed in results:
            results_novelty.append(compute_novelty(ind, game_score, actions_performed, archive))

        # Retrieve game scores
        game_scores = [x[1] for x in results_novelty]

        # Compute mean game score
        mean_game_score = np.mean(game_scores)
        mean_game_scores.append(mean_game_score)

        # Find most novel individual
        results_novelty = sorted(results_novelty, key=lambda x: x[3])
        n_elite_ind, n_elite_game_score, n_elite_actions_performed, n_elite_novelty_score = results_novelty[-1]

        # Find individual with best game score
        g_elite_ind, g_elite_game_score, g_elite_actions_performed, g_elite_novelty_score = \
            sorted(results_novelty, key=lambda x: x[1])[-1]

        best_game_scores.append(g_elite_game_score)

        elite_novelty = n_elite_ind.copy()

        file = open('./{}/{}-elite_game-{}-score-{}.txt'.format(exp_name, game_name, gen, g_elite_game_score), 'w')
        file.write(g_elite_ind.to_JSON())
        file.flush()
        file.close()
        file = open('./{}/{}-elite_novelty-{}-score-{}.txt'.format(exp_name, game_name, gen, n_elite_game_score), 'w')
        file.write(n_elite_ind.to_JSON())
        file.flush()
        file.close()

        print('Best Score (Novelty): {} {:.2f} {:.2f}'.format(n_elite_ind, n_elite_game_score, n_elite_novelty_score))
        print('Best Score (Game Score): {} {:.2f} {:.2f}'.format(g_elite_ind, g_elite_game_score, g_elite_novelty_score))
        print('Previous Mean Game Scores: {}'.format(mean_game_scores))
        print('Current Mean Game Score: {:.2f}'.format(mean_game_score))
        print('Previous Best Game Scores: {}'.format(best_game_scores))
        print('Current Best Game Score: {}'.format(best_game_scores[-1]))

        # Truncate based on novelty score
        pop_trunc = [result for result in results_novelty[-T:]]

        # Then select half with highest game score (Not in main experiment)
        pop_trunc = sorted(pop_trunc, key=lambda x: x[1])[-T // 2:]

        # Use cross-validation to select elite
        validation_pop = [result[0] for result in sorted(pop_trunc, key=lambda x: x[1])[-2:]]
        print([(x.init_seed, ns) for x, _, _, ns in pop_trunc[-2:]])
        validation_runs = product(validation_pop, validation_episodes)
        # print([(x.init_seed, y) for x,y in validation_runs])

        print('Running Validation Episodes.')
        validation_results = Parallel(n_jobs=num_cores)(
            delayed(play_atari)(
                ind, game_name, validation_frames, [episode]) for ind, episode in validation_runs
        )

        # Collect scores by individual
        validation_scores = []
        for ind in validation_pop:
            validation_scores.append((ind, np.mean([v_score for v_ind, v_score, _ in validation_results if v_ind == ind])))
        validation_scores = sorted(validation_scores, key=lambda x: x[1])

        print([(x.init_seed, y) for x,y in validation_scores])

        # Sort based on game score and select top individual
        top_validation_ind, top_validation_score = validation_scores[-1]

        # Add best-generalizing individual to reproduction population
        # pop_trunc.append((top_validation_ind, None, None, None))

        mean_validation_scores.append(top_validation_score)
        print('Previous Mean Validation Scores: {}'.format(mean_validation_scores))
        print('Current Mean Validation Score Over {} Episodes: {:.2f}'.format(len(validation_episodes), mean_validation_scores[-1]))

        file = open('./{}/{}-elite_validation-{}-mean_score-{}.txt'.format(exp_name, game_name, gen, mean_validation_scores[-1]), 'w')
        file.write(top_validation_ind.to_JSON())
        file.flush()
        file.close()

        run_file = open('./{}/run_info.txt'.format(exp_name), 'w')
        run_file.write('Mean Game Scores\n{}\n'.format(mean_game_scores))
        run_file.write('Best Game Scores\n{}\n'.format(best_game_scores))
        run_file.write('Mean Validation Scores Over {} Episodes\n{}\n'.format(len(validation_episodes), mean_validation_scores))
        run_file.flush()
        run_file.close()

        if gen == n_generations - 1:
            return g_elite_game_score
        # Initialize population with elite ind
        new_pop = [top_validation_ind]
        for _ in range(pop_size - 1):
            offspring, _, _, _ = pop_trunc[rnd_int(low=0, high=len(pop_trunc))]
            offspring = offspring.copy()
            mutate(offspring)
            new_pop.append(offspring)

        population = new_pop
        gc.collect()
        break


### Testing
import gym.spaces
env = gym.make(game_name)
action_space_size = env.action_space.n
env.close()
evolve_solution(game_name, action_space_size, pop_size=50+1, archive_p=0.2, archive_dump=None,
                training_episodes=[0], T=20,
                training_frames=20000, validation_frames=20000, validation_episodes=list(range(1,6)))