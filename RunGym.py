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
import sys
from itertools import product
import time

"""
Novelty. Select top T novelty scores from each generation.
Non-hybrid. Single form of selection pressure. 
Select top-scoring individual in validation among top T novel as elite.
"""

num_cores = 16
game_name = 'MsPacman-v0'
exp_name = 'GameScore'

def generate_individual(in_shape, out_shape, init_connect_rate, init_seed):
    return Individual(in_shape=in_shape, out_shape=out_shape, init_seed=init_seed, init_connect_rate=init_connect_rate)


def generate_init_population(in_shape, out_shape, num_individuals=10, init_connect_rate=0.5):
    population = []
    for i in range(num_individuals):
        # init_seed = randint(low=0, high=4294967295)
        ind = generate_individual(in_shape, out_shape, init_connect_rate, i)
        population.append(ind)
    return population



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

            if vis:
                env.render()
                time.sleep(sleep)


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

    sys.stdout.write('{}{}\n'.format(individual, total_reward))
    sys.stdout.flush()
    return (individual, total_reward, episode_actions_performed)



### Testing
import gym.spaces
ind = from_JSON('{"in_shape": "(84, 84, 4)", "out_shape": "(9,)", "init_connect_rate": "1.0", "init_seed": "976", "generation_seeds": ["1452488516", "2767836047", "3517343448", "1536683794", "1274168319", "4073902876", "3155924010", "2972768006", "3657926649", "254235909", "1567624117", "2702106432", "1237567372", "1629798871", "2801103686", "3294980023", "2307735679", "1625336938", "3577845034", "1279026622", "2539215302", "2056190793", "2059869481", "1734737592", "365642192", "3638705758", "4257534424", "1522560352", "3042513625", "2984999723", "2818625488", "2510873284", "2621320187", "3392158705", "965913018", "1655224552", "1330191371", "3510076811", "114111129", "1697325268", "3808601653", "2553353034", "3017996306", "2482170125", "1536427179", "2111152152", "802096646", "1812935917", "123046972", "999754614", "1652623579", "216438246", "939885832", "1485980227", "1204997022", "2397250566", "1386686078", "4250249689", "868215653", "1887564529", "3426791123", "1963829002", "2009895069", "3237112679", "3498385552", "3303394820", "2984002096", "381739745", "302037513", "2970052576", "3940850777", "1695472639", "3282572962", "2592435525", "4066664379", "378949116", "284287115", "567366389", "3293785514", "1521330349", "610973202", "2258461577", "2407156622", "3839688205", "105210911", "237360723", "2721110400", "388726124", "2514586443", "2889329296", "1225643129", "2044937341", "3000684775", "668827975", "697997280", "3942659925", "2913767754", "3009963422", "2950951301", "2571861800", "2719652903", "34293296", "3023931102", "16282", "2993628332", "1141189561", "266391742", "1480233664", "3878160712", "2876770989", "1475384264", "1157907489", "4063006464", "2316497171", "1457813580", "1842598357", "794349974", "2282157588", "3480142300", "3648135514", "721918952", "2219193783", "386486140", "3197639630", "1213371687", "935821867", "3764754021", "2288197262", "108667808", "2995594625", "3263372247", "366351962", "1352662233", "50601782", "2719893688", "411689779", "1339669201", "3524225684", "189177297", "1164402627", "3380748151", "2539783537", "38142965", "1298279923", "21318683", "804736195", "2311394553", "1742195613", "2268256251", "3724045865", "2850137963", "3294277608", "1548157039", "2080301507", "1916954520", "175091489", "2470020824"], "in_size": "28224", "out_size": "9", "num_neurons": "28233"}')
ind2 = from_JSON('{"in_shape": "(84, 84, 4)", "out_shape": "(9,)", "init_connect_rate": "1.0", "init_seed": "466", "generation_seeds": ["3647113223", "461100142", "4285677365", "138471480", "2572181088", "2267510845", "824064676", "975219508", "1516765704", "2792074665", "4243464184", "1220549348", "2636431891", "3653150499", "2851406550", "2295437580", "876612224", "1233336661", "472462831", "1587979622", "110071658", "225860696", "1041783931", "2664632128", "2502682185", "2811840794", "4191232835", "2401243574", "1006459753", "3383264800", "1747236957", "160310671", "3797307705", "744185967", "502391812", "2225820761", "3788796214", "3562622889", "1522341776", "4146967464", "442137748", "2131762920", "763185124", "3482749320", "1993359798", "1344938112", "1407948026", "138552979", "1967471584", "629060904", "2968899759", "423081209", "3042980924", "675205044", "3790437803", "923498846", "2048513056", "2346615697", "3256792612"], "in_size": "28224", "out_size": "9", "num_neurons": "28233"}')
# episodes = list(range(31,61))
episodes = list(range(1,31))
i, score, actions = play_atari(ind2, 'MsPacman-v0', 20000, episodes, vis=True, sleep=0.0015)
print(score / len(episodes))