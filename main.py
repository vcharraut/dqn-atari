import argparse

from core.atari.config import Config
from core.atari.algorithm.dqn import DQN
from core.atari.algorithm.double_q_dqn import DoubleQ_DQN
from core.atari.algorithm.dueling_dqn import Dueling_DQN


parser = argparse.ArgumentParser()
parser.add_argument('--train', action='store_true',
                    help='Train a new agent or show a trained one')
parser.add_argument('--env', type=str, default='breakout',
                    help='Gym environment')
parser.add_argument('--algo', type=str, default='dqn',
                    help='Algortihms to be used for the learning')
parser.add_argument('--display', type=int, default=0,
                    help='Display or not the agent in the environment')
parser.add_argument('--record', type=int, default=0,
                    help='Record or not the environment')

args = parser.parse_args()

display = args.display
record = args.record
train = args.train

dict_env = {
    'airaid': 'AirRaid-v0',
    'alien': 'Alien-v0',
    'amidar': 'Amidar-v0',
    'assault': 'Assault-v0',
    'asterix': 'Asterix-v0',
    'asteroids': 'Asteroids-v0',
    'atlantis': 'Atlantis-v0',
    'bankheist': 'BankHeist-v0',
    'battlezone': 'BattleZone-v0',
    'beamrider': 'BeamRider-v0',
    'berkzerk': 'Berzerk-v0',
    'bowling': 'Bowling-v0',
    'boxing': 'Boxing-v0',
    'breakout': 'BreakoutNoFrameskip-v4',
    'carnival': 'Carnival-v0',
    'centipete': 'Centipede-v0',
    'choppercommand': 'ChopperCommand-v0',
    'crazyclimber': 'CrazyClimber-v0',
    'demonattack': 'DemonAttack-v0',
    'doubledunk': 'DoubleDunk-v0',
    'elevatoraction': 'ElevatorAction-v0',
    'enduro': 'Enduro-v0',
    'fishingderby': 'FishingDerby-v0',
    'freeway': 'Freeway-v0',
    'frostbite': 'Frosbite-v0',
    'gopher': 'Gopher-v0',
    'gravitar': 'Gravitar-v0',
    'icehockey': 'IceHockey-v0',
    'jamesbond': 'Jamesbond-v0',
    'journeyescape': 'JourneyEscape-v0',
    'kangaroo': 'Kangaroo-v0',
    'krull': 'Krull-v0',
    'kungfumaster': 'KungFuMaster-v0',
    'montezumarevenge': 'MontezumaRevenge-v0',
    'mspacman': 'MsPacman-v0',
    'namethisgame': 'NameThisGame-v0',
    'phoenix': 'Phoenix-v0',
    'pitfall': 'Pitfall-v0',
    'pong': 'Pong-v0',
    'pooyan': 'Pooyan-v0',
    'privateye': 'PrivateEye-v0',
    'qbert': 'Qbert-v0',
    'riverraid': 'Riverraid-v0',
    'roadrunner': 'RoadRunner-v0',
    'robotank': 'Robotank-v0',
    'seaquest': 'Seaquest-v0',
    'skiing': 'Skiing-v0',
    'solaris': 'Solaris-v0',
    'spaceinvaders': 'SpaceInvaders-v0',
    'stargunner': 'StarGunner-v0',
    'tennis': 'Tennis-v0',
    'timepilot': 'TimePilot-v0',
    'tutankham': 'Tutankham-v0',
    'upndown': 'UpNDown-v0',
    'venture': 'Venture-v0',
    'videopinball': 'VideoPinball-v0',
    'wizardofwor': 'WizardOfWor-v0',
    'yarsrevenge': 'YarsRevenge-v0',
    'zaxxon': 'Zaxxon-v0'
}

if args.env not in dict_env:
    raise TypeError('Environment name not recognized.')

if args.algo == 'dqn':
    agent = DQN(dict_env[args.env], Config(),
                train=train, record=record)
    if train:
        agent.train()
    else:
        print('No agent trained yet.')

elif args.algo == 'doubleq':
    agent = DoubleQ_DQN(dict_env[args.env], Config(),
                        train=train, record=record)
    if train:
        agent.train()
    else:
        print('No agent trained yet.')
elif args.algo == 'dueling':
    agent = Dueling_DQN(dict_env[args.env], Config(),
                        train=train, record=record)
    if train:
        agent.train()
    else:
        agent.test(display=False,
                   model_path='core/atari/save/dqn_doubleq_dueling_1.pt')

else:
    raise TypeError('Algo is not valid')
