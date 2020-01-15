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
    'airaid': 'AirRaidNoFrameskip-v4',
    'alien': 'AlienNoFrameskip-v4',
    'amidar': 'AmidarNoFrameskip-v4',
    'assault': 'AssaultNoFrameskip-v4',
    'asterix': 'AsterixNoFrameskip-v4',
    'asteroids': 'AsteroidsNoFrameskip-v4',
    'atlantis': 'AtlantisNoFrameskip-v4',
    'bankheist': 'BankHeistNoFrameskip-v4',
    'battlezone': 'BattleZoneNoFrameskip-v4',
    'beamrider': 'BeamRiderNoFrameskip-v4',
    'berkzerk': 'BerzerkNoFrameskip-v4',
    'bowling': 'BowlingNoFrameskip-v4',
    'boxing': 'BoxingNoFrameskip-v4',
    'breakout': 'BreakoutNoFrameskip-v4',
    'carnival': 'CarnivalNoFrameskip-v4',
    'centipete': 'CentipedeNoFrameskip-v4',
    'choppercommand': 'ChopperCommandNoFrameskip-v4',
    'crazyclimber': 'CrazyClimberNoFrameskip-v4',
    'demonattack': 'DemonAttackNoFrameskip-v4',
    'doubledunk': 'DoubleDunkNoFrameskip-v4',
    'elevatoraction': 'ElevatorActionNoFrameskip-v4',
    'enduro': 'EnduroNoFrameskip-v4',
    'fishingderby': 'FishingDerbyNoFrameskip-v4',
    'freeway': 'FreewayNoFrameskip-v4',
    'frostbite': 'FrosbiteNoFrameskip-v4',
    'gopher': 'GopherNoFrameskip-v4',
    'gravitar': 'GravitarNoFrameskip-v4',
    'icehockey': 'IceHockeyNoFrameskip-v4',
    'jamesbond': 'JamesbondNoFrameskip-v4',
    'journeyescape': 'JourneyEscapeNoFrameskip-v4',
    'kangaroo': 'KangarooNoFrameskip-v4',
    'krull': 'KrullNoFrameskip-v4',
    'kungfumaster': 'KungFuMasterNoFrameskip-v4',
    'montezumarevenge': 'MontezumaRevengeNoFrameskip-v4',
    'mspacman': 'MsPacmanNoFrameskip-v4',
    'namethisgame': 'NameThisGameNoFrameskip-v4',
    'phoenix': 'PhoenixNoFrameskip-v4',
    'pitfall': 'PitfallNoFrameskip-v4',
    'pong': 'PongNoFrameskip-v4',
    'pooyan': 'PooyanNoFrameskip-v4',
    'privateye': 'PrivateEyeNoFrameskip-v4',
    'qbert': 'QbertNoFrameskip-v4',
    'riverraid': 'RiverraidNoFrameskip-v4',
    'roadrunner': 'RoadRunnerNoFrameskip-v4',
    'robotank': 'RobotankNoFrameskip-v4',
    'seaquest': 'SeaquestNoFrameskip-v4',
    'skiing': 'SkiingNoFrameskip-v4',
    'solaris': 'SolarisNoFrameskip-v4',
    'spaceinvaders': 'SpaceInvadersNoFrameskip-v4',
    'stargunner': 'StarGunnerNoFrameskip-v4',
    'tennis': 'TennisNoFrameskip-v4',
    'timepilot': 'TimePilotNoFrameskip-v4',
    'tutankham': 'TutankhamNoFrameskip-v4',
    'upndown': 'UpNDownNoFrameskip-v4',
    'venture': 'VentureNoFrameskip-v4',
    'videopinball': 'VideoPinballNoFrameskip-v4',
    'wizardofwor': 'WizardOfWorNoFrameskip-v4',
    'yarsrevenge': 'YarsRevengeNoFrameskip-v4',
    'zaxxon': 'ZaxxonNoFrameskip-v4'
}

if args.env not in dict_env:
    raise TypeError('Environment name not recognized.')

if args.algo == 'dqn':
    agent = DQN(dict_env[args.env], Config(),
                train=train, record=record)
    if train:
        agent.train(display=display)
    else:
        print('No agent trained yet.')

elif args.algo == 'doubleq':
    agent = DoubleQ_DQN(dict_env[args.env], Config(),
                        train=train, record=record)
    if train:
        agent.train(display=display)
    else:
        print('No agent trained yet.')
elif args.algo == 'dueling':
    agent = Dueling_DQN(dict_env[args.env], Config(),
                        train=train, record=record)
    if train:
        agent.train(display=display)
    else:
        agent.test(display=display,
                   model_path='./save_model/dqn/dqn_doubleq_dueling_1.pt')

else:
    raise TypeError('Algo is not valid')
