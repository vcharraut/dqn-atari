import argparse

from playground.atari.config import Config, ConfigRainbow
from playground.atari.algorithm.rainbow import Rainbow
from playground.atari.algorithm.dqn import DQN
from playground.cartpole.dqn_cartpole import DQN_Cartpole


parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='', help='')
parser.add_argument('--algo', type=str, default='', help='')
parser.add_argument('--display', type=int, default=1, help='')
parser.add_argument('--record', type=int, default=0, help='')

args = parser.parse_args()

display = args.display
record = args.record

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

if args.env == 'cartpole':
    agent = DQN_Cartpole(None, 8, None, None, None,
                         evaluation=True,
                         record=record)
    agent.play(display=display,
               model_path='playground/cartpole/save/lr=0.01_hidden=8_gamma= \
                   0.99_batchsize=256_steptarget=100.pt')
elif args.env == 'atari':
    if args.algo == 'dqn':
        print("No model saved yet.")
        config = Config()
        agent = DQN('BreakoutNoFrameskip-v4', config, False, False,
                    False, True, evaluation=True, record=record)
    elif args.algo == 'dqn+':
        config = Config()
        agent = DQN('BreakoutNoFrameskip-v4', config, True,
                    True, evaluation=True, record=record)
        agent.test(display=False,
                   model_path='playground/atari/save/dqn_doubleq_dueling_1.pt')
    elif args.algo == 'rainbow':
        config = ConfigRainbow()
        agent = Rainbow('BreakoutNoFrameskip-v4', config,
                        True, evaluation=True, record=record)
        agent.test(display=display,
                   model_path='playground/atari/save/rainbow_adam_1-final.pt')
    else:
        raise ValueError('Algo is not valid')
else:
    raise ValueError('Environnement is not valid')
