### A3C code and models for Atari games in gym

Multi-GPU version of the A3C algorithm in
[Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783),
with <500 lines of code.

Results of the same code trained on 47 different Atari games were uploaded on OpenAI Gym.
You can see them in [my gym page](https://gym.openai.com/users/ppwwyyxx).
Most of them are the best reproducible results on gym.

### To train on an Atari game:

`./train-atari.py --env Breakout-v0 --gpu 0`

In each iteration it trains on a batch of 128 new states.
The speed is about 6~10 iterations/s on 1 GPU plus 12+ CPU cores.
With 2 TitanX + 20+ CPU cores, by setting `SIMULATOR_PROC=240, PREDICT_BATCH_SIZE=30, PREDICTOR_THREAD_PER_GPU=6`, it can improve to 16 it/s (2K images/s).
Note that the network architecture is larger than what's used in the original paper.

The uploaded models are all trained with 4 GPUs for about 2 days.
But on simple games like Breakout, you can get good performance within several hours.
Also note that multi-GPU doesn't give you obvious speedup here,
because the bottleneck in this implementation is not computation but data.

Some practicical notes:

1. Prefer Python 3.
2. Occasionally, processes may not get terminated completely. It is suggested to use `systemd-run` to run any
multiprocess Python program to get a cgroup dedicated for the task.
3. Training with a significant slower speed (e.g. on CPU) will result in very bad score, probably because of the slightly off-policy implementation.

### To test a model:

Download models from [model zoo](https://goo.gl/9yIol2).

Watch the agent play:
`./train-atari.py --task play --env Breakout-v0 --load Breakout-v0.npy`

Generate gym submissions:
`./train-atari.py --task gen_submit --load Breakout-v0.npy --env Breakout-v0 --output output_dir`

Models are available for the following atari environments (click to watch videos of my agent):

| | | | |
| - | - | - | - |
| [AirRaid](https://gym.openai.com/evaluations/eval_zIeNk5MxSGOmvGEUxrZDUw) | [Alien](https://gym.openai.com/evaluations/eval_8NR1IvjTQkSIT6En4xSMA) |  [Amidar](https://gym.openai.com/evaluations/eval_HwEazbHtTYGpCialv9uPhA) | [Assault](https://gym.openai.com/evaluations/eval_tCiHwy5QrSdFVucSbBV6Q) |
| [Asterix](https://gym.openai.com/evaluations/eval_mees2c58QfKm5GspCjRfCA) | [Asteroids](https://gym.openai.com/evaluations/eval_8eHKsRL4RzuZEq9AOLZA) | [Atlantis](https://gym.openai.com/evaluations/eval_Z1B3d7A1QCaQk1HpO1Rg) | [BankHeist](https://gym.openai.com/evaluations/eval_hifoaxFTIuLlPd38BjnOw) |
| [BattleZone](https://gym.openai.com/evaluations/eval_SoLit2bR1qmFoC0AsJF6Q) | [BeamRider](https://gym.openai.com/evaluations/eval_KuOYumrjQjixwL0spG0iCA) | [Berzerk](https://gym.openai.com/evaluations/eval_Yri0XQbwRy62NzWILdn5IA) | [Breakout](https://gym.openai.com/evaluations/eval_NiKaIN4NSUeEIvWqIgVDrA) |
| [Carnival](https://gym.openai.com/evaluations/eval_xJSOlo2lSWaH1wHEOX5vw) | [Centipede](https://gym.openai.com/evaluations/eval_mc1Kp5e6R42rFdjeMLzkIg) | [ChopperCommand](https://gym.openai.com/evaluations/eval_tYVKyh7wQieRIKgEvVaCuw) | [CrazyClimber](https://gym.openai.com/evaluations/eval_bKeBg0QwSgOm6A0I0wDhSw) |
| [DemonAttack](https://gym.openai.com/evaluations/eval_tt21vVaRCKYzWFcg1Kw) | [DoubleDunk](https://gym.openai.com/evaluations/eval_FI1GpF4TlCuf29KccTpQ) | [ElevatorAction](https://gym.openai.com/evaluations/eval_SqeAouMvR0icRivx2xprZg) | [FishingDerby](https://gym.openai.com/evaluations/eval_pPLCnFXsTVaayrIboDOs0g) |
| [Frostbite](https://gym.openai.com/evaluations/eval_qtC3taKFSgWwkO9q9IM4hA) | [Gopher](https://gym.openai.com/evaluations/eval_KVcpR1YgQkEzrL2VIcAQ) | [Gravitar](https://gym.openai.com/evaluations/eval_QudrLdVmTpK9HF5juaZr0w) | [IceHockey](https://gym.openai.com/evaluations/eval_8oWCTwwGS7OUTTGRwBPQkQ) |
| [Jamesbond](https://gym.openai.com/evaluations/eval_mLF7XPi8Tw66pnjP73JsmA) | [JourneyEscape](https://gym.openai.com/evaluations/eval_S9nQuXLRSu7S5x21Ay6AA) | [Kangaroo](https://gym.openai.com/evaluations/eval_TNJiLB8fTqOPfvINnPXoQ) | [Krull](https://gym.openai.com/evaluations/eval_dfOS2WzhTh6sn1FuPS9HA) |
| [KungFuMaster](https://gym.openai.com/evaluations/eval_vNWDShYTRC0MhfIybeUYg) | [MsPacman](https://gym.openai.com/evaluations/eval_kpL9bSsS4GXsYb9HuEfew) | [NameThisGame](https://gym.openai.com/evaluations/eval_LZqfv706SdOMtR4ZZIwIsg) | [Phoenix](https://gym.openai.com/evaluations/eval_uzUruiB3RRKUMvJIxvEzYA) |
| [Pong](https://gym.openai.com/evaluations/eval_8L7SV59nSW6GGbbP3N4G6w) | [Pooyan](https://gym.openai.com/evaluations/eval_UXFVI34MSAuNTtjZcK8N0A) | [Qbert](https://gym.openai.com/evaluations/eval_S8XdrbByQ1eWLUD5jtQYIQ) | [Riverraid](https://gym.openai.com/evaluations/eval_OU4x3DkTfm4uaXy6CIaXg) |
| [RoadRunner](https://gym.openai.com/evaluations/eval_wINKQTwxT9ipydHOXBhg) | [Robotank](https://gym.openai.com/evaluations/eval_Gr5c0ld3QACLDPQrGdzbiw) | [Seaquest](https://gym.openai.com/evaluations/eval_pjjgc9POQJK4IuVw8nXlBw) | [SpaceInvaders](https://gym.openai.com/evaluations/eval_Eduozx4HRyqgTCVk9ltw) |
| [StarGunner](https://gym.openai.com/evaluations/eval_JB5cOJXFSS2cTQ7dXK8Iag) | [Tennis](https://gym.openai.com/evaluations/eval_gDjJD0MMS1yLm1T0hdqI4g) | [Tutankham](https://gym.openai.com/evaluations/eval_gDjJD0MMS1yLm1T0hdqI4g) | [UpNDown](https://gym.openai.com/evaluations/eval_KmkvMJkxQFSED20wFUMdIA) |
| [VideoPinball](https://gym.openai.com/evaluations/eval_PWwzNhVFR2CxjYvEsPfT1g) | [WizardOfWor](https://gym.openai.com/evaluations/eval_1oGQhphpQhmzEMIYRrrp0A) | [Zaxxon](https://gym.openai.com/evaluations/eval_TIQ102EwTrHrOyve2RGfg) | |


Note that atari game settings in gym (AtariGames-v0) are quite different from DeepMind papers, so the scores are not comparable. The most notable differences are:
+ Each action is randomly repeated 2~4 times.
+ Inputs are RGB instead of greyscale.
+ An episode is limited to 10000 steps.
+ Lost of live is not end of episode.

Also see the DQN implementation [here](../DeepQNetwork)
