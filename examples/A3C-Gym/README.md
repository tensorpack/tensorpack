### A3C code and models for Atari games in gym

Multi-GPU version of the A3C algorithm in
[Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783).

Results of the same code trained on 47 different Atari games were uploaded to OpenAI Gym.
Most of them were the best reproducible results on gym.
However OpenAI has later completely removed leaderboard from their site.

### To train on an Atari game:

`./train-atari.py --env Breakout-v0 --gpu 0`

In each iteration it trains on a batch of 128 new states.
The speed is about 6~10 iterations/s on 1 GPU plus 12+ CPU cores.
With 2 TitanX + 20+ CPU cores, by setting `SIMULATOR_PROC=240, PREDICT_BATCH_SIZE=30, PREDICTOR_THREAD_PER_GPU=6`, it can improve to 16 it/s (2K images/s).
Note that the network architecture is larger than what's used in the original paper.

The pretrained models are all trained with 4 GPUs for about 2 days.
But on simple games like Breakout, you can get good performance within several hours.
Also note that multi-GPU doesn't give you obvious speedup here,
because the bottleneck in this implementation is not computation but simulation.

Some practicical notes:

1. Prefer Python 3; Windows not supported.
2. Training with a significant slower speed (e.g. on CPU) will result in very bad score, probably because of the slightly off-policy implementation.
3. Occasionally, processes may not get terminated completely.
	If you're using Linux, install [python-prctl](https://pypi.org/project/python-prctl/) to prevent this.

### To test a model:

Download models from [model zoo](http://models.tensorpack.com/OpenAIGym/).

Watch the agent play:
`./train-atari.py --task play --env Breakout-v0 --load Breakout-v0.npz`

Dump some videos:
`./train-atari.py --task dump_video --load Breakout-v0.npz --env Breakout-v0 --output output_dir --episode 3`

This table lists available pretrained models and __scores__ (average over 100 episodes),
with their submission links.
The old submission site is not maintained any more so the links might become invalid any time.

| | | | |
| - | - | - | - |
| [AirRaid](https://gym.openai.com/evaluations/eval_zIeNk5MxSGOmvGEUxrZDUw)(2727) | [Alien](https://gym.openai.com/evaluations/eval_8NR1IvjTQkSIT6En4xSMA) (2611) |  [Amidar](https://gym.openai.com/evaluations/eval_HwEazbHtTYGpCialv9uPhA)(1376) | [Assault](https://gym.openai.com/evaluations/eval_tCiHwy5QrSdFVucSbBV6Q)(3397) |
| [Asterix](https://gym.openai.com/evaluations/eval_mees2c58QfKm5GspCjRfCA)(407432) | [Asteroids](https://gym.openai.com/evaluations/eval_8eHKsRL4RzuZEq9AOLZA)(1965) | [Atlantis](https://gym.openai.com/evaluations/eval_Z1B3d7A1QCaQk1HpO1Rg)(217186) | [BankHeist](https://gym.openai.com/evaluations/eval_hifoaxFTIuLlPd38BjnOw)(1274) |
| [BattleZone](https://gym.openai.com/evaluations/eval_SoLit2bR1qmFoC0AsJF6Q)(29210) | [BeamRider](https://gym.openai.com/evaluations/eval_KuOYumrjQjixwL0spG0iCA)(5972) | [Berzerk](https://gym.openai.com/evaluations/eval_Yri0XQbwRy62NzWILdn5IA)(2289) | [Breakout](https://gym.openai.com/evaluations/eval_NiKaIN4NSUeEIvWqIgVDrA) (667) |
| [Carnival](https://gym.openai.com/evaluations/eval_xJSOlo2lSWaH1wHEOX5vw)(5211) | [Centipede](https://gym.openai.com/evaluations/eval_mc1Kp5e6R42rFdjeMLzkIg)(2909) | [ChopperCommand](https://gym.openai.com/evaluations/eval_tYVKyh7wQieRIKgEvVaCuw)(6031) | [CrazyClimber](https://gym.openai.com/evaluations/eval_bKeBg0QwSgOm6A0I0wDhSw)(105297) |
| [DemonAttack](https://gym.openai.com/evaluations/eval_tt21vVaRCKYzWFcg1Kw)(33992) | [DoubleDunk](https://gym.openai.com/evaluations/eval_FI1GpF4TlCuf29KccTpQ)(23) | [ElevatorAction](https://gym.openai.com/evaluations/eval_SqeAouMvR0icRivx2xprZg)(11377) | [FishingDerby](https://gym.openai.com/evaluations/eval_pPLCnFXsTVaayrIboDOs0g)(34) |
| [Frostbite](https://gym.openai.com/evaluations/eval_qtC3taKFSgWwkO9q9IM4hA)(6824) | [Gopher](https://gym.openai.com/evaluations/eval_KVcpR1YgQkEzrL2VIcAQ)(22595) | [Gravitar](https://gym.openai.com/evaluations/eval_QudrLdVmTpK9HF5juaZr0w)(2144) | [IceHockey](https://gym.openai.com/evaluations/eval_8oWCTwwGS7OUTTGRwBPQkQ)(19) |
| [Jamesbond](https://gym.openai.com/evaluations/eval_mLF7XPi8Tw66pnjP73JsmA)(640) | [JourneyEscape](https://gym.openai.com/evaluations/eval_S9nQuXLRSu7S5x21Ay6AA)(-407) | [Kangaroo](https://gym.openai.com/evaluations/eval_TNJiLB8fTqOPfvINnPXoQ)(6540) | [Krull](https://gym.openai.com/evaluations/eval_dfOS2WzhTh6sn1FuPS9HA)(6100) |
| [KungFuMaster](https://gym.openai.com/evaluations/eval_vNWDShYTRC0MhfIybeUYg)(34767) | [MsPacman](https://gym.openai.com/evaluations/eval_kpL9bSsS4GXsYb9HuEfew)(5738) | [NameThisGame](https://gym.openai.com/evaluations/eval_LZqfv706SdOMtR4ZZIwIsg)(15321) | [Phoenix](https://gym.openai.com/evaluations/eval_uzUruiB3RRKUMvJIxvEzYA)(75312) |
| [Pong](https://gym.openai.com/evaluations/eval_8L7SV59nSW6GGbbP3N4G6w)(21) | [Pooyan](https://gym.openai.com/evaluations/eval_UXFVI34MSAuNTtjZcK8N0A)(5607) | [Qbert](https://gym.openai.com/evaluations/eval_S8XdrbByQ1eWLUD5jtQYIQ)(20182) | [Riverraid](https://gym.openai.com/evaluations/eval_OU4x3DkTfm4uaXy6CIaXg)(14185) |
| [RoadRunner](https://gym.openai.com/evaluations/eval_wINKQTwxT9ipydHOXBhg)(60615) | [Robotank](https://gym.openai.com/evaluations/eval_Gr5c0ld3QACLDPQrGdzbiw)(60) | [Seaquest](https://gym.openai.com/evaluations/eval_pjjgc9POQJK4IuVw8nXlBw)(46890) | SpaceInvaders(3454) |
| [StarGunner](https://gym.openai.com/evaluations/eval_JB5cOJXFSS2cTQ7dXK8Iag)(93480) | [Tennis](https://gym.openai.com/evaluations/eval_gDjJD0MMS1yLm1T0hdqI4g)(23) | Tutankham(275) | [UpNDown](https://gym.openai.com/evaluations/eval_KmkvMJkxQFSED20wFUMdIA)(92163) |
| [VideoPinball](https://gym.openai.com/evaluations/eval_PWwzNhVFR2CxjYvEsPfT1g)(140156) | [WizardOfWor](https://gym.openai.com/evaluations/eval_1oGQhphpQhmzEMIYRrrp0A)(3824) | [Zaxxon](https://gym.openai.com/evaluations/eval_TIQ102EwTrHrOyve2RGfg)(32894) | |


All models above are trained with the `-v0` variant of atari games.
Note that this variant is quite different from DeepMind papers, so the scores are not directly comparable.
The most notable differences are:
+ Each action is randomly repeated 2~4 times.
+ Inputs are RGB instead of greyscale.
+ An episode is limited to 60000 steps.
+ Lost of live is not end of episode.

Also see the DQN implementation [here](../DeepQNetwork)
