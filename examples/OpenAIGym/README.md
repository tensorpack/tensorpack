
# To run a pretrained Batch-A3C atari model for 100 episodes:

1. install [tensorpack](https://github.com/ppwwyyxx/tensorpack)
2. Download models from [model zoo](https://drive.google.com/open?id=0B9IPQTvr2BBkS0VhX0xmS1c5aFk)
3. `ENV=NAME_OF_ENV ./run-atari.py --load "$ENV".tfmodel --env "$ENV"`

Models are available for the following gym atari environments (click links for videos):

+ [AirRaid-v0](https://gym.openai.com/evaluations/eval_zIeNk5MxSGOmvGEUxrZDUw) (a bit flickering, don't know why)
+ [Alien-v0](https://gym.openai.com/evaluations/eval_8NR1IvjTQkSIT6En4xSMA)
+ [Amidar-v0](https://gym.openai.com/evaluations/eval_HwEazbHtTYGpCialv9uPhA)
+ [Assault-v0](https://gym.openai.com/evaluations/eval_tCiHwy5QrSdFVucSbBV6Q)
+ [Asterix-v0](https://gym.openai.com/evaluations/eval_mees2c58QfKm5GspCjRfCA)
+ [Asteroids-v0](https://gym.openai.com/evaluations/eval_8eHKsRL4RzuZEq9AOLZA)
+ [Atlantis-v0](https://gym.openai.com/evaluations/eval_Z1B3d7A1QCaQk1HpO1Rg)
+ [BattleZone-v0](https://gym.openai.com/evaluations/eval_SoLit2bR1qmFoC0AsJF6Q)
+ [BankHeist-v0](https://gym.openai.com/evaluations/eval_hifoaxFTIuLlPd38BjnOw)
+ [BeamRider-v0](https://gym.openai.com/evaluations/eval_KuOYumrjQjixwL0spG0iCA)
+ [Breakout-v0](https://gym.openai.com/evaluations/eval_L55gczPrQJamMGihq9tzA)
+ [Carnival-v0](https://gym.openai.com/evaluations/eval_xJSOlo2lSWaH1wHEOX5vw)
+ [ChopperCommand-v0](https://gym.openai.com/evaluations/eval_tYVKyh7wQieRIKgEvVaCuw)
+ [CrazyClimber-v0](https://gym.openai.com/evaluations/eval_bKeBg0QwSgOm6A0I0wDhSw)
+ [DemonAttack-v0](https://gym.openai.com/evaluations/eval_tt21vVaRCKYzWFcg1Kw)
+ [DoubleDunk-v0](https://gym.openai.com/evaluations/eval_FI1GpF4TlCuf29KccTpQ)
+ [ElevatorAction-v0](https://gym.openai.com/evaluations/eval_SqeAouMvR0icRivx2xprZg)
+ [FishingDerby-v0](https://gym.openai.com/evaluations/eval_pPLCnFXsTVaayrIboDOs0g)
+ [Gravitar-v0](https://gym.openai.com/evaluations/eval_QudrLdVmTpK9HF5juaZr0w)
+ [IceHockey-v0](https://gym.openai.com/evaluations/eval_8oWCTwwGS7OUTTGRwBPQkQ)
+ [JourneyEscape-v0](https://gym.openai.com/evaluations/eval_S9nQuXLRSu7S5x21Ay6AA)
+ [Krull-v0](https://gym.openai.com/evaluations/eval_dfOS2WzhTh6sn1FuPS9HA)
+ [KungFuMaster-v0](https://gym.openai.com/evaluations/eval_vNWDShYTRC0MhfIybeUYg)
+ [MsPacman-v0](https://gym.openai.com/evaluations/eval_kpL9bSsS4GXsYb9HuEfew)
+ [Pooyan-v0](https://gym.openai.com/evaluations/eval_UXFVI34MSAuNTtjZcK8N0A)
+ [Pong-v0](https://gym.openai.com/evaluations/eval_8L7SV59nSW6GGbbP3N4G6w)
+ [Phoenix-v0](https://gym.openai.com/evaluations/eval_uzUruiB3RRKUMvJIxvEzYA)
+ [Qbert-v0](https://gym.openai.com/evaluations/eval_wekCJkrWQm9NrOUzltXg)
+ [Riverraid-v0](https://gym.openai.com/evaluations/eval_OU4x3DkTfm4uaXy6CIaXg)
+ [RoadRunner-v0](https://gym.openai.com/evaluations/eval_wINKQTwxT9ipydHOXBhg)
+ [Robotank-v0](https://gym.openai.com/evaluations/eval_Gr5c0ld3QACLDPQrGdzbiw)
+ [Seaquest-v0](https://gym.openai.com/evaluations/eval_N2624y3NSJWrOgoMSpOi4w)
+ [Tennis-v0](https://gym.openai.com/evaluations/eval_gDjJD0MMS1yLm1T0hdqI4g)
+ [UpNDown-v0](https://gym.openai.com/evaluations/eval_KmkvMJkxQFSED20wFUMdIA)
+ [VideoPinball-v0](https://gym.openai.com/evaluations/eval_PWwzNhVFR2CxjYvEsPfT1g)
+ [WizardOfWor-v0](https://gym.openai.com/evaluations/eval_1oGQhphpQhmzEMIYRrrp0A)
+ [Zaxxon-v0](https://gym.openai.com/evaluations/eval_TIQ102EwTrHrOyve2RGfg)

Note that atari game settings in gym are quite different from DeepMind papers, so the scores are not comparable. The most notable differences are:
+ In gym, each action is randomly repeated 2~4 times.
+ In gym, inputs are RGB instead of greyscale.
+ In gym, an episode is limited to 10000 steps.
