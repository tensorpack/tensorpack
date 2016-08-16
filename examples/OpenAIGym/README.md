
# To run pretrained atari model for 100 episodes:

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
+ [Breakout-v0](https://gym.openai.com/evaluations/eval_L55gczPrQJamMGihq9tzA)
+ [Pong-v0](https://gym.openai.com/evaluations/eval_8L7SV59nSW6GGbbP3N4G6w)
+ [Seaquest-v0](https://gym.openai.com/evaluations/eval_N2624y3NSJWrOgoMSpOi4w)

Note that atari game settings in gym is quite different from DeepMind papers, so the scores are not comparable.
