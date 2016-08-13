
# To run pretrained atari model for 100 episodes:

1. install [tensorpack](https://github.com/ppwwyyxx/tensorpack)
2. Download models from [model zoo](https://drive.google.com/open?id=0B9IPQTvr2BBkS0VhX0xmS1c5aFk)
3. `ENV=NAME_OF_ENV ./run-atari.py --load "$ENV".tfmodel --env "$ENV"`

Models are available for the following gym atari environments (click links for videos):

+ [Breakout-v0](https://gym.openai.com/evaluations/eval_L55gczPrQJamMGihq9tzA)
+ [AirRaid-v0](https://gym.openai.com/evaluations/eval_zIeNk5MxSGOmvGEUxrZDUw)
+ [Asterix-v0](https://gym.openai.com/evaluations/eval_mees2c58QfKm5GspCjRfCA)
+ [Amidar-v0](https://gym.openai.com/evaluations/eval_HwEazbHtTYGpCialv9uPhA)
+ [Seaquest-v0](https://gym.openai.com/evaluations/eval_N2624y3NSJWrOgoMSpOi4w)

Note that atari game settings in gym is more difficult than the settings DeepMind papers, therefore the scores are not comparable.
