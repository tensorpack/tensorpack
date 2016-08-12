
# To run pretrained model:

1. install [tensorpack](https://github.com/ppwwyyxx/tensorpack)
2. Download models from [model zoo](https://drive.google.com/open?id=0B9IPQTvr2BBkS0VhX0xmS1c5aFk)
3. `ENV=NAME_OF_ENV ./run-atari.py --load "$ENV".tfmodel --env "$ENV"`

Models are available for the following gym environments:

+ [Breakout-v0](https://gym.openai.com/envs/Breakout-v0)
+ [AirRaid-v0](https://gym.openai.com/envs/AirRaid-v0)

Note that atari game settings in gym is more difficult than the settings DeepMind papers, therefore the scores are not comparable.
