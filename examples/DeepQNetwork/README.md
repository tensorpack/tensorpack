![breakout](breakout.jpg)

[video demo](https://youtu.be/o21mddZtE5Y)

Reproduce (performance of) the following reinforcement learning methods:

+ Nature-DQN in:
[Human-level Control Through Deep Reinforcement Learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)

+ Double-DQN in:
[Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)

+ Dueling-DQN in: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

+ A3C in [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783). (I
used a modified version where each batch contains transitions from different simulators, which I called "Batch-A3C".)

## Performance & Speed
Claimed performance in the paper can be reproduced, on several games I've tested with.

![DQN](curve-breakout.png)

On one GTX 1080Ti,
the ALE version took
__~2 hours__ of training to reach 21 (maximum) score on Pong,
__~10 hours__ of training to reach 400 score on Breakout.
It runs at 100 batches (6.4k trained frames, 400 seen frames, 1.6k game frames) per second on GTX 1080Ti.
This is likely the fastest open source TF implementation of DQN.

## How to use

### With ALE (paper's setting):
Install [ALE](https://github.com/mgbellemare/Arcade-Learning-Environment) and gym.

Download an [atari rom](https://github.com/openai/atari-py/tree/master/atari_py/atari_roms), e.g.:
```
wget https://github.com/openai/atari-py/raw/master/atari_py/atari_roms/breakout.bin
```

Start Training:
```
./DQN.py --env breakout.bin
# use `--algo` to select other DQN algorithms. See `-h` for more options.
```

Watch the agent play:
```
# Download pretrained models or use one you trained:
wget http://models.tensorpack.com/DeepQNetwork/DoubleDQN-Breakout.npz
./DQN.py --env breakout.bin --task play --load DoubleDQN-Breakout.npz
```

### With gym's Atari:

Install gym and atari_py.

```
./DQN.py --env BreakoutDeterministic-v4
```

A3C code and models for Atari games in OpenAI Gym are released in [examples/A3C-Gym](../A3C-Gym)
