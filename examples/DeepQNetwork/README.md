![breakout](breakout.jpg)

[video demo](https://youtu.be/o21mddZtE5Y)

Reproduce the following reinforcement learning methods:

+ Nature-DQN in:
[Human-level Control Through Deep Reinforcement Learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)

+ Double-DQN in:
[Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)

+ Dueling-DQN in: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

+ A3C in [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783). (I
used a modified version where each batch contains transitions from different simulators, which I called "Batch-A3C".)

Claimed performance in the paper can be reproduced, on several games I've tested with.

![DQN](curve-breakout.png)

DQN typically took 1 day of training to reach a score of 400 on breakout game (same as the paper).
My Batch-A3C implementation only took <2 hours.
Both were trained on one GPU with an extra GPU for simulation.

Double-DQN runs at 18 batches/s (1152 frames/s) on TitanX.
Note that I wasn't using the network architecture in the paper.
If switched to the network in the paper it could run 2x faster.

## How to use

Download an [atari rom](https://github.com/openai/atari-py/tree/master/atari_py/atari_roms) to
`$TENSORPACK_DATASET/atari_rom/` (defaults to ~/tensorpack_data/atari_rom/), e.g.:
```
mkdir -p ~/tensorpack_data/atari_rom
wget https://github.com/openai/atari-py/raw/master/atari_py/atari_roms/breakout.bin -O ~/tensorpack_data/atari_rom/breakout.bin
```

Start Training:
```
./DQN.py --rom breakout.bin
# use `--algo` to select other DQN algorithms. See `-h` for more options.
```

Watch the agent play:
```
./DQN.py --rom breakout.bin --task play --load trained.model
```
A pretrained model on breakout can be downloaded [here](https://drive.google.com/open?id=0B9IPQTvr2BBkN1Jrei1xWW0yR28).

A3C code and models for Atari games in OpenAI Gym are released in [examples/A3C-Gym](../A3C-Gym)
