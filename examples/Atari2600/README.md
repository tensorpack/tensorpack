Reproduce the following methods:

+ Nature-DQN in:
[Human-level Control Through Deep Reinforcement Learning](http://www.nature.com/nature/journal/v518/n7540/full/nature14236.html)

+ Double-DQN in:
[Deep Reinforcement Learning with Double Q-learning](http://arxiv.org/abs/1509.06461)

+ A3C in [Asynchronous Methods for Deep Reinforcement Learning](http://arxiv.org/abs/1602.01783). (I
used a modified version where each batch contains transitions from different simulators, which I called "Batch A3C".)

Claimed performance in the paper can be reproduced, on several games I've tested with.

![DQN](curve-breakout.png)

A demo trained with Double-DQN on breakout game is available at [youtube](https://youtu.be/o21mddZtE5Y).

DQN would typically take 2~3 days of training to reach a score of 400 on breakout, but it only takes <4 hours on 1 GPU with my A3C implementation.
This is probably the fastest RL trainer you'd find.

## How to use

Download [atari roms](https://github.com/openai/atari-py/tree/master/atari_py/atari_roms) to
`$TENSORPACK_DATASET/atari_rom` (defaults to tensorpack/dataflow/dataset/atari_rom).

To train:
```
./DQN.py --rom breakout.bin --gpu 0
```
Training speed is about 7.3 iteration/s on 1 Tesla M40
(faster than this at the beginning, but will slow down due to exploration annealing).
It takes days to learn well (see figure above).

To visualize the agent:
```
./DQN.py --rom breakout.bin --task play --load pretrained.model
```

A3C code will be released at the end of August.
