Reproduce DQN in:

**Human-level Control Through Deep Reinforcement Learning**

and Double-DQN in:

**Deep Reinforcement Learning with Double Q-learning**

Can reproduce the claimed performance, on several games I've tested with.

![DQN](https://github.com/ppwwyyxx/tensorpack/raw/master/examples/Atari2600/DoubleDQN-breakout.png)

A demo trained with Double-DQN on breakout is available at [youtube](https://youtu.be/o21mddZtE5Y).

## How to use

Download [atari roms](https://github.com/openai/atari-py/tree/master/atari_py/atari_roms) to
`$TENSORPACK_DATASET/atari_rom` (defaults to tensorpack/dataflow/dataset/atari_rom).

To train:
```
./DQN.py --rom breakout.bin --gpu 0
```
Training speed is about 7.3 iteration/s on 1 Tesla M40. It takes days to learn well (see figure above).

To play:
```
./DQN.py --rom breakout.bin --task play --load pretrained.model
```

A3C code and curve will be available soon. It learns much faster.
