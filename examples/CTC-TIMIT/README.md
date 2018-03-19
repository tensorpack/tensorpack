
## LSTM + CTC on TIMIT speech recognition dataset

### Install Dependencies:
+ python binding for `lmdb`
  + `pip install --user lmdb`
+ `bob.ap` package for MFCC extraction
  + install [blitz](https://github.com/blitzpp/blitz) and openblas as dependencies of bob.ap
  + `pip install --user bob.extension bob.blitz bob.core bob.sp bob.ap`

### Prepare Data:
We assume the following file structure:
```
TRAIN/
  DR1/
    FCJF0/
      *.WAV     # NIST WAV file
      *.TXT
      *.PHN
  ...
```

Convert NIST wav format to RIFF wav format:
```
cd /PATH/TO/TIMIT
find . -name '*.WAV' | parallel -P20 sox {} '{.}.wav'
```

Extract MFCC features and phoneme labels, and save everything to LMDB database. The preprocessing
follows the setup in
+ Connectionist Temporal Classification: Labelling Unsegmented Sequence Data with RNN - Alex Graves

```
./create-lmdb.py build --dataset /PATH/TO/TIMIT/TRAIN --db train.mdb
./create-lmdb.py build --dataset /PATH/TO/TIMIT/TEST --db test.mdb
```

Compute mean/std of the training set (and save to `stats.data` by default):
```
./create-lmdb.py stat --db train.mdb
```

### Train:
```
./train-timit.py --train train.mdb --test test.mdb --stat stats.data
```

### Results:
Get 0.28 LER (normalized edit distance) after about 40 epochs.
