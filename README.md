# HaystackCiphers

## Setup

Almost everything can be ran with pure python except `attacks.py` which uses [SageMath](https://www.sagemath.org) for linear algebra. Some optional packages help too:

```sh
$ python3 -m pip install -U tqdm
```

Every file has some basic tests, can run all of them (incl. Sage one) quickly using

```sh
$ sage -t . 

# or (the same)
$ make sagetests
```

Pure python tests can be ran much faster with `pytest`:

```sh
$ python3 -m pip install -U pytest
$ pytest *.py
```

Running every file also executes the tests but without good interface, e.g.

```sh
$ python3 encs.py
```

## Information

Messages/plaintexts/ciphertexts/etc. are all lists of bits represented by ints, e.g. `[0, 0, 1]`, `[0]`, etc. Again, even single bit needs to be in its own list. Normally, there are no nested lists and splitting into blocks needs to be done manually (relevant block sizes should be available as attributes).


## Modules

[encs.py](./encs.py) contains encodings: ISW, and CopyStrict which copies its input and on decoding checks that all copies are equal. Both have 1-bit input, but can be parallel-copied e.g. `ISW(l=5).parallel(3)` takes as input 3 bits and splits each of them into 5 shares, making a 15-bit output. You can also compose encodings using `*`, but you have to apply `parallel` by yourself to match input sizes, e.g.
```py
# Copy 2 inputs x3 -> 6 outputs
# then ISW 6 inputs x 5 shares-> 30 shares
enc = ISW(5).parallel(6) * CopyStrict(d=3).parallel(2)
```
See the test in the file for examples.

[haystack.py](./haystack.py) contains basic Haystack cipher definition, which takes and encoding and makes a cipher out of it (by padding with randomness and doing static shuffle, which can be turned off).
```py
enc = ISW(5).parallel(3)
H = HaystackCipher(encoding=enc, r=10)
assert len(H.encrypt([0, 1, 0])) == 25
assert [0, 1, 0] == H.decrypt(H.encrypt([0, 1, 0]))
```

[games.py](./games.py) contains CPA/CCA1/CCA2/CCA3 games and GameRunner which can run an attack multiple times to collect statistics. See the test for example games and attacks.

[attacks.py](./attacks.py) contains some attack examples (LDA on ISW and simple forgery for CopyStrict Haystack cipher).