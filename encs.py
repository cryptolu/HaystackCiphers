import random

from operator import xor
from functools import reduce
from operator import and_

from common import Randomized, SeededRand

# p -Enc-> x -HS-> c
# n        s       m  bits

# ISW: l shares
# Copy: d copies
# DS: d slots


class Encoding:
    """Base class for encodings."""

    n: int = NotImplemented
    "Number of input bits (message)."

    s: int = NotImplemented
    "Number of output bits (encoding)."

    def encode(self, p: list) -> list:
        raise NotImplementedError()

    def decode(self, x: list) -> list:  # or None = ⊥
        raise NotImplementedError()

    def parallel(self, num):
        """Concatenate `num` copies of the encoding.

        >>> CopyStrict(d=3).parallel(2).encode([0, 1])
        [0, 0, 0, 1, 1, 1]
        >>> CopyStrict(d=3).parallel_collate(2).encode([0, 1])
        [0, 1, 0, 1, 0, 1]
        """
        return Parallel(num=num, encoding=self)

    def parallel_collate(self, num):
        """Concatenate and collate bit-by-bit outputs of `num` copies of the encoding.

        >>> CopyStrict(d=3).parallel(2).encode([0, 1])
        [0, 0, 0, 1, 1, 1]
        >>> CopyStrict(d=3).parallel_collate(2).encode([0, 1])
        [0, 1, 0, 1, 0, 1]
        """
        return Parallel(num=num, encoding=self, collate=True)

    def fork(self, num):
        """Concatenate `num` copies of the encoding applied to the same input.

        >>> ISW(l=5, rand=SeededRand()).fork(2).encode([0])  # doctest: +NORMALIZE_WHITESPACE
        [0, 1, 0, 0, 1,  1, 1, 0, 0, 0]
        """
        return ForkStrict(num=num, encoding=self)

    def fork_collate(self, num):
        """Concatenate and collate bit-by-bit outputs of `num` copies of the encoding applied to the same input.

        >>> ISW(l=5, rand=SeededRand()).fork_collate(2).encode([0])  # doctest: +NORMALIZE_WHITESPACE
        [0, 1,  1, 1,  0, 0,  0, 0,  1, 0]
        """
        return ForkStrict(num=num, encoding=self, collate=True)

    def __mul__(self, other):
        return Composite(other, self)

    def __rmul__(self, other):
        return Composite(self, other)


class Parallel(Encoding):
    """Parallel application of the same encoding to SEPARATE inputs."""

    def __init__(self, encoding: Encoding, num: int, collate=False):
        self.encoding = encoding
        self.num = int(num)
        self.n = self.encoding.n * self.num
        self.s = self.encoding.s * self.num
        self.collate = bool(collate)
        assert self.n >= 1

    @staticmethod
    def _transpose(lists):
        # https://stackoverflow.com/a/6473724/1868332
        return list(map(list, zip(*lists)))

    def encode(self, p):
        assert len(p) == self.n
        xs = [
            self.encoding.encode(p[i:i+self.encoding.n])
            for i in range(0, len(p), self.encoding.n)
        ]
        if self.collate:
            xs = Parallel._transpose(xs)
        return sum(xs, [])

    def decode(self, x):
        assert len(x) == self.s
        s = self.encoding.s

        # split into chunks
        if self.collate:
            xs = [x[i:i+self.num] for i in range(0, self.s, self.num)]
            xs = Parallel._transpose(xs)
        else:
            xs = [x[i:i+s] for i in range(0, self.s, s)]

        ps = [self.encoding.decode(x) for x in xs]
        if None in ps:
            return None  # ⊥
        return sum(ps, [])

    def __str__(self):
        return f"Par{self.num}-{str(self.encoding)}"


class ForkStrict(Encoding):
    """Parallel application of the same encoding to the SAME input."""

    def __init__(self, encoding: Encoding, num: int, collate=False):
        self.encoding = encoding
        self.num = int(num)
        self.n = self.encoding.n
        self.s = self.encoding.s * self.num
        self.collate = bool(collate)

    def encode(self, p):
        assert len(p) == self.n
        xs = [
            self.encoding.encode(p)
            for i in range(self.num)
        ]
        if self.collate:
            xs = Parallel._transpose(xs)
        return sum(xs, [])

    def decode(self, x):
        assert len(x) == self.s
        s = self.encoding.s
        ps = [self.encoding.decode(x[i:i+s]) for i in range(0, self.s, s)]

        # split into chunks
        if self.collate:
            xs = [x[i:i+self.num] for i in range(0, self.s, self.num)]
            xs = Parallel._transpose(xs)
        else:
            xs = [x[i:i+s] for i in range(0, self.s, s)]

        ps = [self.encoding.decode(x) for x in xs]
        if len(set(map(tuple, ps))) != 1:
            return None  # ⊥
        return ps[0]

    def __str__(self):
        return f"Fork{self.num}-{str(self.encoding)}"



class Composite(Encoding):
    """Composition (sequence) of encodings. Can be constructed using `*`."""

    def __init__(self, *encodings):
        self.sequence = tuple(Composite._flatten(encodings))
        self.n = self.sequence[0].n
        self.s = self.sequence[-1].s
        for i in range(len(self.sequence)-1):
            assert self.sequence[i].s == self.sequence[i+1].n

    @staticmethod
    def _flatten(encodings):
        res = []
        for enc in encodings:
            if not isinstance(enc, Composite):
                res.append(enc)
            else:
                res.extend(Composite._flatten(enc.sequence))
        return res

    def encode(self, p):
        t = p
        for enc in self.sequence:
            t = enc.encode(t)
        x = t
        return x

    def decode(self, x):
        t = x
        for enc in reversed(self.sequence):
            t = enc.decode(t)
            if t is None:
                return None  # ⊥
        p = t
        return p


class ISW(Encoding, Randomized):
    """Simple sharing of 1 bit into `l` bits."""

    def __init__(self, l, *, rand=None):
        Randomized.__init__(self, rand=rand)
        self.l = int(l)
        assert l >= 1
        self.n = 1
        self.s = self.l

    def encode(self, p: list) -> list:
        assert len(p) == self.n == 1
        p, = p
        x = self.rand_bits(self.l - 1)
        x.append(reduce(xor, x, p))
        return x

    def decode(self, x: list) -> list:
        assert len(x) == self.s
        return [reduce(xor, x)]

    def __str__(self):
        return f"ISW{self.l}"

    @property
    def n_shares(self):
        return self.l

class SEL(Encoding, Randomized):
    """SEL masking scheme from CHES 2021"""
    """In the Haystack representation, SEl(1,2) is equivalent to BU18"""

    def __init__(self, l, d, *, rand=None):
        Randomized.__init__(self, rand=rand)
        self.l = int(l)
        self.d = int(d)
        assert l >= 1
        assert d >= 1
        self.n = 1
        self.s = self.l + self.d

    def encode(self, p: list) -> list:
        assert len(p) == self.n == 1
        p, = p
        x = self.rand_bits(self.s - 1)
        x.insert(0, reduce(xor, x[:self.l - 1], p) ^ reduce(and_, x[self.l - 1:]))
        return(x)

    def decode(self, x:list) -> list:
        assert len(x) == self.s
        return [reduce(xor, x[:self.l]) ^ reduce(and_, x[self.l:])]

    def __str__(self):
        return f"SEL{self.l}_{self.d}"

    @property
    def n_shares(self):
        return self.l + self.d


class DumShuf(Encoding, Randomized):
    """Dummy Shuffling countermeasure from Asiacrypt 2021"""
    # n : number of bits encrypted
    # d : number of slots

    def __init__(self, n, d, *, rand=None):
        Randomized.__init__(self, rand=rand)
        self.d = int(d)
        self.n = int(n)
        assert d >= 1
        assert n >= 1
        self.s = self.n * self.d

    def encode(self, p: list, rnd = 128) -> list:
        assert len(p) == self.n
        X = [[p[i]] + self.rand_bits(self.d - 1) for i in range(self.n)]
        rand = random.Random()
        seed = self.rand.getrandbits(rnd)
        for x in X:
            rand.seed(seed)
            rand.shuffle(x)
        out = []
        for x in X:
            out += x
        return(out+[seed >> i & 1 for i in range(rnd - 1,-1,-1)])

    def decode(self, x: list, rnd = 128) -> list:
        assert len(x) > rnd
        seedBits = x[-rnd:]
        seed = 0
        for bit in seedBits:
            seed = (seed << 1) | bit
        rand = random.Random()
        rand.seed(seed)
        DS_shuffle = list(range(self.d))
        rand.shuffle(DS_shuffle)
        DS_shuffle_inv = [0] * self.d
        for i in range(self.d):
            DS_shuffle_inv[DS_shuffle[i]] = i
        X = [[x[j] for j in range(i * self.d, (i + 1) * self.d)] for i in range(self.n)]
        for i in range(self.n):
            X[i] = [X[i][j] for j in DS_shuffle_inv]
        out = []
        for e in X :
            out += [e[0]]
        return(out)

    def __str__(self):
        return f"DumShuf{self.n}_{self.d}"

    @property
    def n_shares(self):
        return self.n * self.d

class S5(Encoding, Randomized):
    """Semi-Shuffled Secret Sharing Scheme from CU2025 eprint"""
    # n : number of bits encrypted
    # l : number of linear shares
    # d : number of slots

    def __init__(self, n, l, d, *, rand=None):
        Randomized.__init__(self, rand=rand)
        self.d = int(d)
        self.l = int(l)
        self.n = int(n)
        assert d >= 1
        assert n >= 1
        assert l >= 2
        self.s = self.n * (self.d + self.l - 1)

    def encode(self, p: list, rnd = 128) -> list:
        assert len(p) == self.n
        rand = random.Random()
        seed = self.rand.getrandbits(rnd)
        X = []
        for i in range(self.n):
            tmp = self.rand_bits(self.l - 1)
            X += tmp
            print("ISW rand")
            print(X)
            print()
            TEMP = [reduce(xor, tmp, p[i])] + self.rand_bits(self.d - 1)
            print("before shuffle")
            print(X)
            print()
            rand.seed(seed)
            rand.shuffle(TEMP)
            X += TEMP
        print("Result")
        print(X)
        print()
        print()
        return(X+[seed >> i & 1 for i in range(rnd - 1,-1,-1)])

    def decode(self, x: list, rnd = 128) -> list:
        print()
        print()
        assert len(x) > rnd
        seedBits = x[-rnd:]
        seed = 0
        for bit in seedBits:
            seed = (seed << 1) | bit
        rand = random.Random()
        rand.seed(seed)
        DS_shuffle = list(range(self.d))
        rand.shuffle(DS_shuffle)
        DS_shuffle_inv = [0] * self.d
        for i in range(self.d):
            DS_shuffle_inv[DS_shuffle[i]] = i

        X = []
        for i in range(self.n):
            NonLinearPart = x[(self.s * i) + self.l - 1 : (self.s * i) + self.s]
            print("NonLinearPart")
            print(NonLinearPart)
            print()
            for i in range(self.n):
                NonLinearPart = [NonLinearPart[j] for j in DS_shuffle_inv]
            print("Unshuffled Nonlinear part")
            print(NonLinearPart)
            print()
            X += [reduce(xor, x[(i * self.s):(((i + 1) * self.s) - self.l - 1)]) ^ NonLinearPart[0]]
            print("Decoded")
            print(X)
            print()
        print(X)
        return(X)

class CopyStrict(Encoding):
    """Copy the input bit into `d` copies. When decoding, check that all copies agree."""

    def __init__(self, d):
        self.d = int(d)
        assert self.d >= 1
        self.n = 1
        self.s = self.d

    def encode(self, p: list) -> list:
        assert len(p) == self.n == 1
        p, = p
        return [p] * self.d

    def decode(self, x: list) -> list:
        assert len(x) == self.s
        if x != x[:1] * self.d:
            return
        return x[:1]

    @property
    def n_copies(self):
        return self.d


def test_basic():
    enc = ISW(5).parallel(3)
    assert len(enc.encode([0, 1, 0])) == 15
    assert [0, 1, 0] == enc.decode(enc.encode([0, 1, 0]))
    assert [1, 0, 1] == enc.decode(enc.encode([1, 0, 1]))

    enc = ISW(5).parallel(2) * ISW(2)
    assert enc.n == 1 and enc.s == 10

    enc = SEL(l=3, d=3).parallel(3)
    assert len(enc.encode([0,1,0])) == 18
    assert [0, 1, 0] == enc.decode(enc.encode([0, 1, 0]))
    assert [1, 0, 1] == enc.decode(enc.encode([1, 0, 1]))

    enc = SEL(l=3, d=3).parallel(2)
    assert enc.n == 2 and enc.s == 12

    enc = DumShuf(n=3, d=4)
    assert len(enc.encode([0,1,0])) == 12+128
    assert [0, 1, 0] == enc.decode(enc.encode([0, 1, 0]))
    assert [1, 0, 1] == enc.decode(enc.encode([1, 0, 1]))

    enc = S5(n=3, l=2, d=5)
    assert len(enc.encode([0,1,0])) == (3 * ((2 - 1) + 5))+128
    assert [0, 1, 0] == enc.decode(enc.encode([0, 1, 0]))
    assert [1, 0, 1] == enc.decode(enc.encode([1, 0, 1]))

    enc = CopyStrict(d=3)
    assert enc.decode(enc.encode([0])) == [0]
    assert enc.decode(enc.encode([1])) == [1]
    assert enc.decode([0, 0, 1]) is None

    enc = CopyStrict(d=3).parallel(2)
    assert enc.n == 2 and enc.s == 6

    # ISW 2 inputs x 5 shares-> 10 shares
    # then Copy 10 shares x11 -> 110 outputs
    enc = CopyStrict(d=11).parallel_collate(10) * ISW(5).parallel(2)
    # note: parallel collate means that we want to group copied bits together
    # e.g. normal parallel copy  (a,b,c) -> (a,a,a,b,b,b,c,c,c)
    #      collate parallel copy (a,b,c) -> (a,b,c,a,b,c,a,b,c)
    assert enc.n == 2 and enc.s == 110
    assert enc.decode(enc.encode([0, 0])) == [0, 0]
    assert enc.decode(enc.encode([0, 1])) == [0, 1]
    # note: we copy the output of the ISW, not the ISW (as in ForkStrict below)
    # so that random shares are the same in all copies
    x = enc.encode([0, 1])
    for i in range(0, 110, 10):
        assert x[i:i+10] == x[:10]

    # ISW 2 inputs x 5 shares-> 10 shares
    # applied FRESHLY three times
    enc = ForkStrict(ISW(5).parallel(2), num=3)
    assert enc.n == 2 and enc.s == 30
    assert enc.decode(enc.encode([0, 0])) == [0, 0]
    assert enc.decode(enc.encode([0, 1])) == [0, 1]
    # note: we copy full ISW encoding
    # so that random shares are fresh in all copies
    x = enc.encode([0, 1])
    assert any(x[i:i+10] != x[:10] for i in range(0, 110, 10))

    # Copy 2 inputs x3 -> 6 outputs
    # then ISW 6 inputs x 5 shares-> 30 shares
    enc = ISW(5).parallel(6) * CopyStrict(d=3).parallel(2)
    assert enc.n == 2 and enc.s == 30
    assert enc.decode(enc.encode([0, 0])) == [0, 0]
    assert enc.decode(enc.encode([0, 1])) == [0, 1]


if __name__ == '__main__':
    test_basic()
