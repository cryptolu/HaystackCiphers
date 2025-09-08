from operator import xor
from functools import reduce

from common import Randomized
from encs import Encoding

# p -Enc-> x -HS-> c
# n        s       m  bits

class HaystackCipher(Randomized):
    def __init__(self, encoding: Encoding, r: int = 0, *, static_shuffle=True, rand=None):
        Randomized.__init__(self, rand=rand)

        self.encoding = encoding
        self.r = int(r)
        assert self.r >= 0
        self.n = self.encoding.n
        self.m = self.encoding.s + r
        self.static_shuffle = list(range(self.m))
        if static_shuffle:
            self.rand.shuffle(self.static_shuffle)
        self.static_shuffle_inv = [0] * self.m
        for i, j in enumerate(self.static_shuffle):
            self.static_shuffle_inv[j] = i

    def encrypt(self, p):
        assert len(p) == self.n
        x = self.encoding.encode(p)
        c = x + self.rand_bits(self.r)
        c = [c[i] for i in self.static_shuffle]
        return c

    def decrypt(self, c):
        assert len(c) == self.m
        c = [c[i] for i in self.static_shuffle_inv]
        x = c[:self.encoding.s]
        p = self.encoding.decode(x)
        return p

    def info(self) -> dict:
        return dict(n=self.n, m=self.m, s=self.encoding.s, r=self.r, encoding=self.encoding)

    def __str__(self):
        return f"Haystack{self.r}-{str(self.encoding)}"


if __name__ == '__main__':
    from encs import ISW
    enc = ISW(5).parallel(3)
    assert len(enc.encode([0, 1, 0])) == 15
    assert [0, 1, 0] == enc.decode(enc.encode([0, 1, 0]))

    H = HaystackCipher(encoding=enc, r=10)
    assert len(H.encrypt([0, 1, 0])) == 25
    assert [0, 1, 0] == H.decrypt(H.encrypt([0, 1, 0]))
