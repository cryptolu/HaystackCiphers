import random

class Randomized:
    def __init__(self, *, rand=None):
        if rand is None:
            rand = random.Random()
        self.rand = rand

    def rand_bit(self):
        return self.rand.randint(0, 1)

    def rand_bits(self, t):
        return [self.rand_bit() for _ in range(t)]


def SeededRand(seed=2025):
    R = random.Random()
    R.seed(seed)
    return R


def test_randomized():
    s = [Randomized().rand_bit() for _ in range(100)]
    assert set(s) == {0, 1}

    s = Randomized().rand_bits(100)
    assert set(s) == {0, 1}

    R = random.Random()
    R.seed(2025)
    out1 = Randomized(rand=R).rand_bits(100)
    R.seed(2025)
    out2 = Randomized(rand=R).rand_bits(100)
    assert out1 == out2

    out3 = Randomized(rand=R).rand_bits(100)
    assert out2 != out3


if __name__ == '__main__':
    test_randomized

