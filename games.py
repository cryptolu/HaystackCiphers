try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import math
from common import Randomized


class Game(Randomized):
    def __init__(self, cipher, *, rand=None):
        super().__init__(rand=rand)
        self._cipher = cipher
        self._win = None

    def info(self) -> dict:
        return self._cipher.info()

    def __str__(self):
        return type(self).__name__


class EncGame(Game):
    def encrypt(self, p: list[int]) -> list[int]:
        return self._cipher.encrypt(p)

class DecGame(Game):
    def decrypt(self, c: list[int]) -> list[int]:
        return self._cipher.decrypt(c)


class CPA(EncGame):
    def __init__(self, cipher, *, rand=None):
        super().__init__(cipher, rand=rand)
        self._b = -1
        self._challenge = None

    def challenge(self, p1: list[int], p2: list[int]) -> list[int]:
        assert self._b == -1
        p1 = list(map(int, p1))
        p2 = list(map(int, p2))
        assert p1 != p2
        assert len(p1) == len(p2) == self._cipher.n
        self._b = self.rand_bit()
        self._challenge = self.encrypt([p1, p2][self._b])
        return self._challenge

    def answer(self, b: int) -> bool:
        assert self._win is None
        self._win = b == self._b
        self._b = None
        return self._win

    def is_win(self) -> bool:
        return self._win


class CCA1(CPA, DecGame):
    def decrypt(self, c: list[int]) -> list[int]:
        if self._challenge:
            raise ValueError("Not allowed to decrypt after challenge given!")
        return super().decrypt(c)


class CCA2(CPA, DecGame):
    def decrypt(self, c: list[int]) -> list[int]:
        if self._challenge and tuple(map(int, c)) == tuple(map(int, self._challenge)):
            raise ValueError("Not allowed to decrypt challenge!")
        return super().decrypt(c)


class CCA3(CPA, DecGame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._ciphertexts = set()

    def encrypt(self, p: list[int]) -> list[int]:
        c = super().encrypt(p)
        self._ciphertexts.add(tuple(c))
        return c

    def decrypt(self, c: list[int]) -> list[int]:
        if self._challenge and tuple(map(int, c)) == tuple(map(int, self._challenge)):
            raise ValueError("Not allowed to decrypt challenge!")

        p = super().decrypt(c)
        if p:
            # valid decryption of new ciphertext
            # win by existential forgery
            if tuple(map(int, c)) not in self._ciphertexts:
                self._win = True
        return p


class GameRunner(Randomized):
    def __init__(self, game_gen, cipher_gen, *, rand=None):
        super().__init__(rand=rand)
        self.game_gen = game_gen
        self.cipher_gen = cipher_gen
        self.runs = 0
        self.wins = 0

        self.game_name = None
        self.cipher_name = None
        self.attack_name = None

    def run_once(self, attack_func):
        cipher = self.cipher_gen()
        game = self.game_gen(cipher)
        self.attack_name = attack_func.__name__
        self.cipher_name = str(cipher)
        self.game_name = str(game)

        self.runs += 1
        attack_func(game)
        self.wins += game.is_win()

    def run(self, attack_func, num=50, progress=True):
        rng = range(num)
        if progress and tqdm:
            rng = tqdm(rng)
        for _ in rng:
            self.run_once(attack_func)

    def print_stat(self):
        print(f"Game {self.game_name}: {self.attack_name}: against {self.cipher_name}", end="")
        win_rate = self.wins / self.runs
        print(f"wins {self.wins}/{self.runs} = {win_rate*100:.1f}%% = 2^{math.log(win_rate, 2):.2f}")


def test_games():
    import pytest
    from encs import CopyStrict, ISW
    from haystack import HaystackCipher

    copy_cipher = HaystackCipher(CopyStrict(d=4), r=0, static_shuffle=False)
    isw_cipher = HaystackCipher(ISW(l=5), r=11)

    # CPA
    game = CPA(copy_cipher)
    ct = game.challenge([0], [1])
    # first output bit tells which message was encrypted
    game.answer(ct[0])
    assert game.is_win()

    game = CPA(copy_cipher)
    ct = game.challenge([0], [1])
    game.answer(ct[0] ^ 1)
    assert not game.is_win()

    # CCA3
    game = CCA3(copy_cipher)
    # new valid ciphertext decrypted
    game.decrypt([0] * 4)
    assert game.is_win()

    game = CCA3(copy_cipher)
    game.encrypt([0])
    # not new ciphertext anymore!
    game.decrypt([0] * 4)
    assert not game.is_win()

    # CCA2
    game = CCA2(copy_cipher)
    game.decrypt([0] * 4)
    # existential forgery not a win
    assert not game.is_win()

    game = CCA2(isw_cipher)
    # malleable forgery -> have to use it yourself to win
    ct_chall = game.challenge([0], [1])
    try:
        game.decrypt(ct_chall)
        assert False, "decrypted challenge???"
    except ValueError:
        pass
    # flip all bits
    ct_forged = [1 ^ v for v in ct_chall]
    pt_forged = game.decrypt(ct_forged)
    # flipped plaintext
    game.answer(pt_forged[0]^1)
    assert game.is_win()


if __name__ == '__main__':
    test_games()
