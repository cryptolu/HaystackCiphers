try:
    import pytest
    pytest.importorskip("sage")
except ImportError:
    pass

from sage.all import randrange, matrix, vector, GF

from encs import ISW, SEL
from haystack import HaystackCipher
from games import CPA, CCA1, CCA2, CCA3, GameRunner


def attack_LDA(game):
    # n,s,m,encoding
    # game.info()
    # {'n': 2, 'm': 11, 's': 6, 'r': 5, 'encoding': <ciphers.ParallelEncoding object at 0x7112062f80b0>}

    # collect 50 ciphertexts
    cts = [game.encrypt([1, randrange(2)]) for _ in range(50)]
    # position of shares of the first bit
    sol = matrix(GF(2), cts).solve_right(vector(GF(2), [1]*len(cts)))
    ct = game.challenge([0, 0], [1, 0])
    game.answer(sol * vector(ct))


def attack_zero_forge(game):
    m = game.info()["m"]
    game.decrypt([0] * m)


def attack_FLDA(game):
    Pt = [randrange(2) for _ in range(100)]
    cts = [game.encrypt([Pt[i]]) for i in range(100)]
    print(matrix(cts))
    print(Pt)
    M = matrix(GF(2), cts)
    for i in range(M.ncols()):
        filtLinIdx=[]
        for j in range(M.nrows()):
            if M[j,i] == 1:
                filtLinIdx.append(j)
        MFilt = M[filtLinIdx,[_ for _ in range(i)] + [_ for _ in range(i+1, M.ncols())]]
        PtFilt = [Pt[pos] for pos in filtLinIdx]

        #print(MFilt)
        #print(PtFilt)

        if MFilt.nrows() > (MFilt.ncols() + 20):
            try :
                sol = MFilt.solve_right(vector(GF(2), PtFilt))
                print(sol)
            except ValueError :
                continue




def test_attacks():
    # Note: there some basic attack examples in games.py file

    # one run example
    enc = ISW(3).parallel(2)
    game = CPA(HaystackCipher(enc, r=0))
    print(attack_LDA(game), game.is_win())
    print()

    # statistic runner
    GR = GameRunner(CPA, lambda: HaystackCipher(enc, r=5))
    GR.run(attack_LDA)
    GR.print_stat()
    print()
    assert GR.wins == GR.runs

    # statistic runner
    GR = GameRunner(CCA3, lambda: HaystackCipher(enc, r=5))
    GR.run(attack_zero_forge)
    GR.print_stat()
    print()
    assert GR.wins == GR.runs


    enc = SEL(3,2)
    game = CPA(HaystackCipher(enc, r=0))
    attack_FLDA(game)



if __name__ == '__main__':
    test_attacks()
