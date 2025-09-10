import logging
import sys
import random
random.seed(2025)

from binteger import Bin
from circkit.boolean import OptBooleanCircuit as BooleanCircuit
#from circkit.boolean import BooleanCircuit
from wboxkit.ciphers.aes import BitAES
from wboxkit.ciphers.aes.aes import encrypt

logging.basicConfig(level=logging.DEBUG)

# 2 rounds - 115GB RAM, 4000s (on pypy3)
# 5 rounds - ?GB RAM, ?s (on pypy3)
# 10 rounds - ?GB RAM, ?s (on pypy3)

NR = int(sys.argv[1])
print("Rounds:", NR, "\n==========")

C = BooleanCircuit(name="AES")

key = b"abcdefghABCDEFGH"
plaintext = b"0123456789abcdef"

pt = C.add_inputs(128)

ct, k10 = BitAES(pt, Bin(key).tuple, rounds=NR)

C.add_output(ct)
C.in_place_remove_unused_nodes()

C.print_stats()

ct = C.evaluate(Bin(plaintext).tuple)
ct = Bin(ct).bytes
# print(ct.hex())

ct2 = encrypt(plaintext, key, 10)
# print(ct2.hex())
# print()

if NR == 10:
    assert ct == ct2


from wboxkit.prng import NFSR, Pool

nfsr = NFSR(
    taps=[[2, 77], [0], [7], [29], [50], [100]],
    clocks_initial=128,
    clocks_per_step=3,
)
prng = Pool(prng=nfsr, n=192)


from wboxkit.masking import ISW, MINQ, DumShuf

C0 = C

params = [
    (2, 11),
    (4, 16),
    (5, 18),  # 6 swaps
    (8, 31),  # 7 swaps
    (7, 23), # 7 swaps
]

for d, ell in params:
    print("d", d, "ell", ell)
    C = DumShuf(prng=prng, n_shares=d, max_bias=1/16.0).transform(C0)
    C.in_place_remove_unused_nodes()
    #C.print_stats()

    C = ISW(prng=prng, order=ell).transform(C)
    C.in_place_remove_unused_nodes()
    C.print_stats()
    print("factor", len(C)/len(C0))

    ct = C.evaluate(Bin(plaintext).tuple)
    ct = Bin(ct).bytes
    print(ct.hex())
    print()
    print()
    sys.stdout.flush()
