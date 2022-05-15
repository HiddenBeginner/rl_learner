from collections import namedtuple

Transition = namedtuple('Transition', ('s', 'a', 'r', 's_prime', 'done'))
