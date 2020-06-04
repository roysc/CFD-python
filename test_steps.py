from importlib import import_module

tests = [
    getattr(import_module(f'_{n}'), 'test', None)
    for n in range(1, 8)
]

def test_all():
    for test in tests:
        if test: test()
