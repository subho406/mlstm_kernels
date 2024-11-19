def is_power_of_2(n):
    assert isinstance(n, int)
    return (n & (n - 1)) == 0
