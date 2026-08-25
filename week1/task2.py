# Monty Hall Game
import numpy as np
import random


def dont_change():
    guess = random.randint(0,2)
    # Create doors where True indicates car
    doors = 3 * [False]
    r = random.randint(0,2)
    doors[r] = True

    return doors[guess]

def change():
    # Random guess
    guess = random.randint(0, 2)
    # Create doors where True indicates car
    doors = 3 * [False]
    r = random.randint(0, 2)
    doors[r] = True
    doors = list(enumerate(doors))
    # Pop door that is not guessed on and is not a car
    for door, car in doors:
        if not car and door != guess:
            doors.pop(door)
            break
    # Switch guess
    guess = [(door, car) for door, car in doors if guess != door][0]
    return guess[1]


# Run simulation
n = 100000

dc = sum([dont_change() for _ in range(n)]) / n
print("Proportion correct when we dont change: ", dc)

c = sum([change() for _ in range(n)]) / n
print("Proportion correct when we change: ", c)





