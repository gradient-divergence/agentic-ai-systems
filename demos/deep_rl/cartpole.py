"""A frictionless cart and uniform hinged pole, integrated by explicit Euler.

Let x be cart position and theta the pole's angle from upright. M is the cart
mass, m the pole mass, l half its length, and F the signed push. A uniform pole
has moment of inertia m*l**2/3 about its centre. Eliminating the hinge force from
the horizontal and angular equations gives, with c = cos(theta), s = sin(theta):

    u = (F + m*l*theta_dot**2*s) / (M + m)
    theta_ddot = (g*s - c*u) / (l*(4/3 - m*c**2/(M + m)))
    x_ddot = u - m*l*theta_ddot*c / (M + m)

All four next coordinates use the old state, with a 0.02 second step. A push
pays 1, including the push that first crosses a failure limit. These are stated
simulator choices, not a claim to reproduce any particular published environment.
The 500-step time limit truncates a surviving episode; it does not terminate it.
"""

import math

import numpy as np

GRAVITY = 9.8
CART_MASS = 1.0
POLE_MASS = 0.1
HALF_LENGTH = 0.5
FORCE = 10.0
DT = 0.02
POSITION_LIMIT = 2.4
ANGLE_LIMIT = 12 * math.pi / 180
TIME_LIMIT = 500
STATE_SCALE = np.array([POSITION_LIMIT, 2.0, ANGLE_LIMIT, 2.0])


class CartPole:
    def __init__(self, limit=TIME_LIMIT):
        self.limit = limit
        self.state = np.zeros(4)
        self.steps = 0
        self.finished = True

    def reset(self, seed):
        self.state = np.random.default_rng(seed).uniform(-0.05, 0.05, 4)
        self.steps = 0
        self.finished = False
        return self.state.copy()

    def step(self, action):
        if self.finished:
            raise ValueError("reset before stepping a finished episode")
        if action not in (0, 1):
            raise ValueError("action must be 0 (left) or 1 (right)")
        x, speed, angle, angular_speed = self.state
        c, s = math.cos(angle), math.sin(angle)
        total_mass = CART_MASS + POLE_MASS
        u = FORCE * (2 * action - 1) + POLE_MASS * HALF_LENGTH * angular_speed**2 * s
        u /= total_mass
        angular_acceleration = (GRAVITY * s - c * u) / (
            HALF_LENGTH * (4 / 3 - POLE_MASS * c**2 / total_mass)
        )
        acceleration = u - POLE_MASS * HALF_LENGTH * angular_acceleration * c / total_mass
        self.state = np.array(
            [
                x + DT * speed,
                speed + DT * acceleration,
                angle + DT * angular_speed,
                angular_speed + DT * angular_acceleration,
            ]
        )
        self.steps += 1
        terminated = abs(self.state[0]) > POSITION_LIMIT or abs(self.state[2]) > ANGLE_LIMIT
        truncated = self.steps >= self.limit and not terminated
        self.finished = bool(terminated or truncated)
        return self.state.copy(), 1.0, bool(terminated), bool(truncated)
