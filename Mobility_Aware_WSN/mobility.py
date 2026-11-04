"""
mobility.py
-----------
Moves the virtual sensor nodes around the simulation area.

We use the RANDOM DIRECTION mobility model:

    1. Every node picks a random direction (angle) and a random speed.
    2. It moves in that direction for a random time (5 to 20 seconds).
    3. When it hits the border of the area it BOUNCES back like a ball.
    4. When its time is over it picks a new direction, speed and time.

Why not the more famous "Random Waypoint" model?
    In Random Waypoint, nodes walk to random destination points, so they
    keep crossing the CENTRE of the area. Over time nodes bunch up in the
    middle and the network becomes denser than at the start. That would
    give moving networks an unfair advantage over the stationary network.
    Random Direction keeps the nodes evenly spread over the whole area, so
    the ONLY difference between stationary and mobile networks is movement.

Some nodes can be "fixed" (they never move). We keep the SINK node
(the base station that collects all data) fixed in the centre.
"""

import numpy as np

MIN_TRAVEL_TIME = 5.0   # seconds a node keeps the same direction (minimum)
MAX_TRAVEL_TIME = 20.0  # seconds a node keeps the same direction (maximum)


class RandomDirectionMobility:
    def __init__(self, num_nodes, area_size, max_speed, seed=None, fixed_nodes=None):
        """
        num_nodes   : how many sensor nodes
        area_size   : the area is a square of area_size x area_size metres
        max_speed   : maximum node speed in metres/second (0 = stationary network)
        seed        : random seed so experiments can be repeated exactly
        fixed_nodes : list of node ids that never move (e.g. the sink)
        """
        self.rng = np.random.default_rng(seed)
        self.num_nodes = num_nodes
        self.area_size = area_size
        self.max_speed = max_speed
        self.fixed_nodes = set(fixed_nodes or [])

        # (x, y) position of every node -> array of shape (num_nodes, 2)
        self.positions = self.rng.uniform(0, area_size, size=(num_nodes, 2))

        # Direction of travel (angle in radians) of each node
        self.angles = self.rng.uniform(0, 2 * np.pi, size=num_nodes)

        # Each node gets its own speed
        self.speeds = self._random_speeds(num_nodes)

        # How many seconds until each node changes direction
        self.time_left = self.rng.uniform(MIN_TRAVEL_TIME, MAX_TRAVEL_TIME, size=num_nodes)

        # Velocity vector (vx, vy) measured during the last update
        self.velocities = np.zeros((num_nodes, 2))

    def _random_speeds(self, count):
        """Speed between 50% and 100% of max_speed."""
        return self.rng.uniform(0.5, 1.0, size=count) * self.max_speed

    def set_max_speed(self, new_max_speed):
        """Change the speed of all nodes while the simulation is running."""
        if new_max_speed == self.max_speed:
            return
        self.max_speed = new_max_speed
        self.speeds = self._random_speeds(self.num_nodes)

    def set_position(self, node_id, x, y):
        """Place a node at an exact position (used for the sink)."""
        self.positions[node_id] = [x, y]

    def update(self, dt):
        """Move every node for 'dt' seconds."""
        old_positions = self.positions.copy()

        for i in range(self.num_nodes):
            # Fixed nodes or a stationary network -> no movement
            if i in self.fixed_nodes or self.max_speed <= 0:
                continue

            # Time to choose a new direction and speed?
            self.time_left[i] -= dt
            if self.time_left[i] <= 0:
                self.angles[i] = self.rng.uniform(0, 2 * np.pi)
                self.speeds[i] = self.rng.uniform(0.5, 1.0) * self.max_speed
                self.time_left[i] = self.rng.uniform(MIN_TRAVEL_TIME, MAX_TRAVEL_TIME)

            # Move one step in the current direction
            dx = self.speeds[i] * np.cos(self.angles[i]) * dt
            dy = self.speeds[i] * np.sin(self.angles[i]) * dt
            x = self.positions[i][0] + dx
            y = self.positions[i][1] + dy

            # Bounce off the left / right walls
            if x < 0:
                x = -x
                self.angles[i] = np.pi - self.angles[i]
            elif x > self.area_size:
                x = 2 * self.area_size - x
                self.angles[i] = np.pi - self.angles[i]

            # Bounce off the bottom / top walls
            if y < 0:
                y = -y
                self.angles[i] = -self.angles[i]
            elif y > self.area_size:
                y = 2 * self.area_size - y
                self.angles[i] = -self.angles[i]

            self.positions[i] = [x, y]

        # Safety check: keep everybody inside the area
        self.positions = np.clip(self.positions, 0, self.area_size)

        # velocity = displacement / time
        self.velocities = (self.positions - old_positions) / dt
        return self.positions
