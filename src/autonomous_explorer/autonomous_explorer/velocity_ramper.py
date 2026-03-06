#!/usr/bin/env python3
"""
Velocity Ramper — smooth trapezoidal velocity profiles for motor commands.

Instead of instant step changes (0 → full speed → 0), the ramper interpolates
velocity using configurable acceleration limits. This prevents wheel slip on
tracked chassis and produces smoother, more accurate motion.

Usage:
    ramper = VelocityRamper(publisher, max_linear=0.35, max_angular=1.0)
    ramper.start()

    ramper.set_target(0.2, 0.0)   # smoothly ramp to 0.2 m/s forward
    ramper.set_target(0.0, 0.0)   # smoothly decelerate to stop
    ramper.emergency_stop()        # instant hard stop (no ramp)

    ramper.shutdown()
"""
import threading
import time

from geometry_msgs.msg import Twist


class VelocityRamper:
    """Background-thread velocity smoother with trapezoidal profiles."""

    def __init__(
        self,
        publisher,
        max_linear: float = 0.35,
        max_angular: float = 1.0,
        linear_accel: float = 0.5,      # m/s² — ramp rate for linear velocity
        linear_decel: float = 0.8,      # m/s² — braking rate (faster than accel)
        angular_accel: float = 2.0,     # rad/s² — ramp rate for angular velocity
        angular_decel: float = 3.0,     # rad/s² — angular braking rate
        update_rate: float = 20.0,      # Hz — control loop frequency
        logger=None,
    ):
        self._pub = publisher
        self._max_linear = max_linear
        self._max_angular = max_angular
        self._linear_accel = linear_accel
        self._linear_decel = linear_decel
        self._angular_accel = angular_accel
        self._angular_decel = angular_decel
        self._dt = 1.0 / update_rate
        self._logger = logger

        # State (protected by lock)
        self._lock = threading.Lock()
        self._target_linear = 0.0
        self._target_angular = 0.0
        self._current_linear = 0.0
        self._current_angular = 0.0
        self._e_stopped = False

        # Thread control
        self._running = False
        self._thread = None

        # Callback for last-published values (for logging/monitoring)
        self._last_published = (0.0, 0.0)

    @property
    def current_velocity(self) -> tuple:
        """Return (linear_x, angular_z) currently being sent to motors."""
        with self._lock:
            return (self._current_linear, self._current_angular)

    @property
    def target_velocity(self) -> tuple:
        """Return (linear_x, angular_z) we're ramping toward."""
        with self._lock:
            return (self._target_linear, self._target_angular)

    @property
    def is_moving(self) -> bool:
        with self._lock:
            return abs(self._current_linear) > 0.001 or abs(self._current_angular) > 0.001

    def start(self):
        """Start the background ramping thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name='velocity_ramper')
        self._thread.start()

    def shutdown(self):
        """Stop the ramper thread and zero motors."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        self._publish_zero()

    def set_target(self, linear_x: float, angular_z: float):
        """Set target velocity — ramper will smoothly approach it."""
        linear_x = max(-self._max_linear, min(self._max_linear, linear_x))
        angular_z = max(-self._max_angular, min(self._max_angular, angular_z))
        with self._lock:
            self._target_linear = linear_x
            self._target_angular = angular_z
            self._e_stopped = False

    def stop(self):
        """Smoothly decelerate to zero."""
        with self._lock:
            self._target_linear = 0.0
            self._target_angular = 0.0

    def emergency_stop(self):
        """Immediately zero all velocities (no ramp)."""
        with self._lock:
            self._target_linear = 0.0
            self._target_angular = 0.0
            self._current_linear = 0.0
            self._current_angular = 0.0
            self._e_stopped = True
        self._publish_zero()

    def block_forward(self):
        """If currently moving forward, ramp to zero. Block positive linear."""
        with self._lock:
            if self._target_linear > 0:
                self._target_linear = 0.0
            if self._current_linear > 0:
                self._current_linear = 0.0

    def _loop(self):
        """Main control loop — runs at update_rate Hz.

        Idles (no publishing) when both current and target are zero,
        saving CPU and avoiding unnecessary ROS2 publish overhead.
        """
        idle_count = 0
        while self._running:
            loop_start = time.monotonic()

            with self._lock:
                if self._e_stopped:
                    idle_count = 0
                    time.sleep(self._dt)
                    continue

                # Check if we're fully idle (current=0, target=0)
                at_rest = (
                    abs(self._current_linear) < 0.001
                    and abs(self._current_angular) < 0.001
                    and abs(self._target_linear) < 0.001
                    and abs(self._target_angular) < 0.001
                )

                if at_rest:
                    idle_count += 1
                    if idle_count > 10:  # after 0.5s at rest, sleep longer
                        time.sleep(0.1)  # 10Hz idle vs 20Hz active
                        continue
                else:
                    idle_count = 0

                # Ramp linear velocity toward target
                self._current_linear = self._ramp(
                    self._current_linear,
                    self._target_linear,
                    self._linear_accel,
                    self._linear_decel,
                )

                # Ramp angular velocity toward target
                self._current_angular = self._ramp(
                    self._current_angular,
                    self._target_angular,
                    self._angular_accel,
                    self._angular_decel,
                )

                lin = self._current_linear
                ang = self._current_angular

            # Publish
            msg = Twist()
            msg.linear.x = lin
            msg.angular.z = ang
            self._pub.publish(msg)
            self._last_published = (lin, ang)

            # Maintain loop rate
            elapsed = time.monotonic() - loop_start
            sleep_time = self._dt - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _ramp(self, current: float, target: float, accel: float, decel: float) -> float:
        """Ramp current toward target using acceleration/deceleration limits."""
        diff = target - current
        if abs(diff) < 0.001:
            return target

        # Determine if we're accelerating or decelerating
        # Decelerating = moving toward zero or reversing direction
        if abs(target) < abs(current) or (target * current < 0):
            rate = decel
        else:
            rate = accel

        max_step = rate * self._dt
        if abs(diff) <= max_step:
            return target
        elif diff > 0:
            return current + max_step
        else:
            return current - max_step

    def _publish_zero(self):
        """Publish a zero Twist immediately."""
        msg = Twist()
        self._pub.publish(msg)
        self._last_published = (0.0, 0.0)
