#!/usr/bin/env python3
# encoding: utf-8
"""
Self-contained pygame joystick reader running as a daemon thread.

Polls /dev/input/js0 at ~50Hz, exposes thread-safe axis values,
and fires callbacks on button press/release. Auto-detects SHANWAN
Android Gamepad vs USB WirelessGamepad button layouts and handles
connect/disconnect/reconnect gracefully.

Usage:
    reader = JoystickReader(on_button_press=my_callback)
    reader.start()
    ...
    axes = reader.axes  # thread-safe copy
    reader.stop()
"""
import os
import threading
import time
from typing import Callable, Optional

# pygame is imported lazily inside the thread so it doesn't pollute
# the main process display state until we actually need it.
import pygame as pg


# ---------------------------------------------------------------------------
# Button name maps — index = pygame button number
# ---------------------------------------------------------------------------
_BUTTONS_SHANWAN = (
    'cross', 'circle', '', 'square', 'triangle', '',
    'l1', 'r1', 'l2', 'r2', 'select', 'start', 'mode', 'lc', 'rc',
)

_BUTTONS_WIRELESS = (
    'triangle', 'circle', 'cross', 'square', 'l1', 'r1', 'l2', 'r2',
    'select', 'start', 'lc', 'rc', 'mode', '', '',
)

# Axis names used in the public `axes` dict
_AXIS_NAMES = ('left_x', 'left_y', 'right_x', 'right_y')

DEADZONE = 0.10


class JoystickReader:
    """Background-thread joystick reader with button callbacks."""

    def __init__(
        self,
        on_button_press: Optional[Callable[[str], None]] = None,
        on_button_release: Optional[Callable[[str], None]] = None,
        deadzone: float = DEADZONE,
        logger=None,
    ):
        self._on_press = on_button_press
        self._on_release = on_button_release
        self._deadzone = deadzone
        self._log = logger  # ROS2 logger or None

        # Thread-safe state
        self._lock = threading.Lock()
        self._axes = {name: 0.0 for name in _AXIS_NAMES}
        self._axes['hat_x'] = 0.0
        self._axes['hat_y'] = 0.0
        self._connected = False
        self._joystick_name = ''

        self._running = False
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def joystick_name(self) -> str:
        with self._lock:
            return self._joystick_name

    @property
    def axes(self) -> dict:
        """Return a thread-safe snapshot of current axis values."""
        with self._lock:
            return self._axes.copy()

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            return
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _info(self, msg: str):
        if self._log:
            self._log.info(msg)

    def _warn(self, msg: str):
        if self._log:
            self._log.warn(msg)

    def _apply_deadzone(self, value: float) -> float:
        return 0.0 if abs(value) < self._deadzone else value

    def _run(self):
        """Main polling loop — runs in daemon thread."""
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        pg.display.init()
        pg.joystick.init()

        js = None
        button_map = _BUTTONS_SHANWAN

        self._info('Joystick reader thread started')

        while self._running:
            try:
                # ---- Connection management ----
                if js is None:
                    if os.path.exists('/dev/input/js0'):
                        pg.joystick.quit()
                        pg.joystick.init()
                        if pg.joystick.get_count() > 0:
                            js = pg.joystick.Joystick(0)
                            js.init()
                            name = js.get_name()
                            if name == 'USB WirelessGamepad':
                                button_map = _BUTTONS_WIRELESS
                            else:
                                button_map = _BUTTONS_SHANWAN
                            with self._lock:
                                self._connected = True
                                self._joystick_name = name
                            self._info(f'Joystick connected: {name}')
                    else:
                        time.sleep(0.5)
                        continue
                elif not os.path.exists('/dev/input/js0'):
                    self._warn('Joystick disconnected')
                    js.quit()
                    js = None
                    with self._lock:
                        self._connected = False
                        self._joystick_name = ''
                        # Zero all axes on disconnect
                        for k in self._axes:
                            self._axes[k] = 0.0
                    # Fire a synthetic release for safety
                    if self._on_press:
                        pass  # Caller handles disconnect via .connected
                    time.sleep(0.5)
                    continue

                # ---- Event processing ----
                for event in pg.event.get():
                    if event.type == pg.JOYAXISMOTION:
                        if event.axis < len(_AXIS_NAMES):
                            val = self._apply_deadzone(event.value)
                            with self._lock:
                                self._axes[_AXIS_NAMES[event.axis]] = val

                    elif event.type == pg.JOYHATMOTION:
                        hat_x, hat_y = event.value
                        with self._lock:
                            self._axes['hat_x'] = float(hat_x)
                            self._axes['hat_y'] = float(hat_y)

                    elif event.type == pg.JOYBUTTONDOWN:
                        if event.button < len(button_map):
                            name = button_map[event.button]
                            if name and self._on_press:
                                try:
                                    self._on_press(name)
                                except Exception as e:
                                    self._warn(f'Button press callback error: {e}')

                    elif event.type == pg.JOYBUTTONUP:
                        if event.button < len(button_map):
                            name = button_map[event.button]
                            if name and self._on_release:
                                try:
                                    self._on_release(name)
                                except Exception as e:
                                    self._warn(f'Button release callback error: {e}')

                # ~50Hz polling
                time.sleep(0.02)

            except pg.error:
                # pygame lost the joystick (e.g. USB unplug during read)
                self._warn('Joystick pygame error — reconnecting')
                js = None
                with self._lock:
                    self._connected = False
                    self._joystick_name = ''
                    for k in self._axes:
                        self._axes[k] = 0.0
                time.sleep(1.0)
            except Exception as e:
                self._warn(f'Joystick reader error: {e}')
                time.sleep(1.0)

        # Cleanup
        if js is not None:
            js.quit()
        pg.joystick.quit()
        self._info('Joystick reader thread stopped')
