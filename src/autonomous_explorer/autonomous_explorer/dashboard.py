#!/usr/bin/env python3
# encoding: utf-8
"""
Real-time terminal dashboard for MentorPi Autonomous Explorer.

Subscribes to /explorer/status and renders a curses-based display showing
battery, sensors, motors, LLM reasoning, safety status, and session stats.

Usage:
    python3 scripts/dashboard.py
    # or after colcon build:
    ros2 run autonomous_explorer dashboard.py

No dependencies beyond stdlib curses + rclpy.
"""
import curses
import json
import math
import signal
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STATUS_TOPIC = '/explorer/status'
STALE_THRESHOLD = 5.0  # seconds before data is considered stale


# ---------------------------------------------------------------------------
# Dashboard ROS2 Node (minimal subscriber)
# ---------------------------------------------------------------------------
class DashboardNode(Node):
    """Minimal ROS2 node that subscribes to the explorer status topic."""

    def __init__(self):
        super().__init__('explorer_dashboard')
        self._status = None
        self._last_update = 0.0
        self.create_subscription(
            String, STATUS_TOPIC, self._status_callback, 1,
        )

    def _status_callback(self, msg: String):
        try:
            self._status = json.loads(msg.data)
            self._last_update = time.time()
        except json.JSONDecodeError:
            pass

    @property
    def status(self):
        return self._status

    @property
    def data_age(self):
        if self._last_update == 0:
            return float('inf')
        return time.time() - self._last_update


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------
def fmt_time(seconds: float) -> str:
    """Format seconds as HH:MM:SS."""
    h = int(seconds) // 3600
    m = (int(seconds) % 3600) // 60
    s = int(seconds) % 60
    return f'{h:02d}:{m:02d}:{s:02d}'


def fmt_ago(timestamp: float) -> str:
    """Format a timestamp as 'Xs ago'."""
    if timestamp == 0:
        return 'never'
    elapsed = time.time() - timestamp
    if elapsed < 1:
        return '<1s ago'
    elif elapsed < 60:
        return f'{int(elapsed)}s ago'
    elif elapsed < 3600:
        return f'{int(elapsed / 60)}m ago'
    return f'{int(elapsed / 3600)}h ago'


def battery_bar(voltage, width=16):
    """Return a battery bar string like [########------]."""
    if voltage is None:
        return '[' + '?' * width + ']'
    # Map voltage: 6.0V = 0%, 8.4V = 100% (2S LiPo)
    pct = max(0.0, min(1.0, (voltage - 6.0) / 2.4))
    filled = int(pct * width)
    return '[' + '#' * filled + '-' * (width - filled) + ']'


def battery_status(voltage):
    """Return battery status text."""
    if voltage is None:
        return 'N/A'
    if voltage > 7.4:
        return 'Full'
    elif voltage > 7.0:
        return 'Good'
    elif voltage > 6.7:
        return 'Low'
    return 'Critical'


def deg(rad):
    """Radians to degrees, rounded."""
    if rad is None:
        return 'N/A'
    return f'{math.degrees(rad):.1f}'


# ---------------------------------------------------------------------------
# Color pairs
# ---------------------------------------------------------------------------
C_NORMAL = 0
C_HEADER = 1
C_GREEN = 2
C_YELLOW = 3
C_RED = 4
C_CYAN = 5
C_DIM = 6
C_RED_FLASH = 7


def init_colors():
    """Initialize curses color pairs."""
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(C_HEADER, curses.COLOR_WHITE, curses.COLOR_BLUE)
    curses.init_pair(C_GREEN, curses.COLOR_GREEN, -1)
    curses.init_pair(C_YELLOW, curses.COLOR_YELLOW, -1)
    curses.init_pair(C_RED, curses.COLOR_RED, -1)
    curses.init_pair(C_CYAN, curses.COLOR_CYAN, -1)
    curses.init_pair(C_DIM, curses.COLOR_WHITE, -1)
    curses.init_pair(C_RED_FLASH, curses.COLOR_WHITE, curses.COLOR_RED)


def battery_color(voltage):
    """Return color pair for battery voltage."""
    if voltage is None:
        return C_DIM
    if voltage > 7.0:
        return C_GREEN
    elif voltage >= 6.7:
        return C_YELLOW
    return C_RED


def lidar_color(distance):
    """Return color pair for a lidar distance."""
    if distance is None or distance == float('inf') or distance > 10:
        return C_DIM
    if distance > 0.4:
        return C_GREEN
    elif distance > 0.2:
        return C_YELLOW
    return C_RED


# ---------------------------------------------------------------------------
# Drawing functions
# ---------------------------------------------------------------------------
def safe_addstr(win, y, x, text, attr=0):
    """Write text to window, truncating if it goes beyond the edge."""
    max_y, max_x = win.getmaxyx()
    if y < 0 or y >= max_y or x >= max_x:
        return
    available = max_x - x - 1
    if available <= 0:
        return
    try:
        win.addnstr(y, x, text, available, attr)
    except curses.error:
        pass


def draw_hline(win, y, x, width, char='─'):
    """Draw a horizontal line."""
    max_y, max_x = win.getmaxyx()
    if y < 0 or y >= max_y:
        return
    for i in range(width):
        if x + i >= max_x - 1:
            break
        try:
            win.addstr(y, x + i, char)
        except curses.error:
            pass


def draw_box_line(win, y, left_char, fill_char, right_char, width):
    """Draw a box line like ╠════...════╣."""
    max_y, max_x = win.getmaxyx()
    if y < 0 or y >= max_y:
        return
    try:
        win.addstr(y, 0, left_char)
        for i in range(1, min(width - 1, max_x - 1)):
            win.addstr(y, i, fill_char)
        if width - 1 < max_x:
            win.addstr(y, width - 1, right_char)
    except curses.error:
        pass


def draw_dashboard(stdscr, node: DashboardNode):
    """Main curses rendering loop."""
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(True)  # Non-blocking getch
    stdscr.timeout(500)  # Refresh every 500ms
    init_colors()

    flash = False  # For emergency stop flashing

    while True:
        # Check for quit
        key = stdscr.getch()
        if key in (ord('q'), ord('Q'), 27):  # q, Q, or Esc
            return

        # Spin ROS2 briefly to get messages
        rclpy.spin_once(node, timeout_sec=0.05)

        stdscr.erase()
        max_y, max_x = stdscr.getmaxyx()
        w = max_x  # usable width

        if max_y < 20 or max_x < 60:
            safe_addstr(stdscr, 0, 0,
                        f'Terminal too small ({max_x}x{max_y}). '
                        f'Need at least 60x20.',
                        curses.color_pair(C_RED))
            stdscr.refresh()
            continue

        status = node.status
        data_age = node.data_age

        if status is None:
            safe_addstr(stdscr, 0, 0,
                        '╔' + '═' * (w - 2) + '╗')
            title = '  MentorPi Explorer Dashboard'
            safe_addstr(stdscr, 1, 0, '║',
                        curses.color_pair(C_HEADER))
            safe_addstr(stdscr, 1, 1, title.ljust(w - 2),
                        curses.color_pair(C_HEADER))
            safe_addstr(stdscr, 1, w - 1, '║',
                        curses.color_pair(C_HEADER))
            safe_addstr(stdscr, 2, 0,
                        '╚' + '═' * (w - 2) + '╝')
            safe_addstr(stdscr, 4, 2,
                        'Waiting for data on /explorer/status ...',
                        curses.color_pair(C_YELLOW))
            safe_addstr(stdscr, 5, 2,
                        'Make sure explorer_node is running.',
                        curses.color_pair(C_DIM))
            safe_addstr(stdscr, 7, 2,
                        'q = quit',
                        curses.color_pair(C_DIM))
            stdscr.refresh()
            continue

        # ---- Unpack status ----
        mode = status.get('mode', '?').upper()
        exploring = status.get('exploring', False)
        e_stop = status.get('emergency_stop', False)
        batt_v = status.get('battery_voltage')
        motors = status.get('motors', {})
        servos = status.get('servos', {})
        lidar = status.get('lidar', {})
        imu = status.get('imu', {})
        odom = status.get('odom', {})
        llm = status.get('llm', {})
        provider = status.get('provider', '?')
        session = status.get('session', {})
        joystick = status.get('joystick', {})

        flash = not flash  # Toggle for flashing elements

        row = 0

        # ======== Header ========
        safe_addstr(stdscr, row, 0, '╔' + '═' * (w - 2) + '╗')
        row += 1

        title = '  MentorPi Explorer Dashboard'
        mode_str = f'Mode: {mode}'
        expl_str = f'Exploring: {"YES" if exploring else "NO"}'
        right_text = f'{mode_str}   {expl_str}   '

        safe_addstr(stdscr, row, 0, '║',
                    curses.color_pair(C_HEADER))
        safe_addstr(stdscr, row, 1,
                    ' ' * (w - 2),
                    curses.color_pair(C_HEADER))
        safe_addstr(stdscr, row, 2, title,
                    curses.color_pair(C_HEADER) | curses.A_BOLD)
        right_x = max(len(title) + 4, w - len(right_text) - 1)
        safe_addstr(stdscr, row, right_x, mode_str,
                    curses.color_pair(C_HEADER) | curses.A_BOLD)
        expl_x = right_x + len(mode_str) + 3
        expl_color = (curses.color_pair(C_HEADER) | curses.A_BOLD)
        safe_addstr(stdscr, row, expl_x, expl_str, expl_color)
        safe_addstr(stdscr, row, w - 1, '║',
                    curses.color_pair(C_HEADER))
        row += 1

        # ======== Battery & Motors row ========
        safe_addstr(stdscr, row, 0,
                    '╠' + '═' * (w - 2) + '╣')
        row += 1

        # Battery
        batt_bar = battery_bar(batt_v)
        batt_v_str = f'{batt_v:.1f}V' if batt_v else 'N/A'
        batt_stat = battery_status(batt_v)
        batt_col = curses.color_pair(battery_color(batt_v))

        safe_addstr(stdscr, row, 0, '║')
        safe_addstr(stdscr, row, 2, 'BATTERY', curses.A_BOLD)
        safe_addstr(stdscr, row, 10, f' {batt_bar} {batt_v_str}', batt_col)

        # Motors on the right half
        mid = w // 2
        lin = motors.get('linear', 0.0)
        ang = motors.get('angular', 0.0)
        safe_addstr(stdscr, row, mid, '│')
        safe_addstr(stdscr, row, mid + 2, 'MOTORS', curses.A_BOLD)
        safe_addstr(stdscr, row, mid + 9,
                    f' lin: {lin:+.2f} m/s  ang: {ang:+.2f} rad/s')
        safe_addstr(stdscr, row, w - 1, '║')
        row += 1

        # Battery status / Servos row
        safe_addstr(stdscr, row, 0, '║')
        safe_addstr(stdscr, row, 2,
                    f'Status: {batt_stat}', batt_col)
        safe_addstr(stdscr, row, mid, '│')
        pan = servos.get('pan', 1500)
        tilt = servos.get('tilt', 1500)
        safe_addstr(stdscr, row, mid + 2, 'Servos', curses.A_BOLD)
        safe_addstr(stdscr, row, mid + 9,
                    f' pan: {pan}  tilt: {tilt}')
        safe_addstr(stdscr, row, w - 1, '║')
        row += 1

        # ======== LiDAR & Position ========
        safe_addstr(stdscr, row, 0,
                    '╠' + '═' * (mid - 1) + '╬' + '═' * (w - mid - 2) + '╣')
        row += 1

        front = lidar.get('front') if lidar else None
        back = lidar.get('back') if lidar else None
        left = lidar.get('left') if lidar else None
        right = lidar.get('right') if lidar else None
        overall = lidar.get('overall') if lidar else None

        def fmt_dist(d):
            if d is None:
                return 'N/A'
            if d == float('inf') or d > 10:
                return '>10m'
            return f'{d:.2f}m'

        def dist_warn(d):
            if d is not None and d != float('inf') and d <= 0.4:
                return ' [!]' if d > 0.2 else ' [!!]'
            return ''

        # LiDAR panel
        safe_addstr(stdscr, row, 0, '║')
        safe_addstr(stdscr, row, 2, 'LIDAR', curses.A_BOLD)
        safe_addstr(stdscr, row, mid, '║')
        safe_addstr(stdscr, row, mid + 2, 'POSITION', curses.A_BOLD)
        safe_addstr(stdscr, row, w - 1, '║')
        row += 1

        # Front/Back
        safe_addstr(stdscr, row, 0, '║')
        safe_addstr(stdscr, row, 2,
                    f' Front: {fmt_dist(front)}',
                    curses.color_pair(lidar_color(front)))
        warn = dist_warn(front)
        if warn:
            safe_addstr(stdscr, row, 2 + 8 + len(fmt_dist(front)),
                        warn, curses.color_pair(C_RED) | curses.A_BOLD)
        safe_addstr(stdscr, row, 22,
                    f'Back: {fmt_dist(back)}',
                    curses.color_pair(lidar_color(back)))

        # Position X/Y/Heading
        safe_addstr(stdscr, row, mid, '║')
        ox = odom.get('x', 0.0) if odom else 0.0
        oy = odom.get('y', 0.0) if odom else 0.0
        oth = odom.get('theta', 0.0) if odom else 0.0
        hdg = f'{math.degrees(oth):.1f}' if odom else 'N/A'
        safe_addstr(stdscr, row, mid + 2,
                    f' X: {ox:.2f}   Y: {oy:.2f}   Hdg: {hdg}°')
        safe_addstr(stdscr, row, w - 1, '║')
        row += 1

        # Left/Right
        safe_addstr(stdscr, row, 0, '║')
        safe_addstr(stdscr, row, 2,
                    f' Left:  {fmt_dist(left)}',
                    curses.color_pair(lidar_color(left)))
        warn = dist_warn(left)
        if warn:
            safe_addstr(stdscr, row, 2 + 8 + len(fmt_dist(left)),
                        warn, curses.color_pair(C_RED) | curses.A_BOLD)
        safe_addstr(stdscr, row, 22,
                    f'Right: {fmt_dist(right)}',
                    curses.color_pair(lidar_color(right)))
        warn = dist_warn(right)
        if warn:
            safe_addstr(stdscr, row,
                        22 + 7 + len(fmt_dist(right)),
                        warn, curses.color_pair(C_RED) | curses.A_BOLD)

        # IMU
        safe_addstr(stdscr, row, mid, '║')
        if imu:
            orient = imu.get('orientation', {})
            roll = deg(orient.get('roll'))
            pitch = deg(orient.get('pitch'))
            safe_addstr(stdscr, row, mid + 2,
                        'IMU', curses.A_BOLD)
            safe_addstr(stdscr, row, mid + 6,
                        f' roll: {roll}°  pitch: {pitch}°')
        else:
            safe_addstr(stdscr, row, mid + 2, 'IMU: N/A',
                        curses.color_pair(C_DIM))
        safe_addstr(stdscr, row, w - 1, '║')
        row += 1

        # Min distance / yaw+accel
        safe_addstr(stdscr, row, 0, '║')
        safe_addstr(stdscr, row, 2,
                    f' Min:   {fmt_dist(overall)}',
                    curses.color_pair(lidar_color(overall)))
        safe_addstr(stdscr, row, mid, '║')
        if imu:
            orient = imu.get('orientation', {})
            yaw = deg(orient.get('yaw'))
            accel = imu.get('linear_acceleration', {})
            az = accel.get('z', 0.0)
            safe_addstr(stdscr, row, mid + 2,
                        f' yaw: {yaw}°  az: {az:.2f} m/s²')
        safe_addstr(stdscr, row, w - 1, '║')
        row += 1

        # ======== LLM Brain ========
        safe_addstr(stdscr, row, 0,
                    '╠' + '═' * (w - 2) + '╣')
        row += 1

        llm_time = llm.get('timestamp', 0)
        llm_ago = fmt_ago(llm_time)
        resp_ms = llm.get('response_ms', 0)
        tok_in = llm.get('tokens_in', 0)
        tok_out = llm.get('tokens_out', 0)
        llm_cost = llm.get('cost', 0.0)

        safe_addstr(stdscr, row, 0, '║')
        safe_addstr(stdscr, row, 2,
                    f'LLM BRAIN ({provider})', curses.A_BOLD)
        stats_str = (f'{resp_ms}ms | {tok_in}+{tok_out} tok '
                     f'| ${llm_cost:.4f}')
        safe_addstr(stdscr, row, 2 + len(f'LLM BRAIN ({provider})') + 4,
                    stats_str, curses.color_pair(C_CYAN))
        safe_addstr(stdscr, row, w - 1, '║')
        row += 1

        # Action line
        llm_action = llm.get('action', '')
        llm_speed = llm.get('speed', 0.0)
        llm_dur = llm.get('duration', 0.0)

        safe_addstr(stdscr, row, 0, '║')
        if llm_action:
            action_str = (f' Action: {llm_action}  '
                          f'speed={llm_speed:.1f}  dur={llm_dur:.1f}s')
            safe_addstr(stdscr, row, 1, action_str,
                        curses.color_pair(C_GREEN) | curses.A_BOLD)
            # Ago indicator on right
            ago_x = w - len(llm_ago) - 2
            safe_addstr(stdscr, row, ago_x, llm_ago,
                        curses.color_pair(C_DIM))
        else:
            safe_addstr(stdscr, row, 2,
                        'Action: (none yet)',
                        curses.color_pair(C_DIM))
        safe_addstr(stdscr, row, w - 1, '║')
        row += 1

        # Reasoning separator
        safe_addstr(stdscr, row, 0,
                    '╟' + '─' * (w - 2) + '╢')
        row += 1

        # Reasoning (wrap to multiple lines)
        reasoning = llm.get('reasoning', '')
        if reasoning:
            text_w = w - 4  # 2 chars border on each side
            lines = []
            while len(reasoning) > text_w:
                # Find a space to break at
                brk = reasoning[:text_w].rfind(' ')
                if brk <= 0:
                    brk = text_w
                lines.append(reasoning[:brk])
                reasoning = reasoning[brk:].lstrip()
            if reasoning:
                lines.append(reasoning)

            safe_addstr(stdscr, row, 0, '║')
            safe_addstr(stdscr, row, 2,
                        f'Reasoning: {lines[0]}' if lines else 'Reasoning:')
            safe_addstr(stdscr, row, w - 1, '║')
            row += 1
            for line in lines[1:]:
                if row >= max_y - 6:
                    break
                safe_addstr(stdscr, row, 0, '║')
                safe_addstr(stdscr, row, 2, line)
                safe_addstr(stdscr, row, w - 1, '║')
                row += 1
        else:
            safe_addstr(stdscr, row, 0, '║')
            safe_addstr(stdscr, row, 2, 'Reasoning: (waiting for first cycle)',
                        curses.color_pair(C_DIM))
            safe_addstr(stdscr, row, w - 1, '║')
            row += 1

        # Speech separator
        safe_addstr(stdscr, row, 0,
                    '╟' + '─' * (w - 2) + '╢')
        row += 1

        # Speech
        speech = llm.get('speech', '')
        safe_addstr(stdscr, row, 0, '║')
        if speech:
            display = f' Speech: "{speech}"'
            if len(display) > w - 4:
                display = display[:w - 7] + '..."'
            safe_addstr(stdscr, row, 1, display,
                        curses.color_pair(C_CYAN))
        else:
            safe_addstr(stdscr, row, 2, 'Speech: (none)',
                        curses.color_pair(C_DIM))
        safe_addstr(stdscr, row, w - 1, '║')
        row += 1

        # ======== Safety & Session Stats ========
        safe_addstr(stdscr, row, 0,
                    '╠' + '═' * (w - 2) + '╣')
        row += 1

        # Safety + Stats
        safety_triggered = llm.get('safety_triggered', False)
        safety_reason = llm.get('safety_reason', '')

        safe_addstr(stdscr, row, 0, '║')

        if e_stop:
            if flash:
                safe_addstr(stdscr, row, 2,
                            ' ** EMERGENCY STOP ** ',
                            curses.color_pair(C_RED_FLASH) | curses.A_BOLD)
            else:
                safe_addstr(stdscr, row, 2,
                            '    EMERGENCY STOP    ',
                            curses.color_pair(C_RED) | curses.A_BOLD)
        elif safety_triggered:
            safe_addstr(stdscr, row, 2, 'SAFETY:',
                        curses.A_BOLD)
            safe_addstr(stdscr, row, 10,
                        f' {safety_reason}',
                        curses.color_pair(C_YELLOW))
        else:
            safe_addstr(stdscr, row, 2, 'SAFETY:',
                        curses.A_BOLD)
            safe_addstr(stdscr, row, 10, ' OK',
                        curses.color_pair(C_GREEN))

        # Session stats on right
        cycles = session.get('cycle_count', 0)
        total_cost = session.get('total_cost', 0.0)
        discoveries = session.get('discoveries', 0)
        stats_right = (f'{cycles} cycles  '
                       f'${total_cost:.2f} total  '
                       f'{discoveries} discoveries')
        safe_addstr(stdscr, row, mid, '│')
        safe_addstr(stdscr, row, mid + 2, stats_right)
        safe_addstr(stdscr, row, w - 1, '║')
        row += 1

        # Joystick + Uptime
        safe_addstr(stdscr, row, 0, '║')
        js_connected = joystick.get('connected', False)
        js_name = joystick.get('name', '')
        if js_connected:
            safe_addstr(stdscr, row, 2,
                        f'Joystick: {js_name} (connected)',
                        curses.color_pair(C_GREEN))
        else:
            safe_addstr(stdscr, row, 2,
                        'Joystick: disconnected',
                        curses.color_pair(C_DIM))

        uptime = session.get('uptime', 0)
        safe_addstr(stdscr, row, mid, '│')
        safe_addstr(stdscr, row, mid + 2,
                    f'Uptime: {fmt_time(uptime)}')
        safe_addstr(stdscr, row, w - 1, '║')
        row += 1

        # ======== Footer ========
        safe_addstr(stdscr, row, 0,
                    '╠' + '═' * (w - 2) + '╣')
        row += 1

        safe_addstr(stdscr, row, 0, '║')
        safe_addstr(stdscr, row, 2, ' q=quit',
                    curses.color_pair(C_DIM))

        # Data freshness
        if data_age == float('inf'):
            age_str = 'No data'
        elif data_age < 1:
            age_str = 'Updated: <1s ago'
        else:
            age_str = f'Updated: {data_age:.1f}s ago'

        age_color = C_GREEN if data_age < 2 else (
            C_YELLOW if data_age < STALE_THRESHOLD else C_RED)
        age_x = w - len(age_str) - 2
        safe_addstr(stdscr, row, age_x, age_str,
                    curses.color_pair(age_color))
        safe_addstr(stdscr, row, w - 1, '║')
        row += 1

        safe_addstr(stdscr, row, 0,
                    '╚' + '═' * (w - 2) + '╝')

        stdscr.refresh()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    rclpy.init()
    node = DashboardNode()

    def signal_handler(sig, frame):
        node.destroy_node()
        rclpy.try_shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        curses.wrapper(lambda stdscr: draw_dashboard(stdscr, node))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
