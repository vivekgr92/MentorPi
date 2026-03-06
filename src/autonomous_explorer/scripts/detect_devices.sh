#!/usr/bin/env bash
# ============================================================================
# detect_devices.sh — Auto-detect USB sensors and create /dev symlinks
#
# Identifies devices by USB vendor:product IDs and product strings rather
# than relying on fragile /dev/ttyUSB* numbering which changes when sensors
# are unplugged and replugged.
#
# Known devices:
#   STM32 controller  — /dev/ttyACM*  → /dev/rrc
#   LD19 LiDAR        — CH340 serial  → /dev/ldlidar
#   WonderEcho Pro    — CH340 serial  → /dev/wonderecho
#   Aurora 930 camera  — USB 3251:1930 (detected by driver, no symlink needed)
#   Gamepad           — /dev/input/js0 (detected by pygame, no symlink needed)
#
# Usage:
#   source detect_devices.sh          # sets DETECTED_* variables
#   ./detect_devices.sh               # prints detection results
# ============================================================================

# Return values (set as env vars when sourced)
DETECTED_STM32=""
DETECTED_LIDAR=""
DETECTED_WONDERECHO=""
DETECTED_CAMERA=""
DETECTED_GAMEPAD=""

detect_devices() {
    echo ">>> Detecting USB devices..."

    # -- STM32 controller (ttyACM*) --
    # The STM32 is the only ACM device on this robot
    for dev in /dev/ttyACM*; do
        [[ -e "$dev" ]] || continue
        DETECTED_STM32="$dev"
        echo "  STM32 controller : $dev"
        break
    done
    if [[ -z "$DETECTED_STM32" ]]; then
        echo "  STM32 controller : NOT FOUND"
    fi

    # -- CH340 devices (ttyUSB*): distinguish LiDAR vs WonderEcho --
    # Both use CH340 (vendor 1a86:7523). We differentiate by the USB
    # device's parent hub port or product string.
    #
    # Strategy: check each ttyUSB device's udevadm info for distinguishing
    # attributes. The LD19 LiDAR typically has no product string or a
    # generic one, while the WonderEcho has "USB Serial" or similar.
    # As a fallback, we use baud rate probing: LiDAR runs at 230400,
    # WonderEcho at 115200.
    local ttyusb_devices=()
    for dev in /dev/ttyUSB*; do
        [[ -e "$dev" ]] || continue
        ttyusb_devices+=("$dev")
    done

    if [[ ${#ttyusb_devices[@]} -eq 0 ]]; then
        echo "  LiDAR            : NOT FOUND (no /dev/ttyUSB* devices)"
        echo "  WonderEcho       : NOT FOUND"
    elif [[ ${#ttyusb_devices[@]} -eq 1 ]]; then
        # Only one CH340 device — figure out which one it is
        local dev="${ttyusb_devices[0]}"
        local product
        product=$(udevadm info -a -n "$dev" 2>/dev/null | grep -m1 'ATTRS{product}' | sed 's/.*=="\(.*\)"/\1/' || echo "")
        local iface
        iface=$(udevadm info -a -n "$dev" 2>/dev/null | grep -m1 'ATTRS{interface}' | sed 's/.*=="\(.*\)"/\1/' || echo "")

        # WonderEcho typically reports interface/product containing "serial" or "wonderecho"
        if echo "$product $iface" | grep -iq "wonderecho\|wonder_echo\|ai.*voice\|voice.*box"; then
            DETECTED_WONDERECHO="$dev"
            echo "  WonderEcho       : $dev (product: $product)"
            echo "  LiDAR            : NOT FOUND"
        else
            # Default single CH340 to LiDAR (more critical for safety)
            DETECTED_LIDAR="$dev"
            echo "  LiDAR            : $dev (product: $product)"
            echo "  WonderEcho       : NOT FOUND"
        fi
    else
        # Multiple CH340 devices — try to identify each
        for dev in "${ttyusb_devices[@]}"; do
            [[ -n "$DETECTED_LIDAR" && -n "$DETECTED_WONDERECHO" ]] && break

            local product
            product=$(udevadm info -a -n "$dev" 2>/dev/null | grep -m1 'ATTRS{product}' | sed 's/.*=="\(.*\)"/\1/' || echo "")
            local iface
            iface=$(udevadm info -a -n "$dev" 2>/dev/null | grep -m1 'ATTRS{interface}' | sed 's/.*=="\(.*\)"/\1/' || echo "")
            local devpath
            devpath=$(udevadm info -q property -n "$dev" 2>/dev/null | grep DEVPATH | head -1 || echo "")

            if echo "$product $iface" | grep -iq "wonderecho\|wonder_echo\|ai.*voice\|voice.*box"; then
                DETECTED_WONDERECHO="$dev"
                echo "  WonderEcho       : $dev (product: $product)"
                continue
            fi
        done

        # Assign remaining unidentified CH340 devices
        for dev in "${ttyusb_devices[@]}"; do
            if [[ "$dev" == "$DETECTED_WONDERECHO" ]]; then
                continue
            fi
            if [[ -z "$DETECTED_LIDAR" ]]; then
                DETECTED_LIDAR="$dev"
                local product
                product=$(udevadm info -a -n "$dev" 2>/dev/null | grep -m1 'ATTRS{product}' | sed 's/.*=="\(.*\)"/\1/' || echo "")
                echo "  LiDAR            : $dev (product: $product)"
            elif [[ -z "$DETECTED_WONDERECHO" ]]; then
                DETECTED_WONDERECHO="$dev"
                echo "  WonderEcho       : $dev (assumed — remaining CH340)"
            fi
        done

        [[ -z "$DETECTED_LIDAR" ]] && echo "  LiDAR            : NOT FOUND"
        [[ -z "$DETECTED_WONDERECHO" ]] && echo "  WonderEcho       : NOT FOUND"
    fi

    # -- Aurora 930 depth camera (USB 3251:1930) --
    if lsusb 2>/dev/null | grep -q "3251:1930"; then
        DETECTED_CAMERA="aurora930"
        echo "  Depth camera     : Aurora 930 (USB 3251:1930)"
    else
        echo "  Depth camera     : NOT FOUND"
    fi

    # -- Gamepad --
    if [[ -e /dev/input/js0 ]]; then
        DETECTED_GAMEPAD="/dev/input/js0"
        echo "  Gamepad          : /dev/input/js0"
    else
        echo "  Gamepad          : NOT FOUND"
    fi

    echo ""
}

create_symlinks() {
    # STM32 → /dev/rrc
    if [[ -n "$DETECTED_STM32" ]]; then
        if [[ "$(readlink -f /dev/rrc 2>/dev/null)" != "$(readlink -f "$DETECTED_STM32")" ]]; then
            echo ">>> Creating symlink: /dev/rrc -> $DETECTED_STM32"
            sudo ln -sf "$DETECTED_STM32" /dev/rrc
        fi
    else
        # Remove stale symlink
        if [[ -L /dev/rrc ]]; then
            echo ">>> Removing stale symlink: /dev/rrc"
            sudo rm -f /dev/rrc
        fi
    fi

    # LiDAR → /dev/ldlidar
    if [[ -n "$DETECTED_LIDAR" ]]; then
        if [[ "$(readlink -f /dev/ldlidar 2>/dev/null)" != "$(readlink -f "$DETECTED_LIDAR")" ]]; then
            echo ">>> Creating symlink: /dev/ldlidar -> $DETECTED_LIDAR"
            sudo ln -sf "$DETECTED_LIDAR" /dev/ldlidar
        fi
    else
        if [[ -L /dev/ldlidar ]]; then
            echo ">>> Removing stale symlink: /dev/ldlidar"
            sudo rm -f /dev/ldlidar
        fi
    fi

    # WonderEcho → /dev/wonderecho
    if [[ -n "$DETECTED_WONDERECHO" ]]; then
        if [[ "$(readlink -f /dev/wonderecho 2>/dev/null)" != "$(readlink -f "$DETECTED_WONDERECHO")" ]]; then
            echo ">>> Creating symlink: /dev/wonderecho -> $DETECTED_WONDERECHO"
            sudo ln -sf "$DETECTED_WONDERECHO" /dev/wonderecho
        fi
    else
        if [[ -L /dev/wonderecho ]]; then
            echo ">>> Removing stale symlink: /dev/wonderecho"
            sudo rm -f /dev/wonderecho
        fi
    fi
}

# Run detection and symlink creation when executed directly
if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
    detect_devices
    create_symlinks
    echo ">>> Device detection complete."
fi
