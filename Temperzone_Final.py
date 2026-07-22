# /new1

# sudo fallocate -l 512M /userdata/swapfile
# sudo chmod 600 /userdata/swapfile
# sudo mkswap /userdata/swapfile
# sudo swapon /userdata/swapfile
# echo '/userdata/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
# free -h
# top -b -n 1 | head -20

from flask import Flask, render_template_string, request, jsonify, Response
from pymodbus.client.sync import ModbusSerialClient as ModbusClient
import threading, time, os, subprocess, socket, shutil, glob
from flask import send_from_directory
import mmap, struct
from collections import deque
from datetime import datetime, timedelta
import json, smtplib, csv
from email.message import EmailMessage
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import random
import sqlite3

# ── Occupancy Variables ─────────────────────────────────────────────────────────────────────
OCCUPANCY_DB = "/home/linaro/occupancy.db"
occupancy_manual_override = {"active": False, "until": 0}  # manual "occupied now"

# ── Layer 2: debounced presence state ──
# Raw detection (face/motion) is noisy. Layer 2 holds "occupied" through short
# gaps so the room doesn't flip empty while people sit still or step off-camera.
VACANCY_TIMEOUT   = 1200   # sec of zero detection before room considered vacant (20 min)
presence_state = {
    "present":        False,   # debounced Layer-2 verdict
    "last_detect":    0.0,     # monotonic time of last raw detection
    "since":          0.0,     # when current present/vacant state began
    "evidence_window": deque(maxlen=60),  # last 60 samples of raw detection (0/1)
}
presence_lock = threading.Lock()


# ── Display Global Variable ─────────────────────────────────────────────────────────────────────
screen_on = True
overlay_proc = None
splash_proc = None   # the startup logo process — closed only once Chromium is ready


# ── Files ─────────────────────────────────────────────────────────────────────
DEVICE_CONFIG_FILE = "/home/linaro/device_config.json"
SCHEDULE_FILE      = "/home/linaro/schedule.json"
CSV_FILE   = "/userdata/temp_log.csv"
FAULT_FILE = "/userdata/fault_log.csv"
HEADER             = ["Date Time", "Temperature", "Set Point"]
FAULT_HEADER       = ["Date Time", "Temp at Time", "SP at Time"]

fault_flag = False
last_fault_coil = {}   # per-controller: last successfully-read Coil 56 value — survives comms glitches

DEFAULT_DEVICE_CONFIG = [
    {"name": "Temperzone", "type": "Temperzone EcoNEX PRO", "slave_id": 20, "enabled": True},
]

# Temperzone EcoNEX PRO controllers use the 20-series Modbus address block
TEMPERZONE_SID_BASE = 20
TEMPERZONE_SID_MAX  = 29
def is_temperzone_type(ctype):
    return 'temperzone' in (ctype or '').lower()
def next_slave_id(cfg, ctype="Temperzone EcoNEX PRO"):
    """Return the next free Modbus Slave ID. Temperzone EcoNEX PRO fills the
       20..29 block in order (20, 21, 22 … 29); anything else takes the lowest
       free 1..247. Never returns an ID already in use."""
    used = {c.get('slave_id') for c in (cfg or [])}
    if is_temperzone_type(ctype):
        for sid in range(TEMPERZONE_SID_BASE, TEMPERZONE_SID_MAX + 1):
            if sid not in used:
                return sid
    for sid in range(1, 248):
        if sid not in used:
            return sid
    return TEMPERZONE_SID_BASE

def make_empty_week():
    return {"Mon":[],"Tue":[],"Wed":[],"Thu":[],"Fri":[],"Sat":[],"Sun":[]}

app = Flask(__name__)

@app.after_request
def no_cache(resp):
    resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
    resp.headers['Pragma'] = 'no-cache'
    resp.headers['Expires'] = '0'
    return resp

# ── Economy cycle (free cooling) ──
economy_enabled    = {}     # per-controller: {ctrl_idx: True/False}
economy_active     = {}     # per-controller: is free-cooling currently engaged
co2_enabled        = {}     # per-controller: show CO2 card on dashboard (default True)
ECONOMY_HYSTERESIS = 2.0    # °C outside must be cooler than inside
ECONOMY_DAMPER_REG = 3      # ⚠️ SET THIS — fresh-air damper register

# ── GPIO ──────────────────────────────────────────────────────────────────────
GPIO3_BASE = 0xfe760000
PAGE_SIZE  = 0x1000
DR_H, DDR_H = 0x0004, 0x000C
RED_BIT    = 1 << (24 - 16)
GREEN_BIT  = 1 << (25 - 16)
BLUE_BIT   = 1 << (27 - 16)
MASK       = RED_BIT | GREEN_BIT | BLUE_BIT
current_color = (0, 0, 0)
lock_led = threading.Lock()

# ── Graph ─────────────────────────────────────────────────────────────────────
history_lock     = threading.Lock()
HISTORY_MAXLEN   = 600   # updated live by /api/trend_config
temp_history     = deque(maxlen=HISTORY_MAXLEN)
setpoint_history = deque(maxlen=HISTORY_MAXLEN)
time_history     = deque(maxlen=HISTORY_MAXLEN)
multi_temp_history = {}   # {ctrl_idx: deque} — per-controller temp for overlay

# ── Minute-aggregated trend history ──
# The chart shows a clean minute-by-minute trend (no seconds). We roll the raw
# 1-second samples into a single averaged point per minute so the line stays
# smooth and readable over long windows, and label each point "HH:MM".
MINUTE_HISTORY_MAXLEN     = 720   # up to 12 h of 1-minute points
minute_time_history       = deque(maxlen=MINUTE_HISTORY_MAXLEN)   # "HH:MM"
minute_temp_history       = deque(maxlen=MINUTE_HISTORY_MAXLEN)
minute_setpoint_history   = deque(maxlen=MINUTE_HISTORY_MAXLEN)
minute_multi_temp_history = {}    # {ctrl_idx: deque}
_minute_accum = {"key": None, "temp": [], "sp": [], "multi": {}}

def _minute_avg(vals):
    good = [v for v in vals if v is not None]
    return round(sum(good) / len(good), 1) if good else None

def accumulate_minute(now_dt, t_val, s_val, cur_multi):
    """Roll 1-second samples into one averaged point per minute.
       Must be called while holding history_lock."""
    mkey = now_dt.strftime("%H:%M")
    acc  = _minute_accum
    if acc["key"] is None:
        acc["key"] = mkey
    if mkey != acc["key"]:
        # minute rolled over → flush the completed minute as one point
        minute_time_history.append(acc["key"])
        minute_temp_history.append(_minute_avg(acc["temp"]) or 0)
        minute_setpoint_history.append(_minute_avg(acc["sp"]) or 0)
        for ci, vals in acc["multi"].items():
            if ci not in minute_multi_temp_history:
                minute_multi_temp_history[ci] = deque(maxlen=MINUTE_HISTORY_MAXLEN)
            minute_multi_temp_history[ci].append(_minute_avg(vals))
        acc["key"]   = mkey
        acc["temp"]  = []
        acc["sp"]    = []
        acc["multi"] = {}
    acc["temp"].append(t_val)
    acc["sp"].append(s_val)
    for ci, v in cur_multi.items():
        acc["multi"].setdefault(ci, []).append(v)

# ── Modbus ────────────────────────────────────────────────────────────────────
PORT          = '/dev/ttyS7'
BAUDRATE      = 9600
PARITY        = 'N'
STOPBITS      = 1
BYTESIZE      = 8
TIMEOUT       = 0.3
SCAN_INTERVAL = 1.0

# ── Motion ────────────────────────────────────────────────────────────────────
VIDEO_DEV       = "/dev/video8"
VIDEO_FALLBACKS = ["/dev/video0", "/dev/video1", "/dev/video2", "/dev/video4"]
SCREEN_TIMEOUT  = 5.0
motion_enabled  = True
MIN_PIXELS      = 50
NOISE_THRESH    = 30
BG_ALPHA        = 0.05
WARMUP_FRAMES   = 20
CONFIRM_FRAMES  = 1
SAMPLING_RATE   = 0.2
SKIP_BORDER     = 10
CAM_W           = 160
CAM_H           = 120
USE_FACE        = True
FACE_SCALE      = 1.2
FACE_NEIGHBORS  = 4
FACE_MIN_SIZE   = (20, 20)

# ── Shared state ──────────────────────────────────────────────────────────────
client             = None
all_ctrl_data      = []
all_ctrl_status    = []
all_ctrl_connected = []
modbus_status      = "Disconnected"
screen_on          = True
last_motion        = time.monotonic()
lock               = threading.Lock()
clock_data         = {"time": "--:--:--", "ampm": "--", "date": "----------"}
clock_lock         = threading.Lock()
runtime_serial     = {"port": PORT, "baud": BAUDRATE}

motion_state = {
    "motion_pct":  0.0,
    "changed_px":  0,
    "face_count":  0,
    "consecutive": 0,
    "frame_count": 0,
    "warmup_done": False,
    "status_txt":  "Starting...",
    "history":     [0.0] * 12,
    "last_wake":   "—",
}
motion_state_lock = threading.Lock()

# ── Register maps per controller type ─────────────────────────────────────────


REGISTER_MAP_TEMPERZONE = {
    1:   {"name": "Current Temperature (IR1)", "scale": 0.1, "unit": "°C", "type": "ro"},
    135: {"name": "On/Off Status (IR135)",     "scale": 1,   "unit": "",   "type": "ro",
          "options": {1:"On",2:"Off Alarm",3:"Off BMS",4:"Off Scheduler",5:"Off Digital In",6:"Off Local",7:"Manual"}},
    100: {"name": "Cooling Setpoint (HR100)",  "scale": 0.1, "unit": "°C", "type": "rw"},
    102: {"name": "Heating Setpoint (HR102)",  "scale": 0.1, "unit": "°C", "type": "rw"},
    114: {"name": "Fan Speed Demand (HR114)",  "scale": 0.1,   "unit": "%",  "type": "rw"},
    116: {"name": "Fan Speed Mode (HR116)",    "scale": 1,   "unit": "",   "type": "rw"},
    117: {"name": "Unit Mode (HR117)",         "scale": 1,   "unit": "",   "type": "rw",
          "options": {0:"Auto",1:"Cooling",2:"Heating",3:"Fan"}},
    "onoff":      {"name": "On/Off BMS (Coil 1)",  "scale": 1, "unit": "", "type": "bit", "options": {0:"Off",1:"On"}},
    "onoff2":     {"name": "On/Off Keyboard (Coil 2)", "scale": 1, "unit": "", "type": "bit", "options": {0:"Off",1:"On"}},
    "_coil2":     {"name": "Coil 2 (live on/off)",  "scale": 1, "unit": "", "type": "bit", "options": {0:"Off",1:"On"}},
    "fault_coil": {"name": "Fault (Coil 56)",  "scale": 1, "unit": "", "type": "bit", "options": {0:"Normal",1:"Fault"}},
    "sch_coil": {"name": "schedule (Coil 22)",  "scale": 1, "unit": "", "type": "bit", "options": {0:"False",1:"True"}},
    
}
# Which register means what, per type. The rest of the app uses these names
# instead of hardcoded numbers.
REG_ROLES = {
    "Temperzone": {
        "temp": 1,            "temp_rtype": "input",     # Input Register 1
        "onoff": 1,           "onoff_rtype": "coil",     # Coil 1 (BMS on/off, RW)
        "onoff2": 2,          "onoff2_rtype": "coil",    # Coil 2 (keyboard on/off, RO)
        "fault": 56,          "fault_rtype": "coil",     # Coil 56 (alarm)
        "schedule": 22,       "schedule_rtype": "coil",     # Coil 22 (schedule)
        "status": 135,        "status_rtype": "input",   # Input Register 135 (status enum)
        "setpoint": 100,      "setpoint_rtype": "holding",   # cooling setpoint
        "setpoint_heat": 102, "setpoint_heat_rtype": "holding",
        "fan": 114,           "fan_rtype": "holding",    # fan demand %
        "fan_mode": 116,      "fan_mode_rtype": "holding",
        "mode": 117,          "mode_rtype": "holding",   # 0=Auto 1=Cool 2=Heat 3=Fan
        "sched": 22,          "sched_rtype": "coil",
        "co": None,
        "oat": None,
        "scale": 0.1,
        "map": REGISTER_MAP_TEMPERZONE,
    },
}


def roles_for(ctrl):
    # Temperzone-only system: always the Temperzone role map
    return REG_ROLES["Temperzone"]

def read_field(client, roles, field, sid):
    addr = roles.get(field)
    if addr is None: return None
    rtype = roles.get(field + "_rtype", "holding")
    if rtype == "coil":
        r = client.read_coils(addr, 1, unit=sid)         # Coil 1
        return int(r.bits[0]) if not r.isError() else None
    elif rtype == "input":
        r = client.read_input_registers(addr, 1, unit=sid)  # Input Register 1
        return r.registers[0] if not r.isError() else None
    else:
        r = client.read_holding_registers(addr, 1, unit=sid)
        return r.registers[0] if not r.isError() else None

# Backward-compat: some code still references REGISTER_MAP
REGISTER_MAP = REGISTER_MAP_TEMPERZONE
MAX_CONTROLLERS = 10
# ── Config helpers ────────────────────────────────────────────────────────────
def load_device_config():
    if os.path.exists(DEVICE_CONFIG_FILE):
        try:
            with open(DEVICE_CONFIG_FILE, 'r') as f:
                cfg = json.load(f)
            if cfg and isinstance(cfg, list):
                return cfg[:MAX_CONTROLLERS]      # ← trim to 3
        except Exception:
            pass
    default = [c.copy() for c in DEFAULT_DEVICE_CONFIG]
    save_device_config(default)
    return default

def save_device_config(data):
    with open(DEVICE_CONFIG_FILE, 'w') as f:
        json.dump(data, f)

def ensure_ctrl_lists(n):
    global all_ctrl_data, all_ctrl_status, all_ctrl_connected
    while len(all_ctrl_data) < n:
        all_ctrl_data.append({})
        all_ctrl_status.append("Disconnected")
        all_ctrl_connected.append(False)
    # trim excess
    all_ctrl_data      = all_ctrl_data[:n]
    all_ctrl_status    = all_ctrl_status[:n]
    all_ctrl_connected = all_ctrl_connected[:n]

def get_csv_file(idx):
    return f"/home/linaro/temp_log_ctrl{idx}.csv"

def load_schedule():
    if os.path.exists(SCHEDULE_FILE):
        with open(SCHEDULE_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_schedule(data):
    with open(SCHEDULE_FILE, 'w') as f:
        json.dump(data, f)

# ── Clock ─────────────────────────────────────────────────────────────────────
_CLK_WD = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
_CLK_MO = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def update_clock_data():
    """Refresh the shared clock_data from the current LOCAL time. Called every
       second by clock_thread, and immediately after any timezone/time change so
       the on-screen clock (and the Set-Time spinner that seeds off /api/clock)
       never lag a zone switch — that lag was what made the time appear to slip."""
    now = datetime.now()
    with clock_lock:
        clock_data["time"] = now.strftime("%I:%M:%S")
        clock_data["ampm"] = now.strftime("%p")
        clock_data["date"] = f"{_CLK_WD[now.weekday()]}, {_CLK_MO[now.month-1]} {now.day:02d} {now.year}"
        clock_data["y"]  = now.year;  clock_data["mo"] = now.month; clock_data["d"]  = now.day
        clock_data["h"]  = now.hour;  clock_data["mi"] = now.minute; clock_data["s"] = now.second

def clock_thread():
    while True:
        try:
            update_clock_data()
        except Exception as e:
            print(f"Clock error: {e}", flush=True)
        time.sleep(1)

# ── Screen ────────────────────────────────────────────────────────────────────
_xauth_cache = {"path": None, "ts": 0.0}

def _detect_xauthority():
    """Find the X authority cookie the *currently running* X server is using.

       'Invalid MIT-MAGIC-COOKIE-1 key' / 'Can't open display :0' means we handed
       X the wrong cookie — usually because X was restarted (fresh cookie) or we
       run as a different user than the one that owns the session. Reading the
       cookie straight off the live X server's own '-auth <file>' argument sidesteps
       all of that; we fall back to the common fixed locations if we can't."""
    # 1) Authoritative: the -auth file the running X/Xorg was started with.
    try:
        out = subprocess.run(['ps', '-eo', 'args'], capture_output=True,
                             text=True, timeout=5).stdout
        for line in out.splitlines():
            if '-auth' not in line:
                continue
            if ('Xorg' in line or 'X :' in line or '/X ' in line
                    or line.lstrip().startswith('X ') or 'Xwayland' in line):
                parts = line.split()
                if '-auth' in parts:
                    p = parts[parts.index('-auth') + 1]
                    if os.path.exists(p):
                        return p
    except Exception:
        pass
    # 2) Common fixed locations (first that exists wins).
    for p in ('/home/linaro/.Xauthority',
              '/run/user/1000/gdm/Xauthority',
              '/var/run/lightdm/root/:0',
              os.path.expanduser('~/.Xauthority')):
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            pass
    return '/home/linaro/.Xauthority'

def get_display_env():
    """Environment for launching X clients (Chromium, feh, xset, xrandr …) on the
       panel. Resolves XAUTHORITY from the live X server so a restarted X server
       or a different run-user never breaks the display connection. Result is
       cached briefly and re-detected if the file vanishes or X restarts."""
    env = dict(os.environ)
    env['DISPLAY'] = ':0'
    now = time.time()
    path = _xauth_cache["path"]
    if (path is None or not os.path.exists(path) or (now - _xauth_cache["ts"]) > 15):
        path = _detect_xauthority()
        _xauth_cache["path"] = path
        _xauth_cache["ts"]   = now
    env['XAUTHORITY'] = path
    return env

screen_on = True
overlay_proc = None

import subprocess
import tempfile
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

def _draw_screensaver_png(path="/tmp/screensaver.png"):
    # Same source the dashboard uses, so both always show the identical time/date.
    with clock_lock:
        time_txt = (clock_data.get("time", "") + " " + clock_data.get("ampm", "")).strip()
        date_txt = clock_data.get("date", "")
    W, H = 1280, 800
    bg = Image.new("RGB", (W, H), "black")
    logo = Image.open("/home/linaro/logo.png").convert("RGBA")
    logo_w = 500
    ratio = logo_w / logo.width
    logo = logo.resize((logo_w, int(logo.height * ratio)))
    bg.paste(logo, ((W - logo_w) // 2, (H - logo.height) // 2 - 80), logo)
    draw = ImageDraw.Draw(bg)
    try:
        font_time = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 64)
        font_date = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 36)
    except Exception:
        font_time = ImageFont.load_default(); font_date = ImageFont.load_default()
    tw = draw.textbbox((0, 0), time_txt, font=font_time)[2]
    draw.text(((W - tw) // 2, (H // 2) + 120), time_txt, fill="white", font=font_time)
    dw = draw.textbbox((0, 0), date_txt, font=font_date)[2]
    draw.text(((W - dw) // 2, (H // 2) + 200), date_txt, fill="white", font=font_date)
    tmp = path + ".tmp"
    bg.save(tmp, "PNG")
    os.replace(tmp, path)      # atomic: feh only ever sees a complete file

def _screensaver_refresher():
    last = None
    while not screen_on:
        time.sleep(1)
        if screen_on:
            break
        with clock_lock:
            cur = clock_data.get("time", "") + clock_data.get("date", "")
        if cur != last:
            try:
                _draw_screensaver_png()   # atomic write from edit last round
                last = cur
            except Exception:
                pass

def screen_sleep():
    global screen_on, overlay_proc
    if not screen_on:
        return
    env = get_display_env()
    _draw_screensaver_png()
    screen_on = False
    overlay_proc = subprocess.Popen(
        ["feh", "-F", "--hide-pointer", "--auto-zoom", "--reload", "1", "/tmp/screensaver.png"],
        env=env)
    threading.Thread(target=_screensaver_refresher, daemon=True).start()

def screen_wake():
    global screen_on, overlay_proc
    env = get_display_env()
    if overlay_proc:
        overlay_proc.terminate()
        overlay_proc = None
    subprocess.run(["xset", "dpms", "force", "on"], env=env)
    screen_on = True

def _wait_for_x(env, timeout=30):
    """Block until the X server on DISPLAY is actually accepting connections.
    On a full power-cycle reboot this service can start racing Xorg/the DSI-1
    panel driver — without this wait, the very first xrandr call below used to
    fail silently and the panel was left in its default portrait orientation."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = subprocess.run(['xset', 'q'], env=env, capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    print("[INIT] X server did not come up within timeout", flush=True)
    return False

def _wait_for_output(env, output_name, timeout=15):
    """Block until xrandr reports the given output (e.g. DSI-1) as connected,
    so the rotate command below isn't issued before the panel is enumerated."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = subprocess.run(['xrandr'], env=env, capture_output=True, text=True, timeout=3)
            if r.returncode == 0:
                for line in r.stdout.splitlines():
                    if line.startswith(output_name) and ' connected' in line:
                        return True
        except Exception:
            pass
        time.sleep(0.5)
    print(f"[INIT] output {output_name} never showed up as connected", flush=True)
    return False

def _apply_landscape_rotation(env, output_name='DSI-1', attempts=5):
    """Issue the rotate command and verify it actually took (xrandr reports
    the output as 'right' rotated), retrying a few times. Silent failures here
    were the cause of the panel booting into portrait after a full restart."""
    for attempt in range(1, attempts + 1):
        try:
            r = subprocess.run(['xrandr', '--output', output_name, '--rotate', 'right'],
                                env=env, capture_output=True, text=True, timeout=5)
        except FileNotFoundError:
            print("[INIT] missing tool: xrandr", flush=True)
            return False
        except Exception as e:
            r = None
            print(f"[INIT] xrandr rotate attempt {attempt} error: {e}", flush=True)
        # Verify — don't just trust the return code, confirm xrandr reports "right"
        try:
            v = subprocess.run(['xrandr', '--verbose'], env=env, capture_output=True, text=True, timeout=5)
            for line in v.stdout.splitlines():
                if line.startswith(output_name) and ' connected' in line and ' right ' in (line + ' '):
                    print(f"[INIT] Landscape rotation confirmed on attempt {attempt}", flush=True)
                    return True
        except Exception:
            pass
        if r is not None and r.returncode != 0:
            print(f"[INIT] xrandr rotate attempt {attempt} failed: {r.stderr.strip()}", flush=True)
        time.sleep(1)
    print("[INIT] Landscape rotation could NOT be confirmed after retries — leaving as-is", flush=True)
    return False

def init_screen():
    env = get_display_env()
    def safe(cmd):
        try:
            return subprocess.run(cmd, env=env, capture_output=True, text=True)
        except FileNotFoundError:
            print(f"[INIT] missing tool: {cmd[0]}", flush=True)
            return None

    # Wait for the X server, then for the DSI-1 output, before touching rotation —
    # this is the fix for "screen comes up in portrait after a restart": the old
    # code fired xrandr immediately on process start, which on a full reboot could
    # race Xorg/the panel driver and fail with nobody checking the result.
    if _wait_for_x(env):
        _wait_for_output(env, 'DSI-1')
        _apply_landscape_rotation(env, 'DSI-1')
    else:
        # Still attempt it — better than nothing if our readiness probe itself failed
        safe(['xrandr', '--output', 'DSI-1', '--rotate', 'right'])

    safe(['xset', 's', 'off'])
    safe(['xset', 's', 'noblank'])
    safe(['xset', '+dpms'])
    safe(['xset', 'dpms', '3600', '3600', '3600'])
    result = safe([
        'xinput', 'set-prop', 'pointer:goodix-ts',
        '--type=float', 'Coordinate Transformation Matrix',
        '0', '1', '0', '-1', '0', '1', '0', '0', '1'
    ])
    if result and result.returncode == 0:
        print("[INIT] Touchscreen rotation applied", flush=True)
    elif result:
        print(f"[INIT] xinput failed: {result.stderr}", flush=True)

# ── System cleanup ────────────────────────────────────────────────────────────
def system_cleanup_thread():
    while True:
        time.sleep(300)
        try:
            print("[CLEANUP] Running system cleanup...", flush=True)
            subprocess.run(['sudo', 'journalctl', '--vacuum-size=20M'], capture_output=True, timeout=30)
            subprocess.run(['sudo', 'apt', 'clean'], capture_output=True, timeout=30)
            cleared = 0
            for item in glob.glob('/tmp/*'):
                try:
                    if os.path.isfile(item) or os.path.islink(item):
                        os.remove(item); cleared += 1
                    elif os.path.isdir(item):
                        shutil.rmtree(item, ignore_errors=True); cleared += 1
                except Exception:
                    pass
            print(f"[CLEANUP] Done — cleared {cleared} /tmp items.", flush=True)
        except Exception as e:
            print(f"[CLEANUP] Error: {e}", flush=True)

# ── Motion Detection ──────────────────────────────────────────────────────────
def motion_thread():
    global last_motion, screen_on
    cap = None
    cam_devices = ["/dev/video8"] + VIDEO_FALLBACKS
    for dev in cam_devices:
        gst = ("v4l2src device="+dev+" io-mode=4 ! "
               "video/x-raw,format=NV12,width=640,height=480 ! "
               "videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1")
        try:
            c = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
            if c.isOpened():
                for _ in range(3): c.read()
                ok, test_frame = c.read()
                if ok and test_frame is not None and test_frame.size > 0:
                    cap = c
                    print(f"[CAMERA] Opened GStreamer {dev} ({test_frame.shape})", flush=True)
                    break
                c.release()
            c2 = cv2.VideoCapture(dev, cv2.CAP_V4L2)
            if c2.isOpened():
                c2.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                c2.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                for _ in range(3): c2.read()
                ok, test_frame = c2.read()
                if ok and test_frame is not None and test_frame.size > 0:
                    cap = c2
                    print(f"[CAMERA] Opened V4L2 {dev} ({test_frame.shape})", flush=True)
                    break
                c2.release()
        except Exception as e:
            print(f"[CAMERA] {dev} error: {e}", flush=True)
    if cap is None:
        print("[CAMERA] Cannot open any camera — motion disabled", flush=True)
        with motion_state_lock:
            motion_state["status_txt"] = "No camera"
        return

    with motion_state_lock:
        motion_state["status_txt"] = "Camera open — loading detector..."

    # ── Load face detector ──
    face_det = None
    if USE_FACE:
        try:
            cascade_paths = [
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml",
                "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml",
                "/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml",
                "/userdata/envi/pip_packages/lib/python3.9/site-packages/cv2/data/haarcascade_frontalface_default.xml",
            ]
            for cpath in cascade_paths:
                if os.path.exists(cpath):
                    d = cv2.CascadeClassifier(cpath)
                    if not d.empty():
                        face_det = d
                        print(f"[CAMERA] Face detector loaded: {cpath}", flush=True)
                        break
        except Exception as ex:
            print(f"[CAMERA] Face detector load error: {ex}", flush=True)

    with motion_state_lock:
        motion_state["status_txt"] = "Initializing motion detection..."

    # ── Per-thread state ──
    background     = None
    frame_count    = 0
    consecutive    = 0
    warmup_done    = False
    motion_history = [0.0] * 12
    last_wake      = "—"
    last_sample    = time.time()
    motion_pct     = 0.0
    changed        = 0
    face_count     = 0
    motion_detected = False
    status_txt     = "Initializing..."

    # ── Main loop ──
    while True:
        elapsed = time.time() - last_sample
        if elapsed < SAMPLING_RATE:
            time.sleep(SAMPLING_RATE - elapsed)
        last_sample = time.time()

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1); continue

        gray = cv2.cvtColor(cv2.resize(frame, (CAM_W, CAM_H)), cv2.COLOR_BGR2GRAY).astype(np.float32)
        frame_count += 1

        if background is None:
            background = gray.copy(); continue

        # Warm-up calibration
        if frame_count <= WARMUP_FRAMES:
            background = (1 - BG_ALPHA) * background + BG_ALPHA * gray
            pct = int(frame_count / WARMUP_FRAMES * 100)
            with motion_state_lock:
                motion_state["status_txt"]  = f"Calibrating… {pct}%"
                motion_state["warmup_done"] = False
                motion_state["frame_count"] = frame_count
            continue

        if not warmup_done:
            warmup_done = True
            last_motion = time.monotonic()
            print("[CAMERA] Calibration done — watching", flush=True)
            with motion_state_lock:
                motion_state["warmup_done"] = True

        # Motion detection
        B  = SKIP_BORDER
        fg = gray[B:CAM_H-B, B:CAM_W-B]
        bg = background[B:CAM_H-B, B:CAM_W-B]
        diff    = np.abs(fg - bg)
        changed = int(np.sum(diff > NOISE_THRESH))
        total   = fg.size
        background = (1 - BG_ALPHA) * background + BG_ALPHA * gray
        motion_pct = changed / total * 100
        motion_history.append(motion_pct)
        motion_history.pop(0)
        motion_detected = changed >= MIN_PIXELS

        # Face detection
        face_found = False
        face_count = 0
        if face_det is not None and (motion_detected or frame_count % 4 == 0):
            faces = face_det.detectMultiScale(
                gray.astype(np.uint8),
                scaleFactor=FACE_SCALE,
                minNeighbors=FACE_NEIGHBORS,
                minSize=FACE_MIN_SIZE,
            )
            face_found = len(faces) > 0
            face_count = len(faces)

        triggered = motion_detected or face_found
        if triggered:
            consecutive += 1
        else:
            consecutive = 0

        now = time.monotonic()
        if consecutive >= CONFIRM_FRAMES:
            last_motion = now
            last_wake = datetime.now().strftime("%H:%M:%S")
            if not screen_on:
                screen_wake()
            status_txt = (f"👤 {face_count} face(s) detected" if face_found else f"◉ Motion {motion_pct:.1f}%")
        elif warmup_done:
            idle   = now - last_motion
            remain = max(0, SCREEN_TIMEOUT - idle)
            status_txt = f"Watching… sleep in {remain:.1f}s"
        else:
            status_txt = "Initializing..."

        if motion_enabled and screen_on and (now - last_motion) > SCREEN_TIMEOUT:
            screen_sleep()

        with motion_state_lock:
            motion_state.update({
                "motion_pct":  round(motion_pct, 2),
                "changed_px":  changed,
                "face_count":  face_count,
                "consecutive": consecutive,
                "frame_count": frame_count,
                "warmup_done": warmup_done,
                "status_txt":  status_txt,
                "history":     [round(v, 1) for v in motion_history],
                "last_wake":   last_wake,
            })

        # ── Layer 2: feed debounced presence (inside the loop, after state update) ──
        if warmup_done:
            update_presence(face_count, motion_detected)

def init_occupancy_db():
    conn = sqlite3.connect(OCCUPANCY_DB)
    conn.execute("""CREATE TABLE IF NOT EXISTS occupancy(
        ts TEXT, weekday INTEGER, hour INTEGER, minute INTEGER,
        slot INTEGER, occupied INTEGER, face_count INTEGER)""")
    conn.commit(); conn.close()
def predict_occupancy():
    """Return {slot: probability} over last 14 days, plus per-weekday typical arrival."""
    try:
        cutoff = (datetime.now() - timedelta(days=14)).isoformat()
        conn = sqlite3.connect(OCCUPANCY_DB)
        rows = conn.execute(
            "SELECT slot, occupied FROM occupancy WHERE ts > ?", (cutoff,)).fetchall()
        conn.close()
        if not rows:
            return {"prob": {}, "arrivals": {}, "samples": 0}
        # occupancy probability per slot
        from collections import defaultdict
        tot = defaultdict(int); occ = defaultdict(int)
        for slot, o in rows:
            tot[slot]+=1; occ[slot]+=o
        prob = {s: occ[s]/tot[s] for s in tot}
        # typical arrival per weekday = earliest slot crossing 50% in work hours (6am-)
        arrivals = {}
        for wd in range(7):
            for q in range(96):
                slot = wd*96 + q
                hour = q//4
                if hour >= 6 and prob.get(slot,0) >= 0.5:
                    arrivals[wd] = {"hour": hour, "minute": (q%4)*15, "slot": slot}
                    break
        return {"prob": prob, "arrivals": arrivals, "samples": len(rows)}
    except Exception as e:
        print(f"[OCC] predict error: {e}", flush=True)
        return {"prob": {}, "arrivals": {}, "samples": 0}  
def occupancy_suggestion():
    """Suggest-only: returns a recommendation, does NOT write to Modbus."""
    pred = predict_occupancy()
    now = datetime.now()
    cur_slot = now.weekday()*96 + now.hour*4 + now.minute//15
    # look at next 15 min (next slot) to catch pre-arrival
    next_slot = (cur_slot + 1) % 672
    p_now  = pred["prob"].get(cur_slot, 0)
    p_next = pred["prob"].get(next_slot, 0)

    pres = get_presence()
    actually_occupied = pres["present"]            # ← Layer 2, not raw face count
    manual = occupancy_manual_override["active"] and time.monotonic() < occupancy_manual_override["until"]

    if actually_occupied or manual:
        state, action = "OCCUPIED", "Maintain comfort setpoint"
    elif p_next >= 0.5:
        state, action = "ARRIVING_SOON", "Pre-condition now — people expected within 15 min"
    elif p_now < 0.3:
        state, action = "LIKELY_EMPTY", "Recommend setback ±3°C to save energy"
    else:
        state, action = "UNCERTAIN", "Hold current setpoint"

    today_arrival = pred["arrivals"].get(now.weekday())
    return {
        "state": state, "action": action,
        "prob_now": round(p_now,2), "prob_next": round(p_next,2),
        "actually_occupied": actually_occupied, "manual_override": manual,
        "typical_arrival_today": today_arrival,
        "samples": pred["samples"],
        "days_learned": min(14, pred["samples"]//288 if pred["samples"] else 0),
    }  
def occupancy_logger_thread():
    """Every 5 min, snapshot occupancy state to SQLite."""
    init_occupancy_db()
    while True:
        try:
            now = datetime.now()
            with motion_state_lock:
                ms = dict(motion_state)
            pres = get_presence()              # ← Layer 2 debounced verdict
            occ  = 1 if pres["present"] else 0
            slot = now.weekday()*96 + now.hour*4 + now.minute//15   # 15-min slot of week (0-671)
            conn = sqlite3.connect(OCCUPANCY_DB)
            conn.execute("INSERT INTO occupancy VALUES(?,?,?,?,?,?,?)",
                (now.isoformat(), now.weekday(), now.hour, now.minute, slot, occ, ms.get("face_count",0)))
            conn.commit(); conn.close()
        except Exception as e:
            print(f"[OCC] log error: {e}", flush=True)
        time.sleep(300)   # 5 min



# ── Economy Cycle ──────────────────────────────────────────────────────────
def economy_cycle_thread():
    while True:
        try:
            # outside air temp from weather cache
            oat = None
            if '_weather_cache' in globals() and _weather_cache.get('data'):
                oat = _weather_cache['data'].get('temp')
            cfg = load_device_config()
            with lock: c = client
            if c is None:
                time.sleep(5); continue

            for i, ctrl in enumerate(cfg):
                if i >= len(all_ctrl_connected) or not all_ctrl_connected[i]:
                    continue
                slave = ctrl['slave_id']
                roles = roles_for(ctrl)
                sc    = roles.get('scale', 1)

                # ECONOMY OFF for this controller → ensure AC restored, damper closed
                if not economy_enabled.get(i):
                    if economy_active.get(i):
                        # was in free-cool, now turning economy off → turn AC back ON, close damper
                        try:
                            c.write_register(ECONOMY_DAMPER_REG, 0, unit=slave)
                            c.write_register(roles['onoff'], 1, unit=slave)
                            print(f"[ECONOMY] ctrl{i} OFF → AC restored ON", flush=True)
                        except Exception as e:
                            print(f"[ECONOMY] ctrl{i} restore error: {e}", flush=True)
                        economy_active[i] = False
                    continue

                inside = all_ctrl_data[i].get(roles['temp'])
                sp     = all_ctrl_data[i].get(roles['setpoint'])
                if inside is None or sp is None or oat is None:
                    continue
                inside *= sc; sp *= sc

                cooling_demand = inside > sp
                free_cool_ok   = oat < (inside - ECONOMY_HYSTERESIS)

                if cooling_demand and free_cool_ok:
                    if not economy_active.get(i):
                        try:
                            c.write_register(ECONOMY_DAMPER_REG, 1, unit=slave)  # open damper
                            c.write_register(roles['onoff'], 0, unit=slave)      # compressor off
                            print(f"[ECONOMY] ctrl{i} free-cool ON (out {oat:.1f} < in {inside:.1f})", flush=True)
                        except Exception as e:
                            print(f"[ECONOMY] ctrl{i} engage error: {e}", flush=True)
                        economy_active[i] = True
                else:
                    if economy_active.get(i):
                        # conditions no longer good → close damper, AC back on
                        try:
                            c.write_register(ECONOMY_DAMPER_REG, 0, unit=slave)
                            c.write_register(roles['onoff'], 1, unit=slave)
                            print(f"[ECONOMY] ctrl{i} free-cool OFF → AC ON", flush=True)
                        except Exception as e:
                            print(f"[ECONOMY] ctrl{i} disengage error: {e}", flush=True)
                        economy_active[i] = False
        except Exception as e:
            print(f"[ECONOMY] error: {e}", flush=True)
        time.sleep(15)

# ── Modbus ────────────────────────────────────────────────────────────────────
def connect_modbus():
    global client, modbus_status
    c = ModbusClient(method="rtu", port=PORT, baudrate=BAUDRATE,
                     parity=PARITY, stopbits=STOPBITS, bytesize=BYTESIZE, timeout=TIMEOUT)
    if c.connect():
        with lock:
            client = c
            modbus_status = f"Connected to {PORT}"
        print(f"Modbus connected: {PORT}", flush=True)
        return True
    modbus_status = f"Failed to open {PORT}"
    return False

def poll_modbus():
    global all_ctrl_data, all_ctrl_status, all_ctrl_connected, modbus_status, fault_flag
    FAULT_VALUE = 0   # value on the fault register (1112/UI4) that means FAULT
    while True:
        try:
            with lock: c = client
            if c is None:
                time.sleep(2); connect_modbus(); continue
            
            cfg = load_device_config()
            ensure_ctrl_lists(len(cfg))
            
            for i, ctrl in enumerate(cfg):
                if not ctrl['enabled']:
                    all_ctrl_connected[i] = False
                    all_ctrl_status[i]    = "Disabled"
                    continue

                # ── TEMPERZONE (only controller type) ──
                roles = REG_ROLES["Temperzone"]
                sid   = ctrl["slave_id"]
                scale = roles["scale"]
                # Hold the bus lock for this whole burst of reads. Previously the lock only
                # guarded grabbing the `client` reference, so a /write request from the
                # dashboard could interleave its Modbus transaction in the middle of this
                # controller's read burst on the shared half-duplex RS485 line — corrupting
                # or dropping frames. That was the real cause of both the flaky Coil 56 fault
                # badge and the heating setpoint not reliably following the cooling setpoint.
                with lock:
                    temp_raw = read_field(client, roles, "temp", sid)
                    onoff    = read_field(client, roles, "onoff", sid)   # Coil 1 (BMS)
                    onoff2   = read_field(client, roles, "onoff2", sid)  # Coil 2 (keyboard)
                    fault    = read_field(client, roles, "fault", sid)   # Coil 56 (alarm)
                    if fault is None:
                        # Comms glitch (timeout/CRC) reading Coil 56 — do NOT assume "Normal".
                        # Hold the last known state so a transient RS485 hiccup can't hide an
                        # active fault badge on the dashboard.
                        fault = last_fault_coil.get(i, 0)
                    else:
                        last_fault_coil[i] = fault
                    status   = read_field(client, roles, "status", sid)  # IR135 status enum
                    sched_en = read_field(client, roles, "sched", sid)   # Coil 22 (Enable Scheduler)
                    sp_cool  = read_field(client, roles, "setpoint", sid)
                    sp_heat  = read_field(client, roles, "setpoint_heat", sid)
                    mode     = read_field(client, roles, "mode", sid)
                    fan      = read_field(client, roles, "fan", sid)
                    fan_mode = read_field(client, roles, "fan_mode", sid)  # HR116 (Fan Speed Mode)
                # temp: signed 16-bit (65238 -> -298 = -29.8C, the no-sensor sentinel)
                if temp_raw is not None and temp_raw > 32767:
                    temp_raw = temp_raw - 65536
                # unified on/off: ON if either BMS(Coil1) or keyboard(Coil2) is on
                if onoff is None and onoff2 is None:
                    eff = None
                else:
                    eff = 1 if ((onoff == 1) or (onoff2 == 1)) else 0
                new_data = {}
                new_data[1]            = temp_raw
                new_data[100]          = sp_cool
                new_data[102]          = sp_heat
                new_data[114]          = fan
                new_data[116]          = fan_mode
                new_data[117]          = mode
                new_data[135]          = status
                new_data["onoff"]      = eff                # unified on/off
                new_data["onoff_bms"]  = onoff              # Coil 1 raw
                new_data["onoff_kbd"]  = onoff2             # Coil 2 raw
                new_data["_coil1"]     = onoff
                new_data["_coil2"]     = onoff2
                new_data["fault_coil"] = fault if fault is not None else 0   # Coil 56
                new_data["sched_enable"] = sched_en          # Coil 22 (Enable Scheduler)
                new_data["status_code"]= status
                new_data["co_level"]   = None
                new_data["oat_value"]  = None
                connected = (onoff is not None) or (temp_raw is not None) or (status is not None)
                all_ctrl_data[i]      = new_data
                all_ctrl_connected[i] = connected
                all_ctrl_status[i]    = "Connected" if connected else "Error"
                # fault flag: Coil 56 = 1 means alarm active (change #4)
                if fault == 1 and not fault_flag:
                    fault_flag = True
                    print(f"[FAULT] {ctrl.get('name')} Coil56 alarm ACTIVE", flush=True)
                elif fault == 0 and fault_flag:
                    fault_flag = False
                continue

             # flush serial buffer before next controller
            time.sleep(0.1)
            try:
                if c.socket:
                    c.socket.reset_input_buffer()
                    c.socket.reset_output_buffer()
            except Exception:
                pass

            primary = next((i for i in range(len(cfg)) if all_ctrl_connected[i]), 0)
            trend_p = TREND_PRIMARY_CTRL if (TREND_PRIMARY_CTRL < len(cfg) and all_ctrl_connected[TREND_PRIMARY_CTRL]) else primary
            with history_lock:
                now_dt  = datetime.now()
                now_str = now_dt.strftime("%I:%M:%S %p")
                trp_roles = roles_for(cfg[trend_p]) if trend_p < len(cfg) else REG_ROLES["_default"]
                trp_scale = trp_roles.get("scale", 1)
                t_raw   = all_ctrl_data[trend_p].get(trp_roles["temp"])
                s_raw   = all_ctrl_data[trend_p].get(trp_roles["setpoint"])
                t_val   = round(t_raw * trp_scale, 1) if t_raw is not None else 0
                s_val   = round(s_raw * trp_scale, 1) if s_raw is not None else 0
                time_history.append(now_str)
                temp_history.append(t_val)
                setpoint_history.append(s_val)
                # per-controller temps for multi-zone overlay
                cur_multi = {}
                for ci in range(len(cfg)):
                    if ci not in multi_temp_history:
                        multi_temp_history[ci] = deque(maxlen=HISTORY_MAXLEN)
                    cr = roles_for(cfg[ci]); csc = cr.get("scale", 1)
                    craw = all_ctrl_data[ci].get(cr["temp"]) if ci < len(all_ctrl_data) else None
                    cval = round(craw * csc, 1) if craw is not None else None
                    multi_temp_history[ci].append(cval)
                    cur_multi[ci] = cval
                # roll these 1-second samples into the minute-by-minute chart buffer
                try:
                    accumulate_minute(now_dt, t_val, s_val, cur_multi)
                except Exception as _e:
                    print(f"[TREND] minute accum error: {_e}", flush=True)
            connected_count = sum(all_ctrl_connected[:len(cfg)])
            modbus_status = f"{connected_count}/{sum(c['enabled'] for c in cfg)} online"
        except Exception as e:
            modbus_status = f"Bus error: {e}"
            print(f"Poll error: {e}", flush=True)
            time.sleep(2); connect_modbus()
        time.sleep(SCAN_INTERVAL)

# ── Scheduler ─────────────────────────────────────────────────────────────────
def scheduler_engine():
    last_triggered = set()
    while True:
        try:
            sched = load_schedule(); cfg = load_device_config()
            now = datetime.now()
            day_str = now.strftime("%a"); current_time = now.strftime("%H:%M")
            with lock: c = client
            for i, ctrl in enumerate(cfg):
                if not ctrl['enabled'] or i >= len(all_ctrl_connected) or not all_ctrl_connected[i]: continue
                ctrl_sched = sched.get(str(i), {})
                if i >= len(all_ctrl_data) or all_ctrl_data[i].get("sched_enable") != 1: continue
                for event in ctrl_sched.get(day_str, []):
                    key = f"{i}-{day_str}-{event['time']}-{event['action']}"
                    if event["time"] == current_time and key not in last_triggered:
                        if c: c.write_coil(1, bool(event["action"]), unit=ctrl['slave_id'])   # Coil 1 On/Off (0=off,1=on)
                        last_triggered.add(key)
            if now.second == 0: last_triggered.clear()
        except Exception as e:
            print(f"Sched Error: {e}")
        time.sleep(10)

# ── LED ───────────────────────────────────────────────────────────────────────
def wreg(mem, off, mask, val):
    mem[off:off+4] = struct.pack('<I', (mask << 16) | (val & mask))

def led_logic_thread():
    global current_color, led_mode, led_ctrl_idx
    last_mode = None
    while True:
        try:
            if led_mode != last_mode:
                print(f"[LED] Mode changed: {last_mode} → {led_mode}, ctrl={led_ctrl_idx}", flush=True)
                last_mode = led_mode
            if led_mode == 'off':
                with lock_led: current_color = (0, 0, 0)
                time.sleep(1); continue
            cfg = load_device_config()
            primary = led_ctrl_idx if (led_ctrl_idx < len(all_ctrl_connected) and all_ctrl_connected[led_ctrl_idx]) \
            else next((i for i in range(len(cfg)) if i < len(all_ctrl_connected) and all_ctrl_connected[i]), 0)
            if led_mode == 'iaq':
                co = all_ctrl_data[primary].get('co_level') if all_ctrl_data else None
                if co is None: time.sleep(1); continue
                # Green <300, Yellow 300-700 (use red+green), Red >700
                if co < 300:   color = (0, 1, 0)
                elif co < 700: color = (1, 1, 0)  # amber
                else:          color = (1, 0, 0)
            else:  # temp mode
                roles = roles_for(cfg[primary]) if primary < len(cfg) else REG_ROLES["_default"]
                rt = all_ctrl_data[primary].get(roles["temp"])     if all_ctrl_data else None
                rs = all_ctrl_data[primary].get(roles["setpoint"]) if all_ctrl_data else None
                if rt is None or rs is None: time.sleep(1); continue
                color = (0, 1, 0) if rs > rt else (1, 0, 0) if rt > rs else (0, 0, 1)
            with lock_led: current_color = color
        except Exception as e:
            print(f"LED error: {e}", flush=True)
        time.sleep(1)
LED_COLOR_FILE = "/run/led_color"

def led_pwm_thread():
    last = None
    while True:
        try:
            with lock_led:
                c = current_color
            if c != last:
                with open("/run/led_color", "w") as f:
                    f.write(f"{c[0]} {c[1]} {c[2]}")
                last = c
        except Exception as e:
            print(f"[LED] write error: {e}", flush=True)
        time.sleep(0.2)


# ── Alarm ─────────────────────────────────────────────────────────────────────
def play_alarm_native():
    try:
        proc = subprocess.Popen(["paplay","/data/local/tmp/alert.wav"])
        time.sleep(3); subprocess.run(["pkill","paplay"])
    except Exception as e: print(f"Audio error: {e}")

def alarm_thread():
    last_state=False; alarm_played_at=0
    try:
        subprocess.run(["pactl","set-sink-volume","@DEFAULT_SINK@","75%"])
        subprocess.run(["pactl","set-sink-mute","@DEFAULT_SINK@","0"])
    except: pass
    while True:
        try:
            cfg = load_device_config()
            primary = next((i for i in range(len(cfg)) if i < len(all_ctrl_connected) and all_ctrl_connected[i]), 0)
            vt = all_ctrl_data[primary].get(5) if all_ctrl_data else None
            vs = all_ctrl_data[primary].get(4) if all_ctrl_data else None
            if vt is not None and vs is not None:
                now=time.monotonic()
                if vt>vs and not last_state and (now-alarm_played_at)>=300:
                    last_state=True; alarm_played_at=now
                    threading.Thread(target=play_alarm_native,daemon=True).start()
                elif vt<=(vs-3): last_state=False
        except Exception as e: print(f"Alarm error: {e}", flush=True)
        time.sleep(2)

# ── Logging ───────────────────────────────────────────────────────────────────
def _write_ctrl_log(ctrl_idx, temp_raw, sp_raw):
    csv_file  = get_csv_file(ctrl_idx)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:00")
    temp_val  = f"{temp_raw:.2f}"; sp_val = f"{sp_raw:.2f}"
    rows = []
    if os.path.exists(csv_file) and os.path.getsize(csv_file) > 0:
        with open(csv_file, "r", newline='', encoding='utf-8') as f:
            reader = list(csv.reader(f))
            if reader: rows = reader[1:]
    updated = False
    for row in rows:
        if row[0] == timestamp: row[1], row[2] = temp_val, sp_val; updated = True; break
    if not updated: rows.append([timestamp, temp_val, sp_val])
    # Keep a bounded rolling window (~3 days of minute rows). This retains enough
    # history for any Time Window while stopping the file — and the per-minute
    # rewrite below — from growing without limit.
    MAX_LOG_ROWS = 4400
    if len(rows) > MAX_LOG_ROWS:
        rows = rows[-MAX_LOG_ROWS:]
    with open(csv_file, "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(HEADER); w.writerows(rows)

def _write_ctrl_log_legacy(temp_val, sp_val):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:00")
    temp_str  = f"{temp_val:.2f}"; sp_str = f"{sp_val:.2f}"
    rows = []
    if os.path.exists(CSV_FILE) and os.path.getsize(CSV_FILE) > 0:
        with open(CSV_FILE, "r", newline='', encoding='utf-8') as f:
            reader = list(csv.reader(f))
            if reader: rows = reader[1:]
    updated = False
    for row in rows:
        if row[0] == timestamp: row[1], row[2] = temp_str, sp_str; updated = True; break
    if not updated: rows.append([timestamp, temp_str, sp_str])
    with open(CSV_FILE, "w", newline='', encoding='utf-8') as f:
        w = csv.writer(f); w.writerow(HEADER); w.writerows(rows)

def log_data_thread():
    while True:
        try:
            cfg = load_device_config()
            for i in range(len(cfg)):
                if not cfg[i]['enabled'] or i >= len(all_ctrl_connected) or not all_ctrl_connected[i]: continue
                roles = roles_for(cfg[i])
                sc    = roles.get("scale", 1)
                tr_raw = all_ctrl_data[i].get(roles['temp'])
                sr_raw = all_ctrl_data[i].get(roles['setpoint'])
                if tr_raw is not None and sr_raw is not None:
                    _write_ctrl_log(i, tr_raw * sc, sr_raw * sc)
            primary = next((i for i in range(len(cfg)) if i < len(all_ctrl_connected) and all_ctrl_connected[i]), 0)
            if primary < len(cfg) and all_ctrl_data:
                proles = roles_for(cfg[primary])
                psc    = proles.get("scale", 1)
                tr = all_ctrl_data[primary].get(proles['temp'])
                sr = all_ctrl_data[primary].get(proles['setpoint'])
                if tr is not None and sr is not None:
                    _write_ctrl_log_legacy(tr * psc, sr * psc)
        except Exception as e: print("Log error:", e)
        time.sleep(60)

def fault_data_thread():
    global fault_flag
    while True:
        try:
            if fault_flag:
                cfg = load_device_config()
                primary = next((i for i in range(len(cfg)) if i < len(all_ctrl_connected) and all_ctrl_connected[i]), 0)
                tr = all_ctrl_data[primary].get(5) if all_ctrl_data else None
                sr = all_ctrl_data[primary].get(4) if all_ctrl_data else None
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                tv = f"{tr:.2f}" if tr else "N/A"; sv = f"{sr*0.1:.2f}" if sr else "N/A"
                wh = not os.path.exists(FAULT_FILE) or os.path.getsize(FAULT_FILE)==0
                with open(FAULT_FILE,"a",newline='',encoding='utf-8') as f:
                    w = csv.writer(f)
                    if wh: w.writerow(FAULT_HEADER)
                    w.writerow([ts,tv,sv])
        except Exception as e: print("Fault log error:",e)
        time.sleep(60)

# ── Email ─────────────────────────────────────────────────────────────────────
EMAIL_HISTORY_MINUTES = 30    # updated live by /api/trend_config
TREND_PRIMARY_CTRL    = 0     # which controller feeds the chart
DISPLAY_HISTORY_LEN   = 30    # informational only — actual windowing happens client-side now

def send_csv_email(ctrl='all', receiver=''):
    import io
    sender   = "envi@insightcontrol.net.au"
    password = "Envialerts-789"
    if not receiver:
        return {"status": "error", "msg": "No recipient email entered"}
    cfg = load_device_config()
    if str(ctrl) == 'all':
        ctrl_indices = [i for i in range(len(cfg)) if cfg[i]['enabled']]
        subject_label = "All Controllers"
    else:
        idx = int(ctrl)
        ctrl_indices = [idx]
        subject_label = cfg[idx]['name'] if idx < len(cfg) else f"Controller {idx+1}"
    cutoff = datetime.now() - timedelta(minutes=EMAIL_HISTORY_MINUTES)
    msg = EmailMessage()
    msg["Subject"] = f"Temp Log — {subject_label} — Last {EMAIL_HISTORY_MINUTES}min"
    msg["From"] = sender; msg["To"] = receiver

    combined_buf = io.StringIO()
    writer = csv.writer(combined_buf)
    total_rows = 0
    per_ctrl_counts = []

    for i in ctrl_indices:
        csv_file = get_csv_file(i)
        if not os.path.exists(csv_file):
            if i == 0 and os.path.exists(CSV_FILE): csv_file = CSV_FILE
            else:
                per_ctrl_counts.append((cfg[i]['name'], 0))
                continue
        try:
            rows = []
            with open(csv_file, "r", newline='', encoding='utf-8') as f:
                for row in list(csv.reader(f))[1:]:
                    try:
                        if datetime.strptime(row[0], "%Y-%m-%d %H:%M:00") >= cutoff: rows.append(row)
                    except: continue
            writer.writerow([f"=== {cfg[i]['name']} ({cfg[i]['type']} · Slave ID {cfg[i]['slave_id']}) ==="])
            writer.writerow(HEADER)
            for row in rows: writer.writerow(row)
            writer.writerow([])
            total_rows += len(rows)
            per_ctrl_counts.append((cfg[i]['name'], len(rows)))
        except Exception as e:
            print(f"Email prep error ctrl {i}: {e}")
            per_ctrl_counts.append((cfg[i]['name'], 0))

    if total_rows == 0:
        return {"status": "error", "msg": "No log data found"}

    summary_lines = "\n".join(f"  {name}: {count} records" for name, count in per_ctrl_counts)
    msg.set_content(
        f"Temperature log — {subject_label}\n"
        f"Period: Last {EMAIL_HISTORY_MINUTES} minutes\n"
        f"Total: {total_rows} records\n\n"
        f"Breakdown:\n{summary_lines}\n"
    )

    fname = "temp_log_all_controllers.csv" if str(ctrl) == 'all' else f"temp_log_{cfg[ctrl_indices[0]]['name'].replace(' ','_')}.csv"
    msg.add_attachment(combined_buf.getvalue().encode('utf-8'), maintype="text", subtype="csv", filename=fname)

    try:
        def _send():
            with smtplib.SMTP_SSL("s3367.syd1.stableserver.net", 465, timeout=10) as smtp:
                smtp.login(sender, password); smtp.send_message(msg)
        threading.Thread(target=_send, daemon=True).start()
        return {"status": "sent", "rows": total_rows, "label": subject_label}
    except Exception as e:
        return {"status": "error", "msg": str(e)}

# ── Network info ──────────────────────────────────────────────────────────────
# ── QR encoder (byte mode, ECC L/M, versions 1-10) ───────────────────────────
# Self-contained on purpose: python-qrcode isn't installed on the panel and a
# remote site may have no internet for a CDN JS library. Verified byte-for-byte
# against python-qrcode across 1/2/4/5-block versions, and decode-tested with
# cv2.QRCodeDetector.
# (ec_per_block, blocks_g1, data_g1, blocks_g2, data_g2)
_ECC = {
    'L': {1:(7,1,19,0,0), 2:(10,1,34,0,0), 3:(15,1,55,0,0), 4:(20,1,80,0,0),
          5:(26,1,108,0,0), 6:(18,2,68,0,0), 7:(20,2,78,0,0), 8:(24,2,97,0,0),
          9:(30,2,116,0,0), 10:(18,2,68,2,69)},
    'M': {1:(10,1,16,0,0), 2:(16,1,28,0,0), 3:(26,1,44,0,0), 4:(18,2,32,0,0),
          5:(24,2,43,0,0), 6:(16,4,27,0,0), 7:(18,4,31,0,0), 8:(22,2,38,2,39),
          9:(22,3,36,2,37), 10:(26,4,43,1,44)},
}
_ALIGN = {1:[], 2:[6,18], 3:[6,22], 4:[6,26], 5:[6,30], 6:[6,34],
          7:[6,22,38], 8:[6,24,42], 9:[6,26,46], 10:[6,28,50]}
_FMT_MASK = 0b101010000010010
_ECC_BITS = {'L':0b01, 'M':0b00, 'Q':0b11, 'H':0b10}

# ── GF(256) ──
_EXP = [0]*512; _LOG = [0]*256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _x <<= 1
    if _x & 0x100:
        _x ^= 0x11D
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i-255]

def _gmul(a, b):
    if a == 0 or b == 0: return 0
    return _EXP[_LOG[a] + _LOG[b]]

def _rs_gen(n):
    g = [1]
    for i in range(n):
        g2 = [0]*(len(g)+1)
        for j, c in enumerate(g):
            g2[j]   ^= _gmul(c, 1)
            g2[j+1] ^= _gmul(c, _EXP[i])
        g = g2
    return g

def _rs_encode(data, n):
    gen = _rs_gen(n)
    res = list(data) + [0]*n
    for i in range(len(data)):
        coef = res[i]
        if coef:
            for j in range(1, len(gen)):
                res[i+j] ^= _gmul(gen[j], coef)
    return res[len(data):]

def _bch_format(fmt):
    v = fmt << 10
    for i in range(4, -1, -1):
        if v & (1 << (i+10)):
            v ^= 0b10100110111 << i
    return ((fmt << 10) | v) ^ _FMT_MASK

def _bch_version(ver):
    v = ver << 12
    for i in range(5, -1, -1):
        if v & (1 << (i+12)):
            v ^= 0b1111100100101 << i
    return (ver << 12) | v

def _capacity(ver, ecc):
    e, b1, d1, b2, d2 = _ECC[ecc][ver]
    return b1*d1 + b2*d2

def _pick_version(nbytes, ecc):
    for v in range(1, 11):
        cc = 8 if v <= 9 else 16
        need = (4 + cc + nbytes*8 + 7) // 8
        if need <= _capacity(v, ecc):
            return v
    raise ValueError("data too long for version <=10")

def _encode_data(text, ver, ecc):
    data = text.encode('utf-8')
    cc   = 8 if ver <= 9 else 16
    bits = []
    def put(val, n):
        for i in range(n-1, -1, -1):
            bits.append((val >> i) & 1)
    put(0b0100, 4)          # byte mode
    put(len(data), cc)
    for b in data:
        put(b, 8)
    cap_bits = _capacity(ver, ecc) * 8
    put(0, min(4, cap_bits - len(bits)))          # terminator
    while len(bits) % 8:
        bits.append(0)
    cws = [int(''.join(map(str, bits[i:i+8])), 2) for i in range(0, len(bits), 8)]
    pad = [0xEC, 0x11]; k = 0
    while len(cws) < _capacity(ver, ecc):
        cws.append(pad[k % 2]); k += 1
    return cws

def _interleave(cws, ver, ecc):
    e, b1, d1, b2, d2 = _ECC[ecc][ver]
    blocks, p = [], 0
    for _ in range(b1):
        blocks.append(cws[p:p+d1]); p += d1
    for _ in range(b2):
        blocks.append(cws[p:p+d2]); p += d2
    ecs = [_rs_encode(b, e) for b in blocks]
    out = []
    for i in range(max(len(b) for b in blocks)):
        for b in blocks:
            if i < len(b): out.append(b[i])
    for i in range(e):
        for b in ecs:
            out.append(b[i])
    return out

def _make_matrix(ver):
    size = 17 + 4*ver
    m = [[None]*size for _ in range(size)]
    def finder(r, c):
        for dr in range(-1, 8):
            for dc in range(-1, 8):
                rr, cc = r+dr, c+dc
                if 0 <= rr < size and 0 <= cc < size:
                    inring = (0 <= dr <= 6 and 0 <= dc <= 6)
                    if inring:
                        v = (dr in (0,6) or dc in (0,6) or (2 <= dr <= 4 and 2 <= dc <= 4))
                    else:
                        v = False
                    m[rr][cc] = 1 if v else 0
    finder(0, 0); finder(0, size-7); finder(size-7, 0)
    for i in range(8, size-8):
        m[6][i] = 1 if i % 2 == 0 else 0
        m[i][6] = 1 if i % 2 == 0 else 0
    # Alignment patterns sit at every combination of the version's coordinates,
    # EXCEPT the three corners occupied by finder patterns. Relying on "cell
    # already set" to skip them happens to work for v2-v6 (timing pattern gets
    # in the way) but wrongly skips valid patterns like (6,22) on v7+.
    coords = _ALIGN[ver]
    if coords:
        lo, hi = coords[0], coords[-1]
        for r in coords:
            for c in coords:
                if (r, c) in ((lo, lo), (lo, hi), (hi, lo)):
                    continue                    # finder corners
                for dr in range(-2, 3):
                    for dc in range(-2, 3):
                        v = (abs(dr) == 2 or abs(dc) == 2 or (dr == 0 and dc == 0))
                        m[r+dr][c+dc] = 1 if v else 0
    m[size-8][8] = 1                       # dark module
    for i in range(9):                     # reserve format
        if m[8][i] is None: m[8][i] = 0
        if m[i][8] is None: m[i][8] = 0
    for i in range(8):
        if m[8][size-1-i] is None: m[8][size-1-i] = 0
        if m[size-1-i][8] is None: m[size-1-i][8] = 0
    if ver >= 7:
        for i in range(6):
            for j in range(3):
                m[size-11+j][i] = 0
                m[i][size-11+j] = 0
    return m, size

def _place(m, size, bits, ver):
    reserved = [[m[r][c] is not None for c in range(size)] for r in range(size)]
    idx, up = 0, True
    col = size - 1
    while col > 0:
        if col == 6: col -= 1
        rows = range(size-1, -1, -1) if up else range(size)
        for r in rows:
            for c in (col, col-1):
                if not reserved[r][c]:
                    m[r][c] = bits[idx] if idx < len(bits) else 0
                    idx += 1
        up = not up
        col -= 2
    return reserved

_MASKS = [
    lambda r, c: (r + c) % 2 == 0,
    lambda r, c: r % 2 == 0,
    lambda r, c: c % 3 == 0,
    lambda r, c: (r + c) % 3 == 0,
    lambda r, c: (r//2 + c//3) % 2 == 0,
    lambda r, c: (r*c) % 2 + (r*c) % 3 == 0,
    lambda r, c: ((r*c) % 2 + (r*c) % 3) % 2 == 0,
    lambda r, c: ((r+c) % 2 + (r*c) % 3) % 2 == 0,
]

def _penalty(m, size):
    p = 0
    for r in range(size):
        for run in (m[r], [m[i][r] for i in range(size)]):
            cnt, prev = 1, run[0]
            for v in run[1:]:
                if v == prev: cnt += 1
                else:
                    if cnt >= 5: p += 3 + (cnt-5)
                    cnt, prev = 1, v
            if cnt >= 5: p += 3 + (cnt-5)
    for r in range(size-1):
        for c in range(size-1):
            if m[r][c] == m[r][c+1] == m[r+1][c] == m[r+1][c+1]: p += 3
    # Rule 3: finder-lookalike sequences confuse scanners. The spec penalises
    # BOTH 10111010000 and its mirror 00001011101 — checking only one lets a
    # scan-hostile mask win on score (mask 4 was chosen and wouldn't decode).
    p1 = [1,0,1,1,1,0,1,0,0,0,0]
    p2 = [0,0,0,0,1,0,1,1,1,0,1]
    for r in range(size):
        row = m[r]; col = [m[i][r] for i in range(size)]
        for seq in (row, col):
            for i in range(size-10):
                w = seq[i:i+11]
                if w == p1 or w == p2: p += 40
    dark = sum(sum(r) for r in m)
    pct = dark*100 // (size*size)
    p += 10 * (abs(pct-50)//5)
    return p

def qr_matrix(text, ecc='M'):
    ver  = _pick_version(len(text.encode('utf-8')), ecc)
    cws  = _encode_data(text, ver, ecc)
    fin  = _interleave(cws, ver, ecc)
    bits = []
    for b in fin:
        for i in range(7, -1, -1):
            bits.append((b >> i) & 1)
    base, size = _make_matrix(ver)
    reserved = [[base[r][c] is not None for c in range(size)] for r in range(size)]
    m = [row[:] for row in base]
    _place(m, size, bits, ver)
    best, best_p = None, None
    for mi, fn in enumerate(_MASKS):
        t = [row[:] for row in m]
        for r in range(size):
            for c in range(size):
                if not reserved[r][c] and fn(r, c):
                    t[r][c] ^= 1
        fmt = _bch_format((_ECC_BITS[ecc] << 3) | mi)
        _put_format(t, size, fmt)
        if ver >= 7: _put_version(t, size, _bch_version(ver))
        pen = _penalty(t, size)
        if best_p is None or pen < best_p:
            best, best_p = t, pen
    return best, size

def _put_format(m, size, fmt):
    # Format info is placed MSB-first: bit 14 goes at (8,0), bit 0 at (0,8).
    # Writing it LSB-first produces a structurally perfect but undecodable code.
    for i in range(15):
        b = (fmt >> (14 - i)) & 1
        if i < 6:      m[8][i] = b
        elif i == 6:   m[8][7] = b
        elif i == 7:   m[8][8] = b
        elif i == 8:   m[7][8] = b
        else:          m[14-i][8] = b
    # 2nd copy: bit 14 starts at (size-1, 8) running up (7 cells), then bits 7-0
    # run along row 8 from (8, size-8) to (8, size-1). The dark module owns
    # (size-8, 8) — a format bit there corrupts the code.
    for i in range(15):
        b = (fmt >> (14 - i)) & 1
        if i < 7:      m[size-1-i][8] = b
        else:          m[8][size-15+i] = b
    m[size-8][8] = 1

def _put_version(m, size, vinfo):
    for i in range(18):
        b = (vinfo >> i) & 1
        r, c = i // 3, i % 3
        m[size-11+c][r] = b
        m[r][size-11+c] = b

def qr_svg(text, ecc='M', quiet=4, fg='#0f172a', bg='#ffffff'):
    m, size = qr_matrix(text, ecc)
    total = size + quiet*2
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total} {total}" '
             f'shape-rendering="crispEdges">',
             f'<rect width="{total}" height="{total}" fill="{bg}"/>']
    for r in range(size):
        c = 0
        while c < size:
            if m[r][c]:
                run = 1
                while c+run < size and m[r][c+run]: run += 1
                parts.append(f'<rect x="{c+quiet}" y="{r+quiet}" width="{run}" height="1" fill="{fg}"/>')
                c += run
            else:
                c += 1
    parts.append('</svg>')
    return ''.join(parts)

def get_network_info():
    info = {"hostname": "", "interfaces": []}
    try: info["hostname"] = socket.gethostname()
    except: pass
    try:
        result = subprocess.run(['ip', '-o', 'addr'], capture_output=True, text=True, timeout=5)
        seen = set()
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 4: continue
            iface, family, addr = parts[1], parts[2], parts[3]
            if iface == 'lo': continue
            key = (iface, family, addr)
            if key in seen: continue
            seen.add(key); info["interfaces"].append({"interface": iface, "family": family, "address": addr})
    except Exception as e: info["error"] = str(e)
    try:
        r = subprocess.run(['iwgetid', '-r'], capture_output=True, text=True, timeout=3)
        ssid = r.stdout.strip()
        if ssid: info["wifi_ssid"] = ssid
    except: pass
    return info

# ── Wi-Fi (NetworkManager) ─────────────────────────────────────────────────
def run_priv(cmd, timeout=25):
    """Run a command; if it fails without privileges, transparently retry with
       non-interactive sudo -n (same passwordless-sudo the Wi-Fi feature uses).
       Retries on ANY non-zero exit. Returns (returncode, stdout, stderr)."""
    cmd = list(cmd)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return 0, r.stdout, r.stderr
        try:
            r2 = subprocess.run(['sudo', '-n'] + cmd, capture_output=True,
                                text=True, timeout=timeout)
            if r2.returncode == 0:
                return 0, r2.stdout, r2.stderr
            return r2.returncode, (r2.stdout or r.stdout), (r2.stderr or r.stderr or '').strip()
        except Exception:
            return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, '', 'command timed out'
    except FileNotFoundError:
        return 127, '', f'{cmd[0]} not installed'
    except Exception as e:
        return 1, '', str(e)

def run_nmcli(args, timeout=25):
    """Run `nmcli <args>` with the privilege-aware runner."""
    return run_priv(['nmcli'] + list(args), timeout=timeout)

def _nmcli_split(line):
    """Split an `nmcli -t` line on ':' while honouring '\\:' escaped colons
       (SSIDs and security fields can contain colons)."""
    out, cur, i = [], '', 0
    while i < len(line):
        c = line[i]
        if c == '\\' and i + 1 < len(line):
            cur += line[i + 1]; i += 2; continue
        if c == ':':
            out.append(cur); cur = ''; i += 1; continue
        cur += c; i += 1
    out.append(cur)
    return out

def get_active_wifi_ssid():
    """Return the SSID of the currently-connected Wi-Fi, or '' if none.
       Uses NetworkManager's own view (authoritative) and falls back to
       iwgetid so the popup always reflects the real connection."""
    # 1) The active wireless connection (most reliable)
    rc, out, _ = run_nmcli(['-t', '-f', 'NAME,TYPE', 'connection', 'show', '--active'], timeout=8)
    if rc == 0:
        for line in out.splitlines():
            p = _nmcli_split(line)
            if len(p) >= 2 and 'wireless' in (p[1] or '').lower():
                if p[0]:
                    return p[0]
    # 2) The AP flagged in-use by the device
    rc, out, _ = run_nmcli(['-t', '-f', 'ACTIVE,SSID', 'device', 'wifi'], timeout=8)
    if rc == 0:
        for line in out.splitlines():
            p = _nmcli_split(line)
            if len(p) >= 2 and p[0].strip() in ('yes', '*', '*yes') and p[1]:
                return p[1]
    # 3) Last resort
    try:
        r = subprocess.run(['iwgetid', '-r'], capture_output=True, text=True, timeout=3)
        return r.stdout.strip()
    except Exception:
        return ''


# ── Network info ──────────────────────────────────────────────────────────────
def get_network_info():
    info = {"hostname": "", "interfaces": []}
    try: info["hostname"] = socket.gethostname()
    except: pass
    try:
        result = subprocess.run(['ip', '-o', 'addr'], capture_output=True, text=True, timeout=5)
        seen = set()
        for line in result.stdout.splitlines():
            parts = line.split()
            if len(parts) < 4: continue
            iface, family, addr = parts[1], parts[2], parts[3]
            if iface == 'lo': continue
            key = (iface, family, addr)
            if key in seen: continue
            seen.add(key); info["interfaces"].append({"interface": iface, "family": family, "address": addr})
    except Exception as e: info["error"] = str(e)
    try:
        r = subprocess.run(['iwgetid', '-r'], capture_output=True, text=True, timeout=3)
        ssid = r.stdout.strip()
        if ssid: info["wifi_ssid"] = ssid
    except: pass
    return info

# ── Wi-Fi (NetworkManager) ─────────────────────────────────────────────────
def run_priv(cmd, timeout=25):
    """Run a command; if it fails without privileges, transparently retry with
       non-interactive sudo -n (same passwordless-sudo the Wi-Fi feature uses).
       Retries on ANY non-zero exit. Returns (returncode, stdout, stderr)."""
    cmd = list(cmd)
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if r.returncode == 0:
            return 0, r.stdout, r.stderr
        try:
            r2 = subprocess.run(['sudo', '-n'] + cmd, capture_output=True,
                                text=True, timeout=timeout)
            if r2.returncode == 0:
                return 0, r2.stdout, r2.stderr
            return r2.returncode, (r2.stdout or r.stdout), (r2.stderr or r.stderr or '').strip()
        except Exception:
            return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, '', 'command timed out'
    except FileNotFoundError:
        return 127, '', f'{cmd[0]} not installed'
    except Exception as e:
        return 1, '', str(e)

def run_nmcli(args, timeout=25):
    """Run `nmcli <args>` with the privilege-aware runner."""
    return run_priv(['nmcli'] + list(args), timeout=timeout)

def _nmcli_split(line):
    """Split an `nmcli -t` line on ':' while honouring '\\:' escaped colons
       (SSIDs and security fields can contain colons)."""
    out, cur, i = [], '', 0
    while i < len(line):
        c = line[i]
        if c == '\\' and i + 1 < len(line):
            cur += line[i + 1]; i += 2; continue
        if c == ':':
            out.append(cur); cur = ''; i += 1; continue
        cur += c; i += 1
    out.append(cur)
    return out

def get_active_wifi_ssid():
    """Return the SSID of the currently-connected Wi-Fi, or '' if none.
       Uses NetworkManager's own view (authoritative) and falls back to
       iwgetid so the popup always reflects the real connection."""
    # 1) The active wireless connection (most reliable)
    rc, out, _ = run_nmcli(['-t', '-f', 'NAME,TYPE', 'connection', 'show', '--active'], timeout=8)
    if rc == 0:
        for line in out.splitlines():
            p = _nmcli_split(line)
            if len(p) >= 2 and 'wireless' in (p[1] or '').lower():
                if p[0]:
                    return p[0]
    # 2) The AP flagged in-use by the device
    rc, out, _ = run_nmcli(['-t', '-f', 'ACTIVE,SSID', 'device', 'wifi'], timeout=8)
    if rc == 0:
        for line in out.splitlines():
            p = _nmcli_split(line)
            if len(p) >= 2 and p[0].strip() in ('yes', '*', '*yes') and p[1]:
                return p[1]
    # 3) Last resort
    try:
        r = subprocess.run(['iwgetid', '-r'], capture_output=True, text=True, timeout=3)
        return r.stdout.strip()
    except Exception:
        return ''


# ── Clock source / time-set ────────────────────────────────────────────────
_clock_src_cache = {'ts': 0.0, 'val': None}

def _has_rtc():
    return os.path.exists('/dev/rtc') or os.path.exists('/dev/rtc0')

def get_clock_source():
    """Report how trustworthy the current time is:
         ntp    → synchronised from the network (best)
         rtc    → a hardware real-time clock is present (holds time offline)
         manual → no NTP sync and no RTC (set by hand, may drift)
       Also returns the IANA timezone and a one-word 'location' (the city part,
       e.g. 'Australia/Sydney' → 'Sydney') for display next to the badge."""
    synced = False
    tz = ''
    ntp_service = False
    local_rtc = False
    try:
        # Parse KEY=VALUE (order-independent). Using --value would rely on
        # systemd's output ORDER, which is NOT the order of the -p flags — that
        # bug made the timezone read back as a stray 'yes'/'no'.
        r = subprocess.run(['timedatectl', 'show'],
                           capture_output=True, text=True, timeout=5)
        props = {}
        for line in r.stdout.splitlines():
            if '=' in line:
                k, _, v = line.partition('=')
                props[k.strip()] = v.strip()
        synced      = props.get('NTPSynchronized', '').lower() == 'yes'
        ntp_service = props.get('NTP', '').lower() == 'yes'
        local_rtc   = props.get('LocalRTC', '').lower() == 'yes'
        tz          = props.get('Timezone', '')
    except Exception:
        pass
    rtc = _has_rtc()
    if synced:
        source = 'ntp'
    elif rtc:
        source = 'rtc'
    else:
        source = 'manual'
    # One-word location from the timezone's city component.
    location = ''
    if tz and '/' in tz:
        location = tz.rsplit('/', 1)[1].replace('_', ' ')
    elif tz:
        location = tz
    return {'source': source, 'synced': synced, 'ntp_service': ntp_service,
            'rtc': rtc, 'tz': tz, 'location': location, 'local_rtc': local_rtc}

def get_clock_source_cached():
    now = time.time()
    if _clock_src_cache['val'] is None or (now - _clock_src_cache['ts']) > 10:
        _clock_src_cache['val'] = get_clock_source()
        _clock_src_cache['ts'] = now
    return _clock_src_cache['val']

# ── Time management (final) ────────────────────────────────────────────────
# Priority of truth:
#   1. VERIFIED network time — NTP that has genuinely synchronised — always wins.
#   2. Otherwise the last-known-good time saved on disk (a manual entry, or the
#      last good NTP value). This survives reboots even on boards with no
#      battery-backed RTC.
# Boot sequence: restore the saved time immediately so the clock is sane while
# offline, then — if a network is present — sync from it (and pull the correct
# timezone from the device's location). When a network later appears we always
# DOUBLE-CHECK that NTP can really sync before trusting it; if it can't, we keep
# following the saved/manual time.
TIME_BACKUP_FILE = "/home/linaro/last_known_time.json"
_geo_tz_done     = {"ok": False}      # resolve timezone-from-location only once/boot

def ntp_is_synced():
    """True only when the system has a GENUINE network time sync. Having an IP is
       not enough — captive portals and blocked NTP look 'online' but never sync.
       This is the double-check performed before trusting network time."""
    try:
        r = subprocess.run(['timedatectl', 'show', '-p', 'NTPSynchronized', '--value'],
                           capture_output=True, text=True, timeout=5)
        return r.stdout.strip().lower() == 'yes'
    except Exception:
        return False

def _read_time_backup():
    try:
        with open(TIME_BACKUP_FILE) as f:
            return json.load(f)
    except Exception:
        return {}

def _valid_timezone(tz):
    """True only for a real IANA zone this system actually ships."""
    if not tz or tz.startswith('/') or '..' in tz:
        return False
    try:
        return os.path.exists('/usr/share/zoneinfo/' + tz)
    except Exception:
        return False

def _apply_tz_to_process(tz):
    """Make THIS running process use `tz` for datetime.now()/logging immediately.
       Without this, libc caches the old zone until the process restarts, so the
       on-screen clock and CSV timestamps would lag behind a timezone change."""
    try:
        if tz:
            os.environ['TZ'] = tz
            time.tzset()
    except Exception:
        pass

def _list_timezones():
    """All IANA zone names for the picker."""
    zones = []
    try:
        import zoneinfo
        zones = sorted(zoneinfo.available_timezones())
    except Exception:
        base = '/usr/share/zoneinfo'
        try:
            for root, _d, files in os.walk(base):
                for fn in files:
                    rel = os.path.relpath(os.path.join(root, fn), base)
                    if (rel and rel[0].isupper() and '/' in rel
                            and not rel.startswith(('posix/', 'right/'))
                            and not rel.endswith(('.tab', '.list'))):
                        zones.append(rel)
            zones = sorted(set(zones))
        except Exception:
            zones = []
    return zones or ['UTC']

def save_time_backup(source=None, tz_manual=None):
    """Persist the current system time (+ source + timezone) to disk. Cheap and
       safe to call often. Preserves the 'tz_manual' flag — a timezone the
       installer chose by hand, which geo-IP must never override."""
    try:
        prev = _read_time_backup()
        if source is None:
            source = 'ntp' if ntp_is_synced() else prev.get('source', 'manual')
        try:
            tz = get_clock_source().get('tz', '') or prev.get('tz', '')
        except Exception:
            tz = prev.get('tz', '')
        if tz_manual is None:
            tz_manual = bool(prev.get('tz_manual', False))
        with open(TIME_BACKUP_FILE, 'w') as f:
            json.dump({"epoch": time.time(),
                       "iso": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                       "source": source, "tz": tz, "tz_manual": tz_manual}, f)
    except Exception as e:
        print(f"[TIME] backup save failed: {e}", flush=True)

def set_system_timezone(tz, manual=False):
    """Set the system AND process timezone to `tz`, then persist it. When
       `manual` is True the choice is LOCKED so geo-IP won't second-guess it
       (geo-IP can't tell e.g. Brisbane from Sydney). Returns True on success."""
    if not _valid_timezone(tz):
        print(f"[TIME] rejected invalid timezone: {tz!r}", flush=True)
        return False
    rc, _o, _e = run_priv(['timedatectl', 'set-timezone', tz], timeout=8)
    if rc != 0:
        print(f"[TIME] set-timezone failed: {tz}", flush=True)
        return False
    _apply_tz_to_process(tz)
    _clock_src_cache['val'] = None
    if manual:
        _geo_tz_done["ok"] = True          # don't let geo-IP re-guess this boot
    save_time_backup(tz_manual=(True if manual else None))
    try:
        update_clock_data()                # reflect the new zone immediately
    except Exception:
        pass
    print(f"[TIME] timezone set -> {tz}{' (manual/locked)' if manual else ''}", flush=True)
    return True

def apply_saved_timezone():
    """Reapply the last-known timezone (from disk) so LOCAL time reads correctly
       even before a network is available — and keep this process in sync."""
    try:
        tz = _read_time_backup().get('tz', '')
        if tz and _valid_timezone(tz):
            if tz != get_clock_source().get('tz', ''):
                run_priv(['timedatectl', 'set-timezone', tz], timeout=8)
                _clock_src_cache['val'] = None
            _apply_tz_to_process(tz)
    except Exception as e:
        print(f"[TIME] apply saved tz failed: {e}", flush=True)

def detect_timezone_from_location():
    """Resolve the timezone from the device's location (geo-IP) and apply it, so
       the panel shows correct LOCAL time out of the box. Best-effort, needs a
       network, runs at most once per boot — and NEVER overrides a timezone the
       installer set by hand."""
    if _geo_tz_done["ok"]:
        return
    if _read_time_backup().get('tz_manual'):
        _geo_tz_done["ok"] = True           # installer's choice wins
        return
    import urllib.request
    tz = None
    sources = (
        ("http://ip-api.com/json/?fields=status,timezone",
         lambda b: (lambda j: j.get("timezone") if j.get("status") == "success" else None)(json.loads(b))),
        ("https://ipapi.co/timezone/", lambda b: (b.strip() or None)),
    )
    for url, parse in sources:
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                body = resp.read().decode("utf-8", "replace")
            tz = parse(body)
            if tz:
                break
        except Exception:
            continue
    if tz and _valid_timezone(tz):
        rc, _out, _err = run_priv(['timedatectl', 'set-timezone', tz], timeout=8)
        if rc == 0:
            _geo_tz_done["ok"] = True
            _apply_tz_to_process(tz)
            _clock_src_cache['val'] = None
            save_time_backup()            # remember tz for offline use (not manual)
            print(f"[TIME] timezone from location -> {tz} (auto; installer can override)", flush=True)

def verified_ntp_sync(wait_s=25):
    """Enable NTP and WAIT until it has genuinely synchronised (or give up).
       Returns True only on a real sync. On success the good time is written to
       the RTC and saved to disk; on failure the saved/manual time is left intact
       and NTP stays armed to correct the clock the instant a server answers."""
    try:
        run_priv(['timedatectl', 'set-ntp', 'true'], timeout=10)
    except Exception as e:
        print(f"[TIME] enabling NTP failed: {e}", flush=True)
        return False
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        if ntp_is_synced():
            try:
                if _has_rtc():
                    run_priv(['timedatectl', 'set-local-rtc', '0'], timeout=10)  # RTC in UTC
                    run_priv(['hwclock', '-w'], timeout=10)     # persist good time
            except Exception:
                pass
            save_time_backup(source='ntp')
            _clock_src_cache['val'] = None
            return True
        time.sleep(2)
    _clock_src_cache['val'] = None
    return False

def sync_time_from_network():
    """Non-blocking: verify+sync network time and pull timezone from location.
       Network time wins when it genuinely syncs; otherwise the saved time stands."""
    def _job():
        try:
            ok = verified_ntp_sync()
            detect_timezone_from_location()
            print(f"[TIME] network sync {'OK' if ok else 'unavailable — keeping saved time'}",
                  flush=True)
        except Exception as e:
            print(f"[TIME] network sync error: {e}", flush=True)
    threading.Thread(target=_job, daemon=True).start()

def restore_time_from_backup():
    """At boot, roll the clock forward to the last saved time if the system clock
       is clearly behind it (RTC-less boards reset on power-off), and reapply the
       saved timezone. Only ever moves the clock FORWARD, and only for a real gap
       (>30 s), so a good RTC/NTP time is never clobbered."""
    try:
        saved = _read_time_backup()
        saved_epoch = float(saved.get("epoch", 0))
        if saved_epoch and (saved_epoch - time.time() > 30):
            ts = datetime.fromtimestamp(saved_epoch).strftime("%Y-%m-%d %H:%M:%S")
            rc, _out, _err = run_priv(['date', '-s', ts], timeout=10)
            if rc != 0:
                run_priv(['timedatectl', 'set-ntp', 'false'], timeout=8)
                run_priv(['timedatectl', 'set-time', ts], timeout=10)
            if _has_rtc():
                run_priv(['hwclock', '-w'], timeout=10)
            _clock_src_cache['val'] = None
            print(f"[TIME] restored clock from backup -> {ts}", flush=True)
        apply_saved_timezone()
    except Exception as e:
        print(f"[TIME] backup restore failed: {e}", flush=True)

def _have_network():
    """True if any real interface (Ethernet or Wi-Fi) has a routable IPv4
       address (not loopback and not a 169.254 link-local)."""
    try:
        for iface in get_network_info().get("interfaces", []):
            if iface.get("family") != "inet":
                continue
            name = iface.get("interface", "")
            if name.startswith(("lo", "docker", "br-", "veth", "virbr", "tun", "tap")):
                continue
            addr = (iface.get("address") or "").split('/')[0]
            if not addr or addr.startswith(("127.", "169.254.")):
                continue
            return True
    except Exception:
        pass
    return False

def time_startup():
    """Boot-time sequence: get the clock sane immediately from the saved time,
       then — if a network is up — sync from it (NTP + timezone-from-location)."""
    restore_time_from_backup()
    # Make sure THIS process uses the system's timezone from the start (so the
    # on-screen clock and logs are correct even before any change).
    _apply_tz_to_process(get_clock_source().get('tz', ''))
    save_time_backup()                    # guarantee a baseline backup file exists
    if _have_network():
        sync_time_from_network()

def network_time_thread():
    """Keep the clock correct for the life of the product:
         • when a network appears (Ethernet OR Wi-Fi), DOUBLE-CHECK that NTP can
           really sync and, if so, let it take over (+ timezone from location);
         • if it can't sync, keep following the saved/manual time;
         • back the current time up to disk ~once a minute so a manually-set time
           survives the next reboot."""
    was_online        = False
    last_sync_attempt = 0.0
    backup_tick       = 0
    while True:
        try:
            online = _have_network()
            now    = time.monotonic()
            if online:
                # Attempt a sync when we just came online or NTP still isn't
                # locked — throttled so we never thrash timesyncd.
                if ((not was_online) or (not ntp_is_synced())) and (now - last_sync_attempt) > 60:
                    sync_time_from_network()
                    last_sync_attempt = now
            elif was_online:
                # Just went offline — make sure local time still reads right.
                apply_saved_timezone()
            was_online = online

            backup_tick += 1
            if backup_tick >= 6:          # ~60 s
                backup_tick = 0
                save_time_backup()
        except Exception as e:
            print(f"[TIME] network watch error: {e}", flush=True)
        time.sleep(10)

def startup_once():
    """Show logo fullscreen + play sound. Runs every boot (lock file is boot-volatile).
       The logo STAYS ON SCREEN — it is only closed later, once Chromium is ready,
       via close_splash()."""
    global splash_proc
    lock_file = "/run/envi_startup_done"
    if os.path.exists(lock_file):
        return
    env = get_display_env()
    try:
        os.makedirs("/run/envi", exist_ok=True)
    except Exception:
        pass
    try:
        subprocess.Popen(
            ["aplay", "-D", "plughw:0,0", "/home/linaro/start.wav"],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception as e:
        print(f"[STARTUP] Sound error: {e}", flush=True)
    try:
        from PIL import Image as _Img
        _logo_src = "/home/linaro/logo.png"
        _logo_tmp = "/tmp/envi_splash.png"
        W, H = 1280, 800
        try:
            img = _Img.open(_logo_src).convert("RGBA")
            if img.height > img.width:
                img = img.rotate(90, expand=True)
            img.thumbnail((W, H), _Img.LANCZOS)
            bg = _Img.new("RGB", (W, H), (0, 0, 0))
            x = (W - img.width) // 2
            y = (H - img.height) // 2
            bg.paste(img, (x, y), img)
            bg.save(_logo_tmp)
            logo_path = _logo_tmp
        except Exception as e:
            print(f"[STARTUP] Logo prep error: {e}", flush=True)
            logo_path = _logo_src

        splash_proc = subprocess.Popen(
            ["feh", "-F", "--hide-pointer", "--auto-zoom", logo_path],
            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print("[STARTUP] Logo displayed — staying up until Chromium is ready", flush=True)
    except Exception as e:
        print(f"[STARTUP] Logo error: {e}", flush=True)
    try:
        with open(lock_file, "w") as f:
            f.write("done")
    except Exception:
        pass

def close_splash():
    """Kill the startup logo. Safe to call even if it's already gone."""
    global splash_proc
    if splash_proc:
        try:
            splash_proc.terminate()
            splash_proc.wait(timeout=3)
        except Exception:
            try: splash_proc.kill()
            except Exception: pass
        splash_proc = None
        print("[STARTUP] Logo closed", flush=True)

def _wait_for_chromium_window(env, timeout=15):
    """Poll for a mapped Chromium window using wmctrl/xdotool if available;
       otherwise fall back to a short fixed grace period for first paint."""
    deadline = time.monotonic() + timeout
    tool = None
    for t in ['wmctrl', 'xdotool']:
        r = subprocess.run(['which', t], capture_output=True, text=True)
        if r.returncode == 0:
            tool = t
            break
    if not tool:
        time.sleep(3)  # no window tool available — just give Chromium a moment
        return
    while time.monotonic() < deadline:
        try:
            if tool == 'wmctrl':
                r = subprocess.run(['wmctrl', '-l'], env=env, capture_output=True, text=True, timeout=2)
                if 'chromium' in r.stdout.lower():
                    time.sleep(0.5)
                    return
            else:
                r = subprocess.run(['xdotool', 'search', '--name', 'chromium'], env=env, capture_output=True, text=True, timeout=2)
                if r.stdout.strip():
                    time.sleep(0.5)
                    return
        except Exception:
            pass
        time.sleep(0.3)
    # Timed out — close anyway so the splash doesn't get stuck forever


def update_presence(face_count, motion_recent):
    """Layer 2: convert noisy raw detection into a stable presence verdict.

    Occupied triggers INSTANTLY on any detection.
    Vacant triggers only after VACANCY_TIMEOUT of continuous zero-detection.
    This mirrors how commercial PIR occupancy sensors use a 'time delay' so
    they don't switch off while someone is sitting still.
    """
    now = time.monotonic()
    detected = (face_count > 0) or motion_recent

    manual = occupancy_manual_override["active"] and now < occupancy_manual_override["until"]
    if manual:
        detected = True

    with presence_lock:
        presence_state["evidence_window"].append(1 if detected else 0)
        if detected:
            presence_state["last_detect"] = now
            if not presence_state["present"]:
                presence_state["present"] = True
                presence_state["since"]   = now
        else:
            # only go vacant after a full timeout of no detection
            if presence_state["present"] and (now - presence_state["last_detect"]) >= VACANCY_TIMEOUT:
                presence_state["present"] = False
                presence_state["since"]   = now
        return presence_state["present"]


def get_presence():
    """Return the current Layer-2 verdict + supporting evidence."""
    now = time.monotonic()
    with presence_lock:
        win = list(presence_state["evidence_window"])
        present = presence_state["present"]
        last    = presence_state["last_detect"]
        since   = presence_state["since"]
    # evidence ratio = fraction of recent samples with raw detection (0..1)
    evidence = (sum(win) / len(win)) if win else 0.0
    return {
        "present":     present,
        "evidence":    round(evidence, 2),
        "idle_sec":    int(now - last) if last else None,
        "state_age":   int(now - since) if since else 0,
        "vacancy_in":  max(0, int(VACANCY_TIMEOUT - (now - last))) if (present and last) else 0,
    }

def open_browser():
    """Wait for Flask to be ready, then launch Chromium in kiosk mode.
       Keeps the splash logo on screen until Chromium has rendered, then closes it."""
    import urllib.request
    env = get_display_env()
    browser = None
    for b in ['chromium-browser', 'chromium', 'google-chrome', 'firefox']:
        r = subprocess.run(['which', b], capture_output=True, text=True)
        if r.returncode == 0:
            browser = r.stdout.strip()
            break
    if not browser:
        print("[BROWSER] No browser found!", flush=True)
        close_splash()
        return

    print(f"[BROWSER] Found browser: {browser}", flush=True)
    print("[BROWSER] Waiting for Flask...", flush=True)
    for _ in range(60):
        try:
            urllib.request.urlopen('http://127.0.0.1:5002/', timeout=1)
            print("[BROWSER] Flask is ready", flush=True)
            break
        except Exception:
            time.sleep(1)

    try:
        subprocess.run(['pkill', '-f', 'chromium'], capture_output=True)
        time.sleep(1)
    except Exception:
        pass

    # Belt-and-suspenders: re-confirm landscape rotation right before Chromium
    # opens. Cheap no-op if it's already rotated; guards against a driver that
    # re-negotiates the panel mode once the GPU/compositor starts drawing.
    _apply_landscape_rotation(env, 'DSI-1', attempts=2)

    print("[BROWSER] Launching Chromium...", flush=True)
    subprocess.Popen([
        browser,
        '--kiosk',
        '--noerrdialogs',
        '--disable-infobars',
        '--no-first-run',
        '--disable-default-apps',
        '--disable-restore-session-state',
        '--disable-session-crashed-bubble',
        '--disable-translate',
        '--no-default-browser-check',
        '--disable-features=TranslateUI',
        '--overscroll-history-navigation=0',
        '--enable-features=VirtualKeyboard',
        '--virtual-keyboard',
        '--disable-pinch',
        '--overscroll-history-navigation=0',
        '--touch-events=enabled',
        '--user-data-dir=/tmp/chromium-envi',
        '--renderer-process-limit=1',
        '--disable-extensions',
        '--disable-background-networking',
        '--disable-sync',
        '--disable-gpu-compositing',
        '--memory-pressure-off',
        '--single-process',
        '--disable-low-res-tiling',
        '--enable-low-end-device-mode',
        'http://127.0.0.1:5002/?kiosk=1'
    ], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Don't drop the logo until Chromium's window is actually on screen
    _wait_for_chromium_window(env, timeout=15)
    close_splash()

# ── Kiosk health watchdog ───────────────────────────────────────────────────
# Chromium already runs in --kiosk, but on a live install a few things can still
# appear OVER it — most commonly NetworkManager's "Wi-Fi Network Authentication
# Required" password dialog, which grabs focus every morning when NM retries a
# saved network it has no key for. That is what makes the taskbar and a window
# title bar suddenly show over the panel. This watchdog keeps the ENVI page
# running cleanly and uninterrupted, with no scheduled blackout/restart:
#   1. it stops NM auto-retrying networks the panel isn't using (kills the
#      pop-up at the source),
#   2. it dismisses any auth/secret dialog the instant one appears (safety net),
#   3. it keeps the kiosk window raised, and relaunches Chromium if it ever dies.

KIOSK_RELOAD_HOURS = 0     # >0 also reloads the page every N hours (0 = never; not needed)

_WIN_TOOL = {'val': 'unset'}
def _win_tool():
    """Which window tool is present — 'xdotool' preferred, then 'wmctrl'. Cached."""
    if _WIN_TOOL['val'] == 'unset':
        tool = None
        if subprocess.run(['which', 'xdotool'], capture_output=True).returncode == 0:
            tool = 'xdotool'
        elif subprocess.run(['which', 'wmctrl'], capture_output=True).returncode == 0:
            tool = 'wmctrl'
        _WIN_TOOL['val'] = tool
    return _WIN_TOOL['val']

_POPUP_KEYWORDS = ('authentication required', 'network authentication',
                   'wi-fi network authentication', 'wifi network authentication')

def _dismiss_intruder_dialogs(env):
    """Close NetworkManager Wi-Fi secret dialogs (and similar auth pop-ups) that
       appear over the kiosk. Closing == pressing Cancel, so the kiosk is never
       left sitting on a password box. Returns True if one was closed."""
    tool = _win_tool()
    closed = False
    try:
        if tool == 'xdotool':
            for pat in ('Wi-Fi Network Authentication', 'Authentication [Rr]equired',
                        'Network Authentication'):
                r = subprocess.run(['xdotool', 'search', '--name', pat],
                                   env=env, capture_output=True, text=True, timeout=3)
                for wid in r.stdout.split():
                    if wid.strip():
                        subprocess.run(['xdotool', 'windowclose', wid.strip()],
                                       env=env, capture_output=True, timeout=3)
                        closed = True
        elif tool == 'wmctrl':
            r = subprocess.run(['wmctrl', '-l'], env=env, capture_output=True, text=True, timeout=3)
            for line in r.stdout.splitlines():
                parts = line.split(None, 3)
                title = parts[3].lower() if len(parts) >= 4 else ''
                if any(k in title for k in _POPUP_KEYWORDS):
                    subprocess.run(['wmctrl', '-i', '-c', parts[0]],
                                   env=env, capture_output=True, timeout=3)
                    closed = True
    except Exception:
        pass
    return closed

def _raise_kiosk(env):
    """Bring the ENVI Chromium kiosk window back to the front and focus it."""
    tool = _win_tool()
    try:
        if tool == 'xdotool':
            wids = []
            for pat in ('ENVI', 'Chromium', 'chromium'):
                r = subprocess.run(['xdotool', 'search', '--name', pat],
                                   env=env, capture_output=True, text=True, timeout=3)
                wids = [w for w in r.stdout.split() if w.strip()]
                if wids:
                    break
            for wid in wids[:1]:
                subprocess.run(['xdotool', 'windowactivate', '--sync', wid],
                               env=env, capture_output=True, timeout=3)
        elif tool == 'wmctrl':
            subprocess.run(['wmctrl', '-a', 'ENVI'], env=env, capture_output=True, timeout=3)
    except Exception:
        pass

def _chromium_alive():
    try:
        return bool(subprocess.run(['pgrep', '-f', 'chromium'],
                                   capture_output=True, text=True, timeout=3).stdout.strip())
    except Exception:
        return True     # assume alive on error — don't relaunch on a false negative

def _reload_kiosk(env):
    """Optional: reload the page in place (no blackout, no boot logo)."""
    tool = _win_tool()
    try:
        if tool == 'xdotool':
            r = subprocess.run(['xdotool', 'search', '--name', 'ENVI'],
                               env=env, capture_output=True, text=True, timeout=3)
            for wid in r.stdout.split()[:1]:
                subprocess.run(['xdotool', 'key', '--window', wid, 'F5'],
                               env=env, capture_output=True, timeout=3)
                print("[KIOSK] periodic in-place page reload", flush=True)
    except Exception:
        pass

def quiet_stray_wifi_autoconnect():
    """Stop NetworkManager from repeatedly retrying — and popping a password
       dialog for — saved Wi-Fi networks the panel isn't using. Disables
       autoconnect on every saved Wi-Fi profile that is NOT currently active;
       the active network keeps autoconnect. Best-effort and safe to fail.
       (To keep a specific network on standby, set it back with:
        nmcli connection modify "<name>" connection.autoconnect yes)"""
    try:
        rc, out, _ = run_nmcli(['-t', '-f', 'NAME', 'connection', 'show', '--active'], timeout=8)
        active = set(l.strip() for l in out.splitlines() if l.strip()) if rc == 0 else set()
        rc, out, _ = run_nmcli(['-t', '-f', 'NAME,TYPE', 'connection', 'show'], timeout=8)
        if rc != 0:
            return
        for line in out.splitlines():
            p = _nmcli_split(line)
            if len(p) < 2:
                continue
            name, ctype = p[0], p[1]
            if 'wireless' not in ctype and 'wifi' not in ctype:
                continue
            if name in active or not name:
                continue
            run_nmcli(['connection', 'modify', name, 'connection.autoconnect', 'no'], timeout=8)
            print(f"[KIOSK] disabled autoconnect for stray Wi-Fi profile: {name}", flush=True)
    except Exception as e:
        print(f"[KIOSK] quiet stray wifi error: {e}", flush=True)

def kiosk_watchdog_thread():
    """Keep the ENVI kiosk clean and uninterrupted for the life of the product."""
    env = get_display_env()
    quiet_stray_wifi_autoconnect()          # kill the morning pop-up at the source
    start       = time.monotonic()
    last_launch = 0.0
    last_reload = time.monotonic()
    while True:
        try:
            # Safety net: keep any intruding auth dialog off the kiosk.
            if _dismiss_intruder_dialogs(env):
                time.sleep(0.3)
                _raise_kiosk(env)
            # Police Chromium's existence only after the initial boot launch settles.
            if (time.monotonic() - start) > 30 and not _chromium_alive():
                if (time.monotonic() - last_launch) > 20:
                    print("[KIOSK] Chromium not running — relaunching", flush=True)
                    open_browser()
                    last_launch = time.monotonic()
            if KIOSK_RELOAD_HOURS and (time.monotonic() - last_reload) > KIOSK_RELOAD_HOURS * 3600:
                _reload_kiosk(env)
                last_reload = time.monotonic()
        except Exception as e:
            print(f"[KIOSK] watchdog error: {e}", flush=True)
        time.sleep(3)
        
# ─────────────────────────────────────────────────────────────────────────────
# HTML
# ─────────────────────────────────────────────────────────────────────────────
DASHBOARD_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, user-scalable=no, initial-scale=1"/>
<title>ENVI Panel</title>
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo:wght@300;400;500;600;700;800;900&display=swap');
/* Global font: Exo (whole system) */
*{font-family:'Exo','Exo 2',sans-serif;}
/* Hide on-screen scrollbars everywhere (kiosk) */
*{scrollbar-width:none;-ms-overflow-style:none;}
*::-webkit-scrollbar{display:none;width:0;height:0;background:transparent;}
.header{z-index:1000;}
.tabs{z-index:999;}
.tab{position:relative;z-index:1000;pointer-events:auto;}
.page{position:relative;z-index:1;}
.ctrl-wrap,.ctrl-bg-overlay,.ctrl-bg-overlay::after{pointer-events:none;}
.ctrl-header,.ctrl-body{pointer-events:auto;}
/* Cursor is hidden ONLY on the physical panel (body gets .kiosk when the page
   is opened with ?kiosk=1, which is how Chromium is launched here). A tech
   browsing the panel's IP from a PC or phone keeps a normal cursor.
   NOTE: '#' is not a CSS comment — the two '#' lines that used to sit here were
   silently discarded by the browser and never did anything. */
body.kiosk, body.kiosk *{cursor:none!important;}
*{box-sizing:border-box;margin:0;padding:0;font-family:'Exo','Exo 2',sans-serif;}
html,body{zoom:1.25;margin:0;padding:0;}
html{width:100%;height:100%;}
body{background:#f0f2f5;color:#222;width:100%;min-height:100vh;overflow-x:hidden;font-family:'Exo','Exo 2',sans-serif;}
.header{background:#1e3a5f;color:white;padding:8px 15px;display:flex;justify-content:space-between;align-items:center;position:sticky;top:0;z-index:100;}
.header-right{text-align:right;display:flex;flex-direction:column;align-items:flex-end;gap:4px;max-width:230px;}
#conn-text{font-size:13px;font-weight:bold;color:#22c55e;margin-top:2px;}
#motion-text{font-size:10px;color:#7ec8e3;margin-top:2px;}
#fault-text{font-size:10px;color:red;margin-top:2px;}
.unit-status-line{font-size:11px;font-weight:700;color:#aed6f1;margin-top:3px;}
.fault-box{margin-top:4px;font-size:11px;font-weight:700;padding:0;max-width:100%;word-break:break-word;text-align:right;}
.fault-box.active{background:#fee2e2;color:#b91c1c;padding:4px 10px;border-radius:6px;display:inline-block;border:1px solid #fca5a5;box-shadow:none;}
.tabs{display:flex;background:#fff;border-bottom:2px solid #e5e7eb;position:sticky;top:44px;z-index:99;overflow-x:auto;}
.tab{flex:1;min-width:58px;padding:9px 4px;text-align:center;font-size:11px;cursor:pointer;color:#666;border-bottom:3px solid transparent;white-space:nowrap;user-select:none;}
.tab.active{color:#1e3a5f;font-weight:bold;border-bottom:3px solid #1e3a5f;}
.tab.locked{color:#bbb;cursor:not-allowed;}
.tab.locked::after{content:"🔒";font-size:9px;margin-left:3px;}
.page{display:none;padding:8px 6px;}
.page.active{display:block;}
.cards{display:grid;grid-template-columns:repeat(2,1fr);gap:8px;margin-bottom:8px;}
.card{background:white;border-radius:8px;padding:12px;border:1px solid #ddd;}
.card-label{font-size:10px;color:#888;text-transform:uppercase;margin-bottom:3px;}
.card-value{font-size:22px;font-weight:bold;color:#1e3a5f;line-height:1;}
.card-unit{font-size:12px;color:#888;margin-left:2px;}
.card-sub{font-size:10px;color:#bbb;margin-top:3px;}
.ctrl-header{display:flex;align-items:center;justify-content:space-between;
  background:linear-gradient(135deg,#1e3a5f,#274b7a);color:white;padding:11px 14px;
  border-radius:12px 12px 0 0;margin-top:0;cursor:pointer;user-select:none;position:relative;transition:border-radius .2s;}.ctrl-header.collapsed{border-radius:8px;}
.ctrl-header::after{content:"";position:absolute;bottom:3px;left:50%;transform:translateX(-50%);width:36px;height:3px;background:rgba(255,255,255,0.35);border-radius:3px;}
.ctrl-chev{display:inline-block;font-size:10px;transition:transform .25s;margin-right:6px;}
.ctrl-header.collapsed .ctrl-chev{transform:rotate(-90deg);}
.ctrl-body.collapsed{display:none;}
.ctrl-badge{font-size:11px;padding:2px 10px;border-radius:12px;background:#374151;}
.ctrl-badge.ok{background:#166534;color:#bbf7d0;}
.ctrl-badge.err{background:#7f1d1d;color:#fecaca;}
.ctrl-badge.dis{background:#374151;color:#9ca3af;}
.ctrl-badge.fault{background:#dc2626;color:#fff;font-weight:800;letter-spacing:.4px;}
.onoff-btn{padding:4px 16px;border:none;border-radius:12px;font-size:12px;font-weight:bold;cursor:pointer;}
.onoff-btn.on{background:#22c55e;color:white;}
.onoff-btn.off{background:#e5e7eb;color:#666;}
.toggle-row{background:white;border-radius:8px;padding:11px 12px;border:1px solid #ddd;margin-bottom:7px;display:flex;align-items:center;justify-content:space-between;}
.toggle-btn{padding:5px 0;width:90px;text-align:center;border:none;border-radius:20px;font-size:12px;font-weight:bold;cursor:pointer;}
.toggle-btn.on{background:#1e3a5f;color:white;}
.toggle-btn.off{background:#e5e7eb;color:#666;}
.reg-section{background:white;border-radius:8px;border:1px solid #ddd;margin-bottom:10px;overflow:hidden;}
.reg-section-title{background:#1e3a5f;color:white;padding:8px 12px;font-size:12px;font-weight:bold;}
table{width:100%;border-collapse:collapse;font-size:12px;}
td,th{padding:7px 10px;border-bottom:1px solid #f0f0f0;}
th{color:#888;font-weight:normal;font-size:10px;text-transform:uppercase;background:#fafafa;}
.val-cell{font-weight:bold;color:#1e3a5f;text-align:right;}
.write-panel{background:white;border-radius:8px;padding:14px;border:1px solid #ddd;margin-bottom:10px;}
/* write-panel h2 defined in unified accordion CSS below */
.write-row{display:flex;flex-direction:column;gap:4px;margin-bottom:12px;}
.write-row label{font-size:12px;color:#555;font-weight:bold;}
.write-row input,.write-row select{flex:1;padding:7px 10px;border:1px solid #ccc;border-radius:5px;font-size:13px;}
.write-row button{padding:7px 16px;background:#1e3a5f;color:white;border:none;border-radius:5px;font-size:12px;cursor:pointer;}
.sched-ctrl-bar{display:flex;gap:6px;margin-bottom:6px;flex-wrap:wrap;}
.sched-ctrl-btn{padding:6px 14px;border:1px solid #1e3a5f;border-radius:16px;font-size:12px;cursor:pointer;background:white;color:#1e3a5f;}
.sched-ctrl-btn.active{background:#1e3a5f;color:white;}
.sched-ctrl-btn.all-btn{border-color:#7c3aed;color:#7c3aed;}
.sched-ctrl-btn.all-btn.active{background:#7c3aed;color:white;border-color:#7c3aed;}
/* ════ BMS Schedule Tab ════ */
.sched-panel{background:#fff;border:1px solid #e8edf3;border-radius:12px;margin-bottom:10px;overflow:hidden;}
.sched-panel-hdr{display:flex;justify-content:space-between;align-items:center;background:#1e3a5f;color:#fff;padding:10px 14px;font-size:13px;font-weight:700;}
.sched-panel-body{padding:12px 14px;}
.sched-enabled-pill{font-size:10px;font-weight:800;padding:3px 10px;border-radius:12px;letter-spacing:.3px;}
.sched-enabled-pill.on{background:rgba(34,197,94,.2);color:#bbf7d0;}
.sched-enabled-pill.off{background:rgba(239,68,68,.2);color:#fecaca;}
.sched-enable-note{margin-top:10px;padding:8px 10px;border-radius:8px;font-size:11px;background:#fffbeb;border:1px dashed #f59e0b;color:#92400e;}

/* ── BMS header status box ── */
.hdr-status-box{background:rgba(255,255,255,0.06);border:1px solid rgba(255,255,255,0.14);
  border-radius:9px;padding:5px 8px;display:inline-flex;flex-direction:column;gap:5px;align-items:stretch;}
.hdr-status-row{display:flex;align-items:center;gap:6px;font-size:11px;font-weight:700;
  color:#e2e8f0;letter-spacing:.3px;white-space:nowrap;}
.hdr-status-dot{width:8px;height:8px;border-radius:50%;background:#64748b;flex-shrink:0;}
.hdr-status-dot.ok{background:#22c55e;box-shadow:0 0 6px rgba(34,197,94,.7);}
.hdr-status-dot.warn{background:#f59e0b;box-shadow:0 0 6px rgba(245,158,11,.7);}
.hdr-status-dot.err{background:#ef4444;box-shadow:0 0 6px rgba(239,68,68,.7);}
.hdr-btn-row{display:flex;gap:5px;justify-content:space-between;}
.hdr-icon-btn img{width:17px;height:17px;object-fit:contain;filter:brightness(0) invert(1);}
.hdr-icon-btn{flex:1;background:rgba(255,255,255,0.16);border:1px solid rgba(255,255,255,0.3);
  border-radius:6px;padding:5px 7px;cursor:pointer;display:flex;align-items:center;justify-content:center;
  transition:background .15s;-webkit-tap-highlight-color:transparent;}
.hdr-icon-btn:active{background:rgba(255,255,255,0.2);}
.hdr-icon-btn{flex:1;background:rgba(255,255,255,0.08);border:1px solid rgba(255,255,255,0.16);
  border-radius:6px;padding:4px 6px;cursor:pointer;display:flex;align-items:center;justify-content:center;
  transition:background .15s;-webkit-tap-highlight-color:transparent;}
.refresh-check{color:#22c55e;font-weight:800;font-size:15px;line-height:1;}
body.theme-dark .hdr-status-box{background:rgba(255,255,255,0.04);border-color:#1e293b;}
body.theme-dark .hdr-icon-btn img{filter:brightness(1.9);}
body.theme-dark .ctrl-wrap{background:#1e293b;border-color:#334155;box-shadow:0 2px 10px rgba(0,0,0,0.3);}
body.theme-dark .ctrl-body{background:#1e293b !important;}
/* view toggle */
.sched-view-toggle{display:flex;gap:8px;margin-bottom:10px;}
.sview-btn{flex:1;padding:9px;border:1.5px solid #e5e7eb;background:#f3f4f6;color:#64748b;border-radius:8px;font-weight:700;font-size:12px;cursor:pointer;}
.sview-btn.active{background:#1e3a5f;color:#fff;border-color:#1e3a5f;}
/* week grid */
.week-grid-wrap{background:#fff;border:1px solid #e8edf3;border-radius:12px;padding:12px;margin-bottom:10px;overflow-x:auto;}
.week-grid{display:grid;grid-template-columns:34px repeat(24,1fr);gap:2px;min-width:440px;align-items:center;}
.wg-corner{background:transparent;}
.wg-hh{font-size:8px;color:#94a3b8;font-weight:800;line-height:1;white-space:nowrap;overflow:visible;padding-bottom:3px;}
.wg-dayrow{font-size:10px;font-weight:800;color:#1e3a5f;text-align:right;padding-right:6px;letter-spacing:.5px;}
.wg-track{height:20px;background:#f1f5f9;border-radius:3px;cursor:pointer;transition:background .12s;border:1px solid transparent;}
.wg-track:active{background:#e2e8f0;}
.wg-track.has-on{background:#22c55e;border-color:#16a34a;}
.wg-track.has-off{background:#ef4444;border-color:#dc2626;}
.wr-ctrl-btns{display:flex;gap:6px;flex-wrap:wrap;}
.wr-ctrl-btn{padding:9px 14px;border:1.5px solid #cbd5e1;border-radius:9px;background:#f8fafc;
  color:#1e293b;font-size:13px;font-weight:700;cursor:pointer;transition:all .15s;}
.wr-ctrl-btn.sel{background:#1e3a5f;color:#fff;border-color:#1e3a5f;}
body.theme-dark .wr-ctrl-btn{background:#0f172a;color:#e2e8f0;border-color:#334155;}
body.theme-dark .wr-ctrl-btn.sel{background:#3b82f6;border-color:#3b82f6;}
body.theme-dark .wg-dayrow{color:#93c5fd;}
body.theme-dark .wg-track{background:#0f172a;}
.week-legend{display:flex;gap:12px;flex-wrap:wrap;font-size:10px;color:#475569;margin-top:10px;align-items:center;}
.wl-dot{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:3px;vertical-align:middle;}
.wl-dot.on{background:#22c55e;}.wl-dot.off{background:#ef4444;}
/* add event */
.add-label{font-size:10px;color:#94a3b8;text-transform:uppercase;font-weight:800;letter-spacing:.5px;margin-bottom:6px;}
.time-display-btn{width:100%;padding:12px;border:1.5px solid #cbd5e1;border-radius:9px;background:#f8fafc;color:#1e293b;font-size:22px;font-weight:800;cursor:pointer;letter-spacing:1px;}
.action-seg{display:flex;background:#eef2f7;border:1px solid #dbe2ea;border-radius:9px;padding:3px;gap:3px;height:46px;}
.act-btn{flex:1;border:none;background:transparent;color:#64748b;border-radius:7px;font-weight:800;font-size:14px;cursor:pointer;transition:all .15s;}
.act-btn.active{background:#1e3a5f;color:#fff;}
.sched-preview{display:flex;align-items:center;gap:12px;background:#f8fafc;border:1px solid #e8edf3;border-radius:10px;padding:12px;}
.prev-icon{font-size:26px;}
.prev-title{font-size:13px;font-weight:700;color:#1e293b;}
.prev-sub{font-size:11px;color:#64748b;margin-top:2px;}
/* time wheel */
.time-wheel-box{background:#fff;border-radius:16px;padding:16px 20px;width:94%;max-width:600px;max-height:94vh;overflow-y:auto;text-align:center;}
/* Landscape layout: analog clock on the left, controls on the right (fits short/wide displays) */
.tw-land{display:flex;gap:22px;align-items:center;justify-content:center;flex-wrap:wrap;}
.tw-land-clock{flex:0 0 auto;display:flex;align-items:center;justify-content:center;}
.tw-land-ctrls{flex:1;min-width:240px;display:flex;flex-direction:column;justify-content:center;}
.time-bar{display:flex;gap:6px;align-items:stretch;}
.time-bar .time-display-btn{flex:1;}
.ampm-seg{display:flex;flex-direction:column;gap:3px;width:44px;}
.ampm-btn{flex:1;border:1.5px solid #cbd5e1;background:#f8fafc;color:#64748b;border-radius:7px;font-size:12px;font-weight:800;cursor:pointer;padding:0;}
.ampm-btn.active{background:#1e3a5f;color:#fff;border-color:#1e3a5f;}
body.theme-dark .ampm-btn{background:#0f172a;color:#94a3b8;border-color:#334155;}
body.theme-dark .ampm-btn.active{background:#3b82f6;border-color:#3b82f6;color:#fff;}
.tw-display{font-size:40px;font-weight:800;color:#1e3a5f;letter-spacing:2px;margin-bottom:12px;}
.tw-colon{color:#94a3b8;}
.tw-cols{display:flex;gap:20px;justify-content:center;}
.tw-col{flex:1;}
.tw-label{font-size:10px;color:#94a3b8;text-transform:uppercase;font-weight:800;margin-bottom:6px;}
.tw-stepper{display:flex;flex-direction:column;align-items:center;gap:4px;}
.tw-stepper button{width:100%;padding:10px;border:none;border-radius:8px;background:#1e3a5f;color:#fff;font-size:16px;cursor:pointer;}
.tw-wheel{font-size:30px;font-weight:800;color:#1e293b;padding:6px 0;width:100%;background:#f1f5f9;border-radius:8px;}
.tw-quick{display:flex;gap:6px;margin-top:14px;}
.tw-quick button{flex:1;padding:8px 4px;border:1px solid #cbd5e1;border-radius:7px;background:#fff;color:#1e3a5f;font-size:12px;font-weight:700;cursor:pointer;}


.wtl{display:flex;flex-direction:column;gap:5px;}
.wtl-row{display:flex;align-items:center;gap:6px;}
.wtl-daylbl{width:30px;font-size:10px;font-weight:800;color:#1e3a5f;text-align:right;flex-shrink:0;}
.wtl-track{position:relative;flex:1;height:22px;background:#f1f5f9;border-radius:6px;overflow:hidden;cursor:pointer;}
.wtl-ruler .wtl-track{background:transparent;height:14px;overflow:visible;}
.wtl-tick{position:absolute;font-size:8px;color:#94a3b8;transform:translateX(-50%);font-weight:700;}
.wtl-span{position:absolute;top:0;bottom:0;background:linear-gradient(180deg,#22c55e,#16a34a);opacity:.85;}
.wtl-mark{position:absolute;top:2px;width:3px;height:18px;border-radius:2px;transform:translateX(-50%);cursor:pointer;}
.wtl-mark.on{background:#065f46;box-shadow:0 0 0 2px rgba(34,197,94,.35);}
.wtl-mark.off{background:#7f1d1d;box-shadow:0 0 0 2px rgba(239,68,68,.35);}
body.theme-dark .wtl-track{background:#1e293b;}
body.theme-dark .wtl-daylbl{color:#93c5fd;}

.day-quick-btn{padding:5px 11px;border:none;border-radius:12px;font-size:11px;cursor:pointer;font-weight:bold;}
.day-toggle-btn{padding:7px 9px;border:none;border-radius:8px;font-size:12px;cursor:pointer;font-weight:bold;min-width:40px;text-align:center;}
.day-toggle-btn.sel{background:#1e3a5f;color:white;}
.day-toggle-btn.unsel{background:#e5e7eb;color:#555;}
/* ════ Config controller accordion (BMS tidy) ════ */
.cfg-acc{background:#fff;border:1px solid #e8edf3;border-radius:12px;margin-bottom:8px;overflow:hidden;transition:box-shadow .2s;}
.cfg-acc.expanded{box-shadow:0 4px 16px rgba(15,23,42,0.08);border-color:#cbd5e1;}
.cfg-acc-hdr{display:flex;align-items:center;justify-content:space-between;padding:12px 14px;cursor:pointer;user-select:none;gap:10px;}
.cfg-acc-hdr:active{background:#f8fafc;}
.cfg-acc-left{display:flex;align-items:center;gap:9px;min-width:0;flex:1;}
.cfg-acc-dot{width:9px;height:9px;border-radius:50%;background:#cbd5e1;flex-shrink:0;}
.cfg-acc-dot.on{background:#22c55e;box-shadow:0 0 5px rgba(34,197,94,.6);}
.cfg-acc-dot.off{background:#94a3b8;}
.cfg-acc-name{font-size:14px;font-weight:700;color:#1e293b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.cfg-acc-type{font-size:11px;color:#94a3b8;font-weight:600;white-space:nowrap;}
.cfg-acc-right{display:flex;align-items:center;gap:12px;flex-shrink:0;}
.cfg-acc-toggle{display:flex;align-items:center;gap:5px;font-size:11px;color:#64748b;font-weight:700;cursor:pointer;}
.cfg-acc-chev{font-size:11px;color:#94a3b8;transition:transform .2s;width:12px;text-align:center;}
.cfg-acc-body{display:none;padding:0 14px 14px;border-top:1px solid #f0f4f8;}
.cfg-acc-body.open{display:block;}
.cfg-grid2{display:grid;grid-template-columns:1fr 1fr;gap:10px;}
.cfg-inline-row{display:flex;align-items:center;gap:8px;padding:10px 0;border-bottom:1px solid #f0f4f8;}
.cfg-inline-label{font-size:12px;color:#475569;font-weight:700;white-space:nowrap;}
.cfg-inline-row .led-cfg-sel{flex:1;min-width:0;}
.cfg-mini-btn{padding:7px 14px;background:#1e3a5f;color:#fff;border:none;border-radius:7px;font-size:12px;font-weight:700;cursor:pointer;white-space:nowrap;}
.cfg-acc-actions{display:flex;align-items:center;gap:10px;margin-top:14px;padding-top:12px;border-top:1px solid #f0f4f8;}
.cfg-acc-actions .test-result{flex:1;font-size:12px;}
.cfg-del-btn{padding:8px 12px;background:#fee2e2;color:#dc2626;border:1px solid #fecaca;border-radius:7px;font-size:13px;cursor:pointer;font-weight:700;}
.cfg-del-btn:active{background:#fecaca;}
.cfg-acc-body .cfg-field{margin-top:12px;}
.cfg-acc-body .cfg-field:first-child{margin-top:14px;}
/* dark theme */
body.theme-dark .cfg-acc{background:#0e0e12;border-color:#1e1e26;}
body.theme-dark .cfg-acc.expanded{border-color:#334155;box-shadow:0 4px 16px rgba(0,0,0,.4);}
body.theme-dark .cfg-acc-hdr:active{background:#14141a;}
body.theme-dark .cfg-acc-name{color:#f4f4f5;}
body.theme-dark .cfg-acc-body,body.theme-dark .cfg-inline-row,body.theme-dark .cfg-acc-actions{border-color:#1e1e26;}
body.theme-dark .cfg-inline-label{color:#a1a1aa;}
body.theme-dark .cfg-del-btn{background:#3a1a1a;border-color:#7f1d1d;}
.cfg-page{padding:12px;}
.cfg-page h1{font-size:18px;font-weight:bold;color:#1e3a5f;margin-bottom:16px;}
.cfg-row{background:white;border:1px solid #ddd;border-radius:8px;padding:12px;margin-bottom:10px;}
.cfg-row-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px;}
.cfg-row-title{font-size:14px;font-weight:bold;color:#1e3a5f;}
.cfg-enabled-toggle{display:flex;align-items:center;gap:6px;font-size:12px;color:#555;}
.cfg-field{margin-bottom:8px;}
.cfg-field label{display:block;font-size:11px;color:#888;margin-bottom:3px;text-transform:uppercase;}
.cfg-field input,.cfg-field select{width:100%;padding:8px 10px;border:1px solid #ccc;border-radius:6px;font-size:13px;}
.cfg-row-footer{display:flex;align-items:center;justify-content:space-between;margin-top:10px;padding-top:10px;border-top:1px solid #f0f0f0;}
/* Config page bars now match the AI page's ai-panel-hdr bar language (dark-blue bar,
   white icon/text/label) so the whole app reads as one synchronous UI instead of the
   Config tab looking like a different, disconnected style. */
.cfg-tile{display:flex;align-items:center;gap:14px;background:#1e3a5f;border:1px solid #1e3a5f;border-radius:14px;padding:16px;margin-bottom:12px;cursor:pointer;transition:transform .15s,box-shadow .2s;box-shadow:0 2px 8px rgba(0,0,0,0.05);-webkit-tap-highlight-color:transparent;}
.cfg-tile:active{transform:scale(.98);}
.cfg-tile:hover{background:#1e4d7a;}
.cfg-tile-wizard{background:#1e3a5f;border:1px solid #1e3a5f;}
.cfg-tile-wizard:hover{background:#1e4d7a;}
.cfg-tile-wizard .cfg-tile-title{color:#fff;}
.cfg-tile-wizard .cfg-tile-desc{color:#bfdbfe;opacity:1;}
.cfg-tile-wizard .cfg-tile-arrow{color:#bfdbfe;}
.cfg-tile-icon{font-size:30px;line-height:1;}
.cfg-tile-body{flex:1;}
.cfg-tile-title{font-size:15px;font-weight:800;color:#fff;}
.cfg-tile-desc{font-size:11px;color:#bfdbfe;margin-top:2px;}
.cfg-tile-arrow{font-size:18px;color:#bfdbfe;transition:transform .25s;}
.cfg-tile-arrow.open{transform:rotate(90deg);}
body.theme-dark .cfg-tile{background:#1e3a8a;border-color:#1e3a8a;}
body.theme-dark .cfg-tile:hover{background:#1e40af;}
body.theme-dark .cfg-tile-title{color:#fff;}
body.theme-dark .cfg-tile-advanced{background:#1e3a8a;}
body.theme-blue .cfg-tile{background:#1e40af;border-color:#1e40af;}
body.theme-blue .cfg-tile:hover{background:#1d4ed8;}
body.theme-blue .cfg-tile-title{color:#fff;}
body.theme-blue .cfg-tile-desc{color:#bfdbfe;}
body.theme-blue .cfg-tile-arrow{color:#bfdbfe;}

/* Technician Mode + Write Any Register — white BG to match */
config-content .cfg-section{background:#fff !important;border:1px solid #e8edf3 !important;}
config-content .write-panel,
config-content .write-panel h2,
config-content #cfg-write-reg-body{background:#fff !important;color:#1e293b !important;}
config-content .write-panel h2{border-bottom:1px solid #e8edf3 !important;}
/* icon images replacing emoji */
.ic{width:1.1em;height:1.1em;vertical-align:-0.18em;object-fit:contain;}
.tab-ic{display:block;width:20px;height:20px;margin:0 auto 2px;object-fit:contain;}
.tile-icon img{width:22px;height:22px;object-fit:contain;}
.mode-ic{width:15px;height:15px;vertical-align:-2px;margin-right:3px;object-fit:contain;}
/* lighten navy icons on dark theme for contrast */
body.theme-dark .tab-ic,body.theme-dark .ic,body.theme-dark .mode-ic{filter:brightness(1.9) saturate(1.1);}

/* ════ CONFIG PAGE — BMS professional styling ════ */
.cfg-topbar{display:flex;justify-content:space-between;align-items:flex-start;
  margin-bottom:18px;padding-bottom:14px;border-bottom:2px solid #e8edf3;}
.cfg-page h1{font-size:19px;font-weight:800;color:#1e293b;letter-spacing:-.3px;}
.cfg-breadcrumb{font-size:11px;color:#94a3b8;margin-top:3px;letter-spacing:.2px;}
.cfg-logout-btn{background:#fff;color:#dc2626;border:1.5px solid #fecaca;border-radius:8px;
  padding:8px 16px;font-size:12px;font-weight:700;cursor:pointer;transition:all .15s;white-space:nowrap;}
.cfg-logout-btn:hover{background:#fef2f2;}

/* Section card — unified BMS panel */
.cfg-section{background:#fff;border:1px solid #e8edf3;border-radius:12px;
  padding:16px;margin-bottom:12px;}
.cfg-section-row{display:flex;align-items:center;justify-content:space-between;gap:12px;}
.cfg-section-title{font-size:14px;font-weight:700;color:#1e293b;}
.cfg-section-sub{font-size:11px;color:#94a3b8;margin-top:2px;}

/* Section group label */
.cfg-group-label{font-size:10px;font-weight:800;text-transform:uppercase;letter-spacing:1.2px;
  color:#94a3b8;margin:18px 2px 8px;display:flex;align-items:center;gap:8px;}
.cfg-group-label::after{content:'';flex:1;height:1px;background:#e8edf3;}

/* Dark/Pro theme */
body.theme-dark .cfg-topbar{border-color:#1e1e26;}
body.theme-dark .cfg-page h1{color:#f4f4f5;}
body.theme-dark .cfg-section{background:#0e0e12;border-color:#1e1e26;}
body.theme-dark .cfg-section-title{color:#f4f4f5;}
body.theme-dark .cfg-logout-btn{background:#0e0e12;border-color:#3a1a1a;}
body.theme-dark .cfg-group-label::after{background:#1e1e26;}
body.theme-blue .cfg-section{background:#eff6ff;border-color:#bfdbfe;}


/* ── PRO dark theme (premium thermostat look) ── */
/* ════════════════════════════════════════════════════════════
   PRO THEME — premium dark UI (all tabs)
   palette: base #08080a · card #0e0e12 · elevated #14141a
   border #1e1e26 · accent #3b82f6 / glow · text #f4f4f5
   ════════════════════════════════════════════════════════════ */

/* ── Header ── */

/* ── Tabs ── */

/* ── View toggle (Dashboard) ── */

/* ── Controller card shell ── */

/* badges */

/* ── Metric tiles — flat, hairline, thin numerals ── */
/* CO accent states keep a thin colored underline only */

/* CO₂ threshold scale */
.co-scale-wrap{position:relative;width:100%;height:6px;border-radius:3px;margin-top:8px;
  background:linear-gradient(90deg,#10b981 0%,#10b981 33%,#f59e0b 33%,#f59e0b 66%,#ef4444 66%,#ef4444 100%);}
.co-scale-marker{position:absolute;top:50%;width:3px;height:14px;background:#1e293b;
  border:1.5px solid #fff;border-radius:2px;transform:translate(-50%,-50%);
  box-shadow:0 1px 3px rgba(0,0,0,0.3);transition:left .4s ease;}
.co-scale-labels{display:flex;justify-content:space-between;font-size:7px;font-weight:700;
  color:#94a3b8;margin-top:3px;letter-spacing:.2px;text-transform:uppercase;}
body.theme-dark .co-scale-marker{background:#fff;border-color:#0f172a;}
body.theme-dark .co-scale-labels{color:#64748b;}

/* ── Setpoint knob cell ── */

/* ── Mode + Fan buttons — minimal outline, glow on active ── */

/* ── Grid view ── */

/* ── Control tab rows ── */

/* ── Panels / accordions (Trends, Schedule, AI, Config) ── */

/* sleep / sub buttons */

/* ── Trends chart container ── */

/* ── Schedule ── */

/* ── Config tiles ── */

/* ── Modals ── */

/* ── AI motion cards ── */

/* ── Virtual keyboard ── */

.test-btn{padding:7px 18px;background:#1e3a5f;color:white;border:none;border-radius:6px;font-size:12px;cursor:pointer;}
.test-result{font-size:12px;min-height:18px;font-weight:bold;}
.test-result.ok{color:#22c55e;}
.test-result.err{color:#ef4444;}
.test-result.testing{color:#f59e0b;}
.connect-all-btn{width:100%;padding:14px;background:#22c55e;color:white;border:none;border-radius:8px;font-size:15px;font-weight:bold;cursor:pointer;margin-top:6px;}
.connect-all-btn:disabled{background:#9ca3af;cursor:not-allowed;}
.add-ctrl-btn{width:100%;padding:12px;background:#7c3aed;color:white;border:none;border-radius:8px;font-size:14px;font-weight:bold;cursor:pointer;margin-top:8px;}
.unlock-banner{background:#22c55e;color:white;padding:10px 14px;border-radius:8px;text-align:center;font-size:13px;font-weight:bold;margin-top:10px;display:none;cursor:pointer;}
.serial-field{background:white;border:1px solid #ddd;border-radius:8px;padding:12px;margin-bottom:10px;}
.serial-field label{display:block;font-size:11px;color:#888;margin-bottom:3px;text-transform:uppercase;}
.serial-field input,.serial-field select{width:100%;padding:8px 10px;border:1px solid #ccc;border-radius:6px;font-size:13px;}
.serial-save-btn{padding:8px 20px;background:#1e3a5f;color:white;border:none;border-radius:6px;font-size:13px;cursor:pointer;margin-top:8px;}
.overlay-panel{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(15,23,42,0.92);z-index:9000;align-items:center;justify-content:center;}
.overlay-box{background:white;border-radius:14px;padding:20px;width:92%;max-width:380px;max-height:80vh;overflow-y:auto;}

/* Set Date & Time popup */
.st-box{max-width:560px;width:94%;}
.st-source{font-size:12px;font-weight:700;border-radius:8px;padding:8px 10px;margin-bottom:14px;text-align:center;background:#f1f5f9;color:#475569;}
.st-head{display:flex;align-items:center;gap:10px;}
.st-location{margin-left:auto;font-size:12px;font-weight:800;letter-spacing:.02em;color:#1d4ed8;background:#dbeafe;border-radius:999px;padding:4px 12px;white-space:nowrap;}
.st-location:empty{display:none;}
body.theme-dark .st-location{background:var(--d-inset);color:#93c5fd;}
.st-tzrow{margin:-4px 0 14px;}
.st-tzlabel{display:block;font-size:11px;font-weight:700;color:#64748b;text-transform:uppercase;letter-spacing:.04em;margin-bottom:6px;}
.st-tzctrls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;}
.st-tzfilter{flex:1 1 140px;min-width:120px;border:1px solid #cbd5e1;border-radius:8px;padding:8px 10px;font-size:13px;color:#0f172a;background:#fff;}
.st-tzselect{flex:1 1 160px;min-width:140px;border:1px solid #cbd5e1;border-radius:8px;padding:8px 10px;font-size:13px;color:#0f172a;background:#fff;max-width:220px;}
.st-tzbtn{border:0;border-radius:8px;padding:8px 14px;font-size:13px;font-weight:700;background:#1e3a5f;color:#fff;cursor:pointer;white-space:nowrap;}
.st-tzbtn:active{opacity:.85;}
body.theme-dark .st-tzfilter,body.theme-dark .st-tzselect{background:var(--d-inset);color:var(--d-fg);border-color:var(--d-line);}
.st-source.ntp{background:#dcfce7;color:#15803d;}
.st-source.rtc{background:#dbeafe;color:#1d4ed8;}
.st-source.manual{background:#fef3c7;color:#b45309;}
.st-spins{display:flex;align-items:flex-start;justify-content:center;gap:8px;flex-wrap:wrap;}
.st-spin{display:flex;flex-direction:column;align-items:center;gap:6px;}
.st-arrow{width:52px;height:38px;border:none;border-radius:8px;background:#e2e8f0;color:#1e3a5f;font-size:15px;font-weight:800;cursor:pointer;-webkit-tap-highlight-color:transparent;}
.st-arrow:active{background:#cbd5e1;transform:scale(.94);}
.st-val{min-width:52px;text-align:center;font-size:22px;font-weight:800;color:#1e3a5f;padding:4px 0;font-variant-numeric:tabular-nums;}
.st-lbl{font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.5px;color:#94a3b8;}
.st-sep{align-self:center;font-size:20px;color:#cbd5e1;padding:0 2px;}
.st-msg{font-size:12px;font-weight:700;text-align:center;min-height:16px;margin-top:12px;}
.st-msg.ok{color:#16a34a;}.st-msg.err{color:#dc2626;}
.st-btn{border:none;border-radius:8px;font-size:13px;font-weight:700;cursor:pointer;padding:11px;}
.st-btn-now{flex:0 0 auto;padding:11px 16px;background:#e5e7eb;color:#475569;}
.st-btn-apply{flex:1;background:#1e3a5f;color:#fff;}
.st-btn-close{flex:0 0 auto;padding:11px 18px;background:#94a3b8;color:#fff;}
body.theme-dark .st-source{background:var(--d-inset);color:var(--d-dim);}
body.theme-dark .st-arrow{background:var(--d-inset);color:#dbeafe;}
body.theme-dark .st-val{color:#dbeafe;}
/* Wi-Fi setup popup — landscape two-column, fits the screen */.wifi-box{max-width:660px;width:94%;max-height:90vh;display:flex;flex-direction:column;}
.wifi-head{display:flex;align-items:center;gap:12px;margin-bottom:10px;}
.wifi-head h3{margin:0;white-space:nowrap;}
.wifi-status{flex:1;font-size:12px;color:#475569;background:#f1f5f9;border-radius:8px;padding:7px 10px;margin:0;text-align:right;}
.wifi-status.ok{background:#dcfce7;color:#15803d;font-weight:700;}
.wifi-body{display:flex;gap:12px;min-height:0;flex:1;}
.wifi-col{flex:1;min-width:0;display:flex;flex-direction:column;}
.wifi-list{flex:1;min-height:120px;max-height:46vh;overflow-y:auto;display:flex;flex-direction:column;gap:6px;border:1px solid #e2e8f0;border-radius:10px;padding:6px;margin-bottom:8px;}
.wifi-empty{text-align:center;color:#94a3b8;font-size:12px;padding:18px;}
.wifi-item{display:flex;align-items:center;gap:10px;padding:9px 11px;border-radius:8px;background:#f8fafc;border:1px solid transparent;cursor:pointer;transition:all .12s;}
.wifi-item:active{background:#eef2f7;}
.wifi-item.sel{background:#dbeafe;border-color:#3b82f6;}
.wifi-item.active-net{background:#dcfce7;}
.wifi-name{flex:1;font-size:13px;font-weight:700;color:#1e3a5f;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.wifi-meta{display:flex;align-items:center;gap:8px;font-size:11px;color:#64748b;}
.wifi-bars{font-size:13px;letter-spacing:1px;}
.wifi-col-connect{justify-content:flex-start;}
.wifi-placeholder{display:flex;align-items:center;justify-content:center;text-align:center;flex:1;color:#94a3b8;font-size:12px;border:1px dashed #cbd5e1;border-radius:10px;padding:16px;}
.wifi-connect-row{background:#f8fafc;border:1px solid #e2e8f0;border-radius:10px;padding:12px;}
.wifi-sel-label{font-size:12px;color:#475569;margin-bottom:8px;}
.wifi-sel-label b{color:#1e3a5f;}
#wifi-pass{width:100%;padding:11px 12px;border:1px solid #cbd5e1;border-radius:8px;font-size:14px;box-sizing:border-box;}
.wifi-show{display:flex;align-items:center;gap:6px;font-size:11px;color:#64748b;margin:8px 0;cursor:pointer;}
.wifi-msg{font-size:12px;font-weight:700;margin-top:8px;min-height:16px;text-align:center;}
.wifi-msg.err{color:#dc2626;}
.wifi-msg.ok{color:#16a34a;}
.wifi-btn{border:none;border-radius:8px;font-size:13px;font-weight:700;cursor:pointer;padding:11px;}
.wifi-btn-scan{background:#e5e7eb;color:#555;width:100%;}
.wifi-btn-connect{background:#1e3a5f;color:#fff;width:100%;margin-top:10px;}
.wifi-btn-close{background:#94a3b8;color:#fff;padding:10px 22px;}
.wifi-foot{display:flex;justify-content:flex-end;margin-top:12px;}
body.theme-dark .wifi-status{background:var(--d-inset);color:var(--d-dim);}
body.theme-dark .wifi-list{background:var(--d-inset);border-color:var(--d-line);}
body.theme-dark .wifi-item{background:var(--d-surf);}
body.theme-dark .wifi-item.sel{background:#1e3a5f;border-color:var(--d-acc);}
body.theme-dark .wifi-name{color:#dbeafe;}
body.theme-dark .wifi-placeholder{border-color:var(--d-line);color:var(--d-dim);}
body.theme-dark .wifi-connect-row{background:var(--d-inset);border-color:var(--d-line);}
body.theme-dark #wifi-pass{background:var(--d-surf);color:var(--d-txt);border-color:var(--d-line);}
.overlay-box h3{font-size:16px;font-weight:bold;color:#1e3a5f;margin-bottom:14px;border-bottom:2px solid #e5e7eb;padding-bottom:8px;}
.net-row{display:flex;justify-content:space-between;align-items:flex-start;padding:8px 0;border-bottom:1px solid #f0f0f0;font-size:13px;}
.net-label{color:#888;font-size:11px;text-transform:uppercase;flex-shrink:0;margin-right:10px;}
.net-val{color:#1e3a5f;font-weight:bold;word-break:break-all;text-align:right;}
.net-iface-badge{display:inline-block;background:#1e3a5f;color:white;border-radius:6px;padding:2px 8px;font-size:11px;font-weight:bold;margin-bottom:4px;}
.net-iface-badge.wifi{background:#0891b2;}
.email-ctrl-btn{display:block;width:100%;padding:11px 16px;margin-bottom:8px;border:2px solid #e5e7eb;border-radius:8px;font-size:13px;cursor:pointer;background:white;color:#333;text-align:left;font-weight:bold;}
.email-ctrl-btn:hover,.email-ctrl-btn.sel{border-color:#1e3a5f;background:#eef2ff;color:#1e3a5f;}
.email-ctrl-btn.all-opt{border-color:#7c3aed;}
.email-ctrl-btn.all-opt:hover,.email-ctrl-btn.all-opt.sel{background:#f5f3ff;color:#7c3aed;border-color:#7c3aed;}
.sp-step-btn{width:34px;height:34px;flex:0 0 34px;border-radius:50%;border:none;cursor:pointer;font-size:16px;font-weight:bold;display:flex;align-items:center;justify-content:center;transition:transform .1s,opacity .2s;user-select:none;-webkit-tap-highlight-color:transparent;box-shadow:0 2px 6px rgba(0,0,0,0.18);}
.sp-step-btn:active{transform:scale(0.90);}
.sp-step-dn{background:linear-gradient(135deg,#3b82f6,#2563eb);color:white;}
.sp-step-up{background:linear-gradient(135deg,#f97316,#ef4444);color:white;}
/* Setpoint knob layout — dial flanked by ▼ ▲ so it can be larger */
.knob-label{font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:.8px;color:#1d4ed8;opacity:0.75;margin-bottom:1px;display:flex;align-items:center;justify-content:center;gap:4px;}
.knob-row{display:flex;align-items:center;justify-content:center;gap:82px;width:100%;}
.qs-help{margin-top:12px;padding-top:10px;border-top:1px solid #eef2f7;display:flex;flex-direction:column;gap:5px;font-size:11px;color:#64748b;}
.qs-help b{color:#1e3a5f;}
body.theme-dark .qs-help{border-color:var(--d-line);color:var(--d-dim);}
body.theme-dark .qs-help b{color:#dbeafe;}
.sp-knob{width:124px;height:124px;flex:0 0 auto;cursor:grab;touch-action:none;}
.sp-result{font-size:10px;color:#22c55e;font-weight:bold;min-height:13px;text-align:center;margin-top:1px;}
body.theme-dark .knob-label{color:#93c5fd;}
.ai-motion-grid{display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:10px;}
.ai-motion-card{background:#0e1520;color:#cdd9e5;border-radius:10px;padding:14px;border:1px solid #1e2d42;}
.ai-motion-card .ai-lbl{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#4a6070;margin-bottom:5px;}
.ai-motion-card .ai-val{font-size:24px;font-weight:bold;color:#00e5ff;line-height:1;}
.ai-motion-card .ai-val.green{color:#00ff88;}
.ai-motion-card .ai-val.yellow{color:#ffd060;}
.ai-motion-card .ai-val.red{color:#ff3b6e;}
.ai-motion-card .ai-sub{font-size:10px;color:#4a6070;margin-top:4px;font-family:'Courier New',monospace;}
.ai-bar-wrap{height:5px;background:#1e2d42;border-radius:3px;margin-top:8px;overflow:hidden;}
.ai-bar-fill{height:100%;border-radius:3px;background:linear-gradient(90deg,#00e5ff,#00ff88);transition:width .2s ease;}
.ai-status-strip{background:#0e1520;color:#cdd9e5;border-radius:10px;padding:12px 14px;margin-bottom:8px;display:flex;align-items:center;gap:12px;border:1px solid #1e2d42;}
.ai-status-dot{width:10px;height:10px;border-radius:50%;background:#4a6070;flex-shrink:0;}
.ai-status-dot.active{background:#00ff88;animation:pulse 1.5s infinite;}
@keyframes pulse{0%,100%{box-shadow:0 0 0 0 rgba(0,255,136,.4);}50%{box-shadow:0 0 0 8px rgba(0,255,136,0);}}
.ai-status-text{flex:1;font-size:13px;font-weight:bold;color:#fff;}
.ai-history-chart{background:#0e1520;border:1px solid #1e2d42;border-radius:10px;padding:14px;margin-bottom:8px;}
.ai-history-chart .ai-lbl{font-size:9px;letter-spacing:2px;text-transform:uppercase;color:#4a6070;margin-bottom:8px;}
.ai-bars-container{display:flex;align-items:flex-end;gap:3px;height:50px;}
.ai-bar{flex:1;background:#00e5ff;opacity:.5;border-radius:3px 3px 0 0;min-height:3px;transition:height .25s ease;}
.ai-bar:last-child{opacity:1;}
.led-legend{display:flex;flex-direction:column;gap:6px;margin-top:10px;background:#f8f9fa;border-radius:8px;padding:10px;}
.led-legend-row{display:flex;align-items:center;gap:8px;font-size:12px;color:#444;}
.led-dot{width:14px;height:14px;border-radius:50%;flex-shrink:0;}
.confirm-modal{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);z-index:999999;align-items:center;justify-content:center;}
.confirm-box{background:white;border-radius:14px;padding:24px;width:88%;max-width:340px;text-align:center;}
.confirm-box h3{font-size:16px;font-weight:bold;color:#1e3a5f;margin-bottom:10px;}
.confirm-box p{font-size:13px;color:#555;margin-bottom:20px;}
.confirm-btns{display:flex;gap:10px;}
.confirm-btns button{flex:1;padding:12px;border:none;border-radius:8px;font-size:14px;font-weight:bold;cursor:pointer;}
.btn-danger{background:#ef4444;color:white;}
.btn-cancel{background:#e5e7eb;color:#555;}
.ctrl-wrap{position:relative;margin-top:12px;border-radius:12px;overflow:hidden;
  border:1px solid #e2e8f0;background:#fff;box-shadow:0 2px 10px rgba(15,23,42,0.06);}
.ctrl-bg-overlay{display:none !important;}
.ctrl-wrap > .ctrl-header,.ctrl-wrap > div:not(.ctrl-bg-overlay){position:relative;z-index:1;}
.ctrl-bg-overlay.heat{background:radial-gradient(circle at 20% 80%,rgba(239,68,68,.6) 0%,transparent 50%),radial-gradient(circle at 80% 20%,rgba(251,146,60,.5) 0%,transparent 50%),linear-gradient(135deg,#fbbf24,#ef4444);}
.ctrl-bg-overlay.heat::after{content:"🔥";position:absolute;right:14px;top:14px;font-size:42px;opacity:.35;transform:rotate(-5deg);}
.ctrl-bg-overlay.cool{background:radial-gradient(circle at 30% 30%,rgba(147,197,253,.6) 0%,transparent 50%),radial-gradient(circle at 70% 70%,rgba(96,165,250,.5) 0%,transparent 50%),linear-gradient(135deg,#dbeafe,#3b82f6);}
.ctrl-bg-overlay.cool::after{content:"❄";position:absolute;right:14px;top:14px;font-size:42px;opacity:.4;color:#0c4a6e;}
.ctrl-bg-overlay.off{background:linear-gradient(135deg,#e5e7eb,#9ca3af);}
.ctrl-bg-overlay.off::after{content:"⏻";position:absolute;right:14px;top:14px;font-size:38px;opacity:.3;color:#374151;}

@keyframes fanRotate{from{transform:rotate(0deg);}to{transform:rotate(360deg);}}
.wind-bg{position:absolute;inset:0;overflow:hidden;border-radius:10px;pointer-events:none;opacity:0;transition:opacity .4s ease;z-index:0;}
.wind-bg{display:none;}
.wind-low{opacity:.25;}.wind-low::before{animation-duration:8s;}
.wind-mid1{opacity:.45;}.wind-mid1::before{animation-duration:5s;}
.wind-mid2{opacity:.65;}.wind-mid2::before{animation-duration:3s;}
.wind-high{opacity:.9;}.wind-high::before{animation-duration:1.5s;}
@keyframes windMove{from{transform:translateX(-30%);}to{transform:translateX(30%);}}
/* Login gate shared styles */
.login-gate-box{background:white;border-radius:12px;padding:24px;border:1px solid #ddd;max-width:380px;margin:0 auto;}
.login-gate-box h2{font-size:18px;color:#1e3a5f;margin-bottom:6px;text-align:center;}
.login-gate-box p{font-size:12px;color:#888;margin-bottom:18px;text-align:center;}
.login-gate-box label{font-size:11px;color:#888;text-transform:uppercase;display:block;margin-bottom:4px;}
.login-gate-box input{width:100%;padding:10px;border:1px solid #ccc;border-radius:6px;font-size:14px;margin-bottom:12px;}
.login-gate-box button.enter-btn{width:100%;padding:12px;background:#1e3a5f;color:white;border:none;border-radius:8px;font-size:15px;font-weight:bold;cursor:pointer;}
.login-gate-box .err-msg{font-size:12px;color:#ef4444;text-align:center;margin-top:10px;min-height:16px;}
/* DARK theme */
body.theme-dark .metric-box{background:#0f172a !important;border-color:#334155 !important;}
body.theme-dark .metric-box .card-label{color:#94a3b8 !important;}
body.theme-dark .metric-box .card-value,body.theme-dark .metric-box [id^="temp-"],body.theme-dark .metric-box [id^="coValue-"]{color:#e2e8f0 !important;}
body.theme-blue .metric-box{background:#eff6ff !important;border-color:#93c5fd !important;}
body.theme-dark{background:#0f172a;color:#e2e8f0;}
body.theme-dark .header{background:#0f172a;border-bottom:1px solid #1e293b;}
body.theme-dark .tabs{background:#1e293b;border-bottom-color:#334155;}
body.theme-dark .tab{color:#94a3b8;}
body.theme-dark .tab.active{color:#60a5fa;border-bottom-color:#60a5fa;}
body.theme-dark .card,body.theme-dark .reg-section,body.theme-dark .toggle-row,body.theme-dark .cfg-row,body.theme-dark .serial-field{background:#1e293b;border-color:#334155;color:#e2e8f0;}
body.theme-dark .card-label,body.theme-dark .card-sub{color:#94a3b8;}
body.theme-dark .card-value{color:#e2e8f0;}
body.theme-dark th{background:#0f172a;color:#94a3b8;}
body.theme-dark td{border-color:#334155;color:#cbd5e1;}
body.theme-dark .val-cell{color:#60a5fa;}
body.theme-dark .write-row label,body.theme-dark .cfg-field label,body.theme-dark .serial-field label{color:#94a3b8;}
body.theme-dark .write-row input,body.theme-dark .write-row select,body.theme-dark .cfg-field input,body.theme-dark .cfg-field select,body.theme-dark .serial-field input,body.theme-dark .serial-field select{background:#0f172a;color:#e2e8f0;border-color:#334155;}

body.theme-dark .cfg-page h1{color:#60a5fa;}
body.theme-dark .cfg-row-title{color:#60a5fa;}
body.theme-dark .cfg-enabled-toggle{color:#cbd5e1;}
body.theme-dark .ctrl-header,body.theme-dark .reg-section-title{background:#1e3a8a;}
body.theme-dark .ctrl-wrap > div:not(.ctrl-bg-overlay):not(.ctrl-header){background:#1e293b !important;border-color:#334155 !important;}
body.theme-dark .toggle-btn.off{background:#334155;color:#94a3b8;}
body.theme-dark .toggle-btn.on{background:#3b82f6;}
body.theme-dark .day-toggle-btn.unsel{background:#334155;color:#94a3b8;}
body.theme-dark .day-toggle-btn.sel{background:#3b82f6;}
body.theme-dark .onoff-btn.off{background:#334155;color:#94a3b8;}
body.theme-dark .sched-ctrl-btn{background:#1e293b;color:#60a5fa;border-color:#60a5fa;}
body.theme-dark .sched-ctrl-btn.active{background:#60a5fa;color:#0f172a;}
body.theme-dark .sched-ctrl-btn.all-btn{color:#a78bfa;border-color:#a78bfa;}
body.theme-dark .sched-ctrl-btn.all-btn.active{background:#a78bfa;color:#0f172a;}
body.theme-dark #schedule-list > div{background:#0f172a !important;}
body.theme-dark #schedule-list > div > div:nth-child(n+2){background:#1e293b !important;color:#e2e8f0;}
body.theme-dark .overlay-box,body.theme-dark .confirm-box,body.theme-dark .login-gate-box{background:#1e293b;color:#e2e8f0;}
body.theme-dark .overlay-box h3,body.theme-dark .confirm-box h3{color:#60a5fa;border-color:#334155;}
body.theme-dark .confirm-box p{color:#cbd5e1;}
body.theme-dark .net-val{color:#60a5fa;}
body.theme-dark .net-row{border-color:#334155;}
body.theme-dark .net-label{color:#94a3b8;}
body.theme-dark .email-ctrl-btn{background:#0f172a;color:#e2e8f0;border-color:#334155;}
body.theme-dark .email-ctrl-btn:hover,body.theme-dark .email-ctrl-btn.sel{background:#1e40af;color:#fff;border-color:#60a5fa;}
body.theme-dark .test-btn,body.theme-dark .serial-save-btn,body.theme-dark .write-row button{background:#3b82f6;}
body.theme-dark .add-ctrl-btn{background:#7c3aed;}
body.theme-dark .connect-all-btn{background:#16a34a;}
body.theme-dark .led-legend{background:#1e293b;}
body.theme-dark .led-legend-row{color:#cbd5e1;}
body.theme-dark .login-gate-box label{color:#94a3b8;}
body.theme-dark .login-gate-box input{background:#0f172a;color:#e2e8f0;border-color:#334155;}
/* BLUE theme */
body.theme-blue{background:#dbeafe;color:#1e3a5f;}
body.theme-blue .header{background:#1e40af;}
body.theme-blue .tabs{background:#bfdbfe;border-bottom-color:#60a5fa;}
body.theme-blue .tab{color:#1e40af;}
body.theme-blue .tab.active{color:#1e3a8a;border-bottom-color:#1e3a8a;}
body.theme-blue .card,body.theme-blue .reg-section,body.theme-blue .toggle-row,body.theme-blue .cfg-row,body.theme-blue .serial-field{background:#eff6ff;border-color:#93c5fd;}
body.theme-blue th{background:#dbeafe;color:#1e40af;}
body.theme-blue td{border-color:#bfdbfe;}
body.theme-blue .val-cell{color:#1e40af;}
body.theme-blue .card-value{color:#1e3a8a;}
body.theme-blue .card-label,body.theme-blue .card-sub{color:#475569;}
body.theme-blue .ctrl-header,body.theme-blue .reg-section-title{background:#1e40af;}

body.theme-blue .cfg-page h1{color:#1e3a8a;}
body.theme-blue .cfg-row-title{color:#1e40af;}
body.theme-blue .toggle-btn.off{background:#dbeafe;color:#1e40af;}
body.theme-blue .toggle-btn.on{background:#1e40af;color:white;}
body.theme-blue .day-toggle-btn.unsel{background:#dbeafe;color:#1e40af;}
body.theme-blue .day-toggle-btn.sel{background:#1e40af;color:white;}
body.theme-blue .sched-ctrl-btn{background:#eff6ff;color:#1e40af;border-color:#1e40af;}
body.theme-blue .sched-ctrl-btn.active{background:#1e40af;color:white;}
body.theme-blue .write-row input,body.theme-blue .write-row select,body.theme-blue .cfg-field input,body.theme-blue .cfg-field select,body.theme-blue .serial-field input,body.theme-blue .serial-field select{background:#fff;color:#1e3a5f;border-color:#93c5fd;}
body.theme-blue #schedule-list > div{background:#eff6ff !important;}
body.theme-blue .overlay-box,body.theme-blue .confirm-box{background:#fff;}
body.theme-blue .overlay-box h3,body.theme-blue .confirm-box h3{color:#1e40af;}
body.theme-blue .net-val{color:#1e40af;}
body.theme-blue .led-legend{background:#dbeafe;}
#hdr-weather{transition:opacity 0.3s;}
.theme-btn.active{border-color:#3b82f6 !important;transform:scale(1.05);}
/* ── Segmented Fan Speed control ── */
.fan-seg{display:flex;background:#eef2f7;border:1px solid #dbe2ea;border-radius:11px;padding:3px;gap:3px;}
.fan-btn{flex:1;padding:10px 4px;border:none;background:transparent;color:#64748b;border-radius:8px;font-weight:700;cursor:pointer;font-size:12px;transition:color .15s,background .15s;position:relative;display:flex;flex-direction:column;align-items:center;gap:2px;}
.fan-btn .fan-ico{font-size:14px;line-height:1;}
.fan-btn.active{background:linear-gradient(135deg,#0891b2,#0e7490);color:#fff;box-shadow:0 2px 8px rgba(8,145,178,0.45);transform:scale(1.03);font-weight:800;}
.fan-btn:active{transform:scale(0.96);}
body.theme-dark .fan-seg{background:#0f172a;border-color:#334155;}
body.theme-dark .fan-btn{color:#64748b;}
body.theme-dark .fan-btn.active{background:#1e293b;color:#22d3ee;}
body.theme-blue .fan-seg{background:#dbeafe;border-color:#93c5fd;}
body.theme-blue .fan-btn.active{background:#fff;color:#0e7490;}
body.theme-dark .fan-btn{background:#0f172a;color:#64748b;border-color:#334155;}
body.theme-dark .fan-btn.active{background:linear-gradient(135deg,#06b6d4,#0891b2);color:white;border-color:#0e7490;}
body.theme-blue .fan-btn{background:#dbeafe;color:#1e40af;border-color:#93c5fd;}
body.theme-blue .fan-btn.active{background:linear-gradient(135deg,#0891b2,#0e7490);color:white;border-color:#0e7490;}
.mode-sel-btn{flex:1;padding:9px 4px;border:2px solid #e5e7eb;background:#f3f4f6;color:#888;border-radius:8px;font-weight:bold;cursor:pointer;font-size:11px;transition:all 0.2s;}
.mode-sel-btn:active{transform:scale(0.97);}
.mode-sel-btn.heat{border-color:#fecaca;color:#ef4444;}
.mode-sel-btn.cool{border-color:#bfdbfe;color:#3b82f6;}
.mode-sel-btn.auto{border-color:#d1fae5;color:#10b981;}
.mode-sel-btn.vent{border-color:#e9d5ff;color:#8b5cf6;}
.mode-sel-btn.heat.active{background:linear-gradient(135deg,#fb923c,#ef4444);color:white;border-color:#dc2626;box-shadow:0 0 12px 3px rgba(239,68,68,.6);transform:scale(1.06);font-size:13px;}
.mode-sel-btn.cool.active{background:linear-gradient(135deg,#60a5fa,#3b82f6);color:white;border-color:#2563eb;box-shadow:0 0 12px 3px rgba(59,130,246,.6);transform:scale(1.06);font-size:13px;}
.mode-sel-btn.auto.active{background:linear-gradient(135deg,#16a34a,#15803d);color:white;border-color:#166534;box-shadow:0 0 12px 3px rgba(22,163,74,.6);transform:scale(1.06);font-size:13px;}
.mode-sel-btn.vent.active{background:linear-gradient(135deg,#a78bfa,#8b5cf6);color:white;border-color:#7c3aed;box-shadow:0 0 12px 3px rgba(139,92,246,.6);transform:scale(1.06);font-size:13px;}
body.theme-dark .mode-sel-btn{background:#0f172a;color:#64748b;border-color:#334155;}
/* ── Unified accordion panels (all tabs) ── */
.acc-panel{border-radius:10px;overflow:hidden;margin-bottom:10px;border:1px solid #e2e8f0;}
.acc-hdr{display:flex;justify-content:space-between;align-items:center;padding:11px 14px;background:#1e3a5f;color:white;cursor:pointer;font-size:13px;font-weight:bold;user-select:none;}
.acc-hdr:hover{background:#1e4d7a;}
.acc-hdr .acc-chev{font-size:11px;transition:transform .25s;}
.acc-hdr.open .acc-chev{transform:rotate(90deg);}
.acc-body{display:none;padding:14px;background:white;}
/* schedule tab uses write-panel style — unify */
.write-panel{background:white;border-radius:10px;padding:0;border:1px solid #e2e8f0;margin-bottom:10px;overflow:hidden;}
.write-panel h2{font-size:13px;font-weight:bold;margin:0;padding:11px 14px;background:#1e3a5f;color:white;display:flex;justify-content:space-between;align-items:center;cursor:pointer;user-select:none;border:none;}
.write-panel h2:hover{background:#1e4d7a;}
.write-panel-body{display:none;padding:14px;}
/* Compact trend-config controls (BMS style) */
.tcfg-row{display:grid;grid-template-columns:1fr 120px;gap:10px;margin-bottom:12px;}
.tcfg-field{display:flex;flex-direction:column;gap:4px;}
.tcfg-field label,.tcfg-block>label{font-size:10px;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#94a3b8;}
.tcfg-field select{padding:8px 10px;border:1px solid #d5dbe3;border-radius:7px;font-size:13px;background:#fff;color:#1e293b;}
.tcfg-block{display:flex;flex-direction:column;gap:6px;margin-bottom:12px;}
.tcfg-seg{display:flex;gap:6px;flex-wrap:wrap;}
.tcfg-chip{flex:1;min-width:44px;padding:8px 6px;border:1px solid #d5dbe3;background:#f8fafc;color:#64748b;border-radius:7px;font-size:12px;font-weight:700;cursor:pointer;transition:all .12s;}
.tcfg-chip.active{background:#1e3a5f;color:#fff;border-color:#1e3a5f;}
.tcfg-toggle{display:flex;align-items:center;gap:6px;font-size:12px;font-weight:600;color:#475569;cursor:pointer;padding:7px 10px;border:1px solid #e2e8f0;border-radius:7px;background:#f8fafc;}
.tcfg-toggle input{margin:0;}
.tcfg-toggle .dot{width:9px;height:9px;border-radius:50%;display:inline-block;}
.tcfg-toggle .dot.temp{background:#ef4444;}
.tcfg-toggle .dot.sp{background:#3b82f6;}
.tcfg-toggle .dot.co{background:#10b981;}
body.theme-dark .tcfg-field select{background:#0f172a;color:#e2e8f0;border-color:#334155;}
body.theme-dark .tcfg-chip{background:#0f172a;color:#94a3b8;border-color:#334155;}
body.theme-dark .tcfg-chip.active{background:#3b82f6;color:#fff;border-color:#3b82f6;}
body.theme-dark .tcfg-toggle{background:#0f172a;color:#cbd5e1;border-color:#334155;}
/* Professional live-register table */
.lr-table{width:100%;border-collapse:collapse;font-size:13px;}
.lr-table tr{border-bottom:1px solid #eef1f5;}
.lr-table tr:last-child{border-bottom:none;}
.lr-param{padding:9px 4px;color:#475569;font-weight:600;}
.lr-addr{display:inline-block;min-width:34px;margin-right:8px;padding:1px 6px;border-radius:4px;background:#f1f5f9;color:#94a3b8;font-size:10px;font-weight:700;font-family:'Courier New',monospace;text-align:center;}
.lr-value{padding:9px 4px;text-align:right;font-weight:800;color:#1e3a5f;white-space:nowrap;}
.lr-value .u{font-size:10px;font-weight:600;color:#94a3b8;margin-left:2px;}
.lr-pill{display:inline-block;padding:2px 9px;border-radius:11px;font-size:11px;font-weight:800;}
.lr-pill.ok{background:#dcfce7;color:#15803d;}
.lr-pill.off{background:#f1f5f9;color:#64748b;}
.lr-pill.warn{background:#fee2e2;color:#b91c1c;}
.lr-pill.info{background:#dbeafe;color:#1d4ed8;}
body.theme-dark .lr-table tr{border-color:#334155;}
body.theme-dark .lr-param{color:#cbd5e1;}
body.theme-dark .lr-value{color:#60a5fa;}
body.theme-dark .lr-addr{background:#0f172a;color:#64748b;}
/* AI panel — same as acc-panel */
.ai-panel{border-radius:10px;overflow:hidden;margin-bottom:10px;border:1px solid #e2e8f0;}
.ai-panel-hdr{display:flex;justify-content:space-between;align-items:center;padding:11px 14px;background:#1e3a5f;color:white;cursor:pointer;font-size:13px;font-weight:bold;user-select:none;}
.ai-panel-hdr:hover{background:#1e4d7a;}
.ai-panel-body{display:none;padding:14px;background:white;}
/* Dark theme panels */
body.theme-dark .acc-panel,body.theme-dark .ai-panel,body.theme-dark .write-panel{border-color:#334155;}
body.theme-dark .acc-hdr,body.theme-dark .ai-panel-hdr,body.theme-dark .write-panel h2{background:#1e3a8a;}
body.theme-dark .acc-hdr:hover,body.theme-dark .ai-panel-hdr:hover,body.theme-dark .write-panel h2:hover{background:#1e40af;}
body.theme-dark .acc-body,body.theme-dark .ai-panel-body,body.theme-dark .write-panel-body{background:#1e293b;color:#e2e8f0;}
/* Blue theme panels */
body.theme-blue .acc-panel,body.theme-blue .ai-panel,body.theme-blue .write-panel{border-color:#93c5fd;}
body.theme-blue .acc-hdr,body.theme-blue .ai-panel-hdr,body.theme-blue .write-panel h2{background:#1e40af;}
body.theme-blue .acc-hdr:hover,body.theme-blue .ai-panel-hdr:hover,body.theme-blue .write-panel h2:hover{background:#1d4ed8;}
body.theme-blue .acc-body,body.theme-blue .ai-panel-body,body.theme-blue .write-panel-body{background:#eff6ff;color:#1e3a5f;}
/* Help modal */
.help-modal{display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(15,23,42,0.92);z-index:9000;align-items:center;justify-content:center;}
/* Quick Start — landscape popover pinned to the upper-right corner */
.qs-overlay{display:none;position:fixed;inset:0;z-index:9500;background:rgba(15,23,42,0.22);}
.qs-overlay.show{display:block;}
.qs-panel{position:absolute;top:14px;right:14px;width:560px;max-width:94vw;
  background:#fff;border:1px solid #e2e8f0;border-radius:14px;
  box-shadow:0 18px 50px rgba(0,0,0,0.30);padding:14px 16px;
  animation:qsIn .18s ease;}
@keyframes qsIn{from{opacity:0;transform:translateY(-8px) scale(.985);}to{opacity:1;transform:none;}}
.qs-hdr{display:flex;align-items:center;justify-content:space-between;margin-bottom:12px;}
.qs-title{font-size:15px;font-weight:800;color:#1e3a5f;letter-spacing:.3px;}
.qs-close{width:30px;height:30px;border:none;border-radius:8px;background:#f1f5f9;color:#475569;font-size:14px;font-weight:700;cursor:pointer;line-height:1;}
.qs-close:active{transform:scale(.92);}
.qs-steps{display:flex;gap:10px;}
.qs-step{flex:1;display:flex;flex-direction:column;gap:8px;background:#f8fafc;border:1px solid #eef2f7;border-radius:10px;padding:11px 10px;}
/* Step 5 "Live": same card footprint as the other steps, with the QR sized to
   fill the space the other cards use for their description text. */
.qs-step-live{align-items:stretch;}
.qs-qr{margin-top:auto;background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:5px;
  display:flex;align-items:center;justify-content:center;}
.qs-qr img{width:100%;max-width:96px;height:auto;display:block;image-rendering:pixelated;}
.qs-qr-url{font-size:9px;color:#64748b;text-align:center;word-break:break-all;line-height:1.3;}
body.theme-dark .qs-qr{background:#fff;border-color:var(--d-line);}
body.theme-dark .qs-qr-url{color:var(--d-dim);}
.qs-num{width:24px;height:24px;border-radius:50%;background:#1e3a5f;color:#fff;font-size:12px;font-weight:800;display:flex;align-items:center;justify-content:center;}
.qs-txt{display:flex;flex-direction:column;gap:3px;}
.qs-txt b{font-size:12px;color:#1e3a5f;font-weight:800;}
.qs-txt span{font-size:11px;color:#64748b;line-height:1.4;}
.qs-foot{margin-top:11px;font-size:10px;color:#94a3b8;text-align:center;}
body.theme-dark .qs-panel{background:var(--d-surf);border-color:var(--d-line);}
body.theme-dark .qs-title{color:var(--d-acc);}
body.theme-dark .qs-close{background:var(--d-inset);color:var(--d-dim);}
body.theme-dark .qs-step{background:var(--d-inset);border-color:var(--d-line);}
body.theme-dark .qs-num{background:var(--d-acc);color:#0f172a;}
body.theme-dark .qs-txt b{color:#dbeafe;}
body.theme-dark .qs-txt span{color:var(--d-dim);}
body.theme-blue .qs-panel{background:#eff6ff;border-color:#bfdbfe;}
body.theme-blue .qs-step{background:#dbeafe;border-color:#bfdbfe;}
.help-box{background:white;border-radius:14px;padding:20px;width:92%;max-width:420px;max-height:80vh;overflow-y:auto;}
.help-box h3{font-size:16px;font-weight:bold;color:#1e3a5f;margin-bottom:14px;border-bottom:2px solid #e5e7eb;padding-bottom:8px;}
.help-section{margin-bottom:14px;}
.help-section h4{font-size:13px;font-weight:bold;color:#1e3a5f;margin-bottom:6px;}
.help-section p,.help-section li{font-size:12px;color:#555;line-height:1.6;}
.help-section ul{padding-left:16px;}
.help-doc-btn{display:block;width:100%;padding:10px 14px;margin-bottom:8px;border:2px solid #1e3a5f;border-radius:8px;background:white;color:#1e3a5f;font-size:13px;font-weight:bold;cursor:pointer;text-align:left;}
.help-doc-btn:hover{background:#eef2ff;}
body.theme-dark .help-box{background:#1e293b;color:#e2e8f0;}
body.theme-dark .help-box h3{color:#60a5fa;border-color:#334155;}
body.theme-dark .help-section h4{color:#60a5fa;}
body.theme-dark .help-section p,body.theme-dark .help-section li{color:#cbd5e1;}
body.theme-dark .help-doc-btn{background:#0f172a;color:#60a5fa;border-color:#60a5fa;}
body.theme-blue .help-box{background:#eff6ff;}
body.theme-blue .help-box h3{color:#1e40af;border-color:#93c5fd;}
body.theme-blue .help-section h4{color:#1e40af;}
body.theme-blue .help-doc-btn{background:#dbeafe;color:#1e40af;border-color:#1e40af;}
/* LED per-controller config in cfg */
.led-cfg-row{display:flex;align-items:center;justify-content:space-between;padding:8px 10px;background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0;margin-top:10px;}
.led-cfg-row label{font-size:11px;color:#64748b;text-transform:uppercase;font-weight:bold;}
.led-cfg-sel{padding:5px 8px;border:1px solid #ccc;border-radius:5px;font-size:12px;background:white;}
body.theme-dark .led-cfg-row{background:#0f172a;border-color:#334155;}
body.theme-dark .led-cfg-sel{background:#0f172a;color:#e2e8f0;border-color:#334155;}
body.theme-blue .led-cfg-row{background:#dbeafe;border-color:#93c5fd;}
/* LED/Delete bar inside cfg-row */
body.theme-dark [id^="cfg-row-"] > div:nth-child(2){background:rgba(15,23,42,0.8)!important;border-color:#334155!important;}
body.theme-blue [id^="cfg-row-"] > div:nth-child(2){background:rgba(219,234,254,0.8)!important;}
/* Sleep timeout buttons */
.sleep-btn{padding:8px 14px;border:2px solid #e5e7eb;border-radius:8px;background:#f3f4f6;color:#555;font-size:13px;font-weight:bold;cursor:pointer;transition:all 0.2s;}
.sleep-btn.active{background:#1e3a5f;color:white;border-color:#1e3a5f;}
body.theme-dark .sleep-btn{background:#0f172a;color:#64748b;border-color:#334155;}
body.theme-dark .sleep-btn.active{background:#3b82f6;border-color:#3b82f6;color:white;}
body.theme-blue .sleep-btn{background:#dbeafe;color:#1e40af;border-color:#93c5fd;}
body.theme-blue .sleep-btn.active{background:#1e40af;border-color:#1e40af;color:white;}
/* Email recipient input */
.email-recipient-row{margin-bottom:12px;}
.email-recipient-row label{font-size:11px;color:#888;text-transform:uppercase;display:block;margin-bottom:4px;font-weight:bold;}
.email-recipient-row input{width:100%;padding:9px 11px;border:1.5px solid #ccc;border-radius:7px;font-size:13px;}
.email-recipient-row input:focus{outline:none;border-color:#1e3a5f;}
body.theme-dark .email-recipient-row input{background:#0f172a;color:#e2e8f0;border-color:#334155;}
body.theme-blue .email-recipient-row input{background:#fff;color:#1e3a5f;border-color:#93c5fd;}
/* ── Metric Tiles ─────────────────────────────────────────────── */
.metric-row{display:grid;grid-template-columns:1fr 1.3fr 1fr;gap:8px;width:100%;padding:4px 2px 6px;}
.metric-row.no-co2{grid-template-columns:1fr 1.1fr;}
.view-toggle-bar{display:flex;gap:8px;margin-bottom:10px;}
.view-toggle-btn{flex:1;padding:9px 4px;border:2px solid #e5e7eb;background:#f3f4f6;color:#888;border-radius:8px;font-weight:bold;cursor:pointer;font-size:13px;transition:all .2s;}
.view-toggle-btn.active{background:#1e3a5f;color:white;border-color:#1e3a5f;}
body.theme-dark .view-toggle-btn{background:#0f172a;color:#64748b;border-color:#334155;}
body.theme-dark .view-toggle-btn.active{background:#3b82f6;border-color:#3b82f6;color:white;}
body.theme-blue .view-toggle-btn{background:#dbeafe;color:#1e40af;border-color:#93c5fd;}
body.theme-blue .view-toggle-btn.active{background:#1e40af;border-color:#1e40af;color:white;}
.grid-cards{display:grid;grid-template-columns:repeat(2,1fr);gap:10px;}
.grid-tile{background:white;border:1px solid #ddd;border-radius:10px;padding:12px;box-shadow:0 2px 8px rgba(0,0,0,0.06);}
# .grid-tile-name{font-size:13px;font-weight:bold;color:#1e3a5f;margin-bottom:2px;}
# .grid-tile-id{font-size:10px;color:#888;margin-bottom:8px;}
.grid-tile-status{font-size:11px;font-weight:700;margin-bottom:8px;color:#64748b;}
.grid-tile-status.s1{color:#16a34a;}   /* On - green */
.grid-tile-status.s2{color:#dc2626;}   /* Off Alarm - red */
.grid-tile-status.s3{color:#d97706;}   /* Off BMS - amber */
.grid-tile-status.s4{color:#7c3aed;}   /* Off Scheduler - purple */
.grid-tile-status.s5{color:#d97706;}   /* Off Digital Input - amber */
.grid-tile-status.s6{color:#0891b2;}   /* Off Local - teal */
.grid-tile-status.s7{color:#2563eb;}   /* Manual - blue */
.grid-tile-temp{font-size:28px;font-weight:900;color:#dc2626;line-height:1;margin-bottom:6px;}
.grid-tile-badge{display:inline-block;font-size:10px;padding:2px 8px;border-radius:10px;margin-bottom:8px;background:#374151;color:#9ca3af;}
.grid-tile-badge.ok{background:#166534;color:#bbf7d0;}
.grid-tile-badge.err{background:#7f1d1d;color:#fecaca;}
.grid-reason{font-size:10px;font-weight:700;text-align:center;margin-top:4px;min-height:12px;}
.grid-reason.reason-ok{color:#16a34a;}
.grid-reason.reason-off{color:#f59e0b;}
.grid-reason.reason-alarm{color:#ef4444;}
.grid-tile-onoff{width:100%;padding:7px;border:none;border-radius:8px;font-size:12px;font-weight:bold;cursor:pointer;}
.grid-tile-onoff.on{background:#22c55e;color:white;}
.grid-tile-onoff.off{background:#e5e7eb;color:#666;}
body.theme-dark .grid-tile{background:#1e293b;border-color:#334155;}
body.theme-dark .grid-tile-name{color:#60a5fa;}
body.theme-dark .grid-tile-temp{color:#fca5a5;}
body.theme-blue .grid-tile{background:#eff6ff;border-color:#93c5fd;}

/* Base tile */
.metric-tile{
  position:relative;overflow:hidden;
  border-radius:14px;padding:8px 10px 7px;
  display:flex;flex-direction:column;justify-content:space-between;
  min-height:82px;
  box-shadow:0 4px 14px rgba(0,0,0,0.08);
  border:1px solid rgba(255,255,255,0.5);
}
.metric-tile::before{
  content:'';position:absolute;top:-20px;right:-20px;
  width:70px;height:70px;border-radius:50%;
  opacity:0.12;
}
.tile-icon{font-size:16px;line-height:1;margin-bottom:2px;}
.tile-label{font-size:9px;font-weight:800;text-transform:uppercase;letter-spacing:.8px;opacity:0.65;margin-bottom:2px;}
.tile-value{font-size:27px;font-weight:900;line-height:1;letter-spacing:-1px;}
.tile-unit{font-size:11px;font-weight:700;opacity:0.6;margin-top:1px;}
.tile-bar-wrap{width:100%;height:4px;border-radius:2px;background:rgba(0,0,0,0.10);margin-top:5px;overflow:hidden;}
.tile-bar-fill{height:100%;border-radius:2px;transition:width .4s ease,background .4s ease;}
.tile-status{font-size:9px;font-weight:700;margin-top:3px;opacity:0.7;letter-spacing:.3px;}

/* Room Temp — warm red/orange gradient */
.tile-temp{
  background:linear-gradient(135deg,#fff5f5 0%,#ffe4e0 100%);
  border-color:rgba(239,68,68,0.2);
}
.tile-temp::before{background:#ef4444;}
.tile-temp .tile-value{color:#dc2626;}
.tile-temp .tile-label{color:#7f1d1d;}
.tile-temp .tile-bar-wrap{background:rgba(239,68,68,0.12);}
.tile-outside-row{display:flex;align-items:center;gap:5px;margin-top:5px;}
.tile-outside-label{font-size:8px;font-weight:800;text-transform:uppercase;letter-spacing:.5px;color:#7f1d1d;opacity:.6;}
.tile-outside-value{font-size:13px;font-weight:800;color:#dc2626;}
.eco-leaf{
  position:absolute;top:8px;right:4px;
  font-size:26px;line-height:1;
  opacity:.25;filter:grayscale(1);
  transition:opacity .3s,filter .3s,transform .3s;
  z-index:2;
}
.eco-leaf.active{
  opacity:1;filter:none;
  transform:scale(1.1);
  text-shadow:0 0 8px rgba(34,197,94,.5);
}
body.theme-dark .tile-outside-label{color:#fca5a5;}
body.theme-dark .tile-outside-value{color:#fca5a5;}

/* Setpoint — cool blue gradient */
.tile-setpoint{
  background:linear-gradient(135deg,#eff6ff 0%,#dbeafe 100%);
  border-color:rgba(59,130,246,0.25);
}
.tile-setpoint::before{background:#3b82f6;}
.tile-setpoint .tile-value{color:#1d4ed8;}
.tile-setpoint .tile-label{color:#1e3a5f;}
.tile-setpoint .tile-bar-wrap{background:rgba(59,130,246,0.12);}

/* CO Level — green/teal gradient, changes with level */
.tile-co{
  background:linear-gradient(135deg,#f0fdf4 0%,#dcfce7 100%);
  border-color:rgba(16,185,129,0.25);
}
.tile-co::before{background:#10b981;}
.tile-co .tile-value{color:#059669;}
.tile-co .tile-label{color:#064e3b;}
.tile-co .tile-bar-wrap{background:rgba(16,185,129,0.12);}
.tile-co.co-warn{background:linear-gradient(135deg,#fffbeb 0%,#fef3c7 100%);border-color:rgba(245,158,11,0.3);}
.tile-co.co-warn .tile-value{color:#d97706;}
.tile-co.co-bad {background:linear-gradient(135deg,#fff5f5 0%,#fee2e2 100%);border-color:rgba(239,68,68,0.3);}
.tile-co.co-bad  .tile-value{color:#dc2626;}

/* Setpoint knob tile */
.tile-knob{
  background:linear-gradient(135deg,#eff6ff 0%,#dbeafe 100%);
  border-color:rgba(59,130,246,0.2);
  align-items:center;padding:8px 6px 6px;
}

/* Dark theme */
body.theme-dark .tile-temp{background:linear-gradient(135deg,rgba(127,29,29,0.35) 0%,rgba(69,10,10,0.45) 100%);border-color:rgba(239,68,68,0.25);}
body.theme-dark .tile-temp .tile-value{color:#fca5a5;}
body.theme-dark .tile-temp .tile-label{color:#fca5a5;}
body.theme-dark .tile-setpoint{background:linear-gradient(135deg,rgba(30,58,95,0.45) 0%,rgba(29,78,216,0.25) 100%);border-color:rgba(59,130,246,0.3);}
body.theme-dark .tile-setpoint .tile-value{color:#93c5fd;}
body.theme-dark .tile-setpoint .tile-label{color:#bfdbfe;}
body.theme-dark .tile-knob{background:linear-gradient(135deg,rgba(30,58,95,0.5) 0%,rgba(15,23,42,0.7) 100%);border-color:rgba(59,130,246,0.2);}
body.theme-dark .tile-co{background:linear-gradient(135deg,rgba(6,78,59,0.35) 0%,rgba(4,44,34,0.45) 100%);border-color:rgba(16,185,129,0.25);}
body.theme-dark .tile-co .tile-value{color:#6ee7b7;}
body.theme-dark .tile-co .tile-label{color:#a7f3d0;}
body.theme-dark .tile-co.co-warn{background:linear-gradient(135deg,rgba(92,53,7,0.4) 0%,rgba(69,26,3,0.5) 100%);}
body.theme-dark .tile-co.co-warn .tile-value{color:#fcd34d;}
body.theme-dark .tile-co.co-bad{background:linear-gradient(135deg,rgba(127,29,29,0.4) 0%,rgba(69,10,10,0.5) 100%);}
body.theme-dark .tile-co.co-bad .tile-value{color:#fca5a5;}
body.theme-dark .tile-bar-wrap{background:rgba(255,255,255,0.08);}

/* Blue theme */
body.theme-blue .tile-temp{background:linear-gradient(135deg,#fff0ef 0%,#ffe4e0 100%);}
body.theme-blue .tile-setpoint{background:linear-gradient(135deg,#dbeafe 0%,#bfdbfe 100%);border-color:rgba(59,130,246,0.4);}
body.theme-blue .tile-knob{background:linear-gradient(135deg,#dbeafe 0%,#bfdbfe 100%);border-color:rgba(59,130,246,0.3);}
body.theme-blue .tile-co{background:linear-gradient(135deg,#d1fae5 0%,#a7f3d0 100%);border-color:rgba(16,185,129,0.35);}

/* Knob cell wrapper stays compatible */
.knob-cell{flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;
  border-radius:14px;padding:6px 4px;
  background:linear-gradient(135deg,#eff6ff 0%,#dbeafe 100%);
  border:1px solid rgba(59,130,246,0.2);
  box-shadow:0 4px 14px rgba(0,0,0,0.08);}
body.theme-dark .knob-cell{background:linear-gradient(135deg,rgba(30,58,95,0.5) 0%,rgba(15,23,42,0.7) 100%);border-color:rgba(59,130,246,0.2);}
body.theme-blue .knob-cell{background:linear-gradient(135deg,#dbeafe 0%,#bfdbfe 100%);border-color:rgba(59,130,246,0.3);}

/* ── Virtual Keyboard ── */
#vk-overlay{display:none;position:fixed;bottom:0;left:0;right:0;z-index:99999;background:#1a2030;padding:8px 6px 10px;box-shadow:0 -4px 20px rgba(0,0,0,0.5);user-select:none;}
#vk-input-display{background:#0f172a;color:#e2e8f0;font-size:16px;padding:8px 12px;border-radius:6px;margin-bottom:8px;min-height:36px;word-break:break-all;border:1px solid #334155;position:relative;}
#vk-input-display::after{content:'|';animation:vk-blink 1s step-end infinite;color:#60a5fa;}
@keyframes vk-blink{0%,100%{opacity:1;}50%{opacity:0;}}
.vk-row{display:flex;gap:4px;justify-content:center;margin-bottom:4px;}
.vk-key{flex:1;max-width:70px;padding:10px 4px;background:#2d3748;color:#e2e8f0;border:none;border-radius:6px;font-size:14px;font-weight:bold;cursor:pointer;text-align:center;min-height:44px;-webkit-tap-highlight-color:transparent;transition:background .1s;}
.vk-key:active,.vk-key.pressed{background:#4a5568;}
.vk-key.vk-wide{max-width:100px;font-size:12px;}
.vk-key.vk-space{flex:3;max-width:200px;font-size:12px;}
.vk-key.vk-done{background:#1e3a5f;color:white;max-width:90px;}
.vk-key.vk-done:active{background:#2563eb;}
.vk-key.vk-del{background:#7f1d1d;color:#fecaca;max-width:80px;}
.vk-key.vk-del:active{background:#b91c1c;}
.vk-key.vk-shift{background:#374151;max-width:80px;}
.vk-key.vk-shift.active{background:#4338ca;color:white;}
.vk-key.vk-sym{background:#374151;max-width:70px;font-size:12px;}
#vk-target-label{font-size:10px;color:#60a5fa;text-align:center;margin-bottom:6px;text-transform:uppercase;letter-spacing:1px;}

/* ── Pro theme: catch-all so every remaining white card matches the dark BMS look ── */

/* ══════════════════════════════════════════════════════════════════
   PRO DASHBOARD — innoTouch-style hero card
   Fewer data points, bigger/clearer readouts: room temp, setpoint,
   mode, fan, on/off. Secondary sensor clutter (CO₂, outside temp,
   eco leaf) is hidden from the main view — still polled/available
   elsewhere, just not competing for attention here.
   ══════════════════════════════════════════════════════════════════ */

/* Hide secondary sensor tiles for a cleaner hero view */

/* Two-column hero: big room temp on the left, setpoint dial on the right */

/* Setpoint dial cell */

/* Mode + Fan strips — compact, icon-forward, single row each */

/* ══════════════════════════════════════════════════════════════════
   UNIVERSAL DARK THEME  (consolidated — loaded last so it wins)
   Brings every tab onto ONE slate palette. Fixes the white Schedule
   panels and the warm near-black Config panels so all cards, tables,
   inputs and headers match across Dashboard / Trends / Schedule /
   Control / Config. Palette:
     header  #1e3a5f   surface #1e293b   inset #0f172a
     border  #334155   text    #e2e8f0   dim   #94a3b8   accent #60a5fa
   ══════════════════════════════════════════════════════════════════ */
body.theme-dark{--d-hdr:#1e3a5f;--d-hdr-hi:#274c78;--d-surf:#1e293b;--d-inset:#0f172a;
  --d-line:#334155;--d-txt:#e2e8f0;--d-dim:#94a3b8;--d-acc:#60a5fa;background:#0f172a;color:var(--d-txt);}

/* ── Surfaces: cards, panels, sections, modals (unifies Config near-black) ── */
body.theme-dark .card,
body.theme-dark .reg-section,
body.theme-dark .sched-panel,
body.theme-dark .grid-tile,
body.theme-dark .led-legend,
body.theme-dark .acc-body,
body.theme-dark .ai-panel-body,
body.theme-dark .write-panel-body,
body.theme-dark .overlay-box,
body.theme-dark .confirm-box,
body.theme-dark .login-gate-box,
body.theme-dark .help-box,
body.theme-dark .time-wheel-box,
body.theme-dark .cfg-acc,
body.theme-dark .cfg-acc.expanded,
body.theme-dark .cfg-section,
body.theme-dark .week-grid-wrap{
  background:var(--d-surf)!important;border-color:var(--d-line)!important;color:var(--d-txt);}

/* ── Insets: table head, list rows, preview, time/action fields ── */
body.theme-dark th,
body.theme-dark .sched-preview,
body.theme-dark .time-display-btn,
body.theme-dark .action-seg,
body.theme-dark .tw-wheel,
body.theme-dark #schedule-list > div{
  background:var(--d-inset)!important;border-color:var(--d-line)!important;color:var(--d-txt)!important;}

/* ── Universal section / panel / card headers ── */
body.theme-dark .ctrl-header,
body.theme-dark .reg-section-title,
body.theme-dark .acc-hdr,
body.theme-dark .ai-panel-hdr,
body.theme-dark .write-panel h2,
body.theme-dark .sched-panel-hdr{
  background:var(--d-hdr)!important;color:#fff!important;border-color:var(--d-line)!important;}
body.theme-dark .acc-hdr:hover,
body.theme-dark .ai-panel-hdr:hover,
body.theme-dark .write-panel h2:hover{background:var(--d-hdr-hi)!important;}

/* ── Separators / borders ── */
body.theme-dark td,
body.theme-dark .lr-table tr,
body.theme-dark .net-row{border-color:var(--d-line)!important;}

/* ── Text roles ── */
body.theme-dark .prev-title{color:var(--d-txt);}
body.theme-dark .prev-sub,
body.theme-dark .add-label,
body.theme-dark .tw-label,
body.theme-dark .cfg-section-title{color:var(--d-dim);}
body.theme-dark .tw-display{color:var(--d-txt);}

/* ── Inputs & selects (catch-all for any unstyled field) ── */
body.theme-dark input,
body.theme-dark select,
body.theme-dark textarea{background:var(--d-inset);color:var(--d-txt);border-color:var(--d-line);}

/* ── Schedule controls: view toggle, day picker, action/ampm, quick chips ── */
body.theme-dark .sview-btn{background:var(--d-inset);color:var(--d-dim);border-color:var(--d-line);}
body.theme-dark .sview-btn.active{background:#3b82f6;color:#fff;border-color:#3b82f6;}
body.theme-dark .day-toggle-btn.unsel{background:var(--d-inset);color:var(--d-dim);}
body.theme-dark .day-toggle-btn.sel{background:#3b82f6;color:#fff;}
body.theme-dark .act-btn{color:var(--d-dim);}
body.theme-dark .act-btn.active{background:#3b82f6;color:#fff;}
body.theme-dark .tw-quick button{background:var(--d-inset);color:var(--d-acc);border-color:var(--d-line);}

/* ── Config: replace remaining warm-black borders/insets with slate ── */
body.theme-dark .cfg-topbar,
body.theme-dark .cfg-acc-body,
body.theme-dark .cfg-inline-row,
body.theme-dark .cfg-acc-actions{border-color:var(--d-line)!important;}
body.theme-dark .cfg-group-label::after{background:var(--d-line)!important;}
body.theme-dark .cfg-acc-hdr:active{background:var(--d-inset)!important;}

/* ══════════════════════════════════════════════════════════════════
   UNIVERSAL DARK THEME — final consolidation (all tabs, cards & tables)
   Palette:  page #0f172a · surface #1e293b · inset #0f172a
             line #334155 · header #1e3a5f · text #e2e8f0 · dim #94a3b8
   Neutralises scattered near-black / leftover-light inline styles so
   every tab matches. Scoped to body.theme-dark only — light theme
   is untouched.
   ══════════════════════════════════════════════════════════════════ */

/* AI tab — harmonise the neon HUD cards with the slate palette */
body.theme-dark .ai-motion-card,
body.theme-dark .ai-status-strip,
body.theme-dark .ai-history-chart{background:var(--d-inset)!important;border-color:var(--d-line)!important;}
body.theme-dark .ai-bar-wrap{background:var(--d-line)!important;}
body.theme-dark .ai-motion-card .ai-lbl,
body.theme-dark .ai-history-chart .ai-lbl{color:var(--d-dim)!important;}

/* Remap leftover LIGHT inline backgrounds → dark inset surface */
body.theme-dark [style*="background:white"],
body.theme-dark [style*="background:#fff"],
body.theme-dark [style*="background:#f8fafc"],
body.theme-dark [style*="background:#f8f9fa"],
body.theme-dark [style*="background:#f1f5f9"],
body.theme-dark [style*="background:#f3f4f6"],
body.theme-dark [style*="background:#f0f2f5"],
body.theme-dark [style*="background:#fafafa"],
body.theme-dark [style*="background:#eef2f7"],
body.theme-dark [style*="background:#eef2ff"],
body.theme-dark [style*="background:#f5f3ff"],
body.theme-dark [style*="background:#e5e7eb"]{
  background:var(--d-inset)!important;border-color:var(--d-line)!important;}

/* Remap off-palette DARK inline backgrounds (AI panels) → unified surfaces */
body.theme-dark [style*="background:#111827"],
body.theme-dark [style*="background:#0e1520"]{background:var(--d-inset)!important;border-color:var(--d-line)!important;}
body.theme-dark [style*="background:#0b0f14"],
body.theme-dark [style*="background:#080b10"]{background:var(--d-surf)!important;}

/* Status tints — keep the meaning, darken for dark theme (must come after neutrals) */
body.theme-dark [style*="background:#fffbeb"]{background:rgba(245,158,11,0.14)!important;border-color:rgba(245,158,11,0.35)!important;}
body.theme-dark [style*="background:#fef2f2"],
body.theme-dark [style*="background:#fee2e2"],
body.theme-dark [style*="background:#fecaca"]{background:rgba(239,68,68,0.15)!important;border-color:rgba(239,68,68,0.35)!important;}

/* Rescue dark inline TEXT so it stays legible on dark surfaces */
body.theme-dark [style*="color:#1e3a5f"],
body.theme-dark [style*="color:#1e293b"],
body.theme-dark [style*="color:#111"],
body.theme-dark [style*="color:#222"],
body.theme-dark [style*="color:#333"]{color:var(--d-txt)!important;}
body.theme-dark [style*="color:#444"],
body.theme-dark [style*="color:#555"],
body.theme-dark [style*="color:#666"],
body.theme-dark [style*="color:#777"],
body.theme-dark [style*="color:#888"],
body.theme-dark [style*="color:#999"]{color:var(--d-dim)!important;}

/* Config tab — unify tiles + logout with the palette */
body.theme-dark .cfg-tile,
body.theme-dark .cfg-tile-advanced{background:var(--d-hdr)!important;border-color:var(--d-line)!important;}
body.theme-dark .cfg-tile:hover{background:var(--d-hdr-hi)!important;}
body.theme-dark .cfg-logout-btn{background:var(--d-inset)!important;}

/* Trend chart plot area (was solid white) → blend into the dark card */
body.theme-dark #tempChart,
body.theme-dark #tempChart{background:transparent!important;}

/* Every table across every tab: one head + one separator colour */
body.theme-dark table th{background:var(--d-inset)!important;color:var(--d-dim)!important;}
body.theme-dark table td,
body.theme-dark .lr-table tr{border-color:var(--d-line)!important;}
</style>
</head>
<body>

<div class="header">
  <h2 onclick="showPowerPanel()" style="cursor:pointer;">
    <img src="/home/linaro/logo_setting.png" alt="⚙" style="height:34px;width:auto;display:block;"
         onerror="this.replaceWith(document.createTextNode('⚙ ENVI'))">
  </h2>
  <div style="display:flex;align-items:center;gap:10px;">
    <div style="text-align:center;line-height:1.3;cursor:pointer;" onclick="showSetTime()" title="Tap to set date &amp; time">
      <div id="hdr-clock-time" style="font-size:20px;">--:--:--</div>
      <div id="hdr-clock-ampm" style="font-size:13px;">--</div>
      <div id="hdr-clock-date" style="font-size:11px;">----------</div>
      
    </div>
    <div id="hdr-weather" style="text-align:center;line-height:1.2;padding-left:10px;cursor:default;">
      <div id="wx-icon" style="font-size:22px;">☁</div>
      <div id="wx-temp" style="font-size:13px;font-weight:bold;">--°</div>
    </div>
  </div>
  <div class="header-right">
    <div class="hdr-status-box">
      <div class="hdr-status-row">
        <span class="hdr-status-dot" id="hdr-status-dot"></span>
        <span id="status-text">—</span>
      </div>
      <div class="hdr-btn-row">
        <button class="hdr-icon-btn" onclick="manualRefresh()" title="Refresh"><img src="/home/linaro/icons/refresh-blue-128x128.png"></button>
        <button class="hdr-icon-btn" onclick="showNetworkPanel()" title="Network Info"><img src="/home/linaro/icons/network-blue-128x128.png"></button>
        <button class="hdr-icon-btn" onclick="showWifi()" title="Wi-Fi Setup"><span style="font-size:16px;line-height:1;">📶</span></button>
        <button class="hdr-icon-btn" onclick="showQuickStart()" title="Quick Start Guide"><img src="/home/linaro/icons/help-blue-128x128.png"></button>
      </div>
    </div>
    <div id="motion-text" style="display:none;">Waiting...</div>
  </div>
</div>

<div class="tabs">
  <div class="tab locked" id="tab-btn-dashboard"  onclick="tryShowTab('dashboard')"><img class="tab-ic" src="/home/linaro/icons/dashboard-blue-128x128.png">Dashboard</div>
  <div class="tab locked" id="tab-btn-schedule"   onclick="tryShowTab('schedule')"><img class="tab-ic" src="/home/linaro/icons/schedule-blue-128x128.png">Schedule</div>
  <div class="tab locked" id="tab-btn-info"       onclick="tryShowTab('info')"><img class="tab-ic" src="/home/linaro/icons/trend-blue-128x128.png">Trends</div>
  <div class="tab locked" id="tab-btn-insight_ai" onclick="tryShowTab('insight_ai')"><img class="tab-ic" src="/home/linaro/icons/ai-blue-128x128.png">AI</div>
  <div class="tab active" id="tab-btn-config"     onclick="tryShowTab('config')"><img class="tab-ic" src="/home/linaro/icons/config-blue-128x128.png">Config</div>
  <div class="tab locked" id="tab-btn-control"    onclick="tryShowTab('control')"><img class="tab-ic" src="/home/linaro/icons/control-blue-128x128.png">Control</div>
</div>

<!-- ══ DASHBOARD ══ -->
<div class="page" id="tab-dashboard">
  <div class="view-toggle-bar">
    <button class="view-toggle-btn active" id="view-btn-list" onclick="setDashView('list')">☰ List</button>
    <button class="view-toggle-btn" id="view-btn-grid" onclick="setDashView('grid')">▦ Grid</button>
  </div>
  <div id="ctrl-sections"></div>
  <div id="ctrl-grid" style="display:none;"></div>
</div>

<!-- ══ CONTROL ══ -->
<div class="page" id="tab-control">
  <div id="control-sections"></div>
</div>

<!-- ══ TRENDS ══ -->
<div class="page" id="tab-info">

  <!-- Config panel — compact settings at top of trends tab -->
  <div class="write-panel" style="margin-bottom:10px;">
    <h2 onclick="toggleWP('trend-cfg-body','trend-cfg-chev')"><img class="ic" src="/home/linaro/icons/config-blue-128x128.png"> Chart Settings <span id="trend-cfg-chev" class="acc-chev">▶</span></h2>
    <div class="write-panel-body" id="trend-cfg-body">

      <!-- Source + style -->
      <div class="tcfg-row">
        <div class="tcfg-field">
          <label>Zone</label>
          <select id="trend-primary-sel" onchange="applyTrendConfig()"></select>
        </div>
        <div class="tcfg-field">
          <label>Style</label>
          <select id="trend-chart-type" onchange="applyTrendConfig()">
            <option value="area" selected>Area</option>
            <option value="line">Line</option>
          </select>
        </div>
      </div>

      <!-- Time window (minute-by-minute) -->
      <div class="tcfg-block">
        <label>Time Window</label>
        <div class="tcfg-seg" id="hlen-seg">
          <button class="tcfg-chip active" id="hlen-15"  onclick="setHistoryLen(15)">15m</button>
          <button class="tcfg-chip"        id="hlen-30"  onclick="setHistoryLen(30)">30m</button>
          <button class="tcfg-chip"        id="hlen-60"  onclick="setHistoryLen(60)">1h</button>
          <button class="tcfg-chip"        id="hlen-120" onclick="setHistoryLen(120)">2h</button>
          <button class="tcfg-chip"        id="hlen-360" onclick="setHistoryLen(360)">6h</button>
        </div>
      </div>

      <!-- Series toggles -->
      <div class="tcfg-block" style="margin-bottom:0;">
        <label>Series</label>
        <div class="tcfg-seg">
          <label class="tcfg-toggle"><input type="checkbox" id="show-temp" checked onchange="applyTrendConfig()"><span class="dot temp"></span>Temperature</label>
          <label class="tcfg-toggle"><input type="checkbox" id="show-setpoint" checked onchange="applyTrendConfig()"><span class="dot sp"></span>Setpoint</label>
          <label class="tcfg-toggle"><input type="checkbox" id="show-co" onchange="applyTrendConfig()"><span class="dot co"></span>CO₂</label>
        </div>
      </div>

    </div>
  </div>

  <!-- Live chart -->
  <div class="reg-section">
    <div style="display:flex;justify-content:space-between;align-items:center;background:#1e3a5f;color:white;padding:8px 12px;">
      <div style="font-size:12px;font-weight:bold;display:flex;align-items:center;gap:6px;" id="trend-chart-title"><img class="ic" style="filter:brightness(0) invert(1);" src="/home/linaro/icons/trend-blue-128x128.png"> Live Trend</div>
      <div style="display:flex;align-items:center;gap:8px;">
        <span id="trend-ctrl-badge" style="font-size:10px;opacity:0.7;"></span>
        <div style="position:relative;">
          <span onclick="toggleMenu()" style="cursor:pointer;font-size:18px;padding:4px;">⋮</span>
          <div id="trend-menu" style="display:none;position:absolute;right:0;top:28px;background:white;color:black;border:1px solid #ccc;border-radius:6px;min-width:165px;z-index:1000;box-shadow:0 4px 12px rgba(0,0,0,0.15);">
<div onclick="openEmailModal()" style="padding:10px;cursor:pointer;border-bottom:1px solid #eee;font-size:13px;"><img class="ic" src="/home/linaro/icons/email-blue-128x128.png"> Send Email</div>            <div onclick="openFullscreenChart()" style="padding:10px;cursor:pointer;font-size:13px;">🔎 Fullscreen</div>
          </div>
        </div>
      </div>
    </div>
    <div style="padding:10px;background:white;"><canvas id="tempChart" style="width:100%;height:260px;"></canvas></div>
  </div>

</div>

<!-- ══ SCHEDULE ══ -->
<div class="page" id="tab-schedule">

  <!-- Controller selector + global enable status -->
  <div class="sched-panel">
    <div class="sched-panel-hdr">
      <span><img class="ic" src="/home/linaro/icons/scheduleclock-blue-128x128.png"> Time Schedule</span>
      <span id="sched-enabled-pill" class="sched-enabled-pill off">Schedules: —</span>
    </div>
    <div class="sched-panel-body">
      <div class="sched-ctrl-bar" id="sched-ctrl-bar"></div>
      <div id="sched-enable-note" class="sched-enable-note" style="display:none;"></div>
    </div>
  </div>

  <!-- View toggle -->
  <div class="sched-view-toggle">
    <button class="sview-btn active" id="sview-list" onclick="setSchedView('list')">☰ List</button>
    <button class="sview-btn" id="sview-week" onclick="setSchedView('week')">▦ Week Grid</button>
  </div>

  <!-- LIST VIEW -->
  <div id="sched-list-view">
    <div class="sched-panel">
      <div class="sched-panel-hdr"><span>Existing Events — <span id="sched-view-label">—</span></span></div>
      <div class="sched-panel-body"><div id="schedule-list"></div></div>
    </div>
  </div>

  <!-- WEEK GRID -->
  <div id="sched-week-view" style="display:none;">
    <div class="week-grid-wrap">
      <div class="week-grid" id="week-grid"></div>
      <div class="week-legend">
        <span><span class="wl-dot on"></span>ON</span>
        <span><span class="wl-dot off"></span>OFF</span>
        <span style="color:#94a3b8;">Tap an hour to add · tap an event to remove</span>
      </div>
    </div>
  </div>

  <!-- ADD EVENT -->
  <div class="sched-panel">
    <div class="sched-panel-hdr"><span>➕ Add Schedule Event</span></div>
    <div class="sched-panel-body">

      <div class="add-label">Days</div>
      <div style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:8px;">
        <button class="day-quick-btn" onclick="selectWeekdays()" style="background:#1e3a5f;color:white;">Weekdays</button>
        <button class="day-quick-btn" onclick="selectAllDays()" style="background:#374151;color:white;">All Days</button>
        <button class="day-quick-btn" onclick="clearSchedDays()" style="background:#e5e7eb;color:#555;">Clear</button>
      </div>
      <div id="sched-day-btns" style="display:flex;gap:5px;flex-wrap:wrap;margin-bottom:14px;"></div>

      <div style="display:flex;gap:12px;margin-bottom:14px;">
        <div style="flex:1;">
          <div class="add-label">Time</div>
          <div class="time-bar">
            <button class="time-display-btn" id="time-display" onclick="openTimeWheel()">--:--</button>
            <div class="ampm-seg">
              <button class="ampm-btn" id="ampm-am" onclick="setSchedAmPm('AM')">AM</button>
              <button class="ampm-btn" id="ampm-pm" onclick="setSchedAmPm('PM')">PM</button>
            </div>
          </div>
        </div>
        <div style="flex:1;">
          <div class="add-label">Action</div>
          <div class="action-seg">
            <button class="act-btn on" id="act-on" onclick="setSchedAction(1)">ON</button>
            <button class="act-btn" id="act-off" onclick="setSchedAction(0)">OFF</button>
          </div>
        </div>
      </div>

      <!-- Live preview -->
      <div class="sched-preview" id="sched-preview">
        <div class="prev-icon" id="prev-icon">🗓</div>
        <div style="flex:1;">
          <div class="prev-title" id="prev-title">Choose days & time to preview</div>
          <div class="prev-sub" id="prev-sub">—</div>
        </div>
      </div>

      <div style="display:flex;gap:8px;margin-top:12px;">
        <button onclick="addSchedule()" style="flex:2;padding:12px;background:#16a34a;color:white;border:none;border-radius:8px;font-size:14px;font-weight:700;cursor:pointer;">✓ Add Event</button>
        <button onclick="confirmClearAllSchedule()" style="flex:1;padding:12px;background:#ef4444;color:white;border:none;border-radius:8px;font-size:13px;cursor:pointer;font-weight:700;">🗑 Clear All</button>
      </div>
    </div>
  </div>
</div>

<!-- Time wheel modal -->
<div class="confirm-modal" id="time-wheel-modal">
  <div class="time-wheel-box">
    <h3 style="margin-bottom:10px;color:#1e3a5f;font-size:15px;">Set Time</h3>
    <div class="tw-land">
      <div class="tw-land-clock">
        <canvas id="tw-clock" width="150" height="150" style="display:block;"></canvas>
      </div>
      <div class="tw-land-ctrls">
        <div class="tw-display"><span id="tw-hh">08</span><span class="tw-colon">:</span><span id="tw-mm">00</span> <span id="tw-ampm" style="font-size:20px;color:#94a3b8;">AM</span></div>
        <div class="tw-cols">
          <div class="tw-col">
            <div class="tw-label">Hour</div>
            <div class="tw-stepper">
              <button onclick="twStep('h',1)">▲</button>
              <div class="tw-wheel" id="tw-wheel-h"></div>
              <button onclick="twStep('h',-1)">▼</button>
            </div>
          </div>
          <div class="tw-col">
            <div class="tw-label">Minute</div>
            <div class="tw-stepper">
              <button onclick="twStep('m',1)">▲</button>
              <div class="tw-wheel" id="tw-wheel-m"></div>
              <button onclick="twStep('m',-1)">▼</button>
            </div>
          </div>
        </div>
        <div class="tw-quick">
          <button onclick="twQuick(6,30)">06:30</button>
          <button onclick="twQuick(8,0)">08:00</button>
          <button onclick="twQuick(12,0)">12:00</button>
          <button onclick="twQuick(16,30)">16:30</button>
        </div>
        <div class="confirm-btns" style="margin-top:12px;">
          <button class="btn-cancel" onclick="closeTimeWheel()">Cancel</button>
          <button class="btn-danger" style="background:#16a34a;" onclick="applyTimeWheel()">Set Time</button>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ══ CONFIG TAB — login gate INSIDE the page div ══ -->
<div class="page active" id="tab-config">

  <!-- Login gate — shown by default -->
  <div id="config-login-gate" style="padding:20px;">
    <div class="login-gate-box">
      <h2><img class="ic" src="/home/linaro/icons/login-blue-128x128.png"> Configuration Access</h2>
      <p>Login to access device configuration</p>
      <label>Username</label>
      <input type="text" id="cfg-username" placeholder="username" value="admin" onclick="vkOpen(this,'Username')" readonly/>
      <label>Password</label>
      <input type="password" id="cfg-password" placeholder="password" value="envi2026" onclick="vkOpen(this,'Password')" readonly onkeydown="if(event.key==='Enter')cfgLogin()"/>
      <button class="enter-btn" onclick="cfgLogin()">Enter</button>
      <div class="err-msg" id="cfg-login-msg"></div>
    </div>
  </div>

<!-- Config content — hidden until login -->
  <div id="config-content" style="display:none;">
    <div class="cfg-page">
      <div class="cfg-topbar">
        <div>
          <h1 style="margin:0;">System Configuration</h1>
        </div>
        <button class="cfg-logout-btn" onclick="cfgLogout()">🔒 Logout</button>
      </div>

      <!-- TILE 1 — Wizard launcher -->
      <div class="cfg-tile cfg-tile-wizard" onclick="wizOpen()">
        <div class="cfg-tile-icon"><img src="/home/linaro/icons/wizard-blue-128x128.png" style="width:30px;height:30px;filter:brightness(0) invert(1);"></div>
        <div class="cfg-tile-body">
          <div class="cfg-tile-title">Guided Setup Wizard</div>
          <div class="cfg-tile-desc">Step-by-step controller setup — recommended</div>
        </div>
        <div class="cfg-tile-arrow">▶</div>
      </div>

      <!-- TILE 2 — Advanced config (expandable) -->
      <div class="cfg-tile cfg-tile-advanced" onclick="toggleAdvCfg()">
        <div class="cfg-tile-icon"><img src="/home/linaro/icons/config-blue-128x128.png" style="width:30px;height:30px;filter:brightness(0) invert(1);"></div>
        <div class="cfg-tile-body">
          <div class="cfg-tile-title">Advanced Configuration</div>
          <div class="cfg-tile-desc">Serial port, controllers &amp; manual settings</div>
        </div>
        <div class="cfg-tile-arrow" id="adv-cfg-chev">▶</div>
      </div>

      <!-- Advanced body — hidden until tile tapped -->
      <div id="adv-cfg-body" style="display:none;">
      <div class="cfg-group-label"><img class="ic" src="/home/linaro/icons/network-blue-128x128.png"> Communication</div>
      <div class="serial-field">
        <label>Serial Port</label>

        <select id="cfg-port">
          <option value="/dev/ttyS7">ENVI CENTRAL </option>
          <option value="/dev/ttyS1">ENVI GLOBAL </option>
        </select>
        <div style="margin-top:6px;display:flex;gap:8px;align-items:center;">
          <label style="font-size:11px;color:#888;text-transform:uppercase;">Baud Rate</label>
          <select id="cfg-baud" style="padding:5px 8px;border:1px solid #ccc;border-radius:5px;font-size:12px;">
            <option value="9600" selected>9600</option>
            <option value="19200">19200</option>
            <option value="38400">38400</option>
          </select>
          <button class="serial-save-btn" onclick="saveSerialConfig()">Apply Port</button>
          <span id="serial-result" style="font-size:12px;color:#22c55e;"></span>
        </div>
      </div>
      <div class="cfg-group-label"><img class="ic" src="/home/linaro/icons/control-blue-128x128.png"> Controllers</div>
      <div id="ctrl-cfg-list"></div>
      <button class="add-ctrl-btn" onclick="addCtrl()">＋ Add Controller</button>
      <div class="unlock-banner" id="unlock-banner" onclick="enterDashboard()" style="display:block;">✓ Connected! Tap here to open Dashboard →</div>
      <div id="cfg-save-result" style="font-size:12px;margin-top:8px;color:#888;text-align:center;"></div>

      <div class="cfg-group-label"><img class="ic" src="/home/linaro/icons/config-blue-128x128.png"> Advanced Tools</div>
      <!-- Technician Mode toggle -->
      <div class="cfg-section">
        <div class="cfg-section-row">
          <div>
            <div class="cfg-section-title"><img class="ic" src="/home/linaro/icons/config-blue-128x128.png"> Technician Mode</div>
            <div class="cfg-section-sub">Show register-level Control tab &amp; advanced tools</div>
          </div>
          <button class="toggle-btn off" id="tech-mode-btn" onclick="toggleTechMode()">Off</button>
        </div>
      </div>

      <!-- Write Any Register — technical tool (moved from AI tab) -->
      <div class="write-panel" style="margin-top:14px;">
      <h2 onclick="toggleWP('cfg-write-reg-body','cfg-write-reg-chev')"><img class="ic" src="/home/linaro/icons/config-blue-128x128.png"> Write Any Register <span style="font-size:9px;background:#ef4444;color:#fff;padding:1px 7px;border-radius:8px;margin-left:6px;">ADVANCED</span><span id="cfg-write-reg-chev" class="acc-chev" style="margin-left:auto;">▶</span></h2>
        <div class="write-panel-body" id="cfg-write-reg-body">
          <div class="wr-warn">⚠ Direct Modbus write — technicians only. Incorrect values can disrupt a live controller.</div>
          <div class="wr-field">
            <label>Target Controller</label>
            <div id="write-ctrl-btns" class="wr-ctrl-btns"></div>
            <input type="hidden" id="write-ctrl" value="0"/>
          </div>
          <div class="wr-grid">
            <div class="wr-field">
              <label>Register Address</label>
              <input type="number" id="write-reg" class="wr-input" value="4" onclick="vkOpen(this,'Register Address')" readonly/>
            </div>
            <div class="wr-field">
              <label>Value (raw)</label>
              <input type="number" id="write-val" class="wr-input" value="230" onclick="vkOpen(this,'Register Value')" readonly/>
            </div>
          </div>
          <button class="wr-write-btn" onclick="writeReg()">✍ Write Register</button>
          <div id="write-result" class="wr-result"></div>
        </div>
      </div>
      </div><!-- /adv-cfg-body -->
    </div><!-- /cfg-page -->
  </div><!-- /config-content -->

</div><!-- /tab-config -->

<!-- ══ AI TAB — no login, fully open ══ -->
<div class="page" id="tab-insight_ai">

  <!-- Motion Control Config -->
  <div class="ai-panel">
    <div class="ai-panel-hdr" onclick="toggleAiPanel('ai-motion-cfg')"><img class="ic" style="filter:brightness(0) invert(1);" src="/home/linaro/icons/config-blue-128x128.png"> Motion &amp; Screen Control <span id="ai-motion-cfg-chev">▶</span></div>
    <div class="ai-panel-body" id="ai-motion-cfg">
      <div style="margin-bottom:12px;">
        <div style="font-size:11px;color:#888;text-transform:uppercase;font-weight:bold;margin-bottom:8px;">Screen Sleep Timeout</div>
        <div style="display:flex;gap:8px;flex-wrap:wrap;">
          <button class="sleep-btn" id="sleep-btn-3"  onclick="setSleepTimeout(3)">3s</button>
          <button class="sleep-btn active" id="sleep-btn-5"  onclick="setSleepTimeout(5)">5s</button>
          <button class="sleep-btn" id="sleep-btn-10" onclick="setSleepTimeout(10)">10s</button>
          <button class="sleep-btn" id="sleep-btn-20" onclick="setSleepTimeout(20)">20s</button>
          <button class="sleep-btn" id="sleep-btn-60" onclick="setSleepTimeout(60)">60s</button>
        </div>
        <div style="font-size:11px;color:#888;margin-top:6px;">Current: <span id="sleep-timeout-label">5s</span></div>
      </div>
      <div style="display:flex;align-items:center;justify-content:space-between;padding:10px;background:#f8fafc;border-radius:8px;border:1px solid #e2e8f0;">
        <div>
          <div style="font-size:13px;font-weight:bold;color:#1e3a5f;">Motion Detection</div>
          <div style="font-size:11px;color:#888;margin-top:2px;">Enable to auto-sleep screen when no motion</div>
        </div>
        <button class="toggle-btn on" id="motion-enable-btn" onclick="toggleMotionEnabled()">Enabled</button>
      </div>
    </div>
  </div>



  <!-- AI Energy Mode — Predictive Occupancy -->
  <div class="ai-panel">
    <div class="ai-panel-hdr" onclick="toggleAiPanel('ai-energy')"><img class="ic" style="filter:brightness(0) invert(1);" src="/home/linaro/icons/ai-blue-128x128.png"> AI Energy Mode <span id="ai-energy-chev">▶</span></div>
    <div class="ai-panel-body" id="ai-energy" style="background:#0b0f14;padding:14px;">

      <!-- State + recommendation card -->
      <div id="occ-state-card" style="border-radius:12px;padding:16px;margin-bottom:12px;background:#111827;border:1px solid #1f2937;transition:all .4s;">
        <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
          <div style="display:flex;align-items:center;gap:10px;">
            <div id="occ-dot" style="width:12px;height:12px;border-radius:50%;background:#4a6070;flex-shrink:0;"></div>
            <div id="occ-state" style="font-size:16px;font-weight:800;color:#fff;letter-spacing:.5px;">Learning…</div>
          </div>
          <div id="occ-learn" style="font-size:10px;color:#64748b;text-align:right;">learning 0/14 days</div>
        </div>
        <div id="occ-action" style="font-size:13px;color:#cbd5e1;line-height:1.5;margin-bottom:6px;">Gathering occupancy data…</div>
        <div id="occ-reason" style="font-size:11px;color:#64748b;font-style:italic;margin-bottom:10px;"></div>
        <button id="occ-apply-btn" onclick="occApplySetback()" style="display:none;width:100%;padding:12px;border:none;border-radius:10px;background:linear-gradient(135deg,#16a34a,#15803d);color:#fff;font-size:14px;font-weight:800;cursor:pointer;">
          ✓ Apply Setback (±3°C) to save energy
        </button>
        <div id="occ-apply-result" style="font-size:11px;color:#4ade80;text-align:center;margin-top:6px;min-height:14px;"></div>
      </div>

      <!-- Override + arrival row -->
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px;">
        <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:12px;">
          <div style="font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:#4a6070;margin-bottom:6px;">Typical Arrival Today</div>
          <div id="occ-arrival" style="font-size:22px;font-weight:800;color:#38bdf8;">—</div>
          <div style="font-size:10px;color:#64748b;margin-top:2px;">pre-condition 15 min prior</div>
        </div>
        <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:12px;display:flex;flex-direction:column;justify-content:space-between;">
          <div style="font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:#4a6070;margin-bottom:6px;">Manual Override</div>
          <button id="occ-override-btn" onclick="occToggleOverride()" style="padding:10px;border:none;border-radius:8px;background:#1e3a5f;color:#fff;font-size:12px;font-weight:700;cursor:pointer;">🙋 I'm Here Now</button>
        </div>
      </div>

      <!-- 7-day heatmap -->
      <div style="background:#111827;border:1px solid #1f2937;border-radius:10px;padding:12px;">
        <div style="font-size:9px;letter-spacing:1.5px;text-transform:uppercase;color:#4a6070;margin-bottom:8px;">Learned Weekly Pattern <span id="occ-samples" style="float:right;text-transform:none;letter-spacing:0;">—</span></div>
        <div id="occ-heatmap" style="font-size:0;"></div>
        <div style="display:flex;justify-content:space-between;font-size:8px;color:#4a6070;margin-top:4px;">
          <span>12a</span><span>6a</span><span>12p</span><span>6p</span><span>11p</span>
        </div>
        <div style="display:flex;align-items:center;gap:6px;margin-top:8px;font-size:9px;color:#64748b;">
          <span>empty</span>
          <span style="width:14px;height:8px;border-radius:2px;background:#1e293b;"></span>
          <span style="width:14px;height:8px;border-radius:2px;background:#0e7490;"></span>
          <span style="width:14px;height:8px;border-radius:2px;background:#0ea5e9;"></span>
          <span style="width:14px;height:8px;border-radius:2px;background:#38bdf8;"></span>
          <span>occupied</span>
        </div>
      </div>

    </div>
  </div>

  <!-- Register inputs -->
  <div class="ai-panel">
    <div class="ai-panel-hdr" onclick="toggleAiPanel('ai-regs')"><img class="ic" style="filter:brightness(0) invert(1);" src="/home/linaro/icons/control-blue-128x128.png"> Live Registers <span id="ai-regs-chev">▶</span></div>
    <div class="ai-panel-body" id="ai-regs" style="padding:12px;">
      <select id="inputs-ctrl-sel" onchange="refreshInputsTable()" style="padding:7px 10px;border:1px solid #ccc;border-radius:6px;font-size:13px;width:100%;margin-bottom:10px;"></select>
      <table class="lr-table">
        <tbody id="all-regs"></tbody>
      </table>
    </div>
  </div>

  <!-- Motion Detection Live -->
  <div class="ai-panel">
    <div class="ai-panel-hdr" onclick="toggleAiPanel('ai-motion-live')"><img class="ic" style="filter:brightness(0) invert(1);" src="/home/linaro/icons/ai-blue-128x128.png"> Motion Detection — Live <span id="ai-motion-live-chev">▶</span></div>
    <div class="ai-panel-body" id="ai-motion-live" style="padding:10px;background:#080b10;">
      <div class="ai-status-strip">
        <div class="ai-status-dot" id="ai-dot"></div>
        <div class="ai-status-text" id="ai-status-text">Initializing camera...</div>
        <div style="font-size:10px;color:#4a6070;font-family:Courier New,monospace;" id="ai-last-wake">Last: —</div>
      </div>
      <div class="ai-motion-grid">
        <div class="ai-motion-card">
          <div class="ai-lbl">Motion Level</div>
          <div class="ai-val green" id="ai-motion-pct">0.0%</div>
          <div class="ai-sub" id="ai-changed-px">0 px changed</div>
          <div class="ai-bar-wrap"><div class="ai-bar-fill" id="ai-motion-bar" style="width:0%"></div></div>
        </div>
        <div class="ai-motion-card">
          <div class="ai-lbl">Occupancy</div>
          <div class="ai-val yellow" id="ai-faces">0</div>
          <div class="ai-sub" id="ai-consecutive">consecutive: 0</div>
        </div>
        <div class="ai-motion-card">
          <div class="ai-lbl">Frames Processed</div>
          <div class="ai-val" id="ai-frames">0</div>
          <div class="ai-sub" id="ai-warmup">awaiting calibration</div>
        </div>
        <div class="ai-motion-card">
          <div class="ai-lbl">Sleep Timer</div>
          <div class="ai-val red" id="ai-sleep-in">--s</div>
          <div class="ai-sub">until screensaver</div>
        </div>
      </div>
      <div class="ai-history-chart">
        <div class="ai-lbl">Motion History (last 12 samples)</div>
        <div class="ai-bars-container" id="ai-bars">
          <div class="ai-bar"></div><div class="ai-bar"></div><div class="ai-bar"></div>
          <div class="ai-bar"></div><div class="ai-bar"></div><div class="ai-bar"></div>
          <div class="ai-bar"></div><div class="ai-bar"></div><div class="ai-bar"></div>
          <div class="ai-bar"></div><div class="ai-bar"></div><div class="ai-bar"></div>
        </div>
      </div>
    </div>
  </div>

</div><!-- /tab-insight_ai -->

<!-- Help modal -->
<!-- Quick Start Guide — landscape popover, anchored to the upper-right corner -->
<div class="qs-overlay" id="quickstart-overlay" onclick="closeQuickStart()">
  <div class="qs-panel" onclick="event.stopPropagation()">
    <div class="qs-hdr">
      <div class="qs-title">🚀 Quick Start Guide</div>
      <button class="qs-close" onclick="closeQuickStart()" title="Close">✕</button>
    </div>
    <div class="qs-steps">
      <div class="qs-step">
        <div class="qs-num">1</div>
        <div class="qs-txt"><b>Connect</b><span>Config tab → log in → set the Modbus Slave ID → Save &amp; Connect.</span></div>
      </div>
      <div class="qs-step">
        <div class="qs-num">2</div>
        <div class="qs-txt"><b>Monitor</b><span>Dashboard shows room temp, setpoint &amp; CO₂ live for each zone.</span></div>
      </div>
      <div class="qs-step">
        <div class="qs-num">3</div>
        <div class="qs-txt"><b>Adjust</b><span>Drag the dial or tap ▼ ▲ to set the target; pick Fan &amp; Mode below.</span></div>
      </div>
      <div class="qs-step">
        <div class="qs-num">4</div>
        <div class="qs-txt"><b>Automate</b><span>Schedule tab → add timed ON/OFF events on the weekly grid.</span></div>
      </div>
      <div class="qs-step qs-step-live">
        <div class="qs-num">5</div>
        <div class="qs-txt"><b>Live</b><span>Scan to open this panel on your phone.</span></div>
        <div class="qs-qr"><img id="qs-qr-img" src="/api/qr" alt="Scan to open panel"></div>
        <div class="qs-qr-url" id="qs-qr-url">Finding address…</div>
      </div>
    </div>
    <div class="qs-help">
  <div><b>Login</b> admin / envi2026 (Config tab)</div>
  <div><b>Dashboard</b> tap header to expand · CO₂ bar shows IAQ ppm</div>
  <div><b>AI tab</b> motion sleep/wake · adjust timeout · Write Any Register</div>
  <div><b>Support</b> support@insightcontrol.net.au</div>
  </div>
  </div>
</div>

<div class="help-modal" id="help-modal" onclick="closeHelpModal()">
  <div class="help-box" onclick="event.stopPropagation()">
    <h3>❓ Help &amp; Documentation</h3>
    <div class="help-section">
      <h4>📋 Quick Start</h4>
      <ul>
        <li>Go to <strong>Config</strong> tab → login → set Slave ID → Save &amp; Connect</li>
        <li>Once connected all other tabs unlock automatically</li>
        <li>Use <strong>Dashboard</strong> to monitor temp, set fan speed &amp; mode</li>
        <li>Use <strong>Schedule</strong> tab to set timed ON/OFF events</li>
      </ul>
    </div>
    <div class="help-section">
      <h4>🏠 Dashboard</h4>
      <ul>
        <li>Tap controller header to expand/collapse card</li>
        <li>Drag the knob or use ▼▲ buttons to adjust setpoint</li>
        <li>Fan Speed and Mode buttons update live from device</li>
        <li>CO Level bar shows IAQ sensor reading in PPM</li>
      </ul>
    </div>
    <div class="help-section">
      <h4>🛠️ Configuration (Login Required)</h4>
      <ul>
        <li>Username: <strong>admin</strong> / Password: <strong>envi2026</strong></li>
        <li>Set correct Modbus Slave ID for each controller</li>
        <li>Use Test Connection to verify before saving</li>
        <li>LED indicator can be assigned per controller</li>
      </ul>
    </div>
    <div class="help-section">
      <h4>🚀 AI Tab</h4>
      <ul>
        <li>Motion detection controls screen sleep/wake</li>
        <li>Adjust sleep timeout with the second buttons</li>
        <li>Disable motion detection to keep screen always on</li>
        <li>Write Any Register for direct Modbus control</li>
      </ul>
    </div>
    <div class="help-section">
      <h4>📄 Documents</h4>
      <button class="help-doc-btn" onclick="alert('Vector VFC Installation Guide\n\nContact: support@insightcontrol.net.au')">📘 Vector VFC Installation Guide</button>
Mode (0=Auto,1=Cool,2=Heat,3=Fan)\nReg 2: Fan Speed\nReg 4: Setpoint\nReg 5: Room Temp\nReg 1013: Alarms\nReg 1051: Standby/Comfort\nReg 1055: Time Schedules')">      <button class="help-doc-btn" onclick="alert('Modbus Register Map\n\nReg 0: ON/OFF\nReg 1: 📗 Modbus Register Map</button>
      <button class="help-doc-btn" onclick="alert('ENVI Panel v3\nInsight Control Pty Ltd\nsupport@insightcontrol.net.au\n\nFor field support contact your installer.')">📞 Support Contact</button>
    </div>
    <button onclick="closeHelpModal()" style="width:100%;padding:10px;background:#1e3a5f;color:white;border:none;border-radius:8px;font-size:13px;cursor:pointer;margin-top:6px;">Close</button>
  </div>
</div>

<!-- ══ POWER PANEL ══ -->
<div id="power-panel" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(15,23,42,0.98);z-index:9999;flex-direction:column;justify-content:center;align-items:center;color:white;overflow-y:auto;">
  <h2 style="margin-bottom:20px;font-size:22px;">System Controls</h2>
  <div style="background:rgba(255,255,255,0.08);border-radius:14px;padding:16px;width:85%;max-width:360px;margin-bottom:20px;">
    <div style="font-size:11px;text-transform:uppercase;letter-spacing:2px;color:#94a3b8;margin-bottom:10px;text-align:center;">UI Theme</div>
    <div style="display:flex;gap:8px;">
      <button onclick="setTheme('light')" id="theme-btn-light" class="theme-btn" style="flex:1;padding:14px 8px;background:#f8fafc;color:#1e3a5f;border:2px solid transparent;border-radius:10px;font-size:13px;font-weight:bold;cursor:pointer;">☀ Light</button>
      <button onclick="setTheme('dark')"  id="theme-btn-dark"  class="theme-btn" style="flex:1;padding:14px 8px;background:#0f172a;color:#e2e8f0;border:2px solid transparent;border-radius:10px;font-size:13px;font-weight:bold;cursor:pointer;">🌙 Dark</button>
    </div>
  </div>
  <div style="display:flex;flex-direction:column;gap:14px;width:85%;max-width:360px;">
    <button onclick="executePower('restart')"  style="padding:18px;background:#f59e0b;color:white;border:none;border-radius:12px;font-size:16px;font-weight:bold;cursor:pointer;">Restart Display</button>
    <button onclick="executePower('shutdown')" style="padding:18px;background:#ef4444;color:white;border:none;border-radius:12px;font-size:16px;font-weight:bold;cursor:pointer;">Power Off</button>
    <button onclick="hidePowerPanel()"        style="padding:12px;background:#4b5563;color:white;border:none;border-radius:10px;font-size:14px;cursor:pointer;margin-top:8px;">⬅ Cancel</button>
  </div>
</div>

<!-- Fullscreen chart -->
<div id="fullscreen-overlay" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:#0a0a0c;z-index:9998;flex-direction:column;">
  <div style="display:flex;justify-content:space-between;align-items:center;padding:12px 18px;background:#0e0e12;border-bottom:1px solid #1e1e26;color:white;">
    <div>
      <div style="font-size:16px;font-weight:800;display:flex;align-items:center;gap:6px;"><img class="ic" style="filter:brightness(0) invert(1);" src="/home/linaro/icons/trend-blue-128x128.png"> Live Trend</div>
      <div id="fs-title" style="font-size:12px;color:#71717a;margin-top:2px;">—</div>
    </div>
    <button onclick="closeFullscreenChart()" style="background:#ef4444;color:white;border:none;border-radius:8px;padding:9px 18px;font-size:14px;font-weight:700;cursor:pointer;">✕ Close</button>
  </div>
  <div style="flex:1;padding:18px;background:#0a0a0c;"><canvas id="tempChartFS" style="width:100%;height:100%;"></canvas></div>
</div>

<!-- Email modal -->
<div class="overlay-panel" id="email-modal" onclick="closeEmailModal()">
  <div class="overlay-box" onclick="event.stopPropagation()">
    <h3><img class="ic" src="/home/linaro/icons/email-blue-128x128.png"> Send Temperature Log</h3>
    <div class="email-recipient-row">
      <label>Recipient Email</label>
      <input type="email" id="email-recipient-input" placeholder="recipient@example.com" onclick="vkOpen(this,'Recipient Email')" readonly style="margin-bottom:0;"/>
    </div>
    <p style="font-size:12px;color:#666;margin-bottom:6px;">Log period</p>
    <div class="tcfg-seg" style="margin-bottom:12px;">
      <button class="tcfg-chip" id="email-30"  onclick="setEmailMinutes(30)">30m</button>
      <button class="tcfg-chip active" id="email-60"  onclick="setEmailMinutes(60)">1h</button>
      <button class="tcfg-chip" id="email-120" onclick="setEmailMinutes(120)">2h</button>
      <button class="tcfg-chip" id="email-360" onclick="setEmailMinutes(360)">6h</button>
      <button class="tcfg-chip" id="email-720" onclick="setEmailMinutes(720)">12h</button>
      <button class="tcfg-chip" id="email-1440" onclick="setEmailMinutes(1440)">24h</button>
    </div>
    <p style="font-size:12px;color:#666;margin-bottom:10px;">Choose which controller's log to email:</p>
    <div id="email-ctrl-options"></div>
    <div style="display:flex;gap:8px;margin-top:14px;">
      <button onclick="closeEmailModal()" style="flex:1;padding:10px;background:#e5e7eb;color:#555;border:none;border-radius:8px;font-size:13px;cursor:pointer;">Cancel</button>
      <button onclick="confirmSendEmail()" style="flex:2;padding:10px;background:#1e3a5f;color:white;border:none;border-radius:8px;font-size:13px;font-weight:bold;cursor:pointer;">Send Email</button>
    </div>
    <div id="email-send-result" style="font-size:12px;margin-top:8px;text-align:center;min-height:16px;"></div>
  </div>
</div>

<!-- Network panel -->
<div class="overlay-panel" id="network-panel" onclick="closeNetworkPanel()">
  <div class="overlay-box" onclick="event.stopPropagation()">
    <h3>🌐 Network Information</h3>
    <div id="network-info-body"><div style="text-align:center;padding:20px;color:#888;font-size:13px;">Loading…</div></div>
    <div style="margin-top:14px;display:flex;gap:8px;">
    <button onclick="loadNetworkInfo()" style="flex:1;padding:9px;background:#e5e7eb;color:#555;border:none;border-radius:8px;font-size:13px;cursor:pointer;"><img class="ic" src="/home/linaro/icons/refresh-blue-128x128.png"> Refresh</button>
    <button onclick="closeNetworkPanel()" style="flex:1;padding:9px;background:#1e3a5f;color:white;border:none;border-radius:8px;font-size:13px;cursor:pointer;">Close</button>
    </div>
  </div>
</div>

<!-- Wi-Fi setup popup -->
<div class="overlay-panel" id="wifi-panel" onclick="closeWifi()">
  <div class="overlay-box wifi-box" onclick="event.stopPropagation()">
    <div class="wifi-head">
      <h3>📶 Wi-Fi Setup</h3>
      <div id="wifi-status" class="wifi-status">Checking connection…</div>
    </div>
    <div class="wifi-body">
      <div class="wifi-col wifi-col-list">
        <div id="wifi-list" class="wifi-list">
          <div class="wifi-empty">Tap “Scan” to find networks…</div>
        </div>
        <button onclick="scanWifi()" id="wifi-scan-btn" class="wifi-btn wifi-btn-scan">🔄 Scan</button>
      </div>
      <div class="wifi-col wifi-col-connect">
        <div id="wifi-placeholder" class="wifi-placeholder">Select a network on the left to connect</div>
        <div id="wifi-connect-row" class="wifi-connect-row" style="display:none;">
          <div class="wifi-sel-label">Connect to <b id="wifi-sel-ssid"></b></div>
          <input type="password" id="wifi-pass" placeholder="Wi-Fi password" autocomplete="off">
          <label class="wifi-show"><input type="checkbox" id="wifi-showpass" onchange="toggleWifiPass()"> Show password</label>
          <button onclick="connectWifi()" id="wifi-connect-btn" class="wifi-btn wifi-btn-connect">Connect</button>
        </div>
        <div id="wifi-msg" class="wifi-msg"></div>
      </div>
    </div>
    <div class="wifi-foot">
      <button onclick="closeWifi()" class="wifi-btn wifi-btn-close">Close</button>
    </div>
  </div>
</div>

<!-- Set Date & Time popup -->
<div class="overlay-panel" id="settime-panel" onclick="closeSetTime()">
  <div class="overlay-box st-box" onclick="event.stopPropagation()">
    <h3 class="st-head">🕒 Set Date &amp; Time <span id="st-location" class="st-location"></span></h3>
    <div id="st-source" class="st-source">Checking clock source…</div>
    <div class="st-tzrow">
      <label class="st-tzlabel">State / Time zone</label>
      <div class="st-tzctrls">
        <select id="st-tz-select" class="st-tzselect"></select>
        <button class="st-tzbtn" onclick="stApplyTz()">Set Zone</button>
      </div>
    </div>
    <div class="st-spins">
      <div class="st-spin"><button class="st-arrow" onclick="stAdj('day',1)">▲</button><div class="st-val" id="st-day">01</div><button class="st-arrow" onclick="stAdj('day',-1)">▼</button><div class="st-lbl">Day</div></div>
      <div class="st-spin"><button class="st-arrow" onclick="stAdj('month',1)">▲</button><div class="st-val" id="st-month">Jan</div><button class="st-arrow" onclick="stAdj('month',-1)">▼</button><div class="st-lbl">Month</div></div>
      <div class="st-spin"><button class="st-arrow" onclick="stAdj('year',1)">▲</button><div class="st-val" id="st-year">2026</div><button class="st-arrow" onclick="stAdj('year',-1)">▼</button><div class="st-lbl">Year</div></div>
      <div class="st-sep">—</div>
      <div class="st-spin"><button class="st-arrow" onclick="stAdj('hour',1)">▲</button><div class="st-val" id="st-hour">00</div><button class="st-arrow" onclick="stAdj('hour',-1)">▼</button><div class="st-lbl">Hour</div></div>
      <div class="st-spin"><button class="st-arrow" onclick="stAdj('minute',1)">▲</button><div class="st-val" id="st-minute">00</div><button class="st-arrow" onclick="stAdj('minute',-1)">▼</button><div class="st-lbl">Min</div></div>
    </div>
    <div id="st-msg" class="st-msg"></div>
    <div style="margin-top:14px;display:flex;gap:8px;">
      <button onclick="stUseNow()" class="st-btn st-btn-now" title="Load the current device time">↺ Now</button>
      <button onclick="setTimeApply()" id="st-apply" class="st-btn st-btn-apply">✓ Set Time</button>
      <button onclick="closeSetTime()" class="st-btn st-btn-close">Close</button>
    </div>
  </div>
</div>

<!-- Confirm: delete controller -->
<div class="confirm-modal" id="confirm-delete-modal">
  <div class="confirm-box">
    <h3>🗑 Delete Controller?</h3>
    <p id="confirm-delete-msg">This will remove the controller and its schedule data.</p>
    <div class="confirm-btns">
      <button class="btn-cancel" onclick="hideConfirmDelete()">Cancel</button>
      <button class="btn-danger" onclick="executeDeleteCtrl()">Yes, Delete</button>
    </div>
  </div>
</div>

<!-- Confirm: clear schedule -->
<div class="confirm-modal" id="clear-sched-modal">
  <div class="confirm-box">
    <h3>🗑 Clear All Schedule Events?</h3>
    <p id="clear-sched-msg">This will permanently delete all scheduled events for the selected controller(s). Are you sure?</p>
    <div class="confirm-btns">
      <button class="btn-cancel" onclick="hideClearSchedModal()">Cancel</button>
      <button class="btn-danger" onclick="executeClearAllSchedule()">Yes, Clear All</button>
    </div>
  </div>
</div>

<!-- Confirm: change live register -->
<div class="confirm-modal" id="confirm-reg-modal">
  <div class="confirm-box">
    <h3>⚠ Confirm Change</h3>
    <p id="confirm-reg-msg">Change this setting?</p>
    <div class="confirm-btns">
      <button class="btn-cancel" onclick="hideConfirmReg()">Cancel</button>
      <button class="btn-danger" onclick="executeConfirmReg()">Yes, Change</button>
    </div>
  </div>
</div>

<!-- ══ VIRTUAL KEYBOARD ══ -->
<div id="vk-overlay">
  <div id="vk-target-label">Enter text</div>
  <div id="vk-input-display"></div>
  <!-- Row 1: numbers -->
  <div class="vk-row" id="vk-row-num">
    <button class="vk-key" data-k="1">1</button>
    <button class="vk-key" data-k="2">2</button>
    <button class="vk-key" data-k="3">3</button>
    <button class="vk-key" data-k="4">4</button>
    <button class="vk-key" data-k="5">5</button>
    <button class="vk-key" data-k="6">6</button>
    <button class="vk-key" data-k="7">7</button>
    <button class="vk-key" data-k="8">8</button>
    <button class="vk-key" data-k="9">9</button>
    <button class="vk-key" data-k="0">0</button>
    <button class="vk-key vk-del" id="vk-del" onclick="vkDel()">⌫</button>
  </div>
  <!-- Row 2: qwerty -->
  <div class="vk-row" id="vk-row-q">
    <button class="vk-key" data-k="q">q</button>
    <button class="vk-key" data-k="w">w</button>
    <button class="vk-key" data-k="e">e</button>
    <button class="vk-key" data-k="r">r</button>
    <button class="vk-key" data-k="t">t</button>
    <button class="vk-key" data-k="y">y</button>
    <button class="vk-key" data-k="u">u</button>
    <button class="vk-key" data-k="i">i</button>
    <button class="vk-key" data-k="o">o</button>
    <button class="vk-key" data-k="p">p</button>
  </div>
  <!-- Row 3: asdf -->
  <div class="vk-row" id="vk-row-a">
    <button class="vk-key" data-k="a">a</button>
    <button class="vk-key" data-k="s">s</button>
    <button class="vk-key" data-k="d">d</button>
    <button class="vk-key" data-k="f">f</button>
    <button class="vk-key" data-k="g">g</button>
    <button class="vk-key" data-k="h">h</button>
    <button class="vk-key" data-k="j">j</button>
    <button class="vk-key" data-k="k">k</button>
    <button class="vk-key" data-k="l">l</button>
    <button class="vk-key" data-k="." data-ks=".">.</button>
  </div>
  <!-- Row 4: zxcv -->
  <div class="vk-row" id="vk-row-z">
    <button class="vk-key vk-shift" id="vk-shift" onclick="vkShift()">⇧</button>
    <button class="vk-key" data-k="z">z</button>
    <button class="vk-key" data-k="x">x</button>
    <button class="vk-key" data-k="c">c</button>
    <button class="vk-key" data-k="v">v</button>
    <button class="vk-key" data-k="b">b</button>
    <button class="vk-key" data-k="n">n</button>
    <button class="vk-key" data-k="m">m</button>
    <button class="vk-key" data-k="_">_</button>
    <button class="vk-key" data-k="-">-</button>
  </div>
  <!-- Row 5: special -->
  <div class="vk-row" id="vk-row-sp">
    <button class="vk-key vk-sym" onclick="vkSym()">!#@</button>
    <button class="vk-key" data-k="@">@</button>
    <button class="vk-key vk-space" data-k=" ">SPACE</button>
    <button class="vk-key" data-k=".com">.com</button>
    <button class="vk-key vk-done" onclick="vkDone()">Done ✓</button>
  </div>
  <!-- Symbol row (hidden by default) -->
  <div class="vk-row" id="vk-row-sym" style="display:none;">
    <button class="vk-key" data-k="!">!</button>
    <button class="vk-key" data-k="#">#</button>
    <button class="vk-key" data-k="$">$</button>
    <button class="vk-key" data-k="%">%</button>
    <button class="vk-key" data-k="&amp;" data-k2="&">&amp;</button>
    <button class="vk-key" data-k="*">*</button>
    <button class="vk-key" data-k="(">(</button>
    <button class="vk-key" data-k=")">)</button>
    <button class="vk-key" data-k="+"  >+</button>
    <button class="vk-key" data-k="=">=</button>
    <button class="vk-key vk-sym" onclick="vkSym()">abc</button>
  </div>
</div>

<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<script>
// ═══════════════════════════════════════════════════════════════════
// GLOBALS
// ═══════════════════════════════════════════════════════════════════
var deviceConfig    = [];
var allCtrlData     = [];
var knobs           = {};
var ctrlCollapsed   = {};
var activeSchedCtrl = 0;
var tabsUnlocked    = false;
var emailCtrlSel    = 'all';
var ledEnabled      = true;
var technicianMode  = false;   // hides Control tab + register tools from end users
var trendPrimaryIdx = 0;
var DAYS_SHORT  = ['Mon','Tue','Wed','Thu','Fri','Sat','Sun'];

var ICON='/home/linaro/icons/';
function ic(name,cls){ return '<img class="'+(cls||'ic')+'" src="'+ICON+name+'-blue-128x128.png">'; }

var schedSelectedDays = new Set(['Mon']);
var META = {
  0:{opts:{0:"OFF",1:"ON"}},
  1051:{opts:{0:"Comfort",1:"Standby"}},
  1:{scale:0.1},                                      // IR1 Current Temperature (x10 raw)
  2:{opts:{0:"Low",1:"Low",2:"Medium",3:"High"}},
  1055:{opts:{0:"Disabled",1:"Enabled"}},
  2024:{opts:{0:"Summer",1:"Winter"}},
  100:{scale:0.1},                                     // HR100 Cooling Setpoint (x10 raw)
  102:{scale:0.1},                                     // HR102 Heating Setpoint (x10 raw)
  117:{opts:{0:"Auto",1:"Cool",2:"Heat",3:"Fan"}},   // HR117 Unit Mode (actual Temperzone encoding)
  116:{opts:{0:"Auto (%)"}},                          // HR116 Fan Speed Mode
  22:{opts:{0:"Disabled",1:"Enabled"}},               // Coil 22 Enable Scheduler
};
var MIN_C=18, MAX_C=30, MIN_DEG=220, MAX_DEG=500;
function c2a(c){return MIN_DEG+(c-MIN_C)/(MAX_C-MIN_C)*(MAX_DEG-MIN_DEG);}
function a2c(d){return MIN_C+(d-MIN_DEG)/(MAX_DEG-MIN_DEG)*(MAX_C-MIN_C);}
function clamp(v,lo,hi){return Math.max(lo,Math.min(hi,v));}

// ═══════════════════════════════════════════════════════════════════
// CONFIG LOGIN / LOGOUT
// ═══════════════════════════════════════════════════════════════════
// ═══════════════════════════════════════════════════════════════════
// VIRTUAL KEYBOARD
// ═══════════════════════════════════════════════════════════════════
var vkTarget = null;      // the real <input> element being filled
var vkValue  = '';
var vkShiftOn = false;
var vkSymOn   = false;

function vkOpen(inputEl, labelText){
  vkTarget = inputEl;
  vkValue  = inputEl.value || '';
  document.getElementById('vk-target-label').textContent = labelText || 'Enter text';
  document.getElementById('vk-input-display').textContent = vkValue;
  document.getElementById('vk-overlay').style.display = 'block';
  // scroll page so keyboard doesn't cover the input
  setTimeout(function(){ inputEl.scrollIntoView({behavior:'smooth',block:'center'}); }, 100);
  // Try system keyboard as well (onboard fallback)
  fetch('/api/keyboard',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'show'})}).catch(function(){});
}

function vkClose(){
  document.getElementById('vk-overlay').style.display = 'none';
  fetch('/api/keyboard',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action:'hide'})}).catch(function(){});
}

function vkDone(){
  if(vkTarget){ vkTarget.value = vkValue; vkTarget.dispatchEvent(new Event('change')); }
  vkClose();
  // If done on password field in config login, auto-attempt login
  if(vkTarget && vkTarget.id === 'cfg-password') cfgLogin();
}

function vkDel(){
  vkValue = vkValue.slice(0,-1);
  document.getElementById('vk-input-display').textContent = vkValue;
  if(vkTarget) vkTarget.value = vkValue;
}

function vkShift(){
  vkShiftOn = !vkShiftOn;
  var btn = document.getElementById('vk-shift');
  btn.classList.toggle('active', vkShiftOn);
  // update key labels
  document.querySelectorAll('#vk-overlay .vk-key[data-k]').forEach(function(k){
    var ch = k.getAttribute('data-k');
    if(ch && ch.length===1 && ch.match(/[a-z]/)){
      k.textContent = vkShiftOn ? ch.toUpperCase() : ch;
    }
  });
}

function vkSym(){
  vkSymOn = !vkSymOn;
  document.getElementById('vk-row-q').style.display   = vkSymOn ? 'none' : '';
  document.getElementById('vk-row-a').style.display   = vkSymOn ? 'none' : '';
  document.getElementById('vk-row-z').style.display   = vkSymOn ? 'none' : '';
  document.getElementById('vk-row-sym').style.display = vkSymOn ? '' : 'none';
}

// Key press handler — attach to all .vk-key buttons with data-k
document.addEventListener('DOMContentLoaded', function(){
  document.querySelectorAll('#vk-overlay .vk-key[data-k]').forEach(function(btn){
    btn.addEventListener('click', function(){
      var ch = btn.getAttribute('data-k');
      if(ch === '.com'){ vkValue += '.com'; }
      else if(ch === ' '){ vkValue += ' '; }
      else { vkValue += vkShiftOn ? ch.toUpperCase() : ch; }
      // auto-turn off shift after one character
      if(vkShiftOn && ch.length===1 && ch.match(/[a-z]/)){
        vkShiftOn = false;
        document.getElementById('vk-shift').classList.remove('active');
        document.querySelectorAll('#vk-overlay .vk-key[data-k]').forEach(function(k){
          var c2 = k.getAttribute('data-k');
          if(c2 && c2.length===1 && c2.match(/[a-zA-Z]/)) k.textContent = c2.toLowerCase();
        });
      }
      document.getElementById('vk-input-display').textContent = vkValue;
      if(vkTarget) vkTarget.value = vkValue;
    });
  });
});

// ── Attach VK to all text/email/password inputs automatically ────────────
function attachVKToInputs(){
  // Only attach to text/email/password — NOT number (those handled explicitly)
  // Skip anything that already has onclick=vkOpen in HTML
  document.querySelectorAll('input[type=text],input[type=email],input[type=password]').forEach(function(inp){
    if(inp.dataset.vkAttached) return;
    if(inp.getAttribute('onclick') && inp.getAttribute('onclick').indexOf('vkOpen')>-1) return;
    inp.dataset.vkAttached = '1';
    inp.setAttribute('readonly', 'readonly');
    inp.addEventListener('click', function(e){
      e.preventDefault();
      vkOpen(inp, inp.placeholder || 'Enter text');
    });
    inp.addEventListener('focus', function(){ inp.blur(); });
  });
}

// Patch cfg/login inputs specifically — they don't need readonly since they are
// rendered once; we handle them by onclick directly in their HTML
function patchLoginInputs(){
  ['cfg-username','cfg-password','email-recipient-input','write-reg','write-val'].forEach(function(id){
    var el = document.getElementById(id);
    if(!el || el.dataset.vkAttached) return;
    el.dataset.vkAttached = '1';
    el.removeAttribute('readonly');
    el.addEventListener('click', function(){
      vkOpen(el, el.placeholder || el.id);
    });
    el.addEventListener('focus', function(){ el.blur(); });
  });
}

var dashView = 'list';   // default on load

function setDashView(mode){
  dashView = mode;
  document.getElementById('view-btn-list').classList.toggle('active', mode==='list');
  document.getElementById('view-btn-grid').classList.toggle('active', mode==='grid');
  document.getElementById('ctrl-sections').style.display = mode==='list' ? 'block' : 'none';
  document.getElementById('ctrl-grid').style.display     = mode==='grid' ? 'block' : 'none';
  if(mode==='grid') buildGrid();
  poll();   // refresh values immediately for the new view
}

function buildGrid(){
  var grid = document.getElementById('ctrl-grid');
  if(!grid){ console.log('[GRID] ctrl-grid div missing'); return; }
  if(!deviceConfig || !deviceConfig.length){ console.log('[GRID] no deviceConfig yet'); grid.innerHTML='<div style="padding:20px;color:#888;">No controllers</div>'; return; }
  var html = '<div class="grid-cards">';
  deviceConfig.forEach(function(ctrl,idx){
    var dis = !ctrl.enabled;
    html += '<div class="grid-tile" id="grid-tile-'+idx+'" style="position:relative;'+(dis?'opacity:0.4;':'')+'">';
    html +=   '<div class="grid-tile-badge" id="grid-badge-'+idx+'" style="position:absolute;top:10px;right:10px;margin:0;">● --</div>';
    var gUnconfig = /^Controller \d+$/.test(ctrl.name);
    html +=   '<div class="grid-tile-name" style="padding-right:70px;">'+ctrl.name+(gUnconfig?' <span id="grid-setup-badge-'+idx+'" style="font-size:8px;font-weight:600;background:#f59e0b;color:#fff;padding:1px 5px;border-radius:6px;">⚙</span>':'')+'</div>';
    html +=   '<div class="grid-tile-temp" id="grid-temp-'+idx+'">--<span style="font-size:13px;">°C</span></div>';
    html +=   '<button class="grid-tile-onoff off" id="grid-onoff-'+idx+'" onclick="toggleCtrlOnOff('+idx+')">--</button>';
    html += '</div>';
  });
  html += '</div>';
  grid.innerHTML = html;
}

var CFG_USER='admin', CFG_PASS='envi2026', cfgLoggedIn=false;

function cfgLogin(){
  var u=(document.getElementById('cfg-username')||{}).value||'';
  var p=(document.getElementById('cfg-password')||{}).value||'';
  var msg=document.getElementById('cfg-login-msg');
  if(u===CFG_USER && p===CFG_PASS){
    cfgLoggedIn=true;
    document.getElementById('config-login-gate').style.display='none';
    document.getElementById('config-content').style.display='block';
    if(msg) msg.textContent='';
    setTimeout(function(){ patchLoginInputs(); attachVKToInputs(); }, 300);
  } else {
    if(msg) msg.textContent='✗ Invalid username or password';
  }
}

function cfgLogout(){
  cfgLoggedIn=false;
  document.getElementById('config-login-gate').style.display='block';
  document.getElementById('config-content').style.display='none';
  var u=document.getElementById('cfg-username');
  var p=document.getElementById('cfg-password');
  if(u) u.value='admin';
  if(p) p.value='envi2026';
  var msg=document.getElementById('cfg-login-msg');
  if(msg) msg.textContent='';
}

// ═══════════════════════════════════════════════════════════════════
// ECONOMIZER
// ═══════════════════════════════════════════════════════════════════
var economyState={};
var lastOutsideTemp = null;

function updateOutsideTempTiles(){
  if(lastOutsideTemp==null) return;
  deviceConfig.forEach(function(ctrl, idx){
    var el = document.getElementById('outsideTemp-'+idx);
    if(el) el.textContent = lastOutsideTemp.toFixed(1)+'°C';
    updateEcoLeaf(idx);
  });
}

function updateEcoLeaf(idx){
  var leaf = document.getElementById('eco-leaf-'+idx);
  var txt  = document.getElementById('eco-text-'+idx);
  var on = !!economyState[idx];
  if(leaf){ leaf.classList.toggle('active', on); leaf.title = on ? 'Economy cycle: ON' : 'Economy cycle: OFF'; }
  if(txt){ txt.style.display = on ? 'inline-block' : 'none'; }
}

function loadEconomy(){
  fetch('/api/economy_config').then(r=>r.json()).then(d=>{
    economyState=d.economy||{};
    deviceConfig.forEach(function(ctrl,idx){
      var on=!!economyState[idx];
      var b=document.getElementById('economy-btn-'+idx);
      if(b){b.textContent=on?'Enabled':'Disabled';b.className='toggle-btn '+(on?'on':'off');}
      updateEcoLeaf(idx);
    });
    updateOutsideTempTiles();
  }).catch(function(){});
}


function toggleEconomy(idx){
  var enable=!economyState[idx];
  economyState[idx]=enable;
  var b=document.getElementById('economy-btn-'+idx);
  if(b){b.textContent=enable?'Enabled':'Disabled';b.className='toggle-btn '+(enable?'on':'off');}
  updateEcoLeaf(idx);
  fetch('/api/economy_config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ctrl:idx,enabled:enable})});
}

// ── CO₂ card enable/disable (per controller) ──
var co2State={};
function co2Enabled(idx){
  // default ON when not explicitly set
  return (co2State[idx]===undefined) ? true : !!co2State[idx];
}
function loadCo2(){
  fetch('/api/co2_config').then(r=>r.json()).then(d=>{
    co2State=d.co2||{};
    deviceConfig.forEach(function(ctrl,idx){
      var on=co2Enabled(idx);
      var b=document.getElementById('co2-btn-'+idx);
      if(b){b.textContent=on?'Enabled':'Disabled';b.className='toggle-btn '+(on?'on':'off');}
      applyCo2Vis(idx);
    });
  }).catch(function(){});
}
function toggleCo2(idx){
  var enable=!co2Enabled(idx);
  co2State[idx]=enable;
  var b=document.getElementById('co2-btn-'+idx);
  if(b){b.textContent=enable?'Enabled':'Disabled';b.className='toggle-btn '+(enable?'on':'off');}
  applyCo2Vis(idx);
  fetch('/api/co2_config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ctrl:idx,enabled:enable})});
}
// Show CO₂ card when enabled; otherwise hide it (fan speed + mode remain below)
function applyCo2Vis(idx){
  var on=co2Enabled(idx);
  var tile=document.getElementById('co-tile-'+idx);
  if(tile) tile.style.display = on ? '' : 'none';
  var row=document.getElementById('metric-row-'+idx);
  if(row) row.classList.toggle('no-co2', !on);
}

function toggleAdvCfg(){
  var body=document.getElementById('adv-cfg-body');
  var chev=document.getElementById('adv-cfg-chev');
  if(!body) return;
  var open=body.style.display==='none';
  body.style.display=open?'block':'none';
  if(chev){ chev.textContent=open?'▼':'▶'; chev.classList.toggle('open',open); }
}

function toggleTechMode(){
  technicianMode = !technicianMode;
  var btn=document.getElementById('tech-mode-btn');
  if(btn){ btn.textContent=technicianMode?'On':'Off'; btn.className='toggle-btn '+(technicianMode?'on':'off'); }
  applyTechMode();
}

function applyTechMode(){
  // Control tab button — hide unless technician mode is on
  var ctrlTab=document.getElementById('tab-btn-control');
  if(ctrlTab) ctrlTab.style.display = technicianMode ? '' : 'none';
  // If currently on Control tab and tech mode just turned off, bounce to dashboard
  if(!technicianMode){
    var ctrlPage=document.getElementById('tab-control');
    if(ctrlPage && ctrlPage.classList.contains('active')) showTab('dashboard');
  }
}

function buildWriteCtrlBtns(){
  var box=document.getElementById('write-ctrl-btns');
  if(!box) return;
  var cur=(document.getElementById('write-ctrl')||{}).value||'0';
  var html='';
  deviceConfig.forEach(function(ctrl,idx){
    html+='<button class="wr-ctrl-btn'+(String(idx)===String(cur)?' sel':'')+'" onclick="event.stopPropagation();selectWriteCtrl('+idx+')">'+ctrl.name+'</button>';
  });
  box.innerHTML=html;
}
function selectWriteCtrl(idx){
  document.getElementById('write-ctrl').value=idx;
  buildWriteCtrlBtns();
}

// ═══════════════════════════════════════════════════════════════════
// BOOT
// ═══════════════════════════════════════════════════════════════════
window.onload = async function() {
  try { pollClock(); } catch(e){ setTimeout(pollClock, 2000); }
  try { await loadConfigIntoForm(); } catch(e){ setTimeout(loadConfigIntoForm, 3000); }
  try { initChart(); setInterval(updateChart, 2000); } catch(e){}
  try { poll(); } catch(e){ setTimeout(poll, 2000); }
  setTimeout(function(){ patchLoginInputs(); attachVKToInputs(); }, 500);
  applyTechMode();
  setTimeout(function(){ try{ wifiStartupCheck(); }catch(e){} }, 2500);
  
};

function manualRefresh() {
  poll(); updateChart(); loadSchedule(); fetchWeather(); buildAll();
  var btn = document.querySelector('button[onclick="manualRefresh()"]');
  if (btn) {
    var img = btn.querySelector('img');
    if (img) {
      var old = img.src;
      img.style.display = 'none';
      btn.insertAdjacentHTML('beforeend', '<span class="refresh-check">✓</span>');
      setTimeout(function(){
        var chk = btn.querySelector('.refresh-check');
        if (chk) chk.remove();
        img.style.display = '';
      }, 1200);
    }
  }
}

async function loadConfigIntoForm() {
  const res = await fetch('/api/device_config');
  deviceConfig = await res.json();
  buildConfigRows(deviceConfig);
  updateDynamicSelects();
  updateAddBtn();
}



function updateAddBtn() {
  var btn = document.querySelector('.add-ctrl-btn');
  if (!btn) return;
  var atMax = deviceConfig.length >= MAX_CONTROLLERS;
  btn.disabled = atMax;
  btn.style.opacity = atMax ? '0.4' : '1';
  btn.style.cursor  = atMax ? 'not-allowed' : 'pointer';
  btn.textContent   = atMax ? '✗ Maximum ' + MAX_CONTROLLERS + ' controllers reached' : '＋ Add Controller';
}

function updateWindEffect(idx,speed){
  var el=document.getElementById('wind-bg-'+idx);
  if(!el) return;
  el.className='wind-bg';
  if(speed===1) el.classList.add('wind-low');
  else if(speed===2) el.classList.add('wind-mid1');
  else if(speed===3) el.classList.add('wind-mid2');
  else if(speed===4) el.classList.add('wind-high');
}

// ── Occupancy AI ──
var occTimer=null;
function occStart(){
  occLoadSuggestion(); occLoadHeatmap();
  if(occTimer) clearInterval(occTimer);
  occTimer=setInterval(function(){
    var tab=document.getElementById('tab-insight_ai');
    if(tab && tab.classList.contains('active')) occLoadSuggestion();
    else { clearInterval(occTimer); occTimer=null; }
  },10000);
}
function occLoadSuggestion(){
  fetch('/api/occupancy/suggestion').then(r=>r.json()).then(d=>{
    var COL={OCCUPIED:'#22c55e',ARRIVING_SOON:'#f59e0b',LIKELY_EMPTY:'#64748b',UNCERTAIN:'#3b82f6'};
    var LBL={OCCUPIED:'Occupied',ARRIVING_SOON:'Arriving Soon',LIKELY_EMPTY:'Likely Empty',UNCERTAIN:'Uncertain'};
    var c=COL[d.state]||'#64748b';
    var dot=document.getElementById('occ-dot'); if(dot) dot.style.background=c;
    var st=document.getElementById('occ-state'); if(st){ st.textContent=LBL[d.state]||d.state; st.style.color=c; }
    var ac=document.getElementById('occ-action'); if(ac) ac.textContent=d.action||'';
    var rs=document.getElementById('occ-reason');
    if(rs) rs.textContent='now '+Math.round((d.prob_now||0)*100)+'% · next 15m '+Math.round((d.prob_next||0)*100)+'%'+(d.manual_override?' · override active':'');
    var ln=document.getElementById('occ-learn'); if(ln) ln.textContent='learning '+(d.days_learned||0)+'/14 days';
    var ar=document.getElementById('occ-arrival');
    if(ar){ var a=d.typical_arrival_today; ar.textContent=a?(String(a.hour).padStart(2,'0')+':'+String(a.minute).padStart(2,'0')):'—'; }
    var ob=document.getElementById('occ-override-btn');
    if(ob){ if(d.manual_override){ob.textContent='✓ Here (active)';ob.style.background='#16a34a';}else{ob.textContent="🙋 I'm Here Now";ob.style.background='#1e3a5f';} }
    var btn=document.getElementById('occ-apply-btn');
    if(btn) btn.style.display=(d.state==='LIKELY_EMPTY')?'block':'none';
  }).catch(function(){});
}
function occHeatColor(p){
  if(p<0.15) return '#1e293b';
  if(p<0.4)  return '#0e7490';
  if(p<0.7)  return '#0ea5e9';
  return '#38bdf8';
}
function occLoadHeatmap(){
  fetch('/api/occupancy/schedule').then(r=>r.json()).then(d=>{
    var wk=d.weekly||{}; var DN=['M','T','W','T','F','S','S']; var html='';
    for(var wd=0; wd<7; wd++){
      var hrs=wk[wd]||[];
      html+='<div style="display:flex;align-items:center;gap:3px;margin-bottom:2px;">';
      html+='<span style="font-size:8px;color:#4a6070;width:10px;">'+DN[wd]+'</span>';
      for(var h=0; h<24; h++) html+='<span style="display:inline-block;width:9px;height:9px;border-radius:2px;background:'+occHeatColor(hrs[h]||0)+';"></span>';
      html+='</div>';
    }
    var hm=document.getElementById('occ-heatmap'); if(hm) hm.innerHTML=html;
    var sm=document.getElementById('occ-samples'); if(sm) sm.textContent=(d.samples||0)+' samples';
  }).catch(function(){});
}
function occApplySetback(){
  var res=document.getElementById('occ-apply-result');
  if(res){res.style.color='#888';res.textContent='Applying…';}
  fetch('/read').then(r=>r.json()).then(d=>{
    var n=0;
    (d.controllers||[]).forEach(function(c){
      if(!c.connected) return;
      var sp=parseFloat(c.data['4']); var mode=c.data['1'];
      if(isNaN(sp)) return;
      var nv=(mode==1)?sp-3:sp+3;   // heat→lower, cool→raise
      fetch('/write',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({ctrl:c.idx,register:4,value:Math.round(nv)})});
      n++;
    });
    if(res){res.style.color='#4ade80';res.textContent='✓ Setback applied to '+n+' zone(s)';}
    setTimeout(occLoadSuggestion,600);
  }).catch(function(){ if(res){res.style.color='#f87171';res.textContent='✗ Failed';} });
}
function occToggleOverride(){
  fetch('/api/occupancy/override',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({active:true,minutes:60})}).then(r=>r.json()).then(function(){ occLoadSuggestion(); }).catch(function(){});
}

function buildConfigRows(cfg) {
  // Keep the advanced dropdown in sync with the wizard's type list, and always
  // include whatever type the controller already has so it never silently resets.
  var BASE_TYPES = (window.CTRL_TYPES || ['Temperzone EcoNEX PRO','Temperzone UC8','Vector','Daikin']);
  var html = '';
  cfg.forEach(function(ctrl, idx) {
    var isEnabled = ctrl.enabled;
    var typeList = BASE_TYPES.slice();
    if(ctrl.type && typeList.indexOf(ctrl.type) < 0) typeList.unshift(ctrl.type);
    var typeOpts = typeList.map(function(t){
      return '<option'+(ctrl.type===t?' selected':'')+'>'+t+'</option>';
    }).join('');

    html += '<div class="cfg-acc" id="cfg-row-'+idx+'">';

    // ── Accordion header ──
    html += '<div class="cfg-acc-hdr" onclick="toggleCfgRow('+idx+')">';
    html +=   '<div class="cfg-acc-left">';
    html +=     '<span class="cfg-acc-dot" id="cfg-dot-'+idx+'"></span>';
    html +=     '<span class="cfg-acc-name" id="cfg-acc-name-'+idx+'">'+ctrl.name+'</span>';
    html +=     '<span class="cfg-acc-type" id="cfg-acc-type-'+idx+'">'+ctrl.type+' · ID '+ctrl.slave_id+'</span>';
    html +=   '</div>';
    html +=   '<div class="cfg-acc-right">';
    html +=     '<label class="cfg-acc-toggle" onclick="event.stopPropagation();">';
    html +=       '<input type="checkbox" id="en-'+idx+'" '+(isEnabled?'checked':'')+' onchange="toggleCfgEnabled('+idx+')">';
    html +=       '<span>On</span>';
    html +=     '</label>';
    html +=     '<span class="cfg-acc-chev" id="cfg-chev-'+idx+'">▶</span>';
    html +=   '</div>';
    html += '</div>';

    // ── Accordion body ──
    html += '<div class="cfg-acc-body" id="cfg-row-body-'+idx+'">';

    // Row: name
    html += '<div class="cfg-field"><label>Room / Zone Name</label><input id="name-'+idx+'" type="text" value="'+ctrl.name+'" placeholder="e.g. Meeting Room" readonly onclick="vkOpen(this,\'Room / Zone Name\')" onchange="saveCtrlField('+idx+')"/></div>';

    // Row: type + slave id
    html += '<div class="cfg-grid2">';
    html +=   '<div class="cfg-field"><label>Controller Type</label><select id="type-'+idx+'" onchange="saveCtrlField('+idx+')">'+typeOpts+'</select></div>';
    html +=   '<div class="cfg-field"><label>Modbus Slave ID</label><input id="sid-'+idx+'" type="number" min="1" max="247" value="'+ctrl.slave_id+'" readonly onclick="vkOpen(this,\'Modbus Slave ID\')" onchange="saveCtrlField('+idx+')"/></div>';
    html += '</div>';

    // Row: LED assignment
    html += '<div class="cfg-inline-row">';
    html +=   '<label class="cfg-inline-label">💡 LED Indicator</label>';
    html +=   '<select class="led-cfg-sel" id="led-assign-'+idx+'" onchange="applyLedAssign('+idx+')">';
    html +=     '<option value="off">Off</option>';
    html +=     '<option value="temp" selected>Temp Control</option>';
    html +=     '<option value="iaq">IAQ / CO₂</option>';
    html +=   '</select>';
    html +=   '<button class="cfg-mini-btn" id="led-assign-btn-'+idx+'" onclick="setLedToCtrl('+idx+')">Apply</button>';
    html += '</div>';

    // Row: economy
    html += '<div class="cfg-inline-row">';
    html +=   '<label class="cfg-inline-label">🍃 Economy Cycle</label>';
    html +=   '<button class="toggle-btn off" id="economy-btn-'+idx+'" onclick="toggleEconomy('+idx+')" style="margin-left:auto;">Disable</button>';
    html += '</div>';

    // Row: CO₂ sensor card
    html += '<div class="cfg-inline-row">';
    html +=   '<label class="cfg-inline-label">🌿 CO₂ Card</label>';
    html +=   '<button class="toggle-btn on" id="co2-btn-'+idx+'" onclick="toggleCo2('+idx+')" style="margin-left:auto;">Enabled</button>';
    html += '</div>';

    // Row: actions (test + delete)
    html += '<div class="cfg-acc-actions">';
    html +=   '<span class="test-result" id="test-result-'+idx+'"></span>';
    html +=   '<button class="test-btn" onclick="testConnection('+idx+')">Test Connection</button>';
    if (cfg.length > 1) {
      html += '<button class="cfg-del-btn" onclick="showConfirmDelete('+idx+')">🗑</button>';
    }
    html += '</div>';

    html += '</div>'; // body
    html += '</div>'; // cfg-acc
  });
  document.getElementById('ctrl-cfg-list').innerHTML = html;
  loadEconomy();
  loadCo2();
  cfg.forEach(function(ctrl, idx) { toggleCfgEnabled(idx); });
  if(typeof attachVKToInputs==='function') attachVKToInputs();
}

function toggleCfgRow(idx) {
  var body = document.getElementById('cfg-row-body-'+idx);
  var chev = document.getElementById('cfg-chev-'+idx);
  var acc  = document.getElementById('cfg-row-'+idx);
  if (!body) return;
  var open = !body.classList.contains('open');
  body.classList.toggle('open', open);
  if (chev) chev.textContent = open ? '▼' : '▶';
  if (acc)  acc.classList.toggle('expanded', open);
}

function toggleCfgEnabled(idx) {
  var cb = document.getElementById('en-'+idx);
  if(!cb) return;
  var en = cb.checked;
  var body = document.getElementById('cfg-row-body-'+idx);
  if (body) {
    var inputs = body.querySelectorAll('input:not([type=checkbox]),select,.test-btn,.cfg-mini-btn');
    inputs.forEach(function(el){ el.disabled=!en; el.style.opacity=en?'1':'0.4'; });
  }
  var dot = document.getElementById('cfg-dot-'+idx);
  if(dot) dot.className = 'cfg-acc-dot '+(en?'on':'off');
}



async function addCtrl() {
  if (deviceConfig.length >= MAX_CONTROLLERS) {
    document.getElementById('cfg-save-result').textContent = '✗ Maximum ' + MAX_CONTROLLERS + ' controllers allowed. Delete one first.';
    return;
  }
  const r = await fetch('/api/controller/add', {method:'POST'});
  const d = await r.json();
  if (d.status === 'added') {
    await loadConfigIntoForm();
    document.getElementById('cfg-save-result').textContent = '✓ Controller added — configure and test.';
  } else {
    document.getElementById('cfg-save-result').textContent = '✗ ' + (d.msg || 'Add failed');
  }
  updateAddBtn();
}

var _deletePendingIdx = -1;

async function deleteCtrl(idx) {
  showConfirmDelete(idx);
}

async function saveCtrlField(idx){
  // read the current input values for this controller
  var name = (document.getElementById('name-'+idx)||{}).value || deviceConfig[idx].name;
  var type = (document.getElementById('type-'+idx)||{}).value || deviceConfig[idx].type;
  var sid  = parseInt((document.getElementById('sid-'+idx)||{}).value) || deviceConfig[idx].slave_id;
  // update local config
  deviceConfig[idx].name     = name;
  deviceConfig[idx].type     = type;
  deviceConfig[idx].slave_id = sid;
  // persist to backend
  await fetch('/api/device_config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(deviceConfig)});
  // update the config accordion header bar (name + Type · ID) without collapsing the open row
  var accName=document.getElementById('cfg-acc-name-'+idx);
  if(accName) accName.textContent=name;
  var accType=document.getElementById('cfg-acc-type-'+idx);
  if(accType) accType.textContent=type+' · ID '+sid;
  // rebuild dashboard so the new ID/name/type shows immediately
  buildDashboard();
  if(typeof buildGrid==='function') buildGrid();
  buildControlTab();
  if(typeof buildSchedCtrlBar==='function') buildSchedCtrlBar();
  if(typeof loadSchedule==='function') loadSchedule();
  var el=document.getElementById('cfg-save-result');
  if(el){ el.textContent='✓ Saved'; setTimeout(function(){el.textContent='';},1500); }
}

function showConfirmDelete(idx){
  _deletePendingIdx = idx;
  var name = (deviceConfig[idx]||{}).name || ('Controller '+(idx+1));
  var msg = document.getElementById('confirm-delete-msg');
  if(msg) msg.textContent = 'Delete "'+name+'"? Its schedule data will be removed.';
  // Close VK if open
  vkClose();
  document.getElementById('confirm-delete-modal').style.display='flex';
}

function hideConfirmDelete(){
  document.getElementById('confirm-delete-modal').style.display='none';
  _deletePendingIdx = -1;
}

async function executeDeleteCtrl(){
  var idx = _deletePendingIdx;
  hideConfirmDelete();
  if(idx < 0) return;
  const r = await fetch('/api/controller/delete/'+idx, {method:'POST'});
  const d = await r.json();
  if (d.status === 'deleted') {
    var nc={};
    Object.keys(ctrlCollapsed).forEach(function(k){
      var ki=parseInt(k);
      if(ki<idx) nc[ki]=ctrlCollapsed[k];
      else if(ki>idx) nc[ki-1]=ctrlCollapsed[k];
    });
    ctrlCollapsed=nc;
    if(activeSchedCtrl!=='ALL'){
      if(activeSchedCtrl===idx) activeSchedCtrl=0;
      else if(activeSchedCtrl>idx) activeSchedCtrl--;
    }
    knobs={};
    await loadConfigIntoForm();
    var el=document.getElementById('cfg-save-result');
    if(el) el.textContent='Controller deleted.';
    buildAll();
  } else {
    var el=document.getElementById('cfg-save-result');
    if(el) el.textContent='✗ '+(d.msg||'Delete failed');
  }
}

function updateDynamicSelects() {
  var html = '';
  deviceConfig.forEach(function(ctrl, idx) {
    html += '<option value="'+idx+'">'+ctrl.name+'</option>';
  });
  var wc  = document.getElementById('write-ctrl');
  var ic  = document.getElementById('inputs-ctrl-sel');
  var tps = document.getElementById('trend-primary-sel');
  if (wc)  wc.innerHTML  = html;
  if (ic)  ic.innerHTML  = html;
  if (tps){ tps.innerHTML = html; tps.value = trendPrimaryIdx; }
  buildWriteCtrlBtns();
}

async function testConnection(idx) {
  var res = document.getElementById('test-result-'+idx);
  var sid = parseInt(document.getElementById('sid-'+idx).value);
  res.className='test-result testing'; res.textContent='Testing slave '+sid+'...';
  var tmpCfg = deviceConfig.map(function(c,i){
    return i===idx ? Object.assign({},c,{slave_id:sid,enabled:true}) : Object.assign({},c);
  });
  await fetch('/api/device_config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(tmpCfg)});
  const r = await fetch('/api/test_connection',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ctrl:idx})});
  const d = await r.json();
  if(d.connected){ res.className='test-result ok';  res.textContent='✓ '+d.msg; }
  else            { res.className='test-result err'; res.textContent='✗ '+d.msg; }
}

async function saveSerialConfig() {
  var port = document.getElementById('cfg-port').value;
  var baud = parseInt(document.getElementById('cfg-baud').value);
  const r = await fetch('/api/serial_config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({port,baud})});
  const d = await r.json();
  var el = document.getElementById('serial-result');
  el.textContent = d.status==='ok' ? '✓ Applied' : '✗ '+d.msg;
  setTimeout(function(){ el.textContent=''; }, 3000);
}

async function saveAndConnect() {
  // Merge form values onto the existing config so no per-controller field is
  // lost (keeps wizard/advanced in sync). Rows always match deviceConfig 1:1
  // because add/delete go through the server + reload.
  var newCfg = deviceConfig.map(function(c, i){
    return Object.assign({}, c, {
      name:     (document.getElementById('name-'+i)||{}).value || c.name,
      type:     (document.getElementById('type-'+i)||{}).value || c.type,
      slave_id: parseInt((document.getElementById('sid-'+i)||{}).value) || c.slave_id,
      enabled:  (document.getElementById('en-'+i)||{}).checked || false,
    });
  });
  await fetch('/api/device_config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(newCfg)});
  deviceConfig = newCfg;
  document.getElementById('cfg-save-result').textContent = 'Saved — connecting...';
  var anyConnected = false;
  for (var i = 0; i < newCfg.length; i++) {
    if (!newCfg[i].enabled) continue;
    var res = document.getElementById('test-result-'+i);
    res.className='test-result testing'; res.textContent='Connecting...';
    const r = await fetch('/api/test_connection',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ctrl:i})});
    const d = await r.json();
    if(d.connected){ res.className='test-result ok'; res.textContent='✓ '+d.msg; anyConnected=true; }
    else            { res.className='test-result err'; res.textContent='✗ '+d.msg; }
  }
  if (anyConnected) {
    document.getElementById('cfg-save-result').textContent = '';
    document.getElementById('unlock-banner').style.display = 'block';
    unlockTabs(); buildAll();
    setTimeout(function(){ showTab('dashboard'); }, 400);
  } else {
    document.getElementById('cfg-save-result').textContent = 'No controllers responded. Check slave IDs and wiring.';
  }
}

function enterDashboard() { unlockTabs(); buildAll(); showTab('dashboard'); }

function buildAll() {
  buildDashboard(); buildGrid(); buildControlTab(); buildSchedCtrlBar(); initAllKnobs(); loadSchedule(); updateDynamicSelects(); loadEconomy(); loadCo2(); updateOutsideTempTiles();
  deviceConfig.forEach(function(ctrl, idx) {
    ctrlCollapsed[idx] = true;
    applyCtrlCollapseState(idx);
  });
  setTimeout(attachVKToInputs, 200);
}

var UNLOCKABLE_TABS = ['dashboard','control','info','insight_ai','schedule'];
function unlockTabs(){
  tabsUnlocked = true;
  UNLOCKABLE_TABS.forEach(function(t){
    var btn = document.getElementById('tab-btn-'+t);
    if(btn) btn.classList.remove('locked');
  });
  applyTechMode();   // keep Control tab hidden unless technician mode is on
}
function tryShowTab(name){
  if(!tabsUnlocked && name!=='config'){
    document.getElementById('cfg-save-result').textContent='Connect a controller first.';
    showTab('config'); return;
  }
  showTab(name);
}
function showTab(name){
  document.querySelectorAll('.page').forEach(function(p){p.classList.remove('active');});
  document.querySelectorAll('.tab').forEach(function(t){t.classList.remove('active');});
  var page = document.getElementById('tab-'+name);
  var btn  = document.getElementById('tab-btn-'+name);
  if(page) page.classList.add('active');
  if(btn)  btn.classList.add('active');
  if(name==='insight_ai'){ refreshInputsTable(); occStart(); }
  if(name==='schedule'){ buildSchedCtrlBar(); loadSchedule(); }
}

// ── Dashboard builder ─────────────────────────────────────────────────────
function buildDashboard(){
  var html='';
  deviceConfig.forEach(function(ctrl,idx){
    var dis=!ctrl.enabled;
    html+='<div class="ctrl-wrap" id="ctrl-wrap-'+idx+'" style="'+(dis?'opacity:0.35;pointer-events:none;':'')+'">';
    html+='<div class="ctrl-bg-overlay" id="ctrl-bg-'+idx+'"></div>';
    html+='<div class="wind-bg" id="wind-bg-'+idx+'"></div>';
    var isUnconfigured = /^Controller \d+$/.test(ctrl.name);
    html+='<div class="ctrl-header" id="ctrl-header-'+idx+'" onclick="toggleCtrlCollapse('+idx+')">';
    html+='<span style="display:flex;align-items:center;font-weight:bold;font-size:13px;"><span class="ctrl-chev">▼</span>'+ctrl.name+(isUnconfigured?' <span id="ctrl-setup-badge-'+idx+'" style="font-size:9px;font-weight:600;background:#f59e0b;color:#fff;padding:1px 7px;border-radius:8px;margin-left:6px;">⚙ Not set up</span>':'')+'</span>';
    html+='<div style="display:flex;align-items:center;gap:8px;" onclick="event.stopPropagation()">';
    html+='<span class="ctrl-badge dis" id="ctrl-badge-'+idx+'">● --</span>';
    html+='<span class="ctrl-badge fault" id="ctrl-fault-'+idx+'" style="display:none;">● FAULT</span>';
    html+='<button class="onoff-btn off" id="ctrl-onoff-'+idx+'" onclick="toggleCtrlOnOff('+idx+')">--</button>';
    html+='</div></div>';
    html+='<div class="ctrl-body" id="ctrl-body-'+idx+'" style="border:none;border-radius:0 0 12px 12px;padding:10px;background:#fff;">';
    if(isUnconfigured){
      html+='<div id="ctrl-setup-prompt-'+idx+'" onclick="showTab(\'config\')" style="background:#fffbeb;border:1px dashed #f59e0b;border-radius:8px;padding:10px 12px;margin-bottom:6px;display:flex;align-items:center;gap:10px;cursor:pointer;">';
      html+='<span style="font-size:20px;">⚙️</span>';
      html+='<div style="flex:1;"><div style="font-size:12px;font-weight:bold;color:#92400e;">Tap to configure this controller</div><div style="font-size:10px;color:#b45309;margin-top:1px;">Set its name, type and Modbus ID in Config</div></div>';
      html+='<span style="font-size:16px;color:#f59e0b;">→</span>';
      html+='</div>';
    }
    html+='<div class="cards" style="margin:0;">'
    html+='<div class="card" style="grid-column:span 2;padding:6px 4px 4px;background:transparent!important;border:none!important;box-shadow:none!important;">';
    // ── Three tiles: Room Temp | Setpoint Knob | CO Level ──
    html+='<div class="metric-row" id="metric-row-'+idx+'">';

    // Tile 1 — Room Temperature
    html+='<div class="metric-tile tile-temp" id="temp-tile-'+idx+'">';
    html+='<span class="eco-leaf" id="eco-leaf-'+idx+'" title="Economy cycle">🍃</span>';
    html+='<div>';
    html+='<div class="tile-icon"><img src="/home/linaro/icons/temperature-blue-128x128.png"></div>';
    html+='<div class="tile-label">Room Temp</div>';
    html+='<div class="tile-value" id="temp-'+idx+'">--<span style="font-size:14px;font-weight:700;">°C</span></div>';
    html+='<div class="tile-outside-label" style="margin-top:6px;">Outside Temp</div>';
    html+='<div class="tile-outside-value" id="outsideTemp-'+idx+'">--°C</div>';
    html+='</div>';
    html+='<div>';
    html+='<div class="tile-bar-wrap"><div id="tempBar-'+idx+'" class="tile-bar-fill" style="width:50%;background:#ef4444;"></div></div>';
    html+='<div style="display:flex;align-items:center;justify-content:space-between;gap:6px;">';
    html+='<div class="tile-status" id="temp-status-'+idx+'">Reading...</div>';
    html+='<div class="tile-status" id="eco-text-'+idx+'" style="color:#16a34a;display:none;">✓ Economy</div>';    html+='</div>';
    html+='</div>';
    html+='</div>';

    // Tile 2 — Setpoint Knob (arrows flank the dial so the knob can be larger)
    html+='<div class="knob-cell">';
    html+='<div class="knob-label"><img src="/home/linaro/icons/setpoint-blue-128x128.png" style="width:12px;height:12px;">Setpoint</div>';
    html+='<canvas id="sp-knob-'+idx+'" width="110" height="110" style="cursor:grab;touch-action:none;"></canvas>';
    html+='<div class="knob-row">';
    html+='<button class="sp-step-btn sp-step-dn" onclick="adjustSP('+idx+',-1)" title="-1°C">▼</button>';
    html+='<button class="sp-step-btn sp-step-up" onclick="adjustSP('+idx+',+1)" title="+1°C">▲</button>';
    html+='</div>';
    html+='<div id="sp-result-'+idx+'" class="sp-result"></div>';
    html+='</div>';

    // Tile 3 — CO Level
    html+='<div class="metric-tile tile-co" id="co-tile-'+idx+'">';
    html+='<div>';
    html+='<div class="tile-icon"><img src="/home/linaro/icons/co2-blue-128x128.png"></div>';
    html+='<div class="tile-label">CO₂ Level</div>';
    html+='<div class="tile-value" id="coValue-'+idx+'">--<span style="font-size:13px;font-weight:700;"> ppm</span></div>';
    html+='</div>';
    html+='<div>';
    html+='<div class="co-scale-wrap"><div id="coBar-'+idx+'" class="co-scale-marker" style="left:0%;"></div></div>';
    html+='<div class="co-scale-labels"><span>Good</span><span>Moderate</span><span>Poor</span></div>';
    html+='<div class="tile-status" id="co-status-'+idx+'">Normal</div>';
    html+='</div>';
    html+='</div>';

    html+='</div>'; // close metric-row
    // Fan Speed
    html+='<div style="margin-top:2px;padding-top:2px;border-top:1px solid #f0f0f0;">';
    html+='<div class="card-label fan-label" style="margin-bottom:6px;font-weight:700;display:flex;align-items:center;gap:6px;">';
    html+='<img class="ic" src="/home/linaro/icons/modefan-blue-128x128.png"><strong>Fan Speed</strong></div>';
    html+='<div style="display:flex;gap:6px;">';
    html+='<button class="fan-btn" id="fan-1-'+idx+'" onclick="setFanSpeed('+idx+',1)">Low</button>';
    html+='<button class="fan-btn" id="fan-2-'+idx+'" onclick="setFanSpeed('+idx+',2)">Medium</button>';
    html+='<button class="fan-btn" id="fan-3-'+idx+'" onclick="setFanSpeed('+idx+',3)">High</button>';
    html+='</div></div>';
    // Mode
    html+='<div style="margin-top:6px;padding-top:6px;border-top:1px solid #f0f0f0;">';
    html+='<div class="card-label" style="margin-bottom:6px;font-weight:700;display:flex;align-items:center;gap:6px;"><img class="ic" src="/home/linaro/icons/temperature-blue-128x128.png"><strong>Mode</strong></div>';
    // Temperzone unit mode (reg 117): 0=Auto 1=Cool 2=Heat 3=Fan
    html+='<div style="display:flex;gap:6px;">';
    html+='<button class="mode-sel-btn auto" id="mode-auto-'+idx+'" onclick="setModeSelection('+idx+',0)">'+ic('modeauto','mode-ic')+'Auto</button>';
    html+='<button class="mode-sel-btn cool" id="mode-cool-'+idx+'" onclick="setModeSelection('+idx+',1)">'+ic('modecool','mode-ic')+'Cool</button>';
    html+='<button class="mode-sel-btn heat" id="mode-heat-'+idx+'" onclick="setModeSelection('+idx+',2)">'+ic('modeheat','mode-ic')+'Heat</button>';
    html+='<button class="mode-sel-btn vent" id="mode-vent-'+idx+'" onclick="setModeSelection('+idx+',3)">'+ic('modefan','mode-ic')+'Fan</button>';
    html+='</div></div>';
    html+='</div></div></div></div>'; // close card/cards/ctrl-body/ctrl-wrap
  });
  document.getElementById('ctrl-sections').innerHTML=html;
  Object.keys(ctrlCollapsed).forEach(function(k){
    if(ctrlCollapsed[k]) applyCtrlCollapseState(parseInt(k));
  });
  deviceConfig.forEach(function(ctrl,idx){ if(typeof applyCo2Vis==='function') applyCo2Vis(idx); });
}


function buildWeekTimeline(data){
  var grid=document.getElementById('week-grid');
  if(!grid) return;
  var html='<div class="wtl">';
  // hour ruler
  html+='<div class="wtl-row wtl-ruler"><div class="wtl-daylbl"></div><div class="wtl-track">';
  [0,6,12,18,24].forEach(function(h){
    html+='<span class="wtl-tick" style="left:'+(h/24*100)+'%;">'+fmtHour12(h===24?0:h)+'</span>';
  });
  html+='</div></div>';

  DAYS_SHORT.forEach(function(d){
    var events=(data[d]||[]).slice().sort(function(a,b){return a.time.localeCompare(b.time);});
    html+='<div class="wtl-row"><div class="wtl-daylbl">'+d+'</div>';
    html+='<div class="wtl-track" onclick="wtlTrackTap(\''+d+'\',event)">';
    // shade ON→OFF spans
    var openMin=null;
    events.forEach(function(e){
      var mins=parseInt(e.time.split(':')[0])*60+parseInt(e.time.split(':')[1]);
      if(e.action===1){ openMin=mins; }
      else if(e.action===0 && openMin!=null){
        var l=openMin/1440*100, w=(mins-openMin)/1440*100;
        html+='<div class="wtl-span" style="left:'+l+'%;width:'+w+'%;"></div>';
        openMin=null;
      }
    });
    if(openMin!=null){ var l=openMin/1440*100; html+='<div class="wtl-span" style="left:'+l+'%;right:0;"></div>'; }
    // event markers
    events.forEach(function(e,i){
      var mins=parseInt(e.time.split(':')[0])*60+parseInt(e.time.split(':')[1]);
      var pos=mins/1440*100;
      html+='<div class="wtl-mark '+(e.action?'on':'off')+'" style="left:'+pos+'%;" title="'+e.time+'" onclick="event.stopPropagation();wtlDelete(\''+d+'\','+i+')"></div>';
    });
    html+='</div></div>';
  });
  html+='</div>';
  grid.innerHTML=html;
}
function wtlTrackTap(day,e){
  var track=e.currentTarget, rect=track.getBoundingClientRect();
  var pct=(e.clientX-rect.left)/rect.width;
  var mins=Math.round(pct*1440/30)*30;   // snap to 30 min
  var h=Math.floor(mins/60), m=mins%60;
  schedSelectedDays=new Set([day]);
  schedTime=String(h).padStart(2,'0')+':'+String(m).padStart(2,'0');
  document.getElementById('time-display').textContent=fmt12(schedTime);
  setSchedAmPm(h<12?'AM':'PM');
  buildSchedDayBtns(); updateSchedPreview();
  document.getElementById('sched-preview').scrollIntoView({behavior:'smooth',block:'center'});
}
function wtlDelete(day,i){
  var displayIdx=activeSchedCtrl==='ALL'?0:activeSchedCtrl;
  fetch('/api/schedule?ctrl='+displayIdx).then(r=>r.json()).then(data=>{
    var ev=(data[day]||[]).slice().sort(function(a,b){return a.time.localeCompare(b.time);})[i];
    if(ev && confirm('Delete '+day+' '+fmt12(ev.time)+' → '+(ev.action?'ON':'OFF')+'?')){
      data[day]=(data[day]||[]).filter(function(x){return x!==ev;});
      fetch('/api/schedule?ctrl='+displayIdx,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}).then(loadSchedule);
    }
  });
}

function toggleCtrlCollapse(idx){
  ctrlCollapsed[idx] = !ctrlCollapsed[idx];
  applyCtrlCollapseState(idx);
}
function applyCtrlCollapseState(idx){
  var hdr  = document.getElementById('ctrl-header-'+idx);
  var body = document.getElementById('ctrl-body-'+idx);
  if(hdr)  hdr.classList.toggle('collapsed',  !!ctrlCollapsed[idx]);
  if(body) body.classList.toggle('collapsed', !!ctrlCollapsed[idx]);
}
function setModeSelection(idx, value){
  fetch('/write',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ctrl:idx, register:1, value:value})
  }).then(r=>r.json()).then(()=>{ setTimeout(poll,300); });
}
var FAN_RAW = {1:300, 2:600, 3:900};   // Low=30%, Med=60%, High=90% (x0.1)
function setFanSpeed(idx,value){
  // optimistic highlight
  [1,2,3].forEach(function(fs){
    var fb=document.getElementById('fan-'+fs+'-'+idx);
    if(fb) fb.classList.toggle('active', fs===value);
  });
  var raw = FAN_RAW[value] || 0;     // write the raw 300/600/900 to register 114
  fetch('/write',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ctrl:idx,register:2,value:raw})
  }).then(r=>r.json()).then(()=>{ setTimeout(poll,300); });
}

function toggleControlSection(idx){
  var body = document.getElementById('ctrl-ctrl-body-'+idx);
  var chev = document.getElementById('ctrl-chev-c-'+idx);
  if (!body) return;
  var open = body.style.display === 'none';
  body.style.display = open ? 'block' : 'none';
  if (chev) chev.textContent = open ? '▼' : '▶';
}

function buildControlTab(){
  var html='';
  deviceConfig.forEach(function(ctrl,idx){
    html += '<div style="margin-bottom:16px;border-radius:10px;overflow:hidden;box-shadow:0 1px 4px rgba(0,0,0,0.08);">';
    html += '<div style="background:#1e3a5f;color:white;padding:10px 14px;display:flex;justify-content:space-between;align-items:center;cursor:pointer;" onclick="toggleControlSection('+idx+')">';
    html +=   '<div><div style="font-weight:bold;font-size:13px;">'+ctrl.name+'</div>';
    html +=   '<div style="font-size:10px;opacity:0.65;margin-top:1px;">'+ctrl.type+' </div></div>';
    html +=   '<span id="ctrl-chev-c-'+idx+'" style="font-size:12px;">▶</span></div>';
    html += '<div id="ctrl-ctrl-body-'+idx+'" style="display:none;background:white;border:1px solid #e5e7eb;border-top:none;border-radius:0 0 10px 10px;">';
    var REGS = [
      [0,'ON / OFF', ic('control','mode-ic')],
      [117,'Heat / Cool', ic('temperature','mode-ic')],
      [116,'Fan', ic('modefan','mode-ic')],
      [22,'Time Schedules', ic('scheduleclock','mode-ic')],
    ];
        REGS.forEach(function(r, ri){
      var isLast = ri === REGS.length - 1;
      html += '<div style="display:flex;align-items:center;justify-content:space-between;padding:11px 14px;'+(isLast?'':'border-bottom:1px solid #f3f4f6;')+'">';
      html +=   '<div style="display:flex;align-items:center;gap:9px;">';
      html +=     '<span style="width:22px;display:inline-flex;align-items:center;justify-content:center;">'+r[2]+'</span>';
      html +=     '<div><div style="font-size:13px;color:#222;font-weight:500;">'+r[1]+'</div><div style="font-size:10px;color:#aaa;">Reg '+r[0]+'</div></div></div>';
      html +=   '<button class="toggle-btn off" id="cbtn-'+idx+'-'+r[0]+'" onclick="toggleCtrlReg('+idx+','+r[0]+')">--</button></div>';
    });
    html += '</div></div>';
  });
  document.getElementById('control-sections').innerHTML = html;
}

// ── Unified accordion toggle ───────────────────────────────────────────────
function toggleWP(bodyId, chevId){
  var body=document.getElementById(bodyId);
  var chev=document.getElementById(chevId);
  if(!body) return;
  var open=body.style.display==='none'||body.style.display==='';
  body.style.display=open?'block':'none';
  if(chev) chev.textContent=open?'▼':'▶';
}
// Legacy alias kept for any remaining calls
function togglePanel(id){
  var body=document.getElementById(id);
  if(!body) return;
  var open=body.style.display==='none'||body.style.display==='';
  body.style.display=open?'block':'none';
}

// ── Knobs ─────────────────────────────────────────────────────────────────
function initAllKnobs(){
  knobs = {};
  deviceConfig.forEach(function(ctrl,idx){ if(ctrl.enabled) initKnob(idx); });
}

function initKnob(idx){
  var canvas=document.getElementById('sp-knob-'+idx);
  if(!canvas) return;
  var ctx=canvas.getContext('2d');
  var W=140, CX=70, CY=70, R=60;
  // High-DPI backing store so the enlarged dial stays crisp
  var dpr=Math.max(1,Math.min(3,window.devicePixelRatio||1));
  canvas.width=W*dpr; canvas.height=W*dpr;
  ctx.setTransform(dpr,0,0,dpr,0,0);
  var spVal=23;
  var dragging=false, lastAng=null;
  var writeTimer=null;

  function draw(angleDeg){
    var dark=document.body.classList.contains('theme-dark');
    var blue=document.body.classList.contains('theme-blue');
    ctx.clearRect(0,0,W,W);
    angleDeg=clamp(angleDeg,MIN_DEG,MAX_DEG);
    var sr=(MIN_DEG-90)*Math.PI/180;
    var er=(MAX_DEG-90)*Math.PI/180;
    var cr=(angleDeg-90)*Math.PI/180;
    var mr=(c2a((MIN_C+MAX_C)/2)-90)*Math.PI/180;

    

    // Track background
    ctx.beginPath();ctx.arc(CX,CY,R-6,sr,er);
    ctx.strokeStyle=dark?'#334155':blue?'#bfdbfe':'#e2e8f0';
    ctx.lineWidth=13;ctx.lineCap='round';ctx.stroke();
    // Cool (blue) arc
    if(cr>sr){
      ctx.beginPath();ctx.arc(CX,CY,R-6,sr,Math.min(cr,mr));
      ctx.strokeStyle='#3b82f6';ctx.lineWidth=13;ctx.lineCap='round';ctx.stroke();
    }
    // Warm (red) arc
    if(cr>mr){
      ctx.beginPath();ctx.arc(CX,CY,R-6,mr,cr);
      ctx.strokeStyle='#ef4444';ctx.lineWidth=13;ctx.lineCap='round';ctx.stroke();
    }

    // Inner face
    ctx.beginPath();ctx.arc(CX,CY,R-20,0,Math.PI*2);
    var grad=ctx.createRadialGradient(CX-8,CY-8,3,CX,CY,R-20);
    if(dark){ grad.addColorStop(0,'rgba(30,41,59,0.95)'); grad.addColorStop(1,'rgba(8,8,12,0.98)'); }
    else if(blue){ grad.addColorStop(0,'rgba(255,255,255,0.98)'); grad.addColorStop(1,'rgba(219,234,254,0.95)'); }
    else{ grad.addColorStop(0,'rgba(255,255,255,0.98)'); grad.addColorStop(1,'rgba(241,245,249,0.95)'); }
    ctx.fillStyle=grad; ctx.fill();
    // hairline ring around face
    ctx.beginPath();ctx.arc(CX,CY,R-20,0,Math.PI*2);
    ctx.strokeStyle=dark?'rgba(148,163,184,0.15)':'rgba(0,0,0,0.06)';ctx.lineWidth=1;ctx.stroke();

    // Thumb dot
    var dx=CX+(R-6)*Math.cos(cr), dy=CY+(R-6)*Math.sin(cr);
    ctx.beginPath();ctx.arc(dx,dy,7,0,Math.PI*2);
    ctx.fillStyle=dark?'#f8fafc':'#1e3a5f';ctx.fill();
    ctx.beginPath();ctx.arc(dx,dy,4,0,Math.PI*2);
    ctx.fillStyle=dark?'#3b82f6':'#ffffff';ctx.fill();

    // Big temperature number (centred — no label beneath)
    ctx.textAlign='center';ctx.textBaseline='middle';
    ctx.font='300 37px -apple-system,"Segoe UI",Arial';
    ctx.fillStyle=dark?'#ffffff':blue?'#1e3a5f':'#0f172a';
    ctx.fillText(spVal.toFixed(0),CX-5,CY+1);
    // degree mark
    ctx.font='300 15px Arial';
    ctx.fillStyle=dark?'#94a3b8':'#64748b';
    ctx.fillText('°C',CX+23,CY-8);
  }

  function getAng(e){
    var rect=canvas.getBoundingClientRect();
    var scaleX=W/rect.width, scaleY=W/rect.height;
    var cx=(e.touches?e.touches[0].clientX:e.clientX)-rect.left;
    var cy=(e.touches?e.touches[0].clientY:e.clientY)-rect.top;
    var x=cx*scaleX-CX, y=cy*scaleY-CY;
    var d=Math.atan2(y,x)*180/Math.PI+90;
    return d<0?d+360:d;
  }
  function onStart(e){
    e.preventDefault();e.stopPropagation();
    dragging=true;lastAng=getAng(e);
    canvas.style.cursor='grabbing';
  }
  function onMove(e){
    if(!dragging) return;
    e.preventDefault();
    var a=getAng(e), delta=a-lastAng;
    if(delta>180) delta-=360;
    if(delta<-180) delta+=360;
    var na=clamp(c2a(spVal)+delta,MIN_DEG,MAX_DEG);
    spVal=clamp(Math.round(a2c(na)),MIN_C,MAX_C);
    lastAng=a;
    spHold[idx]=spVal;
    lastSP[idx]=spVal;
    draw(c2a(spVal));
    var res=document.getElementById('sp-result-'+idx);   // live regulator readout
    if(res){ res.style.color='#888'; res.textContent=spVal.toFixed(0)+'°C'; }
    clearTimeout(writeTimer);
    writeTimer=setTimeout(function(){ writeSetpoint(idx,spVal); spHold[idx]=null; },400);
  }
  function onEnd(){
    if(!dragging) return;
    dragging=false;
    canvas.style.cursor='grab';
    clearTimeout(writeTimer);
    writeSetpoint(idx,spVal);
    spWriteTimers[idx]=setTimeout(function(){ spHold[idx]=null; },1500);  // release after write settles
  }
  canvas.addEventListener('mousedown',onStart);
  canvas.addEventListener('mousemove',onMove);
  canvas.addEventListener('mouseup',onEnd);
  canvas.addEventListener('mouseleave',onEnd);
  canvas.addEventListener('touchstart',onStart,{passive:false});
  canvas.addEventListener('touchmove',onMove,{passive:false});
  canvas.addEventListener('touchend',onEnd,{passive:false});
  knobs[idx]={
    update:function(raw){
      if(dragging || spHold[idx]!=null) return;   // ← don't snap back while held
      spVal=clamp(raw,MIN_C,MAX_C); draw(c2a(spVal));
    },
    redraw:function(){ draw(c2a(spVal)); }
  };
  draw(c2a(spVal));
}

function writeSetpoint(idx,raw){
  var res=document.getElementById('sp-result-'+idx);
  res.style.color='#888';res.textContent='Writing...';
  fetch('/write',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ctrl:idx,register:4,value:raw})
  }).then(r=>r.json()).then(d=>{
    if(d.status==='success'){res.style.color='#22c55e';res.textContent='✓ '+(raw).toFixed(1)+'°C';}
    else{res.style.color='#ef4444';res.textContent='✗ Failed';}
    setTimeout(()=>{res.textContent='';},3000);
  });
}

var lastSP = {};
var spWriteTimers = {};
var spHold = {};   // tracks the value while rapidly tapping

function adjustSP(idx, delta){
  // start from the value we're holding, or last known, or 23
  var base = spHold[idx] != null ? spHold[idx] : (lastSP[idx] != null ? lastSP[idx] : 23);
  var newVal = clamp(base + delta, MIN_C, MAX_C);
  spHold[idx] = newVal;
  lastSP[idx] = newVal;

  // instant visual feedback — update knob immediately
  if(knobs[idx]) knobs[idx].update(newVal);

  // show pending value
  var res = document.getElementById('sp-result-'+idx);
  if(res){ res.style.color='#888'; res.textContent = newVal.toFixed(1)+'°C'; }

  // debounce the actual write — only send 400ms after the last tap
  clearTimeout(spWriteTimers[idx]);
  spWriteTimers[idx] = setTimeout(function(){
    writeSetpoint(idx, newVal);
    setTimeout(function(){ spHold[idx] = null; }, 1600);   // keep hold until device reports back
  }, 400);
}

// ── Clock ─────────────────────────────────────────────────────────────────
function pollClockOnce(){
  return fetch('/api/clock').then(r=>r.json()).then(d=>{
    var el;
    if(el=document.getElementById('hdr-clock-time'))el.textContent=d.time;
    if(el=document.getElementById('hdr-clock-ampm'))el.textContent=d.ampm;
    if(el=document.getElementById('hdr-clock-date'))el.textContent=d.date;
  }).catch(()=>{});
}
function pollClock(){
  pollClockOnce().finally(()=>{setTimeout(pollClock,1000);});
}

// ── Clock source badge + Set Date & Time ───────────────────────────────────
var ST_MONTHS=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
var stState={year:2026,month:1,day:1,hour:0,minute:0};
function stDaysInMonth(y,m){ return new Date(y,m,0).getDate(); }   // m=1..12
function clkSrcLabel(s){
  return s==='ntp' ? {t:'NTP', full:'Network time (NTP) — synced & trustworthy'} :
         s==='rtc' ? {t:'RTC', full:'Battery clock (RTC) — holds time offline'} :
                     {t:'MANUAL', full:'Manual — no network/RTC, may drift'};
}
function loadClockSource(){
  fetch('/api/clock_source').then(r=>r.json()).then(function(d){
    var s=d.source||'manual', lab=clkSrcLabel(s);
    var b=document.getElementById('hdr-clock-src');
    if(b){ b.style.display='inline-block'; b.className='clk-badge '+s; b.textContent=lab.t; }
    // One-word location (city from the timezone) next to the title.
    var loc=document.getElementById('st-location');
    if(loc){ loc.textContent = d.location || ''; loc.title = d.tz || ''; }
    var src=document.getElementById('st-source');
    if(src){ src.className='st-source '+s; src.textContent=lab.full+(d.tz?(' · '+d.tz):''); }
  }).catch(function(){});
}
function showSetTime(){
  document.getElementById('settime-panel').style.display='flex';
  document.getElementById('st-msg').textContent='';
  stUseNow();
  loadClockSource();
  stLoadTimezones();
}
var ST_TZ_ALL=[];
function stLoadTimezones(){
  fetch('/api/timezones').then(r=>r.json()).then(function(d){
    ST_TZ_ALL=d.timezones||[];
    stRenderTz(ST_TZ_ALL, d.current);
  }).catch(function(){});
}
function stRenderTz(list, current){
  var sel=document.getElementById('st-tz-select'); if(!sel) return;
  var frag=document.createDocumentFragment();
  list.forEach(function(z){
    var o=document.createElement('option'); o.value=z.tz; o.textContent=z.label||z.tz;
    if(current && z.tz===current) o.selected=true;
    frag.appendChild(o);
  });
  sel.innerHTML=''; sel.appendChild(frag);
}
function stApplyTz(){
  var sel=document.getElementById('st-tz-select'); var msg=document.getElementById('st-msg');
  if(!sel||!sel.value){ return; }
  var label=sel.options[sel.selectedIndex] ? sel.options[sel.selectedIndex].text : sel.value;
  fetch('/api/set_timezone',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({tz:sel.value})}).then(r=>r.json()).then(function(d){
    if(d.ok){
      if(msg){ msg.textContent='Time zone set — '+label; msg.className='st-msg ok'; }
      loadClockSource();
      // Re-seed the spinner AND the header clock from the new zone right away
      // so the displayed time follows the choice without slipping.
      stUseNow();
      if(typeof pollClockOnce==='function') pollClockOnce();
    } else {
      if(msg){ msg.textContent=(d.error||'Could not set time zone'); msg.className='st-msg err'; }
    }
  }).catch(function(){ if(msg){ msg.textContent='Could not set time zone'; msg.className='st-msg err'; } });
}
function closeSetTime(){ document.getElementById('settime-panel').style.display='none'; }
function stUseNow(){
  fetch('/api/clock').then(r=>r.json()).then(function(d){
    if(d && d.y){ stState={year:d.y,month:d.mo,day:d.d,hour:d.h,minute:d.mi}; }
    else { var n=new Date(); stState={year:n.getFullYear(),month:n.getMonth()+1,day:n.getDate(),hour:n.getHours(),minute:n.getMinutes()}; }
    stRender();
  }).catch(function(){ var n=new Date(); stState={year:n.getFullYear(),month:n.getMonth()+1,day:n.getDate(),hour:n.getHours(),minute:n.getMinutes()}; stRender(); });
}
function stRender(){
  var p=function(n){return (n<10?'0':'')+n;};
  document.getElementById('st-day').textContent=p(stState.day);
  document.getElementById('st-month').textContent=ST_MONTHS[stState.month-1];
  document.getElementById('st-year').textContent=stState.year;
  document.getElementById('st-hour').textContent=p(stState.hour);
  document.getElementById('st-minute').textContent=p(stState.minute);
}
function stAdj(field,delta){
  var s=stState;
  if(field==='minute'){ s.minute=(s.minute+delta+60)%60; }
  else if(field==='hour'){ s.hour=(s.hour+delta+24)%24; }
  else if(field==='year'){ s.year=Math.min(2040,Math.max(2020,s.year+delta)); }
  else if(field==='month'){ s.month=((s.month-1+delta+12)%12)+1; }
  else if(field==='day'){
    var dim=stDaysInMonth(s.year,s.month);
    s.day=((s.day-1+delta+dim)%dim)+1;
  }
  // keep day valid for the selected month/year
  var dim=stDaysInMonth(s.year,s.month);
  if(s.day>dim) s.day=dim;
  stRender();
}
function setTimeApply(){
  var btn=document.getElementById('st-apply'), msg=document.getElementById('st-msg');
  btn.disabled=true; btn.textContent='Setting…'; msg.className='st-msg'; msg.textContent='';
  fetch('/api/set_time',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify(stState)}).then(function(r){return r.json();}).then(function(d){
      btn.disabled=false; btn.textContent='✓ Set Time';
      if(d.ok){
        msg.className='st-msg ok'; msg.textContent='✓ Clock updated';
        loadClockSource();
        setTimeout(closeSetTime, 1200);
      } else {
        msg.className='st-msg err'; msg.textContent='✗ '+(d.error||'Could not set time');
      }
    }).catch(function(){
      btn.disabled=false; btn.textContent='✓ Set Time';
      msg.className='st-msg err'; msg.textContent='✗ Request failed';
    });
}

function poll(){
  fetch('/read').then(r=>r.json()).then(d=>{updateAll(d);}).catch(()=>{}).finally(()=>{setTimeout(poll,2000);});
}

var UNIT_STATUS = {1:'Running',2:'Off — Alarm',3:'Off — BMS',4:'Off — Scheduler',
                   5:'Off — Digital Input',6:'Off — Local Keyboard',7:'Manual Control'};
function statusReason(data){
  var s = data['status_code'];
  var f = data['fault'];
  if(f==1 && (s==null||s==2)) return 'Alarm active';
  if(s!=null && UNIT_STATUS[s]) return UNIT_STATUS[s];
  return '';
}
function reasonClass(data){
  var s=data['status_code'], f=data['fault'];
  if(f==1||s==2) return 'reason-alarm';
  if(s===1) return 'reason-ok';
  if(s>=3 && s<=6) return 'reason-off';
  return '';
}
function updateAll(d){
  var el;
  if(el=document.getElementById('status-text')) el.textContent=d.status||'—';
  var sdot=document.getElementById('hdr-status-dot');
  if(sdot){
    var total=(d.controllers||[]).filter(function(c){return c.enabled;}).length;
    var online=d.connected_count||0;
    sdot.className='hdr-status-dot '+(online===0?'err':online<total?'warn':'ok');
  }
  var ctrls=d.controllers||[];
  // Fault (Coil 56) with controller name, top-right
  var faultEl=document.getElementById('fault-text');
  var faulted=[];
  ctrls.forEach(function(c){ if(c.enabled && c.data && c.data['fault']==1) faulted.push(c.name); });
  if(faultEl){
    if(faulted.length){
      faultEl.textContent='⚠ Fault: '+faulted.join(', ');
      faultEl.className='fault-box active';
    } else {
      faultEl.textContent='';
      faultEl.className='fault-box';
    }
  }
  var mt=document.getElementById('motion-text');
  if(mt){
    if(d.motion==='present'){mt.textContent='👁 Motion';mt.style.color='#22c55e';}
    else{mt.textContent='👁 Sleep in '+d.sleep_in+'s';mt.style.color='#7ec8e3';}
  }
  var ms=d.motion_state||{};
  var sleepIn=d.sleep_in;
  var aiDot=document.getElementById('ai-dot');
  if(aiDot) aiDot.classList.toggle('active',(ms.consecutive||0)>=2);
  var aiStatus=document.getElementById('ai-status-text');   if(aiStatus)  aiStatus.textContent =(ms.status_txt||'Initializing...');
  var aiPct=document.getElementById('ai-motion-pct');       if(aiPct)     aiPct.textContent    =(ms.motion_pct||0).toFixed(1)+'%';
  var aiChanged=document.getElementById('ai-changed-px');   if(aiChanged) aiChanged.textContent=(ms.changed_px||0)+' px changed';
  var aiBar=document.getElementById('ai-motion-bar');       if(aiBar)     aiBar.style.width    =Math.min(ms.motion_pct||0,100)+'%';
  var aiFaces=document.getElementById('ai-faces');          if(aiFaces)   aiFaces.textContent  =(ms.face_count||0);
  var aiConsec=document.getElementById('ai-consecutive');   if(aiConsec)  aiConsec.textContent ='consecutive: '+(ms.consecutive||0);
  var aiFrames=document.getElementById('ai-frames');        if(aiFrames)  aiFrames.textContent =(ms.frame_count||0).toLocaleString();
  var aiWarmup=document.getElementById('ai-warmup');        if(aiWarmup)  aiWarmup.textContent =ms.warmup_done?'calibration ✓':'awaiting calibration';
  var aiSleepIn=document.getElementById('ai-sleep-in');     if(aiSleepIn) aiSleepIn.textContent=sleepIn+'s';
  var aiLastWake=document.getElementById('ai-last-wake');   if(aiLastWake)aiLastWake.textContent='Last: '+(ms.last_wake||'—');
  if(ms.history){
    var bars=document.querySelectorAll('#ai-bars .ai-bar');
    var max=Math.max.apply(null,ms.history.concat([1]));
    bars.forEach(function(bar,i){ var v=ms.history[i]||0; bar.style.height=Math.max(3,(v/max)*50)+'px'; });
  }
  var cc=d.connected_count||0;
  if(!tabsUnlocked && cc>0){
    unlockTabs(); buildAll();
    document.getElementById('unlock-banner').style.display='block';
    setTimeout(function(){ showTab('dashboard'); }, 400);
  }
  (d.controllers||[]).forEach(function(ctrl){
    var idx=ctrl.idx, data=ctrl.data||{};

    // ── Grid view tiles ──
    if(dashView==='grid'){
      var gt=document.getElementById('grid-temp-'+idx);
      var gb=document.getElementById('grid-badge-'+idx);
      var go=document.getElementById('grid-onoff-'+idx);
      var gtemp=parseFloat(data['5']);
      if(gt && !isNaN(gtemp)) gt.innerHTML=gtemp.toFixed(1)+'<span style="font-size:13px;">°C</span>';
      if(gb){
        if(!ctrl.enabled){gb.textContent='● Disabled';gb.className='grid-tile-badge';}
        else if(ctrl.connected){gb.textContent='● Online';gb.className='grid-tile-badge ok';}
        else{gb.textContent='● Offline';gb.className='grid-tile-badge err';}
      }
      if(go){var v=data['0'];if(v!=null){go.textContent=v?'ON':'OFF';go.className='grid-tile-onoff '+(v?'on':'off');}}
      // status code / reason intentionally not shown in grid view
    }

    var badge=document.getElementById('ctrl-badge-'+idx);
    if(badge){
      if(!ctrl.enabled){badge.textContent='● Disabled';badge.className='ctrl-badge dis';}
      else if(ctrl.connected){badge.textContent='● Online';badge.className='ctrl-badge ok';}
      else{badge.textContent='● Offline';badge.className='ctrl-badge err';}
    }
    // FAULT badge (between the Online badge and the ON/OFF button) — shown only on fault
    var cfb=document.getElementById('ctrl-fault-'+idx);
    if(cfb){ cfb.style.display = (data['fault']==1) ? '' : 'none'; }

    // Once a controller is set up (connected/communicating), clear the "Not set up" indicators
    var setupDone = !!ctrl.connected;
    var _sb=document.getElementById('ctrl-setup-badge-'+idx);   if(_sb) _sb.style.display=setupDone?'none':'';
    var _sp=document.getElementById('ctrl-setup-prompt-'+idx);  if(_sp) _sp.style.display=setupDone?'none':'';
    var _gsb=document.getElementById('grid-setup-badge-'+idx);  if(_gsb) _gsb.style.display=setupDone?'none':'';

    // ── Room Temp tile ──────────────────────────────────────────
    var t=parseFloat(data['5']);
    var sp=parseFloat(data['4']);
    var tempEl=document.getElementById('temp-'+idx);
    var tempBar=document.getElementById('tempBar-'+idx);
    var tempStatus=document.getElementById('temp-status-'+idx);
    var tempTile=document.getElementById('temp-tile-'+idx);
    if(tempEl && !isNaN(t)){
      tempEl.innerHTML = t.toFixed(1)+'<span style="font-size:14px;font-weight:700;">°C</span>';
      if(tempBar){
        // bar: 10°C=0% 40°C=100%
        var tpct=Math.min(Math.max((t-10)/(40-10)*100,0),100);
        tempBar.style.width=tpct+'%';
        tempBar.style.background=t>28?'#ef4444':t>22?'#f59e0b':'#3b82f6';
      }
      if(tempStatus){
        if(!isNaN(sp)){
          var diff=(t-sp).toFixed(1);
          tempStatus.textContent = diff>0 ? '▲ '+diff+'° above SP' : diff<0 ? '▼ '+Math.abs(diff)+'° below SP' : '✓ At setpoint';
          tempStatus.style.color = Math.abs(diff)>2?'#3b82f6':Math.abs(diff)>0.5?'#3b82f6':'#3b82f6';
        } else { tempStatus.textContent=''; }
      }
    }

    // ── Outside temp from UI6 sensor (overrides weather if present) ──
    var oatSensor = data['oat_value'];
    var oatEl = document.getElementById('outsideTemp-'+idx);
    if(oatEl && oatSensor != null){
      oatEl.textContent = oatSensor.toFixed(1)+'°C';
    }

// ── Setpoint knob ───────────────────────────────────────────
    if(!isNaN(sp)){ if(spHold[idx]==null){ lastSP[idx]=sp; if(knobs[idx]) knobs[idx].update(sp); } }

    // ── CO₂ tile ────────────────────────────────────────────────
    var coVal=data['co_level'];
    var coValEl=document.getElementById('coValue-'+idx);
    var coBarEl=document.getElementById('coBar-'+idx);
    var coStatus=document.getElementById('co-status-'+idx);
    var coTile=document.getElementById('co-tile-'+idx);
    if(coValEl){
      if(coVal!=null && coVal>0){
        coValEl.innerHTML=coVal+'<span style="font-size:13px;font-weight:700;"> ppm</span>';
        // scale: 400ppm=Good start … 1500ppm=Poor end; clamp marker 0-100%
        var cpct=Math.min(Math.max((coVal-400)/(1500-400)*100,0),100);
        if(coBarEl){coBarEl.style.left=cpct+'%';}
        if(coTile){coTile.className='metric-tile tile-co'+(coVal>=700?' co-bad':coVal>=300?' co-warn':'');}
        if(coStatus){
          coStatus.textContent=coVal<600?'✓ Good ('+coVal+' ppm)':coVal<1000?'⚠ Moderate ('+coVal+' ppm)':'✗ Poor ('+coVal+' ppm)';
          coStatus.style.color=coVal<600?'#059669':coVal<1000?'#d97706':'#dc2626';
        }
      } else {
        coValEl.innerHTML='--<span style="font-size:13px;font-weight:700;"> ppm</span>';
        if(coBarEl){coBarEl.style.left='0%';}
        if(coTile) coTile.className='metric-tile tile-co';
        if(coStatus){coStatus.textContent='No data';coStatus.style.color='';}
      }
    }

    var oo=document.getElementById('ctrl-onoff-'+idx);
    if(oo){var v=data['0'];if(v!=null){oo.textContent=v?'ON':'OFF';oo.className='onoff-btn '+(v?'on':'off');}}
    var modeVal=data['1'];
    var fanVal=data['2'];
    // fan raw 300/600/900 -> Low/Med/High button (nearest)
    if(fanVal!=null){
      var fanStep = (fanVal>=750)?3 : (fanVal>=450)?2 : (fanVal>=150)?1 : 0;
      [1,2,3].forEach(function(fs){
        var fb=document.getElementById('fan-'+fs+'-'+idx);
        if(!fb) return;
        fb.classList.toggle('active', fanStep==fs);
      });
    }
    // Mode highlight — Temperzone reg 117: 0=Auto 1=Cool 2=Heat 3=Fan
    var modeMap={0:'auto',1:'cool',2:'heat',3:'vent'};
    ['heat','cool','auto','vent'].forEach(function(m){
      var b=document.getElementById('mode-'+m+'-'+idx);
      if(b) b.classList.remove('active');
    });
    var mk=modeMap[modeVal];
    if(mk){ var b=document.getElementById('mode-'+mk+'-'+idx); if(b) b.classList.add('active'); }
    // Controller background
    var bg=document.getElementById('ctrl-bg-'+idx);
    if(bg){
      bg.classList.remove('heat','cool','off');
      if(data['0']==0) bg.classList.add('off');
      else if(modeVal==2) bg.classList.add('heat');
      else if(modeVal==0||modeVal==1) bg.classList.add('cool');
    }
    // Wind effect
    updateWindEffect(idx, fanVal);
    [0,117,116,22].forEach(function(reg){
      var btn=document.getElementById('cbtn-'+idx+'-'+reg);
      if(!btn) return;
      var v=data[String(reg)];
      if(v==null){btn.textContent='?';return;}
      btn.textContent=(META[reg]?META[reg].opts[v]||v:v);
      btn.className='toggle-btn '+(v==1?'on':'off');
    });
  });
  // Live Registers — curated, human-readable view (only meaningful parameters)
  var selIdx=parseInt((document.getElementById('inputs-ctrl-sel')||{}).value||0);
  var c0=(d.controllers||[])[selIdx];
  if(c0){
    var tbody=document.getElementById('all-regs');
    if(tbody) tbody.innerHTML=renderLiveRegisters(c0.data||{});
  }
}

// Option maps for enum registers
var LR_MODE   = {0:'Auto',1:'Cooling',2:'Heating',3:'Fan'};
var LR_STATUS = {1:'On',2:'Off · Alarm',3:'Off · BMS',4:'Off · Scheduler',
                 5:'Off · Digital In',6:'Off · Local',7:'Manual'};
// Curated rows: which value to show, its label, register address, and formatter
var LR_ROWS = [
  {k:'0',           addr:'C1',   label:'Power',            fmt:'onoff'},
  {k:'5',           addr:'IR1',  label:'Room Temperature', fmt:'temp'},
  {k:'4',           addr:'HR100',label:'Setpoint',         fmt:'temp'},
  {k:'100',         addr:'HR100',label:'Cooling Setpoint', fmt:'temp10'},
  {k:'102',         addr:'HR102',label:'Heating Setpoint', fmt:'temp10'},
  {k:'117',         addr:'HR117',label:'Mode',             fmt:'mode'},
  {k:'2',           addr:'HR114',label:'Fan Speed',        fmt:'pct'},
  {k:'status_code', addr:'IR135',label:'Unit Status',      fmt:'status'},
  {k:'22',          addr:'C22',  label:'Scheduler',        fmt:'sched'},
  {k:'fault',       addr:'C56',  label:'Fault',            fmt:'fault'}
];
function lrPill(txt,cls){ return '<span class="lr-pill '+cls+'">'+txt+'</span>'; }
function lrFormat(fmt,v){
  if(v==null) return '<span style="color:#cbd5e1;">—</span>';
  switch(fmt){
    case 'temp':   return (typeof v==='number'?v.toFixed(1):v)+'<span class="u">°C</span>';
    case 'temp10': return (v*0.1).toFixed(1)+'<span class="u">°C</span>';
    case 'pct':    return Math.round(v/10)+'<span class="u">%</span>';
    case 'mode':   return lrPill(LR_MODE[v]||v,'info');
    case 'status': return lrPill(LR_STATUS[v]||('Code '+v), v==1?'ok':'off');
    case 'onoff':  return lrPill(v==1?'On':'Off', v==1?'ok':'off');
    case 'sched':  return lrPill(v==1?'Enabled':'Disabled', v==1?'info':'off');
    case 'fault':  return lrPill(v==1?'Fault':'Normal', v==1?'warn':'ok');
    default:       return v;
  }
}
function renderLiveRegisters(data){
  var html='';
  LR_ROWS.forEach(function(row){
    var v=data[row.k];
    if(v===undefined) return;   // controller doesn't expose this parameter
    html+='<tr><td class="lr-param"><span class="lr-addr">'+row.addr+'</span>'+row.label+'</td>'+
          '<td class="lr-value">'+lrFormat(row.fmt,v)+'</td></tr>';
  });
  return html;
}

function refreshInputsTable(){
  var sel=document.getElementById('inputs-ctrl-sel');
  var title=document.getElementById('inputs-title');
  if(sel && title && deviceConfig[parseInt(sel.value)])
    title.textContent='All Registers — '+deviceConfig[parseInt(sel.value)].name;
}

function toggleCtrlOnOff(idx){
  fetch('/read').then(r=>r.json()).then(d=>{
    var ctrl=(d.controllers||[])[idx];if(!ctrl)return;
    var v=ctrl.data['0'];if(v==null)return;
    fetch('/write',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ctrl:idx,register:0,value:v==1?0:1})});
  });
}
// Registers that materially change AC operation → require confirmation
var CONFIRM_REGS = {117:'Heat/Cool mode', 0:'ON/OFF', 2024:'Summer/Winter'};
var _regPending = null;

function toggleCtrlReg(idx,reg){
  fetch('/read').then(r=>r.json()).then(d=>{
    var ctrl=(d.controllers||[])[idx];if(!ctrl)return;
    var v=ctrl.data[String(reg)];if(v==null)return;
    var newVal = v==1?0:1;
    if(CONFIRM_REGS[reg]){
      _regPending = {idx:idx, reg:reg, val:newVal};
      var name = (deviceConfig[idx]||{}).name || ('Controller '+(idx+1));
      var label = CONFIRM_REGS[reg];
      var newLabel = (META[reg] && META[reg].opts) ? (META[reg].opts[newVal]||newVal) : newVal;
      document.getElementById('confirm-reg-msg').innerHTML =
        'Change <strong>'+label+'</strong> on <strong>'+name+'</strong> to <strong>'+newLabel+'</strong>?<br><span style="font-size:11px;color:#888;">This affects a live controller.</span>';
      document.getElementById('confirm-reg-modal').style.display='flex';
      return;
    }
    fetch('/write',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ctrl:idx,register:reg,value:newVal})});
  });
}

function hideConfirmReg(){
  document.getElementById('confirm-reg-modal').style.display='none';
  _regPending=null;
}
function executeConfirmReg(){
  var p=_regPending;
  hideConfirmReg();
  if(!p) return;
  fetch('/write',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ctrl:p.idx,register:p.reg,value:p.val})
  }).then(()=>{ setTimeout(poll,300); });
}
function writeReg(){
  var ctrl=parseInt(document.getElementById('write-ctrl').value);
  var reg=parseInt(document.getElementById('write-reg').value);
  var val=parseInt(document.getElementById('write-val').value);
  var res=document.getElementById('write-result');
  res.style.color='#888';res.textContent='Writing...';
  fetch('/write',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ctrl,register:reg,value:val})
  }).then(r=>r.json()).then(d=>{
    if(d.status==='success'){res.textContent='Written reg '+reg+' = '+val;}
    else{res.textContent='Failed: '+(d.message||d.status);}
  });
}

// ── Schedule ──────────────────────────────────────────────────────────────
function buildSchedDayBtns(){
  var html='';
  DAYS_SHORT.forEach(function(d){
    var sel=schedSelectedDays.has(d);
    html+='<button class="day-toggle-btn '+(sel?'sel':'unsel')+'" onclick="toggleSchedDay(\''+d+'\')">'+d+'</button>';
  });
  document.getElementById('sched-day-btns').innerHTML=html;
}
function toggleSchedDay(d){ if(schedSelectedDays.has(d))schedSelectedDays.delete(d); else schedSelectedDays.add(d); buildSchedDayBtns(); }
function selectWeekdays(){ schedSelectedDays=new Set(['Mon','Tue','Wed','Thu','Fri']); buildSchedDayBtns(); }
function selectAllDays(){  schedSelectedDays=new Set(['Mon','Tue','Wed','Thu','Fri','Sat','Sun']); buildSchedDayBtns(); }
function clearSchedDays(){ schedSelectedDays=new Set(); buildSchedDayBtns(); }

// ── Schedule ──
var schedAction = 1;      // 1=ON 0=OFF
var schedView = 'list';
var twH = 8, twM = 0;     // time wheel state
var schedTime = '';       // selected HH:MM

function buildSchedDayBtns(){
  var html='';
  DAYS_SHORT.forEach(function(d){
    var sel=schedSelectedDays.has(d);
    html+='<button class="day-toggle-btn '+(sel?'sel':'unsel')+'" onclick="toggleSchedDay(\''+d+'\')">'+d+'</button>';
  });
  document.getElementById('sched-day-btns').innerHTML=html;
  updateSchedPreview();
}
function toggleSchedDay(d){ if(schedSelectedDays.has(d))schedSelectedDays.delete(d); else schedSelectedDays.add(d); buildSchedDayBtns(); }
function selectWeekdays(){ schedSelectedDays=new Set(['Mon','Tue','Wed','Thu','Fri']); buildSchedDayBtns(); }
function selectAllDays(){  schedSelectedDays=new Set(['Mon','Tue','Wed','Thu','Fri','Sat','Sun']); buildSchedDayBtns(); }
function clearSchedDays(){ schedSelectedDays=new Set(); buildSchedDayBtns(); }

function setSchedAction(a){
  schedAction=a;
  document.getElementById('act-on').classList.toggle('active',a===1);
  document.getElementById('act-off').classList.toggle('active',a===0);
  updateSchedPreview();
}

var schedAmPm = 'AM';
function setSchedAmPm(ap){
  schedAmPm = ap;
  document.getElementById('ampm-am').classList.toggle('active', ap==='AM');
  document.getElementById('ampm-pm').classList.toggle('active', ap==='PM');
  // if a time is already set, re-derive 24h value and refresh display
  if(schedTime){
    var p=schedTime.split(':'); var h24=parseInt(p[0]); var m=p[1];
    var h12=h24%12; if(h12===0) h12=12;
    var newH = ap==='PM' ? (h12===12?12:h12+12) : (h12===12?0:h12);
    schedTime=String(newH).padStart(2,'0')+':'+m;
    document.getElementById('time-display').textContent=fmt12(schedTime);
  }
  updateSchedPreview();
}
function fmt12(hhmm){
  var p=hhmm.split(':'); var h=parseInt(p[0]);
  var h12=h%12; if(h12===0) h12=12;
  return String(h12).padStart(2,'0')+':'+p[1];   // AM/PM shown by the buttons, not here
}

function setSchedView(v){
  schedView=v;
  document.getElementById('sview-week').classList.toggle('active',v==='week');
  document.getElementById('sview-list').classList.toggle('active',v==='list');
  document.getElementById('sched-week-view').style.display=v==='week'?'block':'none';
  document.getElementById('sched-list-view').style.display=v==='list'?'block':'none';
  loadSchedule();
}

// ── Live preview ──
function updateSchedPreview(){
  var days=Array.from(schedSelectedDays);
  var t=schedTime;
  var icon=document.getElementById('prev-icon');
  var title=document.getElementById('prev-title');
  var sub=document.getElementById('prev-sub');
  if(!days.length || !t){
    if(icon) icon.textContent='🗓';
    if(title) title.textContent='Choose days & time to preview';
    if(sub) sub.textContent='—';
    return;
  }
  var label = schedAction===1?'Turn ON':'Turn OFF';
  var col = schedAction===1?'#16a34a':'#ef4444';
  if(icon) icon.textContent = schedAction===1?'🟢':'🔴';
  if(title){ title.innerHTML='<span style="color:'+col+';">'+label+'</span> at <strong>'+t+'</strong>'; }
  var dayTxt = days.length===7?'Every day':days.length===5&&!schedSelectedDays.has('Sat')&&!schedSelectedDays.has('Sun')?'Weekdays':days.join(', ');
  var who = activeSchedCtrl==='ALL'?'all controllers':(deviceConfig[activeSchedCtrl]?deviceConfig[activeSchedCtrl].name:'');
  if(sub) sub.textContent=dayTxt+' · '+who;
}

// ── Time wheel ──
function openTimeWheel(){
  if(schedTime){ var p=schedTime.split(':'); twH=parseInt(p[0]); twM=parseInt(p[1]); }
  twRender();
  document.getElementById('time-wheel-modal').style.display='flex';
}
function closeTimeWheel(){ document.getElementById('time-wheel-modal').style.display='none'; }
function twStep(which,dir){
  if(which==='h'){ twH=(twH+dir+24)%24; }
  else { twM=(twM+dir*5+60)%60; }
  twRender();
}
function twQuick(h,m){ twH=h; twM=m; twRender(); }
function twRender(){
  var hh=String(twH).padStart(2,'0'), mm=String(twM).padStart(2,'0');
  document.getElementById('tw-hh').textContent=hh;
  document.getElementById('tw-mm').textContent=mm;
  var ap=document.getElementById('tw-ampm'); if(ap) ap.textContent=twH<12?'AM':'PM';
  var wh=document.getElementById('tw-wheel-h'); if(wh) wh.textContent=hh;
  var wm=document.getElementById('tw-wheel-m'); if(wm) wm.textContent=mm;
  drawTwClock();
}
function drawTwClock(){
  var cv=document.getElementById('tw-clock'); if(!cv) return;
  var ctx=cv.getContext('2d'), W=150, C=75, R=64;
  var dark=document.body.classList.contains('theme-dark');
  ctx.clearRect(0,0,W,W);
  // face
  ctx.beginPath();ctx.arc(C,C,R,0,Math.PI*2);
  ctx.fillStyle=dark?'#0f172a':'#f8fafc';ctx.fill();
  ctx.strokeStyle=dark?'#334155':'#cbd5e1';ctx.lineWidth=2;ctx.stroke();
  // hour ticks
  for(var i=0;i<12;i++){
    var a=i*Math.PI/6-Math.PI/2;
    ctx.beginPath();
    ctx.moveTo(C+(R-7)*Math.cos(a),C+(R-7)*Math.sin(a));
    ctx.lineTo(C+(R-2)*Math.cos(a),C+(R-2)*Math.sin(a));
    ctx.strokeStyle=dark?'#64748b':'#94a3b8';ctx.lineWidth=i%3===0?2.5:1;ctx.stroke();
  }
  // hands
  var hAng=((twH%12)+twM/60)*Math.PI/6-Math.PI/2;
  var mAng=twM*Math.PI/30-Math.PI/2;
  // hour hand
  ctx.beginPath();ctx.moveTo(C,C);
  ctx.lineTo(C+R*0.5*Math.cos(hAng),C+R*0.5*Math.sin(hAng));
  ctx.strokeStyle=dark?'#93c5fd':'#1e3a5f';ctx.lineWidth=4;ctx.lineCap='round';ctx.stroke();
  // minute hand
  ctx.beginPath();ctx.moveTo(C,C);
  ctx.lineTo(C+R*0.78*Math.cos(mAng),C+R*0.78*Math.sin(mAng));
  ctx.strokeStyle='#3b82f6';ctx.lineWidth=3;ctx.lineCap='round';ctx.stroke();
  // center dot
  ctx.beginPath();ctx.arc(C,C,4,0,Math.PI*2);ctx.fillStyle='#1e3a5f';ctx.fill();
}

function applyTimeWheel(){
  schedTime=String(twH).padStart(2,'0')+':'+String(twM).padStart(2,'0');
  document.getElementById('time-display').textContent=fmt12(schedTime);
  setSchedAmPm(twH<12?'AM':'PM');   // sync the buttons to the picked hour
  closeTimeWheel();
  updateSchedPreview();
}

// ── Controller bar + enabled status ──
function buildSchedCtrlBar(){
  var html='';
  var isAll=(activeSchedCtrl==='ALL');
  html+='<button class="sched-ctrl-btn all-btn'+(isAll?' active':'')+'" onclick="setSchedCtrl(\'ALL\')">ALL</button>';
  deviceConfig.forEach(function(ctrl,idx){
    if(!ctrl.enabled) return;
    html+='<button class="sched-ctrl-btn'+(activeSchedCtrl===idx?' active':'')+'" onclick="setSchedCtrl('+idx+')">'+ctrl.name+'</button>';
  });
  document.getElementById('sched-ctrl-bar').innerHTML=html;
  buildSchedDayBtns();
  updateSchedEnabledPill();
  setSchedAction(schedAction);
}
function setSchedCtrl(idx){ activeSchedCtrl=idx; buildSchedCtrlBar(); loadSchedule(); updateSchedPreview(); }

// Point 4: reflect whether Time Schedules - Coil 22 is enabled on the controller
function updateSchedEnabledPill(){
  var pill=document.getElementById('sched-enabled-pill');
  var note=document.getElementById('sched-enable-note');
  if(!pill) return;
  fetch('/read').then(r=>r.json()).then(d=>{
    var ctrls=d.controllers||[];
    var idx = activeSchedCtrl==='ALL' ? (ctrls.findIndex(function(c){return c.enabled;})) : activeSchedCtrl;
    var c = ctrls[idx];
    var en = c && c.data && c.data['schedule']==1;
    pill.textContent = 'Schedules: '+(en?'ENABLED':'DISABLED')+' (tap to toggle)';
    pill.className = 'sched-enabled-pill '+(en?'on':'off');
    pill.style.cursor='pointer';
    pill.onclick = function(){ toggleSchedEnable(en?0:1); };
    if(note){
      if(en){ note.style.display='none'; }
      else {
        note.style.display='block';
        note.innerHTML='⚠ Time Schedules are <strong>disabled</strong>'+(activeSchedCtrl==='ALL'?' on one or more controllers':' on this controller')+' — events won\'t run until enabled. '+
          '<span onclick="toggleSchedEnable(1)" style="color:#1e3a5f;text-decoration:underline;cursor:pointer;font-weight:700;">Enable now</span>';
      }
    }
  }).catch(function(){});
}
// Writes Coil 22 (Enable Scheduler) to the selected controller, or every enabled
// controller in the system when the 'ALL' tab is active.
function toggleSchedEnable(value){
  var targets=[];
  if(activeSchedCtrl==='ALL'){
    deviceConfig.forEach(function(ctrl,idx){ if(ctrl.enabled) targets.push(idx); });
  } else {
    targets.push(activeSchedCtrl);
  }
  var reqs = targets.map(function(idx){
    return fetch('/write',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({ctrl:idx,register:22,value:value})});
  });
  Promise.all(reqs).then(function(){
    setTimeout(function(){ updateSchedEnabledPill(); loadSchedule(); buildSchedCtrlBar(); }, 600);
  });
}
// kept for backward compatibility with any old call-sites
function enableSchedReg(idx){ toggleSchedEnable(1); }

// ── Week grid ──
function fmtHour12(h){
  var ap = h<12 ? 'a' : 'p';
  var hr = h%12; if(hr===0) hr=12;
  return hr+ap;   // 12a, 1a … 11a, 12p, 1p … 11p
}
function buildWeekGrid(data){
  var grid=document.getElementById('week-grid');
  if(!grid) return;
  var map={};
  DAYS_SHORT.forEach(function(d){ map[d]={};
    (data[d]||[]).forEach(function(e){ var h=parseInt(e.time.split(':')[0]); map[d][h]=e.action; });
  });
  // header: corner + 24 hour ticks (label every 3 h)
  var html='<div class="wg-corner"></div>';
  for(var h=0;h<24;h++){
    var lbl = (h%3===0) ? (h===0?'12a':h<12?h+'a':h===12?'12p':(h-12)+'p') : '';
    html+='<div class="wg-hh">'+lbl+'</div>';
  }
  // one compact row per day, 24-hour timeline across
  DAYS_SHORT.forEach(function(d){
    html+='<div class="wg-dayrow">'+d+'</div>';
    for(var h=0;h<24;h++){
      var a=map[d][h];
      var cls='wg-track'+(a===1?' has-on':a===0?' has-off':'');
      html+='<div class="'+cls+'" title="'+d+' '+fmtHour12(h)+'" onclick="weekCellTap(\''+d+'\','+h+')"></div>';
    }
  });
  grid.innerHTML=html;
}

function weekCellTap(day,hour){
  // tap existing → offer delete; tap empty → prefill add form
  var displayIdx=activeSchedCtrl==='ALL'?0:activeSchedCtrl;
  fetch('/api/schedule?ctrl='+displayIdx).then(r=>r.json()).then(data=>{
    var ev=(data[day]||[]).find(function(e){return parseInt(e.time.split(':')[0])===hour;});
    if(ev){
      if(confirm('Delete '+day+' '+ev.time+' → '+(ev.action?'ON':'OFF')+'?')){
        data[day]=data[day].filter(function(e){return e!==ev;});
        fetch('/api/schedule?ctrl='+displayIdx,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)}).then(loadSchedule);
      }
    } else {
      schedSelectedDays=new Set([day]);
      schedTime=String(hour).padStart(2,'0')+':00';
      document.getElementById('time-display').textContent=schedTime;
      buildSchedDayBtns();
      updateSchedPreview();
      document.getElementById('sched-preview').scrollIntoView({behavior:'smooth',block:'center'});
    }
  });
}

async function addSchedule(){
  if(!schedTime){ alert('Tap the Time field to set a time'); return; }
  if(schedSelectedDays.size===0){ alert('Select at least one day'); return; }
  var ctrlsToUpdate=[];
  if(activeSchedCtrl==='ALL'){ deviceConfig.forEach(function(ctrl,idx){ if(ctrl.enabled) ctrlsToUpdate.push(idx); }); }
  else { ctrlsToUpdate=[activeSchedCtrl]; }
  for(var ci of ctrlsToUpdate){
    var res=await fetch('/api/schedule?ctrl='+ci); var data=await res.json();
    schedSelectedDays.forEach(function(day){
      if(!data[day]) data[day]=[];
      if(!data[day].some(function(e){return e.time===schedTime&&e.action===schedAction;}))
        data[day].push({time:schedTime,action:schedAction});
    });
    await fetch('/api/schedule?ctrl='+ci,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  }
  loadSchedule();
}

async function loadSchedule(){
  var displayIdx=0;
  if(activeSchedCtrl==='ALL'){ for(var i=0;i<deviceConfig.length;i++){ if(deviceConfig[i].enabled){displayIdx=i;break;} } }
  else { displayIdx=activeSchedCtrl; }
  var labelEl=document.getElementById('sched-view-label');
  if(labelEl) labelEl.textContent=(activeSchedCtrl==='ALL'?'ALL':(deviceConfig[displayIdx]?deviceConfig[displayIdx].name:'—'));
  const res=await fetch('/api/schedule?ctrl='+displayIdx);
  const data=await res.json();
  if(schedView==='week'){ buildWeekGrid(data); return; }
  // list view
  const DF={Mon:"Monday",Tue:"Tuesday",Wed:"Wednesday",Thu:"Thursday",Fri:"Friday",Sat:"Saturday",Sun:"Sunday"};
  var html='';
  if(activeSchedCtrl==='ALL') html+='<div style="background:#f5f3ff;border:1px solid #7c3aed;border-radius:7px;padding:8px 10px;font-size:11px;color:#5b21b6;margin-bottom:10px;">📌 Events apply to ALL enabled controllers.</div>';
  DAYS_SHORT.forEach(function(day){
    var events=data[day]||[];
    html+='<div style="margin-bottom:10px;background:#f8f9fa;border-radius:8px;padding:10px;">';
    html+='<div style="font-weight:bold;color:#1e3a5f;font-size:13px;margin-bottom:5px;">'+DF[day]+'</div>';
    if(!events.length){ html+='<div style="font-size:11px;color:#aaa;">No events</div>'; }
    else events.forEach(function(e,i){
      var c=e.action==1?'#22c55e':'#ef4444',l=e.action==1?'ON':'OFF';
      html+='<div style="display:flex;justify-content:space-between;align-items:center;padding:5px 8px;margin-bottom:4px;background:white;border-radius:6px;border-left:3px solid '+c+';">';
      html+='<span style="font-size:12px;">'+e.time+' → <strong style="color:'+c+';">'+l+'</strong></span>';
      html+='<button onclick="deleteSchedule(\''+day+'\','+i+')" style="background:#ef4444;color:white;border:none;border-radius:4px;padding:2px 8px;font-size:11px;cursor:pointer;">✕</button></div>';
    });
    html+='</div>';
  });
  document.getElementById('schedule-list').innerHTML=html;
}

async function deleteSchedule(day,index){
  var displayIdx=activeSchedCtrl==='ALL'?0:activeSchedCtrl;
  var res=await fetch('/api/schedule?ctrl='+displayIdx); var data=await res.json();
  data[day].splice(index,1);
  await fetch('/api/schedule?ctrl='+displayIdx,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});
  loadSchedule();
}
function confirmClearAllSchedule(){
  var msg=activeSchedCtrl==='ALL'?'This will clear ALL schedule events for every enabled controller.':'This will clear all schedule events for "'+(deviceConfig[activeSchedCtrl]?deviceConfig[activeSchedCtrl].name:'this controller')+'".';
  document.getElementById('clear-sched-msg').textContent=msg;
  document.getElementById('clear-sched-modal').style.display='flex';
}
function hideClearSchedModal(){ document.getElementById('clear-sched-modal').style.display='none'; }
async function executeClearAllSchedule(){
  hideClearSchedModal();
  var emptyWeek={"Mon":[],"Tue":[],"Wed":[],"Thu":[],"Fri":[],"Sat":[],"Sun":[]};
  var ctrlsToUpdate=[];
  if(activeSchedCtrl==='ALL'){ deviceConfig.forEach(function(ctrl,idx){ if(ctrl.enabled) ctrlsToUpdate.push(idx); }); }
  else { ctrlsToUpdate=[activeSchedCtrl]; }
  for(var ci of ctrlsToUpdate) await fetch('/api/schedule?ctrl='+ci,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(emptyWeek)});
  loadSchedule();
}


function hideClearSchedModal(){ document.getElementById('clear-sched-modal').style.display='none'; }
async function executeClearAllSchedule(){
  hideClearSchedModal();
  var emptyWeek={"Mon":[],"Tue":[],"Wed":[],"Thu":[],"Fri":[],"Sat":[],"Sun":[]};
  var ctrlsToUpdate=[];
  if(activeSchedCtrl==='ALL'){ deviceConfig.forEach(function(ctrl,idx){ if(ctrl.enabled) ctrlsToUpdate.push(idx); }); }
  else { ctrlsToUpdate=[activeSchedCtrl]; }
  for(var ci of ctrlsToUpdate){
    await fetch('/api/schedule?ctrl='+ci,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(emptyWeek)});
  }
  loadSchedule();
}

// ── AI panel accordion ────────────────────────────────────────────────────
function toggleAiPanel(id){
  var body=document.getElementById(id);
  var chev=document.getElementById(id+'-chev');
  if(!body) return;
  var open=body.style.display==='none'||body.style.display==='';
  body.style.display=open?'block':'none';
  if(chev) chev.textContent=open?'▼':'▶';
}

// ── Motion config ─────────────────────────────────────────────────────────
var currentSleepTimeout = 5;
var motionDetectionEnabled = true;

function setSleepTimeout(secs){
  currentSleepTimeout = secs;
  [3,5,10,20,60].forEach(function(s){
    var b=document.getElementById('sleep-btn-'+s);
    if(b) b.classList.toggle('active', s===secs);
  });
  document.getElementById('sleep-timeout-label').textContent = secs+'s';
  fetch('/api/motion_config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({timeout:secs})});
}

function toggleMotionEnabled(){
  motionDetectionEnabled = !motionDetectionEnabled;
  var btn=document.getElementById('motion-enable-btn');
  if(btn){ btn.textContent=motionDetectionEnabled?'Enabled':'Disabled'; btn.className='toggle-btn '+(motionDetectionEnabled?'on':'off'); }
  fetch('/api/motion_config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({enabled:motionDetectionEnabled})});
}

// ── LED per-controller assign ─────────────────────────────────────────────
function setLedToCtrl(idx){
  var sel = document.getElementById('led-assign-'+idx);
  var mode = sel ? sel.value : 'temp';
  console.log('[LED] setLedToCtrl idx='+idx+' mode='+mode);
  fetch('/api/led_ctrl',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ctrl:idx, mode:mode})})
  .then(function(r){ return r.json(); })
  .then(function(d){
    // Reset all assign buttons
    document.querySelectorAll('[id^="led-assign-btn-"]').forEach(function(b){
      b.style.background='#1e3a5f'; b.textContent='▶ Apply';
    });
    var btn=document.getElementById('led-assign-btn-'+idx);
    if(btn){
      if(mode==='off'){ btn.style.background='#6b7280'; btn.textContent='● Off'; }
      else{ btn.style.background='#16a34a'; btn.textContent='✓ Active'; }
    }
  })
  .catch(function(){ console.error('LED assign failed'); });
}

function applyLedAssign(idx){
  // Always apply immediately when dropdown changes
  setLedToCtrl(idx);
}


// ── Email ─────────────────────────────────────────────────────────────────
function openEmailModal(){
  document.getElementById('trend-menu').style.display='none';
  var html='';
  html+='<button class="email-ctrl-btn all-opt'+(emailCtrlSel==='all'?' sel':'')+'" onclick="selectEmailCtrl(\'all\')">📊 All Controllers</button>';
  deviceConfig.forEach(function(ctrl,idx){
    if(!ctrl.enabled) return;
    html+='<button class="email-ctrl-btn'+(emailCtrlSel==idx?' sel':'')+'" onclick="selectEmailCtrl('+idx+')">'+ctrl.name+'<span style="float:right;font-size:11px;color:#888;">'+ctrl.type+' · ID:'+ctrl.slave_id+'</span></button>';
  });
  document.getElementById('email-ctrl-options').innerHTML=html;
  document.getElementById('email-send-result').textContent='';
  document.getElementById('email-modal').style.display='flex';
}
function selectEmailCtrl(v){
  emailCtrlSel=v;
  document.querySelectorAll('.email-ctrl-btn').forEach(function(btn){ btn.classList.remove('sel'); });
  event.target.closest('.email-ctrl-btn').classList.add('sel');
}
function closeEmailModal(){ document.getElementById('email-modal').style.display='none'; }
async function confirmSendEmail(){
  var resEl=document.getElementById('email-send-result');
  var recipient=(document.getElementById('email-recipient-input')||{}).value||'';
  if(!recipient){ resEl.style.color='#ef4444'; resEl.textContent='✗ Please enter a recipient email'; return; }
  resEl.style.color='#888'; resEl.textContent='Sending…';
  try{
    const r=await fetch('/send-log',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ctrl:emailCtrlSel,recipient:recipient})});
    const d=await r.json();
    if(d.status==='sent'){resEl.style.color='#22c55e';resEl.textContent='✓ Sent — '+d.rows+' rows ('+d.label+')';setTimeout(closeEmailModal,2000);}
    else{resEl.style.color='#ef4444';resEl.textContent='✗ '+(d.msg||d.status);}
  }catch(e){resEl.style.color='#ef4444';resEl.textContent='✗ Network error';}
}

// ── Network panel ─────────────────────────────────────────────────────────
function showNetworkPanel(){ document.getElementById('network-panel').style.display='flex'; loadNetworkInfo(); }
function closeNetworkPanel(){ document.getElementById('network-panel').style.display='none'; }

// ── Wi-Fi setup popup ──────────────────────────────────────────────────────
var wifiSelSSID='', wifiSelSecure=false, wifiStatusTimer=null;
function showWifi(){
  document.getElementById('wifi-panel').style.display='flex';
  resetWifiSel();
  scanWifi();
  if(wifiStatusTimer) clearInterval(wifiStatusTimer);
  wifiStatusTimer=setInterval(refreshWifiStatus, 4000);   // keep "connected to" accurate
}
function closeWifi(){
  document.getElementById('wifi-panel').style.display='none';
  if(wifiStatusTimer){ clearInterval(wifiStatusTimer); wifiStatusTimer=null; }
}
function refreshWifiStatus(){
  fetch('/api/wifi/status').then(function(r){return r.json();}).then(function(d){
    var st=document.getElementById('wifi-status');
    if(!st) return;
    if(d.connected){ st.className='wifi-status ok'; st.textContent='Connected to '+d.ssid; }
    else { st.className='wifi-status'; st.textContent='Not connected to Wi-Fi'; }
    // keep the list highlight in step with the real connection
    document.querySelectorAll('.wifi-item').forEach(function(el){
      var nm=el.querySelector('.wifi-name');
      if(!nm) return;
      var name=nm.textContent.replace(/^✓\s*/,'');
      el.classList.toggle('active-net', !!d.ssid && name===d.ssid);
    });
  }).catch(function(){});
}
function resetWifiSel(){
  wifiSelSSID=''; wifiSelSecure=false;
  document.getElementById('wifi-connect-row').style.display='none';
  var ph=document.getElementById('wifi-placeholder'); if(ph) ph.style.display='flex';
  document.getElementById('wifi-msg').textContent='';
  var p=document.getElementById('wifi-pass'); if(p) p.value='';
}
function wifiBars(sig){
  var n = sig>=75?4 : sig>=50?3 : sig>=30?2 : sig>=10?1 : 0;
  return '▂▄▆█'.slice(0,n).padEnd(4,'·');
}
function scanWifi(){
  var list=document.getElementById('wifi-list');
  var btn=document.getElementById('wifi-scan-btn');
  list.innerHTML='<div class="wifi-empty">Scanning…</div>';
  if(btn){ btn.disabled=true; btn.textContent='Scanning…'; }
  fetch('/api/wifi/scan').then(function(r){return r.json();}).then(function(d){
    if(btn){ btn.disabled=false; btn.textContent='🔄 Scan'; }
    var st=document.getElementById('wifi-status');
    if(d.connected){ st.className='wifi-status ok'; st.textContent='Connected to '+d.connected; }
    else { st.className='wifi-status'; st.textContent='Not connected to Wi-Fi'; }
    if(!d.ok && (!d.networks||!d.networks.length)){
      list.innerHTML='<div class="wifi-empty">'+(d.error||'No networks found')+'</div>'; return;
    }
    if(!d.networks.length){ list.innerHTML='<div class="wifi-empty">No networks found</div>'; return; }
    list.innerHTML='';
    d.networks.forEach(function(n){
      var div=document.createElement('div');
      div.className='wifi-item'+(n.active?' active-net':'');
      div.onclick=function(){ selectWifi(n, div); };
      div.innerHTML='<span class="wifi-name">'+(n.active?'✓ ':'')+escHtml(n.ssid)+'</span>'+
        '<span class="wifi-meta">'+(n.secure?'🔒':'🔓')+
        '<span class="wifi-bars">'+wifiBars(n.signal)+'</span></span>';
      list.appendChild(div);
    });
  }).catch(function(){
    if(btn){ btn.disabled=false; btn.textContent='🔄 Scan'; }
    list.innerHTML='<div class="wifi-empty">Scan failed — check Wi-Fi hardware</div>';
  });
}
function selectWifi(net, el){
  wifiSelSSID=net.ssid; wifiSelSecure=net.secure;
  document.querySelectorAll('.wifi-item').forEach(function(x){ x.classList.remove('sel'); });
  if(el) el.classList.add('sel');
  document.getElementById('wifi-sel-ssid').textContent=net.ssid;
  var row=document.getElementById('wifi-connect-row');
  var pass=document.getElementById('wifi-pass');
  var ph=document.getElementById('wifi-placeholder'); if(ph) ph.style.display='none';
  // hide password box for open networks
  pass.style.display = net.secure ? '' : 'none';
  document.querySelector('.wifi-show').style.display = net.secure ? '' : 'none';
  row.style.display='block';
  document.getElementById('wifi-msg').textContent='';
  if(net.secure){ setTimeout(function(){ pass.click(); }, 150); } // opens on-screen keyboard
}
function toggleWifiPass(){
  var p=document.getElementById('wifi-pass');
  p.type=document.getElementById('wifi-showpass').checked?'text':'password';
}
function connectWifi(){
  if(!wifiSelSSID) return;
  var pass=document.getElementById('wifi-pass').value||'';
  var msg=document.getElementById('wifi-msg');
  var btn=document.getElementById('wifi-connect-btn');
  if(wifiSelSecure && !pass){ msg.className='wifi-msg err'; msg.textContent='Enter the Wi-Fi password'; return; }
  msg.className='wifi-msg'; msg.textContent='Connecting to '+wifiSelSSID+'…';
  btn.disabled=true; btn.textContent='Connecting…';
  fetch('/api/wifi/connect',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({ssid:wifiSelSSID,password:pass})})
    .then(function(r){return r.json();}).then(function(d){
      btn.disabled=false; btn.textContent='Connect';
      if(d.ok){
        msg.className='wifi-msg ok'; msg.textContent='✓ Connected to '+d.ssid;
        setTimeout(function(){ scanWifi(); }, 1500);
        setTimeout(function(){ closeWifi(); }, 2500);
        setTimeout(function(){ if(typeof loadClockSource==='function') loadClockSource(); }, 16000);
      } else {
        msg.className='wifi-msg err'; msg.textContent='✗ '+(d.error||'Could not connect');
      }
    }).catch(function(){
      btn.disabled=false; btn.textContent='Connect';
      msg.className='wifi-msg err'; msg.textContent='✗ Connection request failed';
    });
}
function escHtml(s){ return String(s).replace(/[&<>"]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c];}); }
// On startup / display restart: if not on Wi-Fi, pop the setup dialog once
function wifiStartupCheck(){
  fetch('/api/wifi/status').then(function(r){return r.json();}).then(function(d){
    if(!d.connected){ showWifi(); }
  }).catch(function(){});
}

function showHelpModal(){ document.getElementById('help-modal').style.display='flex'; }
function closeHelpModal(){ document.getElementById('help-modal').style.display='none'; }
function showQuickStart(){ var o=document.getElementById('quickstart-overlay'); if(o) o.classList.add('show'); loadQuickStartQR(); }
function loadQuickStartQR(){
  // QR + IP reflect the Ethernet IPv4 address (Wi-Fi only as a fallback). Refresh
  // each time the guide opens so the code always tracks the panel's current IP.
  var img=document.getElementById('qs-qr-img');
  var urlEl=document.getElementById('qs-qr-url');
  if(img) img.src='/api/qr?t='+Date.now();     // cache-bust
  fetch('/api/lan_url').then(function(r){return r.json();}).then(function(d){
    if(!urlEl) return;
    if(d && d.url){
      urlEl.textContent=d.url.replace(/^https?:\/\//,'').replace(/\/$/,'');
    } else {
      urlEl.textContent='No network connection';
    }
  }).catch(function(){ if(urlEl) urlEl.textContent='—'; });
}
function closeQuickStart(){ var o=document.getElementById('quickstart-overlay'); if(o) o.classList.remove('show'); }
async function loadNetworkInfo(){
  var body=document.getElementById('network-info-body');
  body.innerHTML='<div style="text-align:center;padding:20px;color:#888;font-size:13px;">Fetching…</div>';
  try{
    const r=await fetch('/api/network'); const d=await r.json();
    var html='';
    html+='<div class="net-row"><span class="net-label">Hostname</span><span class="net-val">'+(d.hostname||'—')+'</span></div>';
    if(d.wifi_ssid) html+='<div class="net-row"><span class="net-label">Wi-Fi SSID</span><span class="net-val" style="color:#0891b2;">📶 '+d.wifi_ssid+'</span></div>';
    if(d.interfaces&&d.interfaces.length>0){
      var grouped={};
      d.interfaces.forEach(function(ifc){ if(!grouped[ifc.interface]) grouped[ifc.interface]=[]; grouped[ifc.interface].push(ifc); });
      Object.keys(grouped).forEach(function(ifname){
        var isWifi=ifname.startsWith('w');
        html+='<div style="margin-top:10px;"><span class="net-iface-badge '+(isWifi?'wifi':'')+'">'+ifname+'</span>';
        grouped[ifname].forEach(function(ifc){
          html+='<div class="net-row" style="padding-left:4px;"><span class="net-label">'+(ifc.family==='inet6'?'IPv6':'IPv4')+'</span><span class="net-val" style="font-size:12px;">'+ifc.address+'</span></div>';
        });
        html+='</div>';
      });
    }
    body.innerHTML=html;
  }catch(e){ body.innerHTML='<div style="color:#ef4444;padding:10px;">Failed to load network info.</div>'; }
}

// ── Chart ─────────────────────────────────────────────────────────────────
// ── Trend config state ────────────────────────────────────────────────────
// trendHistoryLen is now the chart window in MINUTES (one plotted point/minute).
var trendHistoryLen  = 15;
var trendEmailMins   = 60;
var trendChartType   = 'area';

function setHistoryLen(n){
  trendHistoryLen = n;
  [15,30,60,120,360].forEach(function(v){
    var b=document.getElementById('hlen-'+v);
    if(b) b.classList.toggle('active',v===n);
  });
  updateChart();   // redraw immediately
}


function setEmailMinutes(m){
  trendEmailMins = m;
  [30,60,120,360,720,1440].forEach(function(v){
    var b=document.getElementById('email-'+v);
    if(b) b.classList.toggle('active',v===m);
  });
  fetch('/api/trend_config',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({email_minutes:m})});
}

function applyTrendConfig(){
  var sel = document.getElementById('trend-primary-sel');
  var typ = document.getElementById('trend-chart-type');
  if(typ){ trendChartType=typ.value; }
  if(sel){
    trendPrimaryIdx=parseInt(sel.value)||0;
    // Switch zones on the server (which also clears the old zone's trend buffers),
    // THEN rebuild — so we never briefly re-plot the previous controller's data.
    fetch('/api/trend_config',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({primary_ctrl:trendPrimaryIdx})})
      .then(function(){ rebuildChart(); })
      .catch(function(){ rebuildChart(); });
  } else {
    rebuildChart();
  }
}

function applyTrendPrimary(){ applyTrendConfig(); }
function applyChartHistory(){}  // legacy stub

// ── Chart (minute-by-minute) ───────────────────────────────────────────────
var tempChart;
function getChartDatasets(){
  var showTemp    = (document.getElementById('show-temp')    ||{checked:true}).checked;
  var showSP      = (document.getElementById('show-setpoint')||{checked:true}).checked;
  var showCO      = (document.getElementById('show-co')      ||{checked:false}).checked;
  var dark        = document.body.classList.contains('theme-dark');
  var gridColor   = dark ? 'rgba(148,163,184,0.14)' : '#eef1f5';
  var fill        = (trendChartType!=='line');
  var datasets    = [];
  if(showTemp)
    datasets.push({label:'Temperature',borderColor:'#ef4444',
      backgroundColor:'rgba(239,68,68,0.12)',
      data:[],borderWidth:2.5,pointRadius:0,fill:fill,tension:0.35,spanGaps:true});
  if(showSP)
    datasets.push({label:'Setpoint',borderColor:'#3b82f6',borderDash:[6,4],
      backgroundColor:'transparent',
      data:[],borderWidth:2,pointRadius:0,fill:false,tension:0.2,spanGaps:true});
  if(showCO)
    datasets.push({label:'CO₂',borderColor:'#10b981',
      backgroundColor:'rgba(16,185,129,0.10)',
      data:[],borderWidth:2,pointRadius:0,fill:false,tension:0.35,spanGaps:true});
  return {datasets:datasets, gridColor:gridColor};
}

function rebuildChart(){
  if(!tempChart) return;
  var cfg = getChartDatasets();
  tempChart.data.datasets = cfg.datasets;
  tempChart.options.scales.x.grid.color = cfg.gridColor;
  tempChart.options.scales.y.grid.color = cfg.gridColor;
  tempChart.data.labels   = [];
  tempChart.update('none');
  updateChart();
  var badge = document.getElementById('trend-ctrl-badge');
  if(badge && deviceConfig[trendPrimaryIdx])
    badge.textContent = deviceConfig[trendPrimaryIdx].name;
}

function initChart(){
  var ctx=document.getElementById('tempChart').getContext('2d');
  var cfg=getChartDatasets();
  var dark=document.body.classList.contains('theme-dark');
  var tickColor=dark?'#94a3b8':'#64748b';
  tempChart=new Chart(ctx,{type:'line',data:{labels:[],datasets:cfg.datasets},
    options:{responsive:true,maintainAspectRatio:false,animation:{duration:0},
      interaction:{mode:'index',intersect:false},
      scales:{
        y:{beginAtZero:false,grace:'10%',grid:{color:cfg.gridColor,drawBorder:false},
           ticks:{color:tickColor,font:{size:11},padding:6,
                  callback:function(v){return Number(v).toFixed(1)+'°';}}},   // 23.4, 28.3 …
        x:{grid:{color:cfg.gridColor,display:false,drawBorder:false},
           ticks:{color:tickColor,maxTicksLimit:7,maxRotation:0,autoSkip:true,font:{size:10},
                  callback:function(v){return to12h(this.getLabelForValue(v));}}}   // 15:56 → 3:56
      },
      plugins:{
        legend:{position:'top',align:'end',labels:{color:tickColor,boxWidth:9,boxHeight:9,
          usePointStyle:true,pointStyle:'circle',font:{size:11,weight:'600'},padding:14}},
        tooltip:{enabled:true,mode:'index',intersect:false,
          callbacks:{label:function(c){return ' '+c.dataset.label+': '+(c.parsed.y!=null?c.parsed.y.toFixed(1):'—')+(c.dataset.label==='CO₂'?' ppm':'°C');}}}
      }}});
}

// Build a fixed rolling window of `n` minute-slots ending at the most recent
// data minute. Every window size therefore spans a real, fixed time range
// (15 slots = 15 min, 60 = 1 h …) and data plots at the right edge — so
// changing the Time Window visibly changes the chart even with little history.
function _hm2min(s){ var p=(s||'0:0').split(':'); return (+p[0])*60+(+p[1]); }
function _min2hm(m){ m=((m%1440)+1440)%1440; var h=Math.floor(m/60), mm=m%60; return (h<10?'0':'')+h+':'+(mm<10?'0':'')+mm; }
function to12h(hm){ var p=(hm||'').split(':'); if(p.length<2) return hm; var h=+p[0]%12; if(h===0) h=12; return h+':'+p[1]; }
function buildMinuteWindow(d, n){
  var labels=d.labels||[], temp=d.temp||[], sp=d.setpoint||[], co=d.co||[];
  var mapT={}, mapS={}, mapC={};
  for(var i=0;i<labels.length;i++){ mapT[labels[i]]=temp[i]; mapS[labels[i]]=sp[i]; mapC[labels[i]]=co[i]; }
  var anchor = labels.length ? _hm2min(labels[labels.length-1])
                             : (function(){ var t=new Date(); return t.getHours()*60+t.getMinutes(); })();
  var L=[],T=[],S=[],C=[];
  for(var k=n-1;k>=0;k--){
    var hm=_min2hm(anchor-k);
    L.push(hm);
    T.push(hm in mapT ? mapT[hm] : null);
    S.push(hm in mapS ? mapS[hm] : null);
    C.push(hm in mapC ? mapC[hm] : null);
  }
  return {labels:L, temp:T, setpoint:S, co:C};
}

async function updateChart(){
  try{
    var tab=document.getElementById('tab-info');
    if(!tab || !tab.classList.contains('active')) return;   // skip if not on Trends
    var n = trendHistoryLen;                 // window in minutes
    // Persisted, per-controller history for the SELECTED zone (survives restarts
    // and zone switches; always reflects the chosen Time Window).
    var r=await fetch('/api/trend_data?ctrl='+trendPrimaryIdx+'&minutes='+n);
    var d=await r.json();
    if(!tempChart) return;
    var showTemp = (document.getElementById('show-temp')    ||{checked:true}).checked;
    var showSP   = (document.getElementById('show-setpoint')||{checked:true}).checked;
    var showCO   = (document.getElementById('show-co')      ||{checked:false}).checked;

    var w = buildMinuteWindow({labels:d.labels,temp:d.temp,setpoint:d.setpoint,co:[]}, n);
    var labels = w.labels, temp = w.temp, sp = w.setpoint, co = w.co;

    tempChart.data.labels = labels;
    var di=0;
    if(showTemp && tempChart.data.datasets[di]) tempChart.data.datasets[di++].data=temp;
    if(showSP   && tempChart.data.datasets[di]) tempChart.data.datasets[di++].data=sp;
    if(showCO   && tempChart.data.datasets[di]) tempChart.data.datasets[di++].data=co.map(function(v){return v?v/10:null;});
    tempChart.update('none');

    var badge=document.getElementById('trend-ctrl-badge');
    if(badge && deviceConfig[trendPrimaryIdx]) badge.textContent=deviceConfig[trendPrimaryIdx].name;
  }catch(e){}
}

var tempChartFS=null,fsInterval=null;
function openFullscreenChart(){
  document.getElementById('trend-menu').style.display='none';
  document.getElementById('fullscreen-overlay').style.display='flex';
  var dark=document.body.classList.contains('theme-dark');
  var grid=dark?'rgba(148,163,184,0.12)':'rgba(255,255,255,0.08)';
  if(!tempChartFS){
    var ctx=document.getElementById('tempChartFS').getContext('2d');
    tempChartFS=new Chart(ctx,{type:'line',data:{labels:[],datasets:[]},
      options:{responsive:true,maintainAspectRatio:false,animation:{duration:0},
        interaction:{mode:'index',intersect:false},
        scales:{
          y:{beginAtZero:false,grace:'8%',grid:{color:grid,drawBorder:false},
            ticks:{color:'#94a3b8',font:{size:13},padding:8,callback:function(v){return v+'°';}},
            title:{display:true,text:'Temperature (°C)',color:'#94a3b8',font:{size:13,weight:'600'}}},
          x:{grid:{display:false,drawBorder:false},
            ticks:{color:'#94a3b8',maxTicksLimit:10,font:{size:11},maxRotation:0,autoSkip:true}}
        },
        plugins:{
          legend:{position:'top',align:'end',labels:{color:'#e2e8f0',boxWidth:10,boxHeight:10,
            usePointStyle:true,pointStyle:'circle',font:{size:13,weight:'600'},padding:18}},
          tooltip:{enabled:true,backgroundColor:'#0e0e12',titleColor:'#fff',bodyColor:'#cbd5e1',
            borderColor:'#3b82f6',borderWidth:1,padding:12,cornerRadius:8,
            callbacks:{label:function(c){return ' '+c.dataset.label+': '+(c.parsed.y!=null?c.parsed.y.toFixed(1)+'°C':'—');}}}
        }}});
  }
  async function upFS(){
    try{
      var n=trendHistoryLen;
      var labels, w, d;
      if(trendPrimaryIdx==='all'){
        var r=await fetch('/api/history_minute'); d=await r.json();
        w=buildMinuteWindow(d, n);
      } else {
        // Persisted per-controller history for the selected zone.
        var r=await fetch('/api/trend_data?ctrl='+trendPrimaryIdx+'&minutes='+n); d=await r.json();
        w=buildMinuteWindow({labels:d.labels,temp:d.temp,setpoint:d.setpoint,co:[]}, n);
      }
      labels=w.labels;
      var allVals=[];
      var ds=[];
      if(trendPrimaryIdx==='all' && d.multi){
      var COLORS=['#ef4444','#3b82f6','#22c55e','#f59e0b','#a78bfa','#06b6d4','#ec4899','#84cc16','#f97316','#14b8a6'];
      var ds=[]; var ci=0;
      Object.keys(d.multi).slice(0,3).forEach(function(k){   // max 3 zones
        var z=d.multi[k], data=z.temp.slice(-n);
        data.forEach(function(v){ if(v!=null) allVals.push(v); });
        ds.push({label:z.name,borderColor:COLORS[ci%COLORS.length],
          backgroundColor:'transparent',data:data,
          borderWidth:2.5,pointRadius:0,fill:false,tension:0.35,type:'line',spanGaps:true});
        ci++;
      });
      tempChartFS.options.scales.y.title.text='Zone Temperature (°C)';
    } else {
        var temp=w.temp, sp=w.setpoint;
        temp.forEach(function(v){ if(v!=null&&v>0) allVals.push(v); });
        sp.forEach(function(v){ if(v!=null&&v>0) allVals.push(v); });
        ds.push({label:'Temperature (°C)',borderColor:'#ef4444',backgroundColor:'rgba(239,68,68,0.10)',
          data:temp,borderWidth:3,pointRadius:0,fill:true,tension:0.35,spanGaps:true});
        ds.push({label:'Setpoint (°C)',borderColor:'#3b82f6',borderDash:[6,4],backgroundColor:'transparent',
          data:sp,borderWidth:2.5,pointRadius:0,fill:false,tension:0.35,spanGaps:true});
      }
      tempChartFS.data.labels=labels;
      tempChartFS.data.datasets=ds;
      if(allVals.length){
        var lo=Math.min.apply(null,allVals),hi=Math.max.apply(null,allVals),span=hi-lo;
        var pad=span<2?1.5:span*0.20;
        tempChartFS.options.scales.y.min=Math.floor(lo-pad);
        tempChartFS.options.scales.y.max=Math.ceil(hi+pad);
      }
      tempChartFS.update('none');
      var t=document.getElementById('fs-title');
      if(t) t.textContent=(trendPrimaryIdx==='all'?'All Zones':(deviceConfig[trendPrimaryIdx]?deviceConfig[trendPrimaryIdx].name:'Live Trend'));
    }catch(e){}
  }
  upFS(); fsInterval=setInterval(upFS,2000);
}
function closeFullscreenChart(){
  document.getElementById('fullscreen-overlay').style.display='none';
  if(fsInterval){clearInterval(fsInterval);fsInterval=null;}
}

// ── Power panel ───────────────────────────────────────────────────────────
function showPowerPanel(){document.getElementById('power-panel').style.display='flex';}
function hidePowerPanel(){document.getElementById('power-panel').style.display='none';}
function executePower(type){
  if(confirm('Are you sure you want to '+type+'?')){
    document.getElementById('power-panel').innerHTML='<h1 style="color:white;">'+(type==='restart'?'Restarting...':'Shutting down...')+'</h1>';
    fetch('/power/'+type,{method:'POST'});
  }
}
function toggleMenu(){
  var m=document.getElementById('trend-menu');
  m.style.display=m.style.display==='block'?'none':'block';
}
document.addEventListener('click',function(e){
  var m=document.getElementById('trend-menu');
  if(m&&!e.target.closest('#trend-menu')&&!e.target.closest('span')) m.style.display='none';
});

// ── Theme ─────────────────────────────────────────────────────────────────
function setTheme(theme){
  if(theme!=='light' && theme!=='dark') theme='light';
  document.body.classList.remove('theme-light','theme-dark','theme-blue');
  document.body.classList.add('theme-'+theme);
  try{ window._savedTheme=theme; }catch(e){}
  ['light','dark'].forEach(function(t){
    var btn=document.getElementById('theme-btn-'+t);
    if(btn) btn.classList.toggle('active',t===theme);
  });
  Object.keys(knobs).forEach(function(k){ if(knobs[k]) knobs[k].redraw(); });
  // Re-theme the trend chart (grid / ticks / legend follow the palette)
  try{
    if(typeof tempChart!=='undefined' && tempChart){
      tempChart.destroy(); tempChart=null;
      initChart(); updateChart();
    }
  }catch(e){}
}
function initTheme(){ setTheme(window._savedTheme||'light'); }

// ── Weather ───────────────────────────────────────────────────────────────
var WX_ICONS={0:'🌞',1:'🌤',2:'⛅',3:'⛅',45:'🌫',48:'🌫',51:'🌦',53:'🌦',55:'🌦',61:'🌧',63:'🌧',65:'🌧',71:'🌨',73:'🌨',75:'🌨',77:'🌨',80:'🌦',81:'🌧',82:'⛈',85:'🌨',86:'🌨',95:'⛈️',96:'⛈️',99:'⛈️'};

function startWeatherPoll(){ fetchWeather(); setInterval(fetchWeather,600000); }
function fetchWeather(){
  fetch('/api/weather').then(r=>r.json()).then(d=>{
    if(d.temp!=null){
      var icon=WX_ICONS[d.code]||'☁';
      var el1=document.getElementById('wx-icon');
      var el2=document.getElementById('wx-temp');
      if(el1) el1.textContent=icon;
      if(el2) el2.textContent=Math.round(d.temp)+'°';
      lastOutsideTemp = d.temp;
      updateOutsideTempTiles();
    } else { setTimeout(fetchWeather,30000); }
  }).catch(function(){ setTimeout(fetchWeather,30000); });
}
window.addEventListener('load',function(){
  setTimeout(function(){
    try{ initTheme(); }catch(e){}
    try{ startWeatherPoll(); }catch(e){ setTimeout(startWeatherPoll,3000); }
  },100);
});
</script>

<!-- ════════════════════════════════════════════════════════════════
     CONFIG SETUP WIZARD  —  paste this block just before </body>
     (after the existing config tab markup). Self-contained.
     ════════════════════════════════════════════════════════════════ -->
<style>
/* ── Wizard overlay ── */
#wiz-overlay{
  display:none; position:fixed; inset:0; z-index:9500;
  background:linear-gradient(160deg,#0f172a 0%,#1e293b 100%);
  color:#e2e8f0; overflow-y:auto;
  animation:wizFade .35s ease;
}
@keyframes wizFade{from{opacity:0;}to{opacity:1;}}
.wiz-shell{max-width:520px;margin:0 auto;padding:18px 16px 40px;}
/* progress */
.wiz-progress{display:flex;align-items:center;gap:6px;margin:6px 0 20px;}
.wiz-step-dot{flex:1;height:6px;border-radius:3px;background:#334155;transition:background .4s ease;}
.wiz-step-dot.done{background:#22c55e;}
.wiz-step-dot.active{background:#3b82f6;box-shadow:0 0 10px 2px rgba(59,130,246,.5);}
.wiz-step-label{font-size:11px;color:#64748b;letter-spacing:2px;text-transform:uppercase;margin-bottom:4px;}
.wiz-title{font-size:22px;font-weight:800;color:#fff;margin-bottom:4px;}
.wiz-sub{font-size:13px;color:#94a3b8;margin-bottom:20px;line-height:1.5;}
/* step panel slide */
.wiz-step{display:none;animation:wizSlide .4s cubic-bezier(.2,.8,.2,1);}
.wiz-step.active{display:block;}
@keyframes wizSlide{from{opacity:0;transform:translateX(30px);}to{opacity:1;transform:translateX(0);}}
/* type picker cards */
.wiz-type-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-bottom:10px;}
.wiz-type-card{
  background:#1e293b;border:2px solid #334155;border-radius:14px;
  padding:18px 12px;text-align:center;cursor:pointer;
  transition:transform .2s,border-color .2s,background .2s;
  -webkit-tap-highlight-color:transparent;
}
.wiz-type-card:active{transform:scale(.95);}
.wiz-type-card.sel{border-color:#3b82f6;background:#1e3a5f;box-shadow:0 0 16px 2px rgba(59,130,246,.35);transform:scale(1.03);}
.wiz-type-icon{font-size:38px;line-height:1;margin-bottom:8px;}
.wiz-type-name{font-size:14px;font-weight:700;color:#e2e8f0;}
.wiz-type-desc{font-size:10px;color:#64748b;margin-top:3px;}
/* fields */
.wiz-field{margin-bottom:16px;}
.wiz-field label{display:block;font-size:11px;color:#94a3b8;text-transform:uppercase;letter-spacing:1px;margin-bottom:6px;}
.wiz-field input{
  width:100%;padding:13px 14px;border:2px solid #334155;border-radius:10px;
  background:#0f172a;color:#e2e8f0;font-size:16px;transition:border-color .2s;
}
.wiz-field input:focus{outline:none;border-color:#3b82f6;}
/* test result chip */
.wiz-test-chip{
  display:inline-flex;align-items:center;gap:6px;font-size:13px;font-weight:700;
  padding:8px 14px;border-radius:20px;margin-top:6px;min-height:20px;transition:all .3s;
}
.wiz-test-chip.testing{background:rgba(245,158,11,.15);color:#fbbf24;}
.wiz-test-chip.ok{background:rgba(34,197,94,.15);color:#4ade80;}
.wiz-test-chip.err{background:rgba(239,68,68,.15);color:#f87171;}
.wiz-spin{width:14px;height:14px;border:2px solid currentColor;border-top-color:transparent;border-radius:50%;animation:wizRot .7s linear infinite;display:inline-block;}
@keyframes wizRot{to{transform:rotate(360deg);}}
/* nav buttons */
.wiz-nav{display:flex;gap:10px;margin-top:26px;}
.wiz-btn{flex:1;padding:15px;border:none;border-radius:12px;font-size:15px;font-weight:800;cursor:pointer;transition:transform .15s,opacity .2s;-webkit-tap-highlight-color:transparent;position:relative;overflow:hidden;}
.wiz-btn:active{transform:scale(.96);}
.wiz-btn.primary{background:linear-gradient(135deg,#3b82f6,#2563eb);color:#fff;}
.wiz-btn.ghost{background:#334155;color:#cbd5e1;flex:0 0 90px;}
.wiz-btn:disabled{opacity:.4;cursor:not-allowed;}
/* ripple */
.wiz-ripple{position:absolute;border-radius:50%;background:rgba(255,255,255,.45);transform:scale(0);animation:wizRippleAnim .55s ease-out;pointer-events:none;}
@keyframes wizRippleAnim{to{transform:scale(4);opacity:0;}}
/* summary list */
.wiz-sum-row{display:flex;align-items:center;flex-wrap:wrap;gap:12px;background:#1e293b;border:1px solid #334155;border-radius:12px;padding:14px;margin-bottom:10px;animation:wizSlide .4s ease backwards;}
.wiz-sum-row:nth-child(2){animation-delay:.08s;}
.wiz-sum-row:nth-child(3){animation-delay:.16s;}
.wiz-sum-icon{font-size:28px;}
.wiz-sum-info{flex:1;}
.wiz-sum-name{font-size:15px;font-weight:700;color:#fff;}
.wiz-sum-meta{font-size:11px;color:#64748b;margin-top:2px;}
.wiz-sum-status{font-size:11px;font-weight:700;padding:4px 10px;border-radius:12px;}
.wiz-sum-status.ok{background:rgba(34,197,94,.15);color:#4ade80;}
.wiz-sum-status.err{background:rgba(239,68,68,.15);color:#f87171;}
/* big success check */
.wiz-success-check{width:90px;height:90px;margin:10px auto 18px;border-radius:50%;background:rgba(34,197,94,.12);display:flex;align-items:center;justify-content:center;animation:wizPop .5s cubic-bezier(.2,1.4,.4,1);}
.wiz-success-check span{font-size:48px;color:#22c55e;}
@keyframes wizPop{from{transform:scale(0);}to{transform:scale(1);}}
/* launcher button on config page */
.wiz-launch-btn{width:100%;padding:14px;margin-bottom:14px;border:none;border-radius:12px;
  background:linear-gradient(135deg,#7c3aed,#6d28d9);color:#fff;font-size:15px;font-weight:800;
  cursor:pointer;display:flex;align-items:center;justify-content:center;gap:8px;transition:transform .15s;}
.wiz-launch-btn:active{transform:scale(.97);}
</style>

<div id="wiz-overlay">
  <div class="wiz-shell">
    <div class="wiz-step-label" id="wiz-step-counter">STEP 1 OF 4</div>
    <div class="wiz-progress">
      <div class="wiz-step-dot active" id="wiz-dot-0"></div>
      <div class="wiz-step-dot" id="wiz-dot-1"></div>
      <div class="wiz-step-dot" id="wiz-dot-2"></div>
      <div class="wiz-step-dot" id="wiz-dot-3"></div>
    </div>

    <!-- STEP 0 — welcome / serial -->
    <div class="wiz-step active" id="wiz-step-0">
      <div class="wiz-title"> Welcome — Let's set up</div>
      <div class="wiz-field">
        <button class="wiz-btn primary" style="width:100%;padding:14px;font-size:15px;" onclick="wizAutoDetect()">🔍 Auto-detect Connection</button>
        <div id="wiz-detect-result" style="font-size:12px;margin-top:10px;min-height:16px;text-align:center;color:#4ade80;font-weight:700;"></div>
      </div>
      <!-- hidden fields hold the detected values -->
      <input type="hidden" id="wiz-port" value="ENVI CENTRAL"/>
      <input type="hidden" id="wiz-baud" value="9600"/>
      <div class="wiz-nav">
        <button class="wiz-btn ghost" onclick="wizClose()">Cancel</button>
        <button class="wiz-btn primary" onclick="wizNext()">Start →</button>
      </div>
    </div>

    <!-- STEP 1 — controller type + details (repeats per controller) -->
    <div class="wiz-step" id="wiz-step-1">
      <div class="wiz-title" id="wiz-ctrl-title">Controller 1</div>
      <div class="wiz-sub">Pick the type and set its details.</div>
      <div class="wiz-type-grid" id="wiz-type-grid"></div>
      <div class="wiz-field">
        <label>Room / Zone Name</label>
        <input type="text" id="wiz-name" placeholder="e.g. Meeting Room"/>
      </div>
      <div class="wiz-field">
        <label>Modbus Slave ID</label>
        <input type="number" id="wiz-sid" min="1" max="247" value="1"/>
      </div>
      <div class="wiz-nav">
        <button class="wiz-btn ghost" onclick="wizPrev()">← Back</button>
        <button class="wiz-btn primary" onclick="wizNext()">Next →</button>
      </div>
    </div>

    <!-- STEP 2 — test all -->
    <div class="wiz-step" id="wiz-step-2">
      <div class="wiz-title">🔌 Testing connections</div>
      <div class="wiz-sub">Checking each controller responds on the bus…</div>
      <div id="wiz-test-list"></div>
      <div class="wiz-nav">
        <button class="wiz-btn ghost" onclick="wizPrev()">← Back</button>
        <button class="wiz-btn primary" id="wiz-test-next" onclick="wizNext()">Continue →</button>
      </div>
    </div>

    <!-- STEP 3 — done -->
    <div class="wiz-step" id="wiz-step-3">
      <div class="wiz-success-check"><span>✓</span></div>
      <div class="wiz-title" style="text-align:center;">All set!</div>
      <div class="wiz-sub" style="text-align:center;">Your controllers are configured and ready.</div>
      <div id="wiz-summary"></div>
      <div class="wiz-nav">
        <button class="wiz-btn primary" onclick="wizFinish()">Open Dashboard →</button>
      </div>
    </div>
  </div>
</div>

<script>

var MAX_CONTROLLERS = 10;
var ALL_REG_NAMES = __ALL_REG_NAMES__;
// ════════════ CONFIG WIZARD ════════════
var WIZ_TYPES = [
  {id:'Temperzone EcoNEX PRO', icon:'🟩', desc:'Ducted'},
  {id:'Temperzone UC8',    icon:'🟧', desc:'Gateway'},
  {id:'Vector',     icon:'🟦', desc:'VFC series'},
  {id:'Daikin',     icon:'🟦', desc:'Split / VRV'},
];
// Shared type list used by BOTH the wizard and the advanced config, so a type
// chosen in one screen is always valid/selectable in the other.
window.CTRL_TYPES = WIZ_TYPES.map(function(t){ return t.id; });
var WIZ_TYPE_ICON = {'Temperzone EcoNEX PRO':'🟩','Temperzone UC8':'🟧','Vector':'🟦','Daikin':'❄️','Temperzone':'🟩','Other':'⚙️'};
var wizStep = 0;
var wizCtrlIdx = 0;          // which controller we're editing in step 1
var wizDraft = [];          // working copy of deviceConfig
var WIZ_PORTS = ['/dev/ttyS7','/dev/ttyS1'];
var WIZ_BAUDS = ['9600','19200','38400'];

function wizRipple(e){
  var b=e.currentTarget, r=document.createElement('span');
  r.className='wiz-ripple';
  var d=Math.max(b.clientWidth,b.clientHeight);
  r.style.width=r.style.height=d+'px';
  var rect=b.getBoundingClientRect();
  r.style.left=(e.clientX-rect.left-d/2)+'px';
  r.style.top=(e.clientY-rect.top-d/2)+'px';
  b.appendChild(r); setTimeout(function(){r.remove();},550);
}
document.addEventListener('DOMContentLoaded',function(){
  document.querySelectorAll('.wiz-btn').forEach(function(b){ b.addEventListener('click',wizRipple); });
});

function wizOpen(){
  // clone current config as the draft
  wizDraft = JSON.parse(JSON.stringify(deviceConfig || []));
  if(!wizDraft.length){ wizDraft=[{name:'Temperzone',type:'Temperzone EcoNEX PRO',slave_id:20,enabled:true}]; }
  wizStep=0; wizCtrlIdx=0;
  document.getElementById('wiz-overlay').style.display='block';
  wizShowStep(0);
}
function wizClose(){ document.getElementById('wiz-overlay').style.display='none'; }

function wizShowStep(n){
  wizStep=n;
  for(var i=0;i<4;i++){
    var s=document.getElementById('wiz-step-'+i);
    if(s) s.classList.toggle('active', i===n);
    var dot=document.getElementById('wiz-dot-'+i);
    if(dot){ dot.classList.toggle('active', i===n); dot.classList.toggle('done', i<n); }
  }
  document.getElementById('wiz-step-counter').textContent='STEP '+(n+1)+' OF 4';
  if(n===1) wizRenderCtrlStep();
  if(n===2) wizRunTests();
  if(n===3) wizRenderSummary();
}

function wizCyclePort(){
  var el=document.getElementById('wiz-port');
  var i=(WIZ_PORTS.indexOf(el.value)+1)%WIZ_PORTS.length;
  el.value=WIZ_PORTS[i];
}
async function wizAutoDetect(){
  var res=document.getElementById('wiz-detect-result');
  res.style.color='#fbbf24';
  res.innerHTML='<span class="wiz-spin"></span> Scanning ports…';
  try{
    var r=await fetch('/api/detect_port',{method:'POST'});
    var d=await r.json();
    if(d.found){
      document.getElementById('wiz-port').value=d.port;
      document.getElementById('wiz-baud').value=String(d.baud);
      res.style.color='#4ade80';
      res.textContent='✓ Found controller on '+d.port+' @ '+d.baud;
    } else {
      res.style.color='#f87171';
      res.textContent='✗ No controller detected — set port manually';
    }
  }catch(e){
    res.style.color='#f87171';
    res.textContent='✗ Detection failed';
  }
}

function wizCycleBaud(){
  var el=document.getElementById('wiz-baud');
  var i=(WIZ_BAUDS.indexOf(el.value)+1)%WIZ_BAUDS.length;
  el.value=WIZ_BAUDS[i];
}

function wizRenderCtrlStep(){
  var ctrl=wizDraft[wizCtrlIdx];
  document.getElementById('wiz-ctrl-title').textContent=(ctrl.name||('Controller '+(wizCtrlIdx+1)))+'  ('+(wizCtrlIdx+1)+'/'+wizDraft.length+')';
  // type cards
  var html='';
  WIZ_TYPES.forEach(function(t){
    var sel=ctrl.type===t.id?' sel':'';
    html+='<div class="wiz-type-card'+sel+'" onclick="wizPickType(\''+t.id+'\')">';
    html+='<div class="wiz-type-icon">'+t.icon+'</div>';
    html+='<div class="wiz-type-name">'+t.id+'</div>';
    html+='<div class="wiz-type-desc">'+t.desc+'</div>';
    html+='</div>';
  });
  document.getElementById('wiz-type-grid').innerHTML=html;
  document.getElementById('wiz-name').value=ctrl.name||'';
  document.getElementById('wiz-sid').value=ctrl.slave_id||1;
}
function wizPickType(t){
  wizDraft[wizCtrlIdx].type=t;
  document.querySelectorAll('.wiz-type-card').forEach(function(c){
    c.classList.toggle('sel', c.querySelector('.wiz-type-name').textContent===t);
  });
}
function wizSaveCtrlStep(){
  wizDraft[wizCtrlIdx].name=(document.getElementById('wiz-name').value||('Controller '+(wizCtrlIdx+1)));
  wizDraft[wizCtrlIdx].slave_id=parseInt(document.getElementById('wiz-sid').value)||1;
  wizDraft[wizCtrlIdx].enabled=true;
}

async function wizRunTests(){
  // persist draft first so /api/test_connection uses it
  await fetch('/api/device_config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(wizDraft)});
  var html='';
  wizDraft.forEach(function(ctrl,idx){
    html+='<div class="wiz-sum-row">';
    html+='<div class="wiz-sum-icon">'+(WIZ_TYPE_ICON[ctrl.type]||'⚙️')+'</div>';
    html+='<div class="wiz-sum-info"><div class="wiz-sum-name">'+ctrl.name+'</div>';
    html+='<div class="wiz-sum-meta">'+ctrl.type+' · ID '+ctrl.slave_id+'</div></div>';
    html+='<div class="wiz-test-chip testing" id="wiz-chip-'+idx+'"><span class="wiz-spin"></span>Testing</div>';
    html+='</div>';
  });
  document.getElementById('wiz-test-list').innerHTML=html;
  // test each sequentially
  for(var i=0;i<wizDraft.length;i++){
    var chip=document.getElementById('wiz-chip-'+i);
    try{
      var r=await fetch('/api/test_connection',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ctrl:i})});
      var d=await r.json();
      if(d.connected){
        chip.className='wiz-test-chip ok'; chip.textContent='✓ Online';
        wizDraft[i].enabled=true;
      } else {
        chip.className='wiz-test-chip err'; chip.textContent='✗ No reply';
        wizShowSkip(i);
      }
    }catch(e){
      chip.className='wiz-test-chip err'; chip.textContent='✗ Error';
      wizShowSkip(i);
    }
  }
}

function wizShowSkip(idx){
  // add a Skip / Retry control row under the failed controller's chip
  var row=document.getElementById('wiz-chip-'+idx);
  if(!row) return;
  var parent=row.closest('.wiz-sum-row');
  if(!parent || document.getElementById('wiz-skip-'+idx)) return;
  var bar=document.createElement('div');
  bar.id='wiz-skip-'+idx;
  bar.style.cssText='flex-basis:100%;display:flex;gap:8px;margin-top:8px;';
  bar.innerHTML=
    '<button onclick="wizSkipCtrl('+idx+')" style="flex:1;padding:9px;border:none;border-radius:8px;background:#475569;color:#e2e8f0;font-size:12px;font-weight:700;cursor:pointer;">⤳ Skip for now</button>'+
    '<button onclick="wizRetryCtrl('+idx+')" style="flex:1;padding:9px;border:none;border-radius:8px;background:#3b82f6;color:#fff;font-size:12px;font-weight:700;cursor:pointer;">↻ Retry</button>';
  parent.appendChild(bar);
}

function wizSkipCtrl(idx){
  // disable this controller so it won't block — keeps its config but inactive
  wizDraft[idx].enabled=false;
  var chip=document.getElementById('wiz-chip-'+idx);
  if(chip){ chip.className='wiz-test-chip'; chip.style.background='rgba(100,116,139,.18)'; chip.style.color='#94a3b8'; chip.textContent='⤳ Skipped'; }
  var bar=document.getElementById('wiz-skip-'+idx);
  if(bar) bar.remove();
}

async function wizRetryCtrl(idx){
  var chip=document.getElementById('wiz-chip-'+idx);
  var bar=document.getElementById('wiz-skip-'+idx);
  if(bar) bar.remove();
  if(chip){ chip.className='wiz-test-chip testing'; chip.innerHTML='<span class="wiz-spin"></span>Testing'; }
  try{
    var r=await fetch('/api/test_connection',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({ctrl:idx})});
    var d=await r.json();
    if(d.connected){ chip.className='wiz-test-chip ok'; chip.textContent='✓ Online'; wizDraft[idx].enabled=true; }
    else{ chip.className='wiz-test-chip err'; chip.textContent='✗ No reply'; wizShowSkip(idx); }
  }catch(e){ chip.className='wiz-test-chip err'; chip.textContent='✗ Error'; wizShowSkip(idx); }
}

function wizRenderSummary(){
  var html='';
  wizDraft.forEach(function(ctrl){
    html+='<div class="wiz-sum-row">';
    html+='<div class="wiz-sum-icon">'+(WIZ_TYPE_ICON[ctrl.type]||'⚙️')+'</div>';
    html+='<div class="wiz-sum-info"><div class="wiz-sum-name">'+ctrl.name+'</div>';
    html+='<div class="wiz-sum-meta">'+ctrl.type+' · Slave ID '+ctrl.slave_id+'</div></div>';
    html+='</div>';
  });
  document.getElementById('wiz-summary').innerHTML=html;
}

function wizNext(){
  if(wizStep===0){
    // apply serial settings
    var port=document.getElementById('wiz-port').value;
    var baud=parseInt(document.getElementById('wiz-baud').value);
    fetch('/api/serial_config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({port:port,baud:baud})}).catch(function(){});
    wizCtrlIdx=0;
    wizShowStep(1);
  } else if(wizStep===1){
    wizSaveCtrlStep();
    if(wizCtrlIdx < wizDraft.length-1){
      wizCtrlIdx++;
      wizRenderCtrlStep();           // next controller, stay on step 1
      // little slide refresh
      var s=document.getElementById('wiz-step-1');
      s.style.animation='none'; void s.offsetWidth; s.style.animation='';
    } else {
      wizShowStep(2);
    }
  } else if(wizStep===2){
    wizShowStep(3);
  }
}
function wizPrev(){
  if(wizStep===1){
    if(wizCtrlIdx>0){ wizSaveCtrlStep(); wizCtrlIdx--; wizRenderCtrlStep(); }
    else wizShowStep(0);
  } else if(wizStep===2){
    wizCtrlIdx=wizDraft.length-1; wizShowStep(1);
  }
}

async function wizFinish(){
  // Persist the wizard draft, then re-read the authoritative copy from the
  // server so the wizard, advanced config and dashboard all agree.
  await fetch('/api/device_config',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(wizDraft)});
  wizClose();
  if(typeof loadConfigIntoForm==='function'){ try{ await loadConfigIntoForm(); }catch(e){ deviceConfig = wizDraft; } }
  else { deviceConfig = wizDraft; }
  if(typeof unlockTabs==='function') unlockTabs();
  if(typeof buildAll==='function') buildAll();
  if(typeof showTab==='function') showTab('dashboard');
}
</script>

</body>
</html>"""


# ── Flask routes ──────────────────────────────────────────────────────────────
led_ctrl_idx = 0
led_mode     = 'temp'   # 'off' | 'temp' | 'iaq'

@app.route('/api/led_ctrl', methods=['POST'])
def set_led_ctrl():
    global led_ctrl_idx, led_mode
    data = request.get_json(silent=True) or {}
    led_ctrl_idx = int(data.get('ctrl', led_ctrl_idx))
    if 'mode' in data:
        led_mode = data['mode']  # 'off' | 'temp' | 'iaq'
    return jsonify({'status':'ok','ctrl':led_ctrl_idx,'mode':led_mode})

@app.route('/api/motion_config', methods=['POST'])
def set_motion_config():
    global SCREEN_TIMEOUT, motion_enabled
    data = request.get_json(silent=True) or {}
    if 'timeout' in data:
        SCREEN_TIMEOUT = float(data['timeout'])
    if 'enabled' in data:
        motion_enabled = bool(data['enabled'])
    return jsonify({'status':'ok','timeout':SCREEN_TIMEOUT,'enabled':motion_enabled})

@app.route('/api/trend_config', methods=['GET','POST'])
def trend_config():
    global EMAIL_HISTORY_MINUTES, TREND_PRIMARY_CTRL, DISPLAY_HISTORY_LEN
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        if 'email_minutes' in data:
            EMAIL_HISTORY_MINUTES = int(data['email_minutes'])
        if 'history_len' in data:
            DISPLAY_HISTORY_LEN = int(data['history_len'])
        if 'primary_ctrl' in data:
            new_primary = int(data['primary_ctrl'])
            if new_primary != TREND_PRIMARY_CTRL:
                TREND_PRIMARY_CTRL = new_primary
                # Zone switched: wipe the primary trend buffers so the chart drops
                # the previous controller's line entirely and rebuilds cleanly from
                # the newly-selected zone's data (old data first, then new).
                with history_lock:
                    time_history.clear()
                    temp_history.clear()
                    setpoint_history.clear()
                    minute_time_history.clear()
                    minute_temp_history.clear()
                    minute_setpoint_history.clear()
                    _minute_accum["key"]   = None
                    _minute_accum["temp"]  = []
                    _minute_accum["sp"]    = []
                    _minute_accum["multi"] = {}
        return jsonify({'status':'ok','email_minutes':EMAIL_HISTORY_MINUTES,
                        'history_len':DISPLAY_HISTORY_LEN,'primary_ctrl':TREND_PRIMARY_CTRL})
    return jsonify({'email_minutes':EMAIL_HISTORY_MINUTES,
                    'history_len':DISPLAY_HISTORY_LEN,'primary_ctrl':TREND_PRIMARY_CTRL})

@app.route('/api/detect_port', methods=['POST'])
def detect_port():
    """Scan available serial ports + common bauds for any responding controller."""
    cfg = load_device_config()
    slave_ids = [c['slave_id'] for c in cfg] or [1]
    test_reg = 1     # Temperzone Input Register 1 (temperature)
    for port in ['/dev/ttyS7', '/dev/ttyS1', '/dev/ttyS3', '/dev/ttyS9']:
        for baud in [9600, 19200, 38400]:
            try:
                tc = ModbusClient(method='rtu', port=port, baudrate=baud,
                                  parity='N', stopbits=1, bytesize=8, timeout=0.3)
                if not tc.connect():
                    continue
                for sid in slave_ids:
                    try:
                        r = tc.read_holding_registers(address=test_reg, count=1, unit=sid)
                        if not r.isError():
                            tc.close()
                            return jsonify({'found': True, 'port': port, 'baud': baud, 'slave': sid})
                    except Exception:
                        pass
                tc.close()
            except Exception:
                pass
    return jsonify({'found': False})

_onboard_proc = None

@app.route('/api/keyboard', methods=['POST'])
def toggle_keyboard():
    global _onboard_proc
    data = request.get_json(silent=True) or {}
    action = data.get('action', 'show')
    env = get_display_env()
    try:
        if action == 'show':
            if _onboard_proc is None or _onboard_proc.poll() is not None:
                _onboard_proc = subprocess.Popen(
                    ['onboard', '--size=800x250', '--layout=Phone',
                     '--theme=Nightshade', '--xid'],
                    env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
        else:
            if _onboard_proc and _onboard_proc.poll() is None:
                _onboard_proc.terminate()
                _onboard_proc = None
    except Exception as e:
        print(f"[KB] onboard error: {e}", flush=True)
    return jsonify({'status': 'ok'})

@app.route('/home/linaro/<path:filename>')
def serve_linaro(filename):
    return send_from_directory('/home/linaro', filename)

@app.route('/api/device_config', methods=['GET','POST'])
def handle_device_config():
    if request.method == 'POST':
        save_device_config(request.json)
        return jsonify({"status":"saved"})
    return jsonify(load_device_config())

@app.route('/api/device_config/reset', methods=['POST'])
def reset_device_config():
    save_device_config([c.copy() for c in DEFAULT_DEVICE_CONFIG])
    return jsonify({"status":"reset"})



@app.route('/api/controller/add', methods=['POST'])
def add_controller():
    global all_ctrl_data, all_ctrl_status, all_ctrl_connected
    cfg = load_device_config()
    if len(cfg) >= MAX_CONTROLLERS:
        return jsonify({"status": "error", "msg": f"Maximum {MAX_CONTROLLERS} controllers allowed"}), 400
    new_idx = len(cfg)
    ctype = "Temperzone EcoNEX PRO"
    new_ctrl = {"name": f"Temperzone {new_idx + 1}", "type": ctype,
                "slave_id": next_slave_id(cfg, ctype), "enabled": True}
    cfg.append(new_ctrl)
    save_device_config(cfg)
    sched = load_schedule()
    sched[str(new_idx)] = make_empty_week()
    save_schedule(sched)
    ensure_ctrl_lists(len(cfg))
    return jsonify({"status": "added", "idx": new_idx, "config": new_ctrl})

@app.route('/api/controller/delete/<int:idx>', methods=['POST'])
def delete_controller(idx):
    global all_ctrl_data, all_ctrl_status, all_ctrl_connected
    cfg = load_device_config()
    if idx < 0 or idx >= len(cfg):
        return jsonify({"status": "error", "msg": "Invalid index"}), 400
    if len(cfg) <= 1:
        return jsonify({"status": "error", "msg": "Cannot delete the last controller"}), 400
    cfg.pop(idx)
    with lock:
        if idx < len(all_ctrl_data):      all_ctrl_data.pop(idx)
        if idx < len(all_ctrl_status):    all_ctrl_status.pop(idx)
        if idx < len(all_ctrl_connected): all_ctrl_connected.pop(idx)
    sched = load_schedule()
    new_sched = {}
    for i in range(len(cfg)):
        old_key = str(i if i < idx else i + 1)
        new_sched[str(i)] = sched.get(old_key, make_empty_week())
    save_schedule(new_sched)
    save_device_config(cfg)
    return jsonify({"status": "deleted", "remaining": len(cfg)})

@app.route('/api/serial_config', methods=['POST'])
def handle_serial_config():
    global client, PORT, BAUDRATE, runtime_serial
    data     = request.json
    new_port = data.get('port', PORT)
    new_baud = int(data.get('baud', BAUDRATE))
    try:
        with lock:
            if client: client.close(); client = None
        PORT = new_port; BAUDRATE = new_baud
        runtime_serial = {"port": new_port, "baud": new_baud}
        if connect_modbus():
            return jsonify({"status":"ok","msg":f"Connected to {new_port} @ {new_baud}"})
        else:
            return jsonify({"status":"error","msg":f"Failed to open {new_port}"})
    except Exception as e:
        return jsonify({"status":"error","msg":str(e)})

@app.route('/api/test_connection', methods=['POST'])
def test_connection():
    ctrl_idx = int(request.json.get('ctrl', 0))
    cfg      = load_device_config()
    if ctrl_idx >= len(cfg):
        return jsonify({'connected': False, 'msg': 'Invalid controller index'})
    roles    = roles_for(cfg[ctrl_idx])
    sc       = roles.get('scale', 1)
    slave_id = cfg[ctrl_idx]['slave_id']
    with lock: c = client
    if c is None:
        if not connect_modbus():
            return jsonify({'connected': False, 'msg': f'Cannot open {PORT}'})
        with lock: c = client
    try:
        raw = read_field(c, roles, 'temp', slave_id)   # honours temp_rtype (input/holding/coil)
        if raw is None:
            return jsonify({'connected': False, 'msg': f'No response from slave {slave_id}'})
        temp_val = raw * sc
        on_raw   = read_field(c, roles, 'onoff', slave_id)
        onoff    = "ON" if on_raw == 1 else "OFF"
        return jsonify({'connected': True, 'msg': f'Temp: {temp_val:.1f}°C  Status: {onoff}', 'temp': temp_val})
    except Exception as e:
        return jsonify({'connected': False, 'msg': str(e)})

@app.route('/api/schedule', methods=['GET','POST'])
def handle_schedule():
    ctrl_idx = str(request.args.get('ctrl', '0'))
    sched    = load_schedule()
    if request.method == 'POST':
        sched[ctrl_idx] = request.json
        save_schedule(sched)
        return jsonify({"status":"saved"})
    return jsonify(sched.get(ctrl_idx, make_empty_week()))

@app.route('/api/clock')
def api_clock():
    with clock_lock: return jsonify(clock_data)

@app.route('/api/clock_source')
def api_clock_source():
    return jsonify(get_clock_source_cached())

@app.route('/api/timezones')
def api_timezones():
    """Australian state/territory timezones for the picker, plus the current one.
       Only these are offered — the product is deployed in Australia and the
       state list makes the (DST-sensitive) choice unambiguous."""
    au = [
        {"tz": "Australia/Sydney",    "label": "New South Wales / ACT — Sydney"},
        {"tz": "Australia/Melbourne", "label": "Victoria — Melbourne"},
        {"tz": "Australia/Brisbane",  "label": "Queensland — Brisbane"},
        {"tz": "Australia/Adelaide",  "label": "South Australia — Adelaide"},
        {"tz": "Australia/Perth",     "label": "Western Australia — Perth"},
        {"tz": "Australia/Hobart",    "label": "Tasmania — Hobart"},
        {"tz": "Australia/Darwin",    "label": "Northern Territory — Darwin"},
    ]
    au = [z for z in au if _valid_timezone(z["tz"])]   # only zones this system ships
    return jsonify({"timezones": au, "current": get_clock_source().get('tz', '')})

@app.route('/api/set_timezone', methods=['POST'])
def api_set_timezone():
    """Manually set (and lock) the timezone. Changing the zone only changes the
       LOCAL time shown — the underlying UTC/RTC is untouched — and geo-IP will
       no longer override it. Use this to pin e.g. Australia/Brisbane."""
    data = request.get_json(silent=True) or {}
    tz = (data.get('tz') or '').strip()
    if set_system_timezone(tz, manual=True):
        return jsonify({"ok": True, **get_clock_source()})
    return jsonify({"ok": False, "error": "Invalid or unknown timezone"}), 400

@app.route('/api/set_time', methods=['POST'])
def api_set_time():
    """Manually set the system date/time (and persist to the hardware RTC if one
       exists). Turns NTP off first, because timedatectl refuses a manual set
       while NTP is active."""
    data = request.get_json(silent=True) or {}
    try:
        y  = int(data['year']);  mo = int(data['month']); d  = int(data['day'])
        h  = int(data['hour']);  mi = int(data['minute'])
        s  = int(data.get('second', 0))
        # validate by constructing a datetime (rejects e.g. Feb 30)
        _ = datetime(y, mo, d, h, mi, s)
    except Exception:
        return jsonify({'ok': False, 'error': 'Invalid date/time'}), 400
    ts = f"{y:04d}-{mo:02d}-{d:02d} {h:02d}:{mi:02d}:{s:02d}"
    run_priv(['timedatectl', 'set-ntp', 'false'], timeout=8)
    rc, out, err = run_priv(['timedatectl', 'set-time', ts], timeout=10)
    if rc != 0:
        rc, out, err = run_priv(['date', '-s', ts], timeout=10)   # fallback
    if rc != 0:
        return jsonify({'ok': False, 'error': (err or 'Failed to set time').strip()}), 200
    if _has_rtc():
        # Force the RTC to UTC mode and write it. A hardware clock read in the
        # wrong mode (local-time vs UTC) is the classic cause of a clean, fixed
        # N-hour offset; standardising to UTC here fixes it permanently. This
        # does NOT shift the running clock — only how the RTC is stored.
        run_priv(['timedatectl', 'set-local-rtc', '0'], timeout=10)
        run_priv(['hwclock', '-w'], timeout=10)                   # persist to RTC (UTC)
    save_time_backup(source='manual')                            # persist manual time (survives reboot)
    _clock_src_cache['val'] = None                                # force refresh
    try:
        update_clock_data()                                       # reflect new time immediately
    except Exception:
        pass
    return jsonify({'ok': True, 'set': ts, **get_clock_source()})

@app.route('/api/history')
def get_history():
    with history_lock:
        cfg = load_device_config()
        multi = {}
        for ci in range(len(cfg)):
            if ci in multi_temp_history:
                multi[str(ci)] = {"name": cfg[ci]["name"], "temp": list(multi_temp_history[ci])}
        return jsonify({"labels":list(time_history),"temp":list(temp_history),
                        "setpoint":list(setpoint_history),"multi":multi})

@app.route('/api/history_minute')
def get_history_minute():
    """Minute-by-minute trend data for the chart (one averaged point per minute).
       The still-accumulating current minute is appended as a live provisional
       point so the trend edge updates without waiting for the minute to close."""
    with history_lock:
        cfg = load_device_config()
        labels = list(minute_time_history)
        temp   = list(minute_temp_history)
        sp     = list(minute_setpoint_history)
        multi  = {}
        for ci in range(len(cfg)):
            if ci in minute_multi_temp_history:
                multi[str(ci)] = {"name": cfg[ci]["name"],
                                  "temp": list(minute_multi_temp_history[ci])}
        # live provisional point for the minute currently in progress
        acc = _minute_accum
        if acc["key"] is not None and acc["temp"]:
            labels.append(acc["key"])
            temp.append(_minute_avg(acc["temp"]) or 0)
            sp.append(_minute_avg(acc["sp"]) or 0)
            for ci, vals in acc["multi"].items():
                k = str(ci)
                if k in multi:
                    multi[k]["temp"] = multi[k]["temp"] + [_minute_avg(vals)]
        return jsonify({"labels": labels, "temp": temp,
                        "setpoint": sp, "multi": multi})

# ── Persisted per-controller trend (CSV-backed) ────────────────────────────
# The chart is fed from each controller's on-disk minute log (temp_log_ctrlN.csv)
# instead of a volatile in-memory buffer. That means any Time Window (15 min …
# many hours) shows the REAL last-N-minutes for the SELECTED controller, and the
# data survives restarts and zone switches. Reads are cached by file mtime so we
# only re-parse when the log actually changes (~once a minute).
_trend_cache = {}   # ctrl_idx -> {"mtime": float, "rows": [(datetime, temp, sp), ...]}

def _read_ctrl_trend_rows(ctrl, max_rows=4400):
    """Return the last ~max_rows minute-rows for a controller as
       [(datetime, temp, setpoint), ...], sorted oldest→newest. Tail-bounded and
       mtime-cached so it stays cheap on the panel's low-end CPU."""
    csv_file = get_csv_file(ctrl)
    try:
        st = os.stat(csv_file)
    except FileNotFoundError:
        return []
    except Exception:
        return []
    cached = _trend_cache.get(ctrl)
    if cached and cached.get("mtime") == st.st_mtime:
        return cached["rows"]
    rows = []
    try:
        with open(csv_file, newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader, None)                 # skip header
            buf = deque(maxlen=max_rows)       # keep only the tail
            for r in reader:
                buf.append(r)
        for r in buf:
            if len(r) < 3:
                continue
            try:
                dt = datetime.strptime(r[0], "%Y-%m-%d %H:%M:%S")
            except Exception:
                continue
            try:
                t = float(r[1]); s = float(r[2])
            except Exception:
                continue
            rows.append((dt, t, s))
    except Exception as e:
        print(f"[TREND] read error ctrl{ctrl}: {e}", flush=True)
        return cached["rows"] if cached else []
    rows.sort(key=lambda x: x[0])
    _trend_cache[ctrl] = {"mtime": st.st_mtime, "rows": rows}
    return rows

@app.route('/api/trend_data')
def api_trend_data():
    """Last-N-minutes trend for one controller, read from its persisted log.
       Query: ?ctrl=<idx>&minutes=<n>. Returns minute-resolution
       {labels:[HH:MM], temp:[], setpoint:[]} plus a fresh live point for the
       minute in progress."""
    cfg = load_device_config()
    try:
        ctrl = int(request.args.get('ctrl', TREND_PRIMARY_CTRL))
    except Exception:
        ctrl = TREND_PRIMARY_CTRL
    if ctrl < 0 or ctrl >= len(cfg):
        ctrl = 0
    try:
        minutes = int(request.args.get('minutes', 60))
    except Exception:
        minutes = 60
    minutes = max(1, min(minutes, 24 * 60))    # clamp 1 min … 24 h

    rows = _read_ctrl_trend_rows(ctrl)

    # Window = last `minutes`, ending at the newest sample or now (whichever is later).
    now_min = datetime.now().replace(second=0, microsecond=0)
    end_dt  = max(rows[-1][0], now_min) if rows else now_min
    start_dt = end_dt - timedelta(minutes=minutes)
    window = [(dt, t, s) for (dt, t, s) in rows if dt >= start_dt]

    # Fresh live point for the minute in progress (right edge tracks reality even
    # between the once-a-minute CSV writes).
    try:
        if ctrl < len(all_ctrl_data) and ctrl < len(all_ctrl_connected) and all_ctrl_connected[ctrl]:
            roles = roles_for(cfg[ctrl]); scv = roles.get("scale", 1)
            traw = all_ctrl_data[ctrl].get(roles["temp"])
            sraw = all_ctrl_data[ctrl].get(roles["setpoint"])
            if traw is not None and sraw is not None:
                tv = round(traw * scv, 1); sv = round(sraw * scv, 1)
                if window and window[-1][0] == now_min:
                    window[-1] = (now_min, tv, sv)
                else:
                    window.append((now_min, tv, sv))
    except Exception:
        pass

    labels = [dt.strftime("%H:%M") for (dt, t, s) in window]
    temp   = [t for (dt, t, s) in window]
    sp     = [s for (dt, t, s) in window]
    return jsonify({"labels": labels, "temp": temp, "setpoint": sp,
                    "ctrl": ctrl, "minutes": minutes})

@app.route('/api/network')
def api_network():
    return jsonify(get_network_info())

@app.route('/api/wifi/status')
def api_wifi_status():
    ssid = get_active_wifi_ssid()
    return jsonify({'connected': bool(ssid), 'ssid': ssid})

@app.route('/api/wifi/scan')
def api_wifi_scan():
    # Best-effort rescan (needs the radio on; ignore failures), then list cells
    run_nmcli(['radio', 'wifi', 'on'], timeout=8)
    run_nmcli(['device', 'wifi', 'rescan'], timeout=15)
    rc, out, err = run_nmcli(
        ['-t', '-f', 'IN-USE,SSID,SIGNAL,SECURITY', 'device', 'wifi', 'list'],
        timeout=15)
    active_ssid = get_active_wifi_ssid()   # authoritative "connected to" name
    if rc != 0 and not out:
        return jsonify({'ok': False, 'error': (err or 'Wi-Fi scan failed').strip(),
                        'networks': [], 'connected': active_ssid}), 200
    nets = {}
    for line in out.splitlines():
        p = _nmcli_split(line)
        if len(p) < 4:
            continue
        inuse, ssid, signal, security = p[0], p[1], p[2], p[3]
        if not ssid:
            continue  # hidden network
        try:
            sig = int(signal)
        except Exception:
            sig = 0
        active = (inuse.strip() in ('*', 'yes', '*yes')) or (ssid == active_ssid)
        secure = bool(security and security.strip() not in ('', '--'))
        cur = nets.get(ssid)
        if not cur or sig > cur['signal']:
            nets[ssid] = {'ssid': ssid, 'signal': sig, 'secure': secure, 'active': active}
        elif active:
            nets[ssid]['active'] = True
    # Make sure the connected SSID is marked/added even if the scan list lagged
    if active_ssid and active_ssid not in nets:
        nets[active_ssid] = {'ssid': active_ssid, 'signal': 0, 'secure': True, 'active': True}
    lst = sorted(nets.values(), key=lambda n: (0 if n['active'] else 1, -n['signal']))
    return jsonify({'ok': True, 'networks': lst, 'connected': active_ssid})

def _iface_rank(name, prefer="wifi"):
    """Interface preference for the QR/LAN URL.

       For the QR we prefer Wi-Fi: a phone scanning the code is almost always on
       the Wi-Fi network, so the Wi-Fi IPv4 is the address it can actually reach.
       (A wired PC on the Ethernet subnet may reach the Ethernet IP, but the phone
       usually can't — that's why the old Ethernet-first QR 'opened on PC but not
       on the phone'.) Ethernet is kept as the fallback when there's no Wi-Fi."""
    is_eth  = name.startswith(("eth", "en"))    # eth0, enp1s0, eno1, ens33 …
    is_wifi = name.startswith(("wlan", "wl"))   # wlan0, wlp2s0 …
    if prefer == "eth":
        if is_eth:  return 0
        if is_wifi: return 1
        return 2
    # default: Wi-Fi first
    if is_wifi: return 0
    if is_eth:  return 1
    return 2

def get_lan_url(port=5002, prefer="wifi"):
    """Best LAN URL for reaching this panel from a phone/PC.

       IPv4 only (never IPv6 — many phones won't open a bracketed IPv6 URL). By
       default prefers the Wi-Fi address (what a scanning phone is on), falling
       back to Ethernet, then to whatever source IP the kernel would route from."""
    cands = []
    try:
        for iface in get_network_info().get("interfaces", []):
            if iface.get("family") != "inet":          # IPv4 only (skip inet6)
                continue
            name = iface.get("interface", "")
            if name.startswith(("lo", "docker", "br-", "veth", "virbr", "tun", "tap")):
                continue
            addr = (iface.get("address") or "").split('/')[0]
            # 169.254.x.x = link-local (DHCP never answered) — useless in a QR
            if not addr or addr.startswith(("127.", "169.254.")):
                continue
            cands.append((_iface_rank(name, prefer), name, addr))
    except Exception:
        pass
    if cands:
        cands.sort(key=lambda x: (x[0], x[1]))        # preferred kind first, then name
        rank, name, addr = cands[0]
        return f"http://{addr}:{port}/"
    # Nothing enumerable — ask the kernel which source IP it would route from.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        addr = s.getsockname()[0]
        s.close()
        return f"http://{addr}:{port}/"
    except Exception:
        return None

def get_lan_details(port=5002, prefer="wifi"):
    """Same selection as get_lan_url(), but reports which interface was chosen
       and what else was available — useful when the QR shows an unexpected IP."""
    out = {"url": None, "interface": None, "kind": None, "candidates": []}
    try:
        for iface in get_network_info().get("interfaces", []):
            if iface.get("family") != "inet":
                continue
            name = iface.get("interface", "")
            if name.startswith(("lo", "docker", "br-", "veth", "virbr", "tun", "tap")):
                continue
            addr = (iface.get("address") or "").split('/')[0]
            if not addr or addr.startswith(("127.", "169.254.")):
                continue
            is_eth  = name.startswith(("eth", "en"))
            is_wifi = name.startswith(("wlan", "wl"))
            out["candidates"].append({"interface": name, "address": addr,
                                      "kind": ("wifi" if is_wifi else
                                               "ethernet" if is_eth else "other")})
    except Exception:
        pass
    out["candidates"].sort(key=lambda c: (_iface_rank(c["interface"], prefer), c["interface"]))
    if out["candidates"]:
        c = out["candidates"][0]
        out["url"], out["interface"], out["kind"] = f'http://{c["address"]}:{port}/', c["interface"], c["kind"]
    else:
        out["url"] = get_lan_url(port, prefer)
    return out

@app.route('/api/qr')
def api_qr():
    """SVG QR of this panel's LAN URL, for the Quick Start 'Live' step.
       Scanning it opens the same dashboard in a phone browser."""
    url = request.args.get('data') or get_lan_url()
    if not url:
        # No network — return a readable placeholder rather than a broken image.
        svg = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">'
               '<rect width="100" height="100" fill="#f1f5f9"/>'
               '<text x="50" y="46" font-size="9" fill="#94a3b8" text-anchor="middle">No</text>'
               '<text x="50" y="58" font-size="9" fill="#94a3b8" text-anchor="middle">network</text></svg>')
        return Response(svg, mimetype='image/svg+xml')
    try:
        svg = qr_svg(url, ecc='M', quiet=2)
    except Exception as e:
        print(f"[QR] encode failed for {url!r}: {e}", flush=True)
        return Response('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 10 10"/>',
                        mimetype='image/svg+xml')
    resp = Response(svg, mimetype='image/svg+xml')
    resp.headers['Cache-Control'] = 'no-store'
    return resp

@app.route('/api/lan_url')
def api_lan_url():
    return jsonify(get_lan_details())

@app.route('/api/wifi/connect', methods=['POST'])
def api_wifi_connect():
    data = request.get_json(force=True, silent=True) or {}
    ssid = (data.get('ssid') or '').strip()
    password = data.get('password') or ''
    if not ssid:
        return jsonify({'ok': False, 'error': 'No network selected'}), 400
    run_nmcli(['radio', 'wifi', 'on'], timeout=8)
    if password:
        # Clear any stale saved profile so we reconnect fresh with the new key
        run_nmcli(['connection', 'delete', ssid], timeout=10)
        rc, out, err = run_nmcli(
            ['device', 'wifi', 'connect', ssid, 'password', password], timeout=45)
    else:
        # No password: try an existing saved profile first, else open network
        rc, out, err = run_nmcli(['connection', 'up', ssid], timeout=45)
        if rc != 0:
            rc, out, err = run_nmcli(['device', 'wifi', 'connect', ssid], timeout=45)
    if rc == 0:
        sync_time_from_network()   # let NTP correct the clock, then persist to RTC
        return jsonify({'ok': True, 'ssid': ssid})
    msg = (err or out or 'Connection failed').strip()
    msg = msg.splitlines()[-1] if msg else 'Connection failed'
    return jsonify({'ok': False, 'error': msg}), 200

@app.route('/read')
def read():
    cfg = load_device_config()
    controllers = []
    for i in range(len(cfg)):
        data = all_ctrl_data[i] if i < len(all_ctrl_data) else {}
        roles = roles_for(cfg[i])
        sc = roles.get("scale", 1)
        norm = {str(k): v for k, v in data.items()}
        if is_temperzone_type(cfg[i].get("type")):
            # coils/inputs stored under string keys / distinct addresses — no collision
            norm['0']           = data.get("onoff")             # Coil 2 (keyboard on/off = real state)
            tp                  = data.get(1)                    # Input Register 1 (filtered)
            # Dashboard shows a SINGLE setpoint knob. In Heat Only mode that's the heating
            # setpoint (HR102); in Auto/Cool Only/Fan it's the cooling setpoint (HR100) —
            # the two are kept a fixed 2.0°C apart automatically (see /write).
            _sp_raw             = data.get(102) if data.get(117) == 2 else data.get(100)
            norm['4']           = round(_sp_raw * sc) if _sp_raw is not None else None
            norm['5']           = round(tp * sc, 1) if tp is not None else None
            norm['1']           = data.get(117)                  # mode
            norm['2']           = data.get(114)                  # fan
            norm['status_code'] = data.get(135)                  # unit status enum 1..7
            norm['fault']       = data.get("fault_coil")         # Coil 56 alarm
            norm['schedule']    = data.get("sched_enable")       # Coil 22 Enable Scheduler
            norm['22']          = data.get("sched_enable")       # same value, keyed by real register number for the Control tab
        else:
            norm['0'] = data.get(roles['onoff'])
            sp = data.get(roles['setpoint'])
            tp = data.get(roles['temp'])
            norm['4'] = round(sp * sc) if sp is not None else None   # setpoint → whole °C
            norm['5'] = round(tp * sc, 1) if tp is not None else None # temp → °C
            norm['1'] = data.get(roles['mode'])
            if roles['fan']:   norm['2'] = data.get(roles['fan'])
            norm['1055'] = data.get(roles['sched'])
        norm['oat_value'] = data.get('oat_value')
        controllers.append({
            "idx":       i,
            "name":      cfg[i]["name"],
            "slave_id":  cfg[i]["slave_id"],
            "enabled":   cfg[i]["enabled"],
            "connected": all_ctrl_connected[i] if i < len(all_ctrl_connected) else False,
            "status":    all_ctrl_status[i]    if i < len(all_ctrl_status)    else "Unknown",
            "data":      norm,
        })
    connected_count = sum(1 for c in controllers if c["connected"])
    now  = time.monotonic()
    idle = now - last_motion
    with motion_state_lock: ms = dict(motion_state)
    return jsonify({
        'controllers':     controllers,
        'connected_count': connected_count,
        'status':          modbus_status,
        'motion':          'present' if idle < 1 else 'absent',
        'sleep_in':        max(0, int(SCREEN_TIMEOUT - idle)),
        'fault':           fault_flag,
        'motion_state':    ms,
    })

@app.route('/write', methods=['POST'])
def write():
    try:
        ctrl_idx = int(request.json.get('ctrl', 0))
        reg      = int(request.json.get('register'))
        val      = int(request.json.get('value'))
        cfg      = load_device_config()
        if ctrl_idx >= len(cfg):
            return jsonify({'status':'error','message':'Invalid controller'}), 400

        # translate generic register → controller-specific register
        roles = roles_for(cfg[ctrl_idx])
        generic_to_role = {0:'onoff', 4:'setpoint', 5:'temp', 1:'mode', 2:'fan', 22:'sched'}
        role_name = generic_to_role.get(reg)
        if role_name:
            real = roles.get(role_name)
            if real is not None:
                reg = real
        # scale setpoint up for controllers that store ×10
        if role_name == 'setpoint':
            val = int(round(val / roles.get("scale", 1)))   # 25 → 250 for Vector

        slave_id = cfg[ctrl_idx]['slave_id']
        with lock: c = client
        if c is None:
            return jsonify({'status':'error','message':'Not connected'}), 500
        # Temperzone: on/off & fault are COILS, mode/fan/setpoint are holding regs
        if is_temperzone_type(cfg[ctrl_idx].get("type")):
            if role_name == 'onoff':
                # Coil 1 is the writable BMS on/off. (Coil 2 keyboard on/off is READ-ONLY.)
                # Note: this unit currently ignores Coil 1 unless configured for BMS control.
                with lock:
                    r = c.write_coil(1, bool(val), unit=slave_id)          # Coil 1 (BMS on/off, writable)

            elif role_name == 'sched':
                with lock:
                    r = c.write_coil(22, bool(val), unit=slave_id)         # Coil 22
            elif role_name == 'fan':
                with lock:
                    c.write_register(116, 0, unit=slave_id)                # 116=0 → 114 is 0-100%
                    r = c.write_register(114, int(val), unit=slave_id)
            elif role_name == 'mode':
                with lock:
                    r = c.write_register(117, int(val), unit=slave_id)     # 0=Auto 1=Cool 2=Heat 3=Fan
            elif role_name == 'setpoint':
                # Auto Mode keeps BOTH a Heating (HR102) and Cooling (HR100) setpoint live at
                # all times, but the dashboard only ever shows/edits ONE knob. Whichever one the
                # person is adjusting is written as-is; the hidden partner is derived to keep a
                # fixed 2.0°C deadband, so the two setpoints can never collide or invert.
                #
                # Both writes are now done while holding the shared bus `lock` (see poll_modbus)
                # so the poller can't interleave a read in between the two writes on the
                # half-duplex RS485 line — that race was why the hidden setpoint sometimes
                # silently failed to update. A short settle delay is also given between the two
                # writes, and the secondary write's result is now actually checked/retried.
                SP_DEADBAND = 20                     # 2.0°C, in raw x10 register units
                SP_MIN, SP_MAX = 180, 300             # 18.0–30.0°C, in raw x10 register units (matches dashboard knob range)
                cur_mode = all_ctrl_data[ctrl_idx].get(117) if ctrl_idx < len(all_ctrl_data) else None
                if cur_mode == 2:
                    # Heat Only — the dashboard's single setpoint IS the heating setpoint (HR102)
                    primary_addr, primary_val   = 102, max(SP_MIN, min(SP_MAX, int(val)))
                    partner_addr, partner_val   = 100, max(SP_MIN, min(SP_MAX, primary_val + SP_DEADBAND))
                else:
                    # Auto / Cool Only / Fan — the dashboard's single setpoint IS the cooling setpoint (HR100)
                    primary_addr, primary_val   = 100, max(SP_MIN, min(SP_MAX, int(val)))
                    partner_addr, partner_val   = 102, max(SP_MIN, min(SP_MAX, primary_val - SP_DEADBAND))
                with lock:
                    r = c.write_register(primary_addr, primary_val, unit=slave_id)
                    time.sleep(0.1)   # let the RS485 bus/unit turn around before the next transaction
                    r2 = c.write_register(partner_addr, partner_val, unit=slave_id)
                    if r2.isError():
                        time.sleep(0.1)
                        r2 = c.write_register(partner_addr, partner_val, unit=slave_id)   # one retry
                    if r2.isError():
                        print(f"[SETPOINT] partner register {partner_addr} write failed: {r2}", flush=True)
            else:
                with lock:
                    r = c.write_register(address=reg, value=val, unit=slave_id)
        else:
            with lock:
                r = c.write_register(address=reg, value=val, unit=slave_id)
        if r.isError():
            return jsonify({'status':'fail','message':str(r)}), 500
        return jsonify({'status':'success'})
    except Exception as e:
        return jsonify({'status':'error','message':str(e)}), 500

def write_temperzone(client, sid, field, value):
    roles = REG_ROLES["Temperzone"]
    rtype = roles.get(field + "_rtype", "holding")
    addr = roles[field]
    if rtype == "coil":
        return client.write_coil(addr, bool(value), unit=sid)
    if field == "fan":
        client.write_register(roles["fan_mode"], 0, unit=sid)  # 0 = interpret 114 as 0-100%
        return client.write_register(addr, int(value), unit=sid)
    if field in ("setpoint", "setpoint_heat"):
        return client.write_register(addr, int(value / roles["scale"]), unit=sid)
    return client.write_register(addr, int(value), unit=sid)

@app.route('/power/restart',  methods=['POST'])
def handle_restart():  os.system('sudo reboot');          return jsonify({"status":"restarting"})
@app.route('/power/shutdown', methods=['POST'])
def handle_shutdown(): os.system('sudo shutdown -h now'); return jsonify({"status":"shutting down"})

@app.route('/send-log', methods=['POST'])
def send_log():
    data = request.get_json(silent=True) or {}
    return jsonify(send_csv_email(data.get('ctrl', 'all'), data.get('recipient', '')))

@app.route('/api/weather')
def api_weather():
    import urllib.request, json as _json
    global _weather_cache
    try:
        if '_weather_cache' not in globals():
            globals()['_weather_cache'] = {'t': 0, 'data': None, 'ok': False}
        cache = _weather_cache
        cache_ttl = 600 if cache.get('ok') else 30
        if time.monotonic() - cache['t'] < cache_ttl and cache['data']:
            return jsonify(cache['data'])
        lat, lon = -27.4698, 153.0251
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}"
               f"&current=temperature_2m,weather_code&timezone=auto")
        with urllib.request.urlopen(url, timeout=5) as resp:
            d = _json.loads(resp.read().decode())
        c = d.get('current', {})
        result = {'temp': c.get('temperature_2m'), 'code': c.get('weather_code', 0)}
        _weather_cache['t']    = time.monotonic()
        _weather_cache['data'] = result
        _weather_cache['ok']   = result.get('temp') is not None
        return jsonify(result)
    except Exception as e:
        _weather_cache['t']    = time.monotonic()
        _weather_cache['data'] = {'temp': None, 'code': 0, 'error': str(e)}
        _weather_cache['ok']   = False
        return jsonify({'temp': None, 'code': 0, 'error': str(e)})

@app.route('/')
def index():
    # per-controller register names so each type shows its own map
    cfg = load_device_config()
    all_reg_names = {}
    for i, ctrl in enumerate(cfg):
        rmap = roles_for(ctrl).get("map", REGISTER_MAP)
        all_reg_names[str(i)] = {str(k): v["name"] for k,v in rmap.items()}
    reg_names = {str(k): v["name"] for k,v in REGISTER_MAP.items()}  # fallback/global
    html = DASHBOARD_HTML.replace('__REG_NAMES__', json.dumps(reg_names))
    html = html.replace('__ALL_REG_NAMES__', json.dumps(all_reg_names))
    # Hide the mouse cursor ONLY on the physical panel. The on-device Chromium
    # always loads over loopback (127.0.0.1) and with ?kiosk=1; a technician
    # browsing the panel's LAN IP from a PC or phone is neither, so they keep a
    # normal cursor. The body.kiosk CSS rule does the actual hiding.
    remote = request.remote_addr or ''
    is_kiosk = (request.args.get('kiosk') == '1') or remote in ('127.0.0.1', '::1', 'localhost')
    if is_kiosk:
        html = html.replace('<body>', '<body class="kiosk">', 1)
    return render_template_string(html)

@app.route('/api/economy_config', methods=['GET','POST'])
def economy_config():
    global economy_enabled
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        idx = int(data.get('ctrl', 0))
        if 'enabled' in data:
            economy_enabled[idx] = bool(data['enabled'])
    return jsonify({'economy': economy_enabled})

@app.route('/api/co2_config', methods=['GET','POST'])
def co2_config():
    global co2_enabled
    if request.method == 'POST':
        data = request.get_json(silent=True) or {}
        idx = int(data.get('ctrl', 0))
        if 'enabled' in data:
            co2_enabled[idx] = bool(data['enabled'])
    # keys serialized as strings for JS lookup; default is True (handled client-side)
    return jsonify({'co2': {str(k): v for k, v in co2_enabled.items()}})

@app.route('/api/occupancy/suggestion')
def api_occ_suggestion():
    return jsonify(occupancy_suggestion())

@app.route('/api/occupancy/schedule')
def api_occ_schedule():
    pred = predict_occupancy()
    # weekly heatmap: prob per slot, compacted to hourly for display
    weekly = {}
    for wd in range(7):
        hours = []
        for h in range(24):
            ps = [pred["prob"].get(wd*96+h*4+q,0) for q in range(4)]
            hours.append(round(sum(ps)/4, 2))
        weekly[wd] = hours
    return jsonify({"weekly": weekly, "arrivals": pred["arrivals"], "samples": pred["samples"]})

@app.route('/api/occupancy/override', methods=['POST'])
def api_occ_override():
    data = request.get_json(silent=True) or {}
    mins = int(data.get('minutes', 60))
    occupancy_manual_override["active"] = bool(data.get('active', True))
    occupancy_manual_override["until"] = time.monotonic() + mins*60
    return jsonify({"status":"ok", **occupancy_manual_override})

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== ENVI PANEL V1.0===", flush=True)

    init_screen()

    # Time: restore the last saved time immediately (sane clock while offline),
    # then sync from the network + location if one is present. The network watch
    # thread (started below) keeps it correct from here on.
    time_startup()

        # Restart LED daemon on app startup
    try:
        subprocess.run(['sudo', 'systemctl', 'restart', 'led-daemon.service'],
                       capture_output=True, timeout=10)
        print("[LED] daemon restarted", flush=True)
    except Exception as e:
        print(f"[LED] daemon restart failed: {e}", flush=True)    # let X / touch rotation settle before drawing the logo


    startup_once()         # logo appears now and STAYS UP — no fixed timeout


    cfg0 = load_device_config()
    all_ctrl_data      = [{} for _ in cfg0]
    all_ctrl_status    = ["Disconnected" for _ in cfg0]
    all_ctrl_connected = [False for _ in cfg0]
    connect_modbus()


    threading.Thread(target=poll_modbus,           daemon=True).start()
    threading.Thread(target=motion_thread,         daemon=True).start()
    threading.Thread(target=led_logic_thread,      daemon=True).start()
    threading.Thread(target=led_pwm_thread,        daemon=True).start()
    threading.Thread(target=clock_thread,          daemon=True, name="Clock").start()
    threading.Thread(target=network_time_thread,   daemon=True, name="NetTime").start()
    threading.Thread(target=scheduler_engine,      daemon=True).start()
    threading.Thread(target=log_data_thread,       daemon=True).start()
    threading.Thread(target=fault_data_thread,     daemon=True).start()
    threading.Thread(target=system_cleanup_thread, daemon=True, name="Cleanup").start()
    threading.Thread(target=economy_cycle_thread,  daemon=True).start()
    threading.Thread(target=occupancy_logger_thread, daemon=True).start()

    threading.Thread(target=open_browser, daemon=True).start()
    threading.Thread(target=kiosk_watchdog_thread, daemon=True, name="KioskWatch").start()

    print("All threads started. Serving at http://0.0.0.0:5002", flush=True)

    app.run(host='0.0.0.0', port=5002, debug=False, use_reloader=False, threaded=True)
