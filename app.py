# merged_app_auto_pro.py — Centralized Enterprise Server
# BUG FIX: Infinite Expansion Loop Destroyed (Sidebars are hard-locked).
# NEW PRO FEATURE: CCTV HUD with blinking REC dot and live timestamps.
# NEW PRO FEATURE: Visual bounding box flash & Audio feedback on successful scan.
# RETAINED: Twin AI, DB Migrator, CCTV Cooldown, Face Updater, Hot-Swapping.

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, simpledialog
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import cv2, sqlite3, os, qrcode, shutil, base64, threading, time
import numpy as np
import pandas as pd
from datetime import datetime
from pyzbar.pyzbar import decode
from insightface.app import FaceAnalysis
from concurrent.futures import ThreadPoolExecutor
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import faiss
from ultralytics import YOLO
from sklearn.cluster import DBSCAN
import numpy as np
from PIL import Image, ImageTk
from scipy.spatial import distance as dist
from collections import OrderedDict

try: import winsound; HAS_SOUND = True
except ImportError: HAS_SOUND = False

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["OPENCV_LOG_LEVEL"] = "FATAL"

# ---------- InsightFace & Threading ----------
_MODEL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(_MODEL_ROOT, exist_ok=True)
FA = FaceAnalysis(name="buffalo_sc", root=_MODEL_ROOT, providers=["CPUExecutionProvider"])
FA.prepare(ctx_id=0, det_size=(640, 480))

ai_executor = ThreadPoolExecutor(max_workers=2)

# ---------- GLOBALS & CONFIG ----------
DB_FILE = "students.db"
SURVEILLANCE_DB_FILE = "surveillance.db"
CAMERA_SOURCE = 0 
SIMILARITY_THRESHOLD = 0.45
CCTV_COOLDOWN_SECONDS = 300 
UNKNOWN_EVENT_COOLDOWN_SECONDS = 45
PROCESS_EVERY_N = 3

# --- MASTER CCTV SHARED MEMORY ---
shared_cctv_frames = {} 
focused_cctv_camera = [None]
cctv_daemon_running = [False]

THEMES = {
    "dark": {
        "bg": "#121212", "fg": "white", "entry_bg": "#1e1e1e", "entry_fg": "#00ffff", 
        "card_bg": "#1f1f1f", "card_fg": "white", "hover_bg": "#008080", 
        "tree_bg": "#1f1f1f", "tree_fg": "white", "tree_header_bg": "#191919", "tree_header_fg": "white",
        "shadow": "#00e5ff", "shadow_opacity": 200, "shadow_blur": 18
    },
    "light": {
        "bg": "#f5f5f5", "fg": "black", "entry_bg": "#ffffff", "entry_fg": "black", 
        "card_bg": "#ffffff", "card_fg": "black", "hover_bg": "#a5d6a7", 
        "tree_bg": "white", "tree_fg": "black", "tree_header_bg": "#dcdcdc", "tree_header_fg": "black",
        "shadow": "#000000", "shadow_opacity": 50, "shadow_blur": 15
    }
}
current_theme = "light"
open_windows = {}

# ---------- DATABASE ENGINE & AUTO-MIGRATOR ----------
def get_conn():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA journal_mode=WAL;") 
    return conn

def get_surveillance_conn():
    conn = sqlite3.connect(SURVEILLANCE_DB_FILE)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_surveillance_db():
    os.makedirs("surveillance_snapshots", exist_ok=True)
    with get_surveillance_conn() as conn:
        c = conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS surveillance_tracks (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_name TEXT,
            mode TEXT,
            event_type TEXT NOT NULL,
            person_id INTEGER,
            legacy_student_id INTEGER,
            person_type TEXT,
            label TEXT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            match_percentage REAL,
            severity TEXT DEFAULT 'info',
            snapshot_path TEXT,
            action_taken TEXT,
            details TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS surveillance_unknowns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_name TEXT,
            mode TEXT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            severity TEXT DEFAULT 'alert',
            snapshot_path TEXT,
            best_match_percentage REAL,
            action_taken TEXT,
            details TEXT,
            embedding BLOB DEFAULT NULL
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS surveillance_summary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            report_date TEXT UNIQUE,
            known_count INTEGER DEFAULT 0,
            unknown_count INTEGER DEFAULT 0,
            alert_count INTEGER DEFAULT 0,
            last_updated TEXT
        )""")
        conn.commit()

def init_db():
    os.makedirs("photos", exist_ok=True); os.makedirs("qrcodes", exist_ok=True); os.makedirs("unknown_faces", exist_ok=True)
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY AUTOINCREMENT, reg_no TEXT UNIQUE, name TEXT, course TEXT, mobile TEXT, photo_path TEXT, qr_path TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER, date TEXT, time TEXT, match_percentage REAL)")
        c.execute("""CREATE TABLE IF NOT EXISTS people (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_type TEXT NOT NULL DEFAULT 'student',
            external_ref TEXT,
            reg_no TEXT,
            name TEXT NOT NULL,
            course TEXT,
            department TEXT,
            mobile TEXT,
            status TEXT DEFAULT 'active',
            photo_path TEXT,
            qr_path TEXT,
            embedding BLOB,
            is_twin INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(person_type, external_ref)
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS classes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_name TEXT UNIQUE NOT NULL,
            department TEXT,
            section TEXT,
            status TEXT DEFAULT 'active'
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_code TEXT UNIQUE,
            department_name TEXT UNIQUE NOT NULL,
            status TEXT DEFAULT 'active'
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_code TEXT,
            subject_name TEXT NOT NULL,
            department TEXT,
            status TEXT DEFAULT 'active',
            UNIQUE(subject_code, subject_name)
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS class_subjects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            class_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            faculty_person_id INTEGER,
            status TEXT DEFAULT 'active',
            UNIQUE(class_id, subject_id, faculty_person_id)
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS class_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_code TEXT UNIQUE NOT NULL,
            class_id INTEGER,
            subject_id INTEGER,
            faculty_person_id INTEGER,
            session_title TEXT,
            session_date TEXT NOT NULL,
            start_time TEXT,
            end_time TEXT,
            status TEXT DEFAULT 'open',
            notes TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS cameras (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_name TEXT UNIQUE NOT NULL,
            source TEXT NOT NULL,
            camera_type TEXT DEFAULT 'USB',
            location TEXT,
            allowed_modes TEXT DEFAULT 'enrollment,cctv,attendance,qr,entry,exit',
            can_surveillance INTEGER DEFAULT 1,
            can_attendance INTEGER DEFAULT 1,
            can_enrollment INTEGER DEFAULT 1,
            can_qr INTEGER DEFAULT 1,
            status TEXT DEFAULT 'active',
            locked_by TEXT,
            locked_at TEXT,
            notes TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            device_name TEXT UNIQUE NOT NULL,
            device_type TEXT DEFAULT 'main',
            location TEXT,
            default_mode TEXT,
            status TEXT DEFAULT 'active',
            last_seen TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS attendance_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_type TEXT NOT NULL,
            class_id INTEGER,
            subject_id INTEGER,
            faculty_person_id INTEGER,
            camera_id INTEGER,
            location TEXT,
            started_at TEXT DEFAULT CURRENT_TIMESTAMP,
            ended_at TEXT,
            status TEXT DEFAULT 'open'
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS attendance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id INTEGER,
            person_id INTEGER,
            legacy_student_id INTEGER,
            person_type TEXT DEFAULT 'student',
            event_type TEXT DEFAULT 'attendance',
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            camera_name TEXT,
            camera_location TEXT,
            class_id INTEGER,
            subject_id INTEGER,
            match_percentage REAL,
            verification_method TEXT DEFAULT 'face',
            status TEXT DEFAULT 'official',
            notes TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS entry_exit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            person_id INTEGER,
            legacy_student_id INTEGER,
            person_type TEXT,
            area_name TEXT,
            event_type TEXT NOT NULL,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            camera_name TEXT,
            verification_method TEXT DEFAULT 'face',
            match_percentage REAL,
            notes TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS camera_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_name TEXT,
            event_type TEXT NOT NULL,
            mode TEXT,
            person_id INTEGER,
            legacy_student_id INTEGER,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            severity TEXT DEFAULT 'info',
            snapshot_path TEXT,
            details TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS unknown_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            camera_name TEXT,
            mode TEXT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            severity TEXT DEFAULT 'log',
            snapshot_path TEXT,
            face_count INTEGER DEFAULT 1,
            action_taken TEXT,
            details TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS qr_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            qr_value TEXT NOT NULL,
            person_id INTEGER,
            legacy_student_id INTEGER,
            area_name TEXT,
            event_type TEXT,
            date TEXT NOT NULL,
            time TEXT NOT NULL,
            camera_name TEXT,
            status TEXT DEFAULT 'accepted',
            details TEXT
        )""")
        c.execute("""CREATE TABLE IF NOT EXISTS qr_access_points (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            area_code TEXT UNIQUE NOT NULL,
            area_name TEXT NOT NULL,
            area_type TEXT DEFAULT 'room',
            default_event TEXT DEFAULT 'in',
            device_type TEXT DEFAULT 'camera',
            device_source TEXT,
            status TEXT DEFAULT 'active',
            notes TEXT
        )""")
        
        c.execute("PRAGMA table_info(students)")
        student_cols = [row[1] for row in c.fetchall()]
        if 'embedding' not in student_cols: c.execute("ALTER TABLE students ADD COLUMN embedding BLOB")
        if 'is_twin' not in student_cols: c.execute("ALTER TABLE students ADD COLUMN is_twin INTEGER DEFAULT 0")

        c.execute("PRAGMA table_info(people)")
        people_cols = [row[1] for row in c.fetchall()]
        if 'designation' not in people_cols: c.execute("ALTER TABLE people ADD COLUMN designation TEXT")
        if 'organization' not in people_cols: c.execute("ALTER TABLE people ADD COLUMN organization TEXT")
        if 'visitor_purpose' not in people_cols: c.execute("ALTER TABLE people ADD COLUMN visitor_purpose TEXT")
        if 'valid_until' not in people_cols: c.execute("ALTER TABLE people ADD COLUMN valid_until TEXT")

        c.execute("PRAGMA table_info(attendance)")
        att_cols = [row[1] for row in c.fetchall()]
        if 'camera_location' not in att_cols: c.execute("ALTER TABLE attendance ADD COLUMN camera_location TEXT DEFAULT 'Main Server'")
        c.execute("""DELETE FROM attendance
                     WHERE id NOT IN (SELECT MIN(id) FROM attendance GROUP BY student_id, date)""")
        c.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_attendance_student_date ON attendance(student_id, date)")

        c.execute("PRAGMA table_info(attendance_logs)")
        log_cols = [row[1] for row in c.fetchall()]
        if 'class_session_id' not in log_cols: c.execute("ALTER TABLE attendance_logs ADD COLUMN class_session_id INTEGER")
        if 'session_code' not in log_cols: c.execute("ALTER TABLE attendance_logs ADD COLUMN session_code TEXT")

        c.execute("INSERT OR IGNORE INTO cameras (camera_name, source, camera_type, location, notes) VALUES (?, ?, ?, ?, ?)",
                  ("Default Camera", str(CAMERA_SOURCE), "USB", "Main Server", "Auto-created from current CAMERA_SOURCE"))
        c.execute("""INSERT OR IGNORE INTO people
            (person_type, external_ref, reg_no, name, course, mobile, photo_path, qr_path, embedding, is_twin)
            SELECT 'student', CAST(id AS TEXT), reg_no, name, course, mobile, photo_path, qr_path, embedding, is_twin
            FROM students
        """)
        
        conn.commit()
init_db()
init_surveillance_db()


def extract_core_biometrics(frame, enforce_single=True):
    """
    CENTRALIZED AI ENGINE: Instantly extracts the biometric embedding and a cropped portrait 
    from a raw camera frame. Selects the most prominent face if multiple are found.
    Returns: (Success_Boolean, Embedding_Bytes, Cropped_Image_Array, Status_Message)
    """
    faces = FA.get(frame)
    if not faces:
        return False, None, None, "No biometric signature detected in frame."
    
    if enforce_single and len(faces) > 1:
        return False, None, None, "Multiple faces detected. Please isolate the subject."
        
    # Isolate the primary target (the largest face in the frame by bounding box area)
    primary_face = max(faces, key=lambda f: (f.bbox[2]-f.bbox[0]) * (f.bbox[3]-f.bbox[1]))
    
    # Generate a padded crop for the ID photo
    x1, y1, x2, y2 = map(int, primary_face.bbox)
    h, w = frame.shape[:2]
    pad = 35 # Generous padding for a professional ID look
    px1, py1 = max(0, x1-pad), max(0, y1-pad)
    px2, py2 = min(w, x2+pad), min(h, y2+pad)
    cropped_portrait = frame[py1:py2, px1:px2]
    
    return True, primary_face.embedding.tobytes(), cropped_portrait, "Success"




# ---------- SERVER MEMORY ----------
KNOWN_EMBEDDINGS, KNOWN_IDS, KNOWN_LABELS, KNOWN_COURSES = [], [], [], []
KNOWN_PERSON_IDS, KNOWN_PERSON_TYPES = [], []

def load_server_memory():
    global KNOWN_EMBEDDINGS, KNOWN_IDS, KNOWN_LABELS, KNOWN_COURSES, KNOWN_PERSON_IDS, KNOWN_PERSON_TYPES
    KNOWN_EMBEDDINGS.clear(); KNOWN_IDS.clear(); KNOWN_LABELS.clear(); KNOWN_COURSES.clear(); KNOWN_PERSON_IDS.clear(); KNOWN_PERSON_TYPES.clear()
    with get_conn() as conn:
        df = pd.read_sql_query("""SELECT id as person_id,
                                  CASE WHEN person_type='student' AND external_ref GLOB '[0-9]*' THEN CAST(external_ref AS INTEGER) ELSE NULL END as legacy_student_id,
                                  person_type, COALESCE(reg_no, external_ref, '') as reg_no, name,
                                  COALESCE(course, department, person_type) as course, embedding
                                  FROM people WHERE embedding IS NOT NULL
                                  UNION ALL
                                  SELECT NULL as person_id, id as legacy_student_id, 'student' as person_type, reg_no, name, course, embedding FROM students
                                  WHERE embedding IS NOT NULL
                                  AND id NOT IN (SELECT CAST(external_ref AS INTEGER) FROM people WHERE person_type='student' AND external_ref GLOB '[0-9]*')""", conn)
    for _, row in df.iterrows():
        KNOWN_EMBEDDINGS.append(np.frombuffer(row["embedding"], dtype=np.float32))
        KNOWN_IDS.append(None if pd.isna(row["legacy_student_id"]) else int(row["legacy_student_id"]))
        KNOWN_PERSON_IDS.append(None if pd.isna(row["person_id"]) else int(row["person_id"]))
        KNOWN_PERSON_TYPES.append(str(row["person_type"]))
        KNOWN_LABELS.append(f"{row['reg_no']} | {row['name']} ({row['person_type']})")
        KNOWN_COURSES.append(str(row["course"]).strip())
load_server_memory()

def face_similarity(known_encodings, probe_enc):
    if not known_encodings: return np.array([])
    k, p = np.array(known_encodings, dtype=np.float32), np.array(probe_enc, dtype=np.float32)
    return (k / (np.linalg.norm(k, axis=1, keepdims=True) + 1e-9)) @ (p / (np.linalg.norm(p) + 1e-9))

# ---------- GLOBAL CAMERA MANAGER ----------

def get_resource_sources():
    with get_conn() as conn:
        cams = conn.cursor().execute(
            """
            SELECT camera_name, source, location
            FROM cameras
            WHERE status='active'
            ORDER BY camera_name
            """
        ).fetchall()

        devs = conn.cursor().execute(
            """
            SELECT device_name, device_type, location
            FROM devices
            WHERE status='active'
            ORDER BY device_name
            """
        ).fetchall()

    values = [
        f"camera | {name} | {src} | {loc or ''}"
        for name, src, loc in cams
    ]
    values.extend([
        f"device | {name} | {typ} | {loc or ''}"
        for name, typ, loc in devs
    ])
    return values

class AsyncCamera:
    def __init__(self, source):
        self.source = source
        self.lock = threading.Lock()
        self.cap = None
        self.frame = None
        self.ret = False
        self.running = True
        self.thread = threading.Thread(
            target=self._loop,
            daemon=True
        )
        self.thread.start()

    def _loop(self):
        try:
            if (
                isinstance(self.source, int)
                or
                (
                    isinstance(self.source, str)
                    and
                    self.source.isdigit()
                )
            ):
                self.cap = cv2.VideoCapture(
                    int(self.source),
                    cv2.CAP_DSHOW
                )
            else:
                self.cap = cv2.VideoCapture(
                    self.source
                )

            if not self.cap.isOpened():
                self.ret = False
                return

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            CAMERA_FPS = 30

            while self.running:
                try:
                    ret, frame = self.cap.read()
                    with self.lock:
                        self.ret = ret
                        if ret:
                            self.frame = frame.copy()
                except cv2.error:
                    self.ret = False
                time.sleep(1 / CAMERA_FPS)
        except Exception:
            self.ret = False

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def stop(self):
        self.running = False
        if self.thread.is_alive():
            self.thread.join(timeout=1.0)
        if self.cap:
            self.cap.release()
            self.cap = None

class CameraManager:
    def __init__(self):
        self.cameras = {}

    def start(self, source):
        self.start_camera("default", source)

    def stop(self):
        self.stop_camera("default")

    def read(self):
        return self.read_camera("default")

    def add_camera(self, camera_name, source):
        if camera_name in self.cameras:
            return False
        self.cameras[camera_name] = AsyncCamera(source)
        return True

    def start_camera(self, camera_name, source):
        if camera_name in self.cameras:
            old = self.cameras[camera_name]
            if old.source == source:
                return
            old.stop()
        self.cameras[camera_name] = AsyncCamera(source)

    def stop_camera(self, camera_name):
        cam = self.cameras.get(camera_name)
        if cam:
            cam.stop()
            del self.cameras[camera_name]

    def remove_camera(self, camera_name):
        self.stop_camera(camera_name)

    def read_camera(self, camera_name):
        cam = self.cameras.get(camera_name)
        if cam:
            return cam.read()
        return False, None

    def get_camera_names(self):
        return list(self.cameras.keys())

    def stop_all(self):
        for cam in list(self.cameras.values()):
            try: cam.stop()
            except: pass
        self.cameras.clear()

cam_manager = CameraManager()

def scan_available_cameras(max_cameras=10):
    detected = []
    for idx in range(max_cameras):
        cap = None
        try:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    detected.append((idx, f"USB Camera {idx}"))
        except:
            pass
        finally:
            if cap:
                cap.release()
    return detected

def load_registered_cameras():
    with get_conn() as conn:
        return conn.cursor().execute(
            """
            SELECT id, camera_name, source, camera_type, location, status
            FROM cameras
            ORDER BY camera_name
            """
        ).fetchall()
    
def test_camera_source(source):
    cap = None
    try:
        if str(source).isdigit():
            cap = cv2.VideoCapture(int(source))
        else:
            cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            return False, None
        ret, frame = cap.read()
        if not ret:
            return False, None
        h, w = frame.shape[:2]
        return True, f"{w}x{h}"
    except Exception:
        return False, None
    finally:
        if cap:
            cap.release()

def add_ip_camera():
    win = tk.Toplevel(root)
    win.title("Add Network Camera")
    win.geometry("500x450")
    tk.Label(win, text="Camera Name").pack(pady=5)
    name_entry = tk.Entry(win)
    name_entry.pack(fill="x", padx=20)
    tk.Label(win, text="Camera Type").pack(pady=5)
    type_var = tk.StringVar(value="IP Webcam")
    ttk.Combobox(win, textvariable=type_var, values=["IP Webcam", "RTSP", "Mobile Camera", "Custom Stream"], state="readonly").pack(fill="x", padx=20)
    tk.Label(win, text="Source URL").pack(pady=5)
    source_entry = tk.Entry(win)
    source_entry.pack(fill="x", padx=20)
    result_label = tk.Label(win, text="")
    result_label.pack(pady=15)

    def test_connection():
        source = source_entry.get().strip()
        ok, resolution = test_camera_source(source)
        if ok:
            result_label.config(text=f"Connected ✓ ({resolution})", fg="green")
        else:
            result_label.config(text="Connection Failed ✗", fg="red")

    def save_camera():
        name = name_entry.get().strip()
        source = source_entry.get().strip()
        cam_type = type_var.get()
        if not name or not source:
            return
        with get_conn() as conn:
            conn.cursor().execute(
                "INSERT INTO cameras (camera_name, source, camera_type, location) VALUES (?, ?, ?, ?)",
                (name, source, cam_type, "Network")
            )
            conn.commit()
        messagebox.showinfo("Saved", "Camera Registered")
        win.destroy()

    tk.Button(win, text="Test Connection", command=test_connection).pack(pady=10)
    tk.Button(win, text="Save Camera", command=save_camera).pack(pady=10)
    
def preview_camera(source):
    preview = tk.Toplevel(root)
    preview.title("Camera Preview")
    preview.geometry("900x650")
    preview.configure(bg="black")
    frame_holder = tk.Frame(preview, bg="black")
    frame_holder.pack(fill="both", expand=True)
    lbl = tk.Label(frame_holder, bg="black")
    lbl.place(relx=0.5, rely=0.5, anchor="center")
    cam_manager.start_camera("preview", source)

    def update():
        ret, frame = cam_manager.read_camera("preview")
        if ret and frame is not None:
            img = ImageTk.PhotoImage(
                Image.fromarray(
                    cv2.cvtColor(
                        resize_to_fit(frame, frame_holder.winfo_width(), frame_holder.winfo_height()),
                        cv2.COLOR_BGR2RGB
                    )
                )
            )
            lbl.imgtk = img
            lbl.configure(image=img)
        if preview.winfo_exists():
            preview.after(30, update)

    update()
    def close_preview():
        cam_manager.stop_camera("preview")
        preview.destroy()
    preview.protocol("WM_DELETE_WINDOW", close_preview)

def open_camera_management():
    win = tk.Toplevel(root)
    win.title("Camera Management Center")
    win.geometry("1200x700")
    win.configure(bg=THEMES[current_theme]["bg"])
    add_window_toolbar(win, "camera_management", stop_camera_instance=None)

    left = tk.Frame(win, bg=THEMES[current_theme]["bg"])
    left.pack(side="left", fill="both", expand=True, padx=10, pady=10)
    right = tk.Frame(win, bg=THEMES[current_theme]["bg"])
    right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

    tk.Label(left, text="Detected Cameras", font=("Segoe UI", 15, "bold")).pack()
    detected_list = tk.Listbox(left, height=20)
    detected_list.pack(fill="both", expand=True)

    tk.Label(right, text="Registered Cameras", font=("Segoe UI", 15, "bold")).pack()
    registered_list = tk.Listbox(right, height=20)
    registered_list.pack(fill="both", expand=True)

    def scan():
        detected_list.delete(0, tk.END)
        cams = scan_available_cameras()
        for cam_id, name in cams:
            detected_list.insert(tk.END, f"{cam_id} | {name}")

    def refresh_registered():
        registered_list.delete(0, tk.END)
        rows = load_registered_cameras()
        for row in rows:
            registered_list.insert(tk.END, f"{row[0]} | {row[1]} | {row[2]}")

    def add_camera():
        sel = detected_list.curselection()
        if not sel: return
        value = detected_list.get(sel[0])
        source = value.split("|")[0].strip()
        cam_name = simpledialog.askstring("Camera Name", "Enter Camera Name:")
        if not cam_name: return
        with get_conn() as conn:
            conn.cursor().execute(
                "INSERT INTO cameras (camera_name, source, camera_type, location) VALUES (?, ?, ?, ?)",
                (cam_name, source, "USB", "Unknown")
            )
            conn.commit()
        refresh_registered()

    def preview_selected():
        sel = registered_list.curselection()
        if not sel: return
        row_text = registered_list.get(sel[0])
        db_id = int(row_text.split("|")[0].strip())
        with get_conn() as conn:
            row = conn.cursor().execute("SELECT source FROM cameras WHERE id=?", (db_id,)).fetchone()
        if row:
            source = int(row[0]) if str(row[0]).isdigit() else row[0]
            preview_camera(source)

    btn_frame = tk.Frame(left, bg=THEMES[current_theme]["bg"])
    btn_frame.pack(fill="x")
    tk.Button(btn_frame, text="Scan Cameras", command=scan).pack(side="left", padx=5)
    tk.Button(btn_frame, text="Add Camera", command=add_camera).pack(side="left", padx=5)
    tk.Button(btn_frame, text="Add IP Camera", command=add_ip_camera).pack(side="left", padx=5)
    tk.Button(right, text="Preview", command=preview_selected).pack(pady=5)

    scan()
    refresh_registered()

def close_all_modules():
    cam_manager.stop_all()
    for win_name in list(open_windows.keys()):
        try: open_windows[win_name].destroy()
        except Exception: pass
        open_windows.pop(win_name, None)

def play_success_beep():
    if HAS_SOUND: threading.Thread(target=lambda: winsound.Beep(1200, 150), daemon=True).start()

unknown_event_memory = {}

def safe_name(value):
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value))[:80]

def save_event_snapshot(frame, folder, camera_name, mode, event_type):
    if frame is None: return None
    os.makedirs(folder, exist_ok=True)
    now_dt = datetime.now()
    filename = f"{now_dt.strftime('%Y-%m-%d_%H-%M-%S')}_{safe_name(camera_name)}_{safe_name(mode)}_{safe_name(event_type)}.jpg"
    path = os.path.join(folder, filename)
    try:
        cv2.imwrite(path, frame)
        return path
    except Exception: return None

def update_surveillance_summary(date_s, event_type, severity):
    with get_surveillance_conn() as conn:
        conn.cursor().execute("""INSERT OR IGNORE INTO surveillance_summary
            (report_date, known_count, unknown_count, alert_count, last_updated)
            VALUES (?, 0, 0, 0, ?)""", (date_s, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        if event_type == "unknown_face":
            conn.cursor().execute("UPDATE surveillance_summary SET unknown_count = unknown_count + 1, last_updated=? WHERE report_date=?",
                                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), date_s))
        else:
            conn.cursor().execute("UPDATE surveillance_summary SET known_count = known_count + 1, last_updated=? WHERE report_date=?",
                                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), date_s))
        if severity == "alert":
            conn.cursor().execute("UPDATE surveillance_summary SET alert_count = alert_count + 1, last_updated=? WHERE report_date=?",
                                  (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), date_s))
        conn.commit()

def log_surveillance_track(camera_name, mode, event_type, frame=None, person_id=None, legacy_student_id=None,
                           person_type=None, label=None, match_percentage=None, severity="info",
                           action_taken="recorded", details=""):
    now_dt = datetime.now()
    dt_s, tm_s = now_dt.strftime("%Y-%m-%d"), now_dt.strftime("%H:%M:%S")
    snapshot_path = save_event_snapshot(frame, "surveillance_snapshots", camera_name, mode, event_type)
    with get_surveillance_conn() as conn:
        conn.cursor().execute("""INSERT INTO surveillance_tracks
            (camera_name, mode, event_type, person_id, legacy_student_id, person_type, label, date, time,
             match_percentage, severity, snapshot_path, action_taken, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (camera_name, mode, event_type, person_id, legacy_student_id, person_type, label, dt_s, tm_s,
             match_percentage, severity, snapshot_path, action_taken, details))
        conn.commit()
    update_surveillance_summary(dt_s, event_type, severity)
    return snapshot_path

def log_unknown_event(camera_name, mode, frame=None, severity="log", action_taken="saved_track", details="Face did not match registered people.", best_match_percentage=None):
    now_dt = datetime.now()
    key = f"{camera_name}|{mode}|{severity}"
    if now_dt.timestamp() - unknown_event_memory.get(key, 0) < UNKNOWN_EVENT_COOLDOWN_SECONDS:
        return False

    unknown_event_memory[key] = now_dt.timestamp()
    dt_s, tm_s = now_dt.strftime("%Y-%m-%d"), now_dt.strftime("%H:%M:%S")
    snapshot_path = save_event_snapshot(frame, "unknown_faces", camera_name, mode, "unknown")

    with get_conn() as conn:
        conn.cursor().execute("""INSERT INTO unknown_events
            (camera_name, mode, date, time, severity, snapshot_path, face_count, action_taken, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (camera_name, mode, dt_s, tm_s, severity, snapshot_path, 1, action_taken, details))
        conn.cursor().execute("""INSERT INTO camera_events
            (camera_name, event_type, mode, date, time, severity, snapshot_path, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (camera_name, "unknown_face", mode, dt_s, tm_s, severity, snapshot_path, details))
        conn.commit()
    with get_surveillance_conn() as conn:
        conn.cursor().execute("""INSERT INTO surveillance_unknowns
            (camera_name, mode, date, time, severity, snapshot_path, best_match_percentage, action_taken, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (camera_name, mode, dt_s, tm_s, severity, snapshot_path, best_match_percentage, action_taken, details))
        conn.cursor().execute("""INSERT INTO surveillance_tracks
            (camera_name, mode, event_type, label, date, time, match_percentage, severity, snapshot_path, action_taken, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (camera_name, mode, "unknown_face", "UNKNOWN", dt_s, tm_s, best_match_percentage, severity, snapshot_path, action_taken, details))
        conn.commit()
    update_surveillance_summary(dt_s, "unknown_face", severity)
    return True

def create_camera_device_selector(parent, target_var):
    source_choice = tk.StringVar()
    row = tk.Frame(parent, bg=THEMES[current_theme]["bg"])
    row.pack(fill="x", pady=6)
    tk.Label(row, text="Saved Camera / Device", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w")
    combo = ttk.Combobox(row, textvariable=source_choice, state="readonly")
    combo.pack(fill="x", pady=3)

    with get_conn() as conn:
        cams = conn.cursor().execute("SELECT camera_name, source, location FROM cameras WHERE status='active' ORDER BY camera_name").fetchall()
    values = [f"camera | {name} | {src} | {loc or ''}" for name, src, loc in cams]
    combo["values"] = values

    if values:
        combo.current(0)
        parts = [p.strip() for p in values[0].split("|")]
        if len(parts) >= 3: target_var.set(parts[2])

    def apply(_event=None):
        parts = [p.strip() for p in source_choice.get().split("|")]
        if len(parts) >= 3: target_var.set(parts[2])

    combo.bind("<<ComboboxSelected>>", apply)
    return combo

def resize_to_fit(image, target_w, target_h):
    if target_w < 10 or target_h < 10: return image
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    return cv2.resize(image, (int(w * scale), int(h * scale)))

def draw_hud(display, cam_name):
    tm = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    blink = int(time.time() * 2) % 2 == 0
    if blink: cv2.circle(display, (30, 30), 8, (0, 0, 255), -1)
    cv2.putText(display, f"REC | {cam_name}", (50, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, tm, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return display

def set_zoomed(win):
    try: win.state("zoomed")
    except Exception: win.geometry(f"{win.winfo_screenwidth()}x{win.winfo_screenheight()}+0+0")

def toggle_zoom(win):
    try: win.state("normal" if win.state() == "zoomed" else "zoomed")
    except Exception: set_zoomed(win)

def show_dashboard(win=None, win_name=None, stop_camera_instance=None):
    if stop_camera_instance:
        cam_manager.stop_camera(stop_camera_instance)
    if win_name:
        open_windows.pop(win_name, None)
    if win is not None:
        try: win.destroy()
        except Exception: pass
    try:
        root.deiconify()
        root.lift()
    except Exception: pass

def add_window_toolbar(win, win_name, stop_camera_instance=None):
    toolbar = tk.Frame(win, bg="#202020", height=42)
    toolbar.pack(fill="x", side="top")
    tk.Button(toolbar, text="Back Home", command=lambda: show_dashboard(win, win_name, stop_camera_instance),
              bg="#14818f", fg="white", bd=0, padx=12, pady=6, cursor="hand2").pack(side="left", padx=8, pady=6)
    tk.Button(toolbar, text="Minimize", command=win.iconify,
              bg="#555", fg="white", bd=0, padx=12, pady=6, cursor="hand2").pack(side="right", padx=6, pady=6)
    tk.Button(toolbar, text="Maximize / Restore", command=lambda: toggle_zoom(win),
              bg="#555", fg="white", bd=0, padx=12, pady=6, cursor="hand2").pack(side="right", padx=6, pady=6)
    win.protocol("WM_DELETE_WINDOW", lambda: show_dashboard(win, win_name, stop_camera_instance))
    return toolbar

def backup_database_file(source_file, title):
    path = filedialog.asksaveasfilename(title=title, defaultextension=".db", filetypes=[("SQLite DB", "*.db")])
    if path:
        shutil.copy2(source_file, path)
        messagebox.showinfo("Backup", "Database backup saved.")

def fetch_named_options(table, label_expr, where="status='active'"):
    with get_conn() as conn:
        rows = conn.cursor().execute(f"SELECT id, {label_expr} FROM {table} WHERE {where} ORDER BY 2").fetchall()
    return [f"{row[0]} | {row[1]}" for row in rows]

def option_id(value):
    if not value: return None
    try: return int(str(value).split("|", 1)[0].strip())
    except Exception: return None

def create_or_get_class_session(session_code, class_id=None, subject_id=None, faculty_person_id=None, title="Attendance Session"):
    code = (session_code or "").strip()
    if not code: code = datetime.now().strftime("SESSION-%Y%m%d-%H%M%S")
    with get_conn() as conn:
        row = conn.cursor().execute("SELECT id FROM class_sessions WHERE session_code=?", (code,)).fetchone()
        if row: return row[0], code
        now_dt = datetime.now()
        cur = conn.cursor()
        cur.execute("""INSERT INTO class_sessions
            (session_code, class_id, subject_id, faculty_person_id, session_title, session_date, start_time)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (code, class_id, subject_id, faculty_person_id, title, now_dt.strftime("%Y-%m-%d"), now_dt.strftime("%H:%M:%S")))
        conn.commit()
        return cur.lastrowid, code

def mark_daily_student_attendance(student_id, date_s, time_s, match_percentage, location):
    with get_conn() as conn:
        existing = conn.cursor().execute("SELECT id, time, match_percentage FROM attendance WHERE student_id=? AND date=?",
                                         (student_id, date_s)).fetchone()
        if existing: return False, existing
        conn.cursor().execute("""INSERT INTO attendance (student_id, date, time, match_percentage, camera_location)
                                 VALUES (?, ?, ?, ?, ?)""",
                              (student_id, date_s, time_s, match_percentage, location))
        conn.commit()
    return True, None

def open_student_detail_window(student_id=None, reg_no=None):
    with get_conn() as conn:
        if student_id is not None:
            row = conn.cursor().execute("SELECT id, reg_no, name, course, mobile, photo_path FROM students WHERE id=?", (student_id,)).fetchone()
        else:
            row = conn.cursor().execute("SELECT id, reg_no, name, course, mobile, photo_path FROM students WHERE reg_no=?", (reg_no,)).fetchone()
    if not row: return messagebox.showerror("Details", "Student not found.")
    sid, reg, name, course, mobile, photo_path = row
    detail_win = tk.Toplevel(root); detail_win.title(f"Attendance Details: {reg}"); detail_win.geometry("520x520")
    detail_win.configure(bg=THEMES[current_theme]["bg"])
    tk.Label(detail_win, text=f"{reg} | {name}", font=("Segoe UI", 16, "bold"), bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(pady=10)
    img_box = tk.Label(detail_win, bg="black", fg="white", width=260, height=180)
    img_box.pack(pady=8)
    if photo_path and os.path.exists(photo_path):
        try:
            img = Image.open(photo_path).resize((220, 170))
            imgtk = ImageTk.PhotoImage(img)
            img_box.imgtk = imgtk; img_box.configure(image=imgtk, text="")
        except Exception: img_box.configure(text="Photo unavailable")
    else: img_box.configure(text="No photo")
    today = datetime.now().strftime("%Y-%m-%d")
    with get_conn() as conn:
        att = conn.cursor().execute("SELECT time, match_percentage, camera_location FROM attendance WHERE student_id=? AND date=?",
                                    (sid, today)).fetchone()
    lines = [
        f"Course: {course or ''}",
        f"Mobile: {mobile or ''}",
        f"Date: {today}",
        f"Status: {'Marked' if att else 'Not marked today'}",
        f"Time: {att[0] if att else ''}",
        f"Match: {att[1]:.1f}%" if att and att[1] is not None else "Match:",
        f"Location: {att[2] if att else ''}",
    ]
    tk.Label(detail_win, text="\n".join(lines), justify="left", font=("Segoe UI", 12),
             bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(pady=8)

def open_simple_bar_plot(title, rows, x_index=0, y_index=1):
    plot_win = tk.Toplevel(root); plot_win.title(title); plot_win.geometry("820x520")
    plot_win.configure(bg=THEMES[current_theme]["bg"])
    tk.Label(plot_win, text=title, font=("Segoe UI", 16, "bold"),
             bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(pady=10)
    canvas = tk.Canvas(plot_win, bg=THEMES[current_theme]["card_bg"], highlightthickness=0)
    canvas.pack(fill="both", expand=True, padx=18, pady=18)

    def draw(_event=None):
        canvas.delete("all")
        if not rows:
            canvas.create_text(400, 230, text="No data available", fill=THEMES[current_theme]["fg"], font=("Segoe UI", 14))
            return
        w, h = canvas.winfo_width(), canvas.winfo_height()
        pad_l, pad_b, pad_t, pad_r = 60, 70, 30, 25
        values = [float(r[y_index] or 0) for r in rows]
        labels = [str(r[x_index]) for r in rows]
        max_v = max(values) or 1
        bar_w = max(18, min(70, (w - pad_l - pad_r) / max(len(rows), 1) - 8))
        canvas.create_line(pad_l, h-pad_b, w-pad_r, h-pad_b, fill="#888")
        canvas.create_line(pad_l, pad_t, pad_l, h-pad_b, fill="#888")
        for i, val in enumerate(values):
            x1 = pad_l + i * ((w - pad_l - pad_r) / max(len(rows), 1)) + 6
            x2 = x1 + bar_w
            y2 = h - pad_b
            y1 = y2 - (val / max_v) * (h - pad_t - pad_b)
            canvas.create_rectangle(x1, y1, x2, y2, fill="#14818f", outline="")
            canvas.create_text((x1+x2)/2, y1-10, text=f"{val:.0f}", fill=THEMES[current_theme]["fg"], font=("Segoe UI", 9))
            canvas.create_text((x1+x2)/2, h-pad_b+22, text=labels[i][:10], fill=THEMES[current_theme]["fg"], font=("Segoe UI", 8), angle=25)
    canvas.bind("<Configure>", draw)
    draw()

def lookup_qr_identity(qr_value):
    value = (qr_value or "").strip()
    if not value: return None
    with get_conn() as conn:
        row = conn.cursor().execute("""SELECT id, person_type, name,
                                       CASE WHEN person_type='student' AND external_ref GLOB '[0-9]*' THEN CAST(external_ref AS INTEGER) ELSE NULL END
                                       FROM people
                                       WHERE external_ref=? OR reg_no=?
                                       ORDER BY CASE WHEN status='active' THEN 0 ELSE 1 END, id DESC LIMIT 1""",
                                    (value, value)).fetchone()
        if row: return {"person_id": row[0], "person_type": row[1], "name": row[2], "legacy_student_id": row[3]}
        row = conn.cursor().execute("SELECT id, name FROM students WHERE reg_no=?", (value,)).fetchone()
        if row: return {"person_id": None, "person_type": "student", "name": row[1], "legacy_student_id": row[0]}
    return None

def log_qr_access(qr_value, area_name, event_type, camera_name, method="qr", details=""):
    now_dt = datetime.now()
    identity = lookup_qr_identity(qr_value)
    status = "accepted" if identity else "unknown_qr"
    person_id = identity["person_id"] if identity else None
    legacy_student_id = identity["legacy_student_id"] if identity else None
    label = identity["name"] if identity else "Unknown QR"
    with get_conn() as conn:
        conn.cursor().execute("""INSERT INTO qr_logs
            (qr_value, person_id, legacy_student_id, area_name, event_type, date, time, camera_name, status, details)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (qr_value, person_id, legacy_student_id, area_name, event_type, now_dt.strftime("%Y-%m-%d"),
             now_dt.strftime("%H:%M:%S"), camera_name, status, details))
        conn.cursor().execute("""INSERT INTO entry_exit_logs
            (person_id, legacy_student_id, person_type, area_name, event_type, date, time, camera_name,
             verification_method, match_percentage, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (person_id, legacy_student_id, identity["person_type"] if identity else None, area_name, event_type,
             now_dt.strftime("%Y-%m-%d"), now_dt.strftime("%H:%M:%S"), camera_name, method, None, details))
        conn.commit()
    return status, label, now_dt.strftime("%Y-%m-%d"), now_dt.strftime("%H:%M:%S")


def open_identity_management_hub1():
    win_name = "identity_hub"
    if win_name in open_windows and open_windows[win_name].winfo_exists():
        open_windows[win_name].deiconify()
        open_windows[win_name].lift()
        return

    win = tk.Toplevel(root)
    win.title("Unified Biometric & Identity Management Hub")
    win.geometry("1350x840")
    open_windows[win_name] = win
    win.configure(bg=THEMES[current_theme]["bg"])
    
    cam_slot_name = "identity_updater_stream"
    add_window_toolbar(win, win_name, stop_camera_instance=cam_slot_name)

    # State variables & Historical Biometric Cache Holders
    selected_person = {"id": None, "reg": None, "name": None, "type": None}
    target_historical_embedding = [None] 
    live_verification_score = tk.StringVar(value="Verification: Pending Target Selection")
    
    # --- MODERN SPLIT DASHBOARD LAYOUT ---
    main_container = tk.Frame(win, bg=THEMES[current_theme]["bg"])
    main_container.pack(fill="both", expand=True, padx=20, pady=15)
    
    # Left Column: Search & Registry Database
    left_card = tk.LabelFrame(main_container, text=" Global Identity Registry ", 
                             bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"],
                             font=("Segoe UI", 11, "bold"), padx=15, pady=10, bd=1, relief="solid", width=550)
    left_card.pack_propagate(False)
    left_card.pack(side="left", fill="both", padx=(0, 10))

    # Right Column: Live Biometric Alignment Viewport
    right_card = tk.LabelFrame(main_container, text=" Live Biometric Alignment Viewport ", 
                              bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"],
                              font=("Segoe UI", 11, "bold"), padx=15, pady=10, bd=1, relief="solid")
    right_card.pack(side="right", fill="both", expand=True, padx=(10, 0))

    # --- LEFT PANE: REGISTRY BROWSER ---
    search_frame = tk.Frame(left_card, bg=THEMES[current_theme]["card_bg"])
    search_frame.pack(fill="x", pady=(0, 10))
    tk.Label(search_frame, text="Search:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(side="left")
    search_var = tk.StringVar()
    search_entry = tk.Entry(search_frame, textvariable=search_var, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"], font=("Segoe UI", 11), bd=1, relief="solid")
    search_entry.pack(side="left", fill="x", expand=True, padx=(5, 0), ipady=3)

    # Treeview Columns Configuration
    cols = ("ID", "Reg No", "Name", "Type")
    tv_registry = ttk.Treeview(left_card, columns=cols, show="headings", height=20)
    tv_registry.heading("ID", text="DB ID"); tv_registry.column("ID", width=50, anchor="center")
    tv_registry.heading("Reg No", text="Registration"); tv_registry.column("Reg No", width=120, anchor="center")
    tv_registry.heading("Name", text="Full Name"); tv_registry.column("Name", width=200, anchor="w")
    tv_registry.heading("Type", text="Type"); tv_registry.column("Type", width=100, anchor="center")
    tv_registry.pack(fill="both", expand=True, pady=5)

    def load_registry(event=None):
        tv_registry.delete(*tv_registry.get_children())
        query = search_var.get().strip().lower()
        with get_conn() as conn:
            df = pd.read_sql_query("""
                SELECT id, reg_no, name, 'student' as type FROM students
                UNION ALL
                SELECT id, COALESCE(external_ref, reg_no) as reg_no, name, person_type as type FROM people WHERE person_type != 'student'
                ORDER BY name
            """, conn)
        
        for _, r in df.iterrows():
            if query and query not in str(r["reg_no"]).lower() and query not in str(r["name"]).lower():
                continue
            tv_registry.insert("", tk.END, values=(r["id"], r["reg_no"], r["name"], r["type"]))
            
    search_var.trace_add("write", lambda *args: load_registry())
    load_registry()

    target_frame = tk.Frame(left_card, bg="#1a1a1a", bd=1, relief="solid", pady=10)
    target_frame.pack(fill="x", pady=10)
    lbl_target = tk.Label(target_frame, text="TARGET: NONE SELECTED", font=("Segoe UI", 12, "bold"), bg="#1a1a1a", fg="#ff4444")
    lbl_target.pack()

    # --- POPULATE OLD BIOMETRICS FOR VERIFICATION ON SELECT ---
    def on_person_select(event):
        sel = tv_registry.selection()
        if not sel: return
        vals = tv_registry.item(sel[0])["values"]
        selected_person.update({"id": vals[0], "reg": vals[1], "name": vals[2], "type": vals[3]})
        lbl_target.config(text=f"TARGET: {vals[1]} | {vals[2]}", fg="#00ff00")
        
        # Pull historical biometric fingerprint array to run matching checks against live camera stream
        target_historical_embedding[0] = None
        live_verification_score.set("Verification: Evaluating Live Stream...")
        with get_conn() as conn:
            if vals[3] == "student":
                row = conn.cursor().execute("SELECT embedding FROM students WHERE id=?", (vals[0],)).fetchone()
            else:
                row = conn.cursor().execute("SELECT embedding FROM people WHERE id=?", (vals[0],)).fetchone()
                
        if row and row[0] is not None:
            target_historical_embedding[0] = np.frombuffer(row[0], dtype=np.float32)
        else:
            live_verification_score.set("Verification: NO PREVIOUS BIOMETRICS FOUND (New Enrollment)")
        
    tv_registry.bind("<<TreeviewSelect>>", on_person_select)

    # --- RIGHT PANE: HARDWARE ROUTING & ALIGNMENT ---
    routing_frame = tk.Frame(right_card, bg=THEMES[current_theme]["card_bg"])
    routing_frame.pack(fill="x", pady=(0, 10))
    tk.Label(routing_frame, text="Active Viewport Source:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(side="left")
    
    selected_source = tk.StringVar()
    camera_combo = create_camera_device_selector(routing_frame, selected_source)
    camera_combo.pack(side="left", fill="x", expand=True, padx=(10, 0))

    # Clean Digital HUD Display Core Viewport
    feed_viewport = tk.Frame(right_card, bg="black", bd=1, relief="solid")
    feed_viewport.pack(fill="both", expand=True, pady=12)
    feed_viewport.pack_propagate(False)
    
    lbl_video = tk.Label(feed_viewport, bg="black", text="Initializing camera stream...", fg="#888", font=("Segoe UI", 10))
    lbl_video.place(relx=0.5, rely=0.5, anchor="center")

    # Dynamic Live Verification Widget Hud
    lbl_verify_hud = tk.Label(right_card, textvariable=live_verification_score, font=("Consolas", 11, "bold"), bg="#1a1a1a", fg="yellow", pady=6, bd=1, relief="solid")
    lbl_verify_hud.pack(fill="x", pady=5)

    def switch_routing_stream(*args):
        source = selected_source.get()
        if not source: return
        actual_src = int(source) if source.isdigit() else source
        cam_manager.start_camera(cam_slot_name, actual_src)

    selected_source.trace_add("write", switch_routing_stream)
    win.after(400, switch_routing_stream)

    def draw_alignment_reticle(frame):
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        box_w, box_h = 240, 320
        x1, y1 = center_x - box_w // 2, center_y - box_h // 2
        x2, y2 = center_x + box_w // 2, center_y + box_h // 2
        
        color = (0, 255, 255) 
        thick = 2; length = 30
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thick)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thick)
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thick)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thick)
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thick)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thick)
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thick)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thick)
        
        cv2.putText(frame, "ALIGN FACE WITHIN BRACKETS", (center_x - 110, y2 + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return frame

    # --- FIXED: RENAME CHANNELS TO RESOLVE NAMEERROR BUG ---
    def run_viewport_refresh():
        if win.winfo_exists():
            ret, frame = cam_manager.read_camera(cam_slot_name)
            if ret and frame is not None:
                display_frame = draw_hud(frame.copy(), "BIOMETRIC ALIGNMENT MODE")
                display_frame = draw_alignment_reticle(display_frame)
                
                # Dynamic real-time verification matching check directly on viewport loop
                faces = FA.get(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
                if len(faces) == 1:
                    fx1, fy1, fx2, fy2 = map(int, faces[0].bbox * 2)
                    
                    if target_historical_embedding[0] is not None:
                        # Compare live face array against loaded historical profile array structure
                        sim = float(face_similarity([target_historical_embedding[0]], faces[0].embedding)[0])
                        match_percent = sim * 100
                        
                        if sim >= SIMILARITY_THRESHOLD:
                            hud_color = (0, 255, 0) # Green border box for match verification
                            live_verification_score.set(f"Verification: IDENTITY CONFIRMED MATCH ({match_percent:.1f}%)")
                            lbl_verify_hud.config(fg="#00ff00")
                        else:
                            hud_color = (0, 0, 255) # Red warning color box on mismatch tracking
                            live_verification_score.set(f"CRITICAL MISMATCH ALERT: UNEXPECTED TARGET PROFILE ({match_percent:.1f}%)")
                            lbl_verify_hud.config(fg="#ff4444")
                    else:
                        hud_color = (255, 165, 0) # Orange if starting clean enrollment check
                        if selected_person["id"]:
                            live_verification_score.set("Verification: New Profile Structure Detected (No history setup)")
                            lbl_verify_hud.config(fg="orange")
                    
                    cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), hud_color, 2)
                else:
                    if selected_person["id"] and target_historical_embedding[0] is not None:
                        live_verification_score.set("Verification: Target missing from viewport visibility area...")
                        lbl_verify_hud.config(fg="yellow")

                # FIXED: Reads container coordinates cleanly via feed_viewport box reference
                w, h = feed_viewport.winfo_width(), feed_viewport.winfo_height()
                resized = resize_to_fit(display_frame, w, h)
                imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
                lbl_video.imgtk = imgtk
                lbl_video.configure(image=imgtk, text="")
            win.after(33, run_viewport_refresh)
    run_viewport_refresh()

    # --- ADVANCED BIOMETRIC COMMIT TRANSACTION ENGINE ---
    def commit_new_biometrics():
        if not selected_person["id"]:
            messagebox.showerror("Validation Fault", "You must select a target profile from the Global Registry first.")
            return

        ret, frame = cam_manager.read_camera(cam_slot_name)
        if not ret or frame is None:
            messagebox.showerror("Hardware Fault", "Cannot capture frame. Stream interrupted.")
            return

        # Core central extraction handler pipeline invocation
        success, embedding_bytes, cropped_img, msg = extract_core_biometrics(frame, enforce_single=True)
        if not success:
            messagebox.showerror("Extraction Failed", msg)
            return

        new_emb = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        # Security Matrix: Run global collision array scan for cross-entity matches
        sims = face_similarity(KNOWN_EMBEDDINGS, new_emb)
        if sims.size > 0 and sims.max() >= 0.55: 
            match_idx = int(sims.argmax())
            matched_global_id = KNOWN_IDS[match_idx]
            
            is_same_person = False
            if selected_person["type"] == "student" and matched_global_id == selected_person["id"]:
                is_same_person = True
            elif selected_person["type"] != "student" and KNOWN_PERSON_IDS[match_idx] == selected_person["id"]:
                is_same_person = True

            if not is_same_person:
                matched_label = KNOWN_LABELS[match_idx]
                ans = messagebox.askyesno("CRITICAL SECURITY COLLISION", 
                                          f"DUPLICATE BIOMETRIC SIGNAL ENCOUNTERED!\n\n"
                                          f"This live face footprint profile is registered to:\n[{matched_label}]\n\n"
                                          f"Are you sure you want to duplicate this link onto '{selected_person['name']}'?")
                if not ans: return

        p_id = selected_person["id"]
        reg = selected_person["reg"]
        
        with get_conn() as conn:
            cur = conn.cursor()
            if selected_person["type"] == "student":
                photo_path = os.path.join("photos", f"{reg}.jpg")
                cv2.imwrite(photo_path, cropped_img)
                cur.execute("UPDATE students SET embedding=?, photo_path=? WHERE id=?", (embedding_bytes, photo_path, p_id))
                cur.execute("UPDATE people SET embedding=?, photo_path=? WHERE person_type='student' AND external_ref=?", (embedding_bytes, photo_path, str(p_id)))
            else:
                photo_path = os.path.join("photos", f"person_{p_id}.jpg")
                cv2.imwrite(photo_path, cropped_img)
                cur.execute("UPDATE people SET embedding=?, photo_path=? WHERE id=?", (embedding_bytes, photo_path, p_id))
            conn.commit()

        # Update cache matrices across live tracking memory pipelines instantly
        load_server_memory()
        play_success_beep()
        
        # Update current view states seamlessly
        target_historical_embedding[0] = new_emb
        messagebox.showinfo("Biometrics Updated", f"Successfully committed fresh centralized biometrics layer onto '{selected_person['name']}'.")

    tk.Button(right_card, text="📸 CAPTURE & OVERWRITE BIOMETRICS", command=commit_new_biometrics, 
              bg="#d9534f", fg="white", font=("Segoe UI", 12, "bold"), bd=0, cursor="hand2", pady=12).pack(fill="x", side="bottom", pady=5)
    

# ---------- 1. ENROLLMENT ----------
def open_enrollment():
    win = tk.Toplevel(root); win.title("Enrollment Server"); win.geometry("1100x650")
    open_windows["enrollment"] = win; win.configure(bg=THEMES[current_theme]["bg"])
    add_window_toolbar(win, "enrollment", stop_camera_instance="enrollment")

    form = tk.Frame(win, bg=THEMES[current_theme]["bg"], width=350)
    form.pack_propagate(False) 
    form.pack(side="left", padx=20, pady=20, fill="y")
    
    selected_source = tk.StringVar()
    camera_combo = create_camera_device_selector(form, selected_source)
    
    def switch_enrollment_camera(_event=None):
        source = selected_source.get()
        if not source: return
        actual_src = int(source) if str(source).isdigit() else source
        cam_manager.start_camera("enrollment", actual_src)

    camera_combo.bind("<<ComboboxSelected>>", switch_enrollment_camera, add="+")
    win.after(500, switch_enrollment_camera)
    
    tk.Label(form, text="Register Student", font=("Segoe UI", 18, "bold"), bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(pady=(0, 10))

    vars_map = {}
    with get_conn() as conn: courses = [r[0] for r in conn.cursor().execute("SELECT DISTINCT course FROM students").fetchall()]
    for label in ["Reg No", "Name", "Course", "Mobile"]:
        tk.Label(form, text=f"{label}:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w")
        ent = ttk.Combobox(form, values=courses) if label == "Course" else tk.Entry(form, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"])
        ent.pack(fill="x", pady=5); vars_map[label] = ent

    video_frame = tk.Frame(win, bg="black"); video_frame.pack(side="right", padx=20, pady=20, fill="both", expand=True)
    video_frame.pack_propagate(False)
    lbl_video = tk.Label(video_frame, bg="black")
    lbl_video.pack(fill="both", expand=True)
    
    def update_cam():
        ret, frame = cam_manager.read_camera("enrollment")
        if ret and frame is not None:
            frame = cv2.resize(frame, (640, 480))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = ImageTk.PhotoImage(Image.fromarray(frame))
            lbl_video.configure(image=img, text="")
            lbl_video.image = img
        if win.winfo_exists():
            win.after(30, update_cam)
    update_cam()

    def save_student():
        reg, name, crs, mob = (vars_map[k].get().strip() for k in ["Reg No", "Name", "Course", "Mobile"])
        if not all([reg, name, crs, mob]): return messagebox.showerror("Error", "All fields required")
        with get_conn() as conn:
            if conn.cursor().execute("SELECT 1 FROM students WHERE reg_no=?", (reg,)).fetchone():
                return messagebox.showerror("Error", "Student exists!")

        ret, frame = cam_manager.read_camera("enrollment")
        if not ret: return messagebox.showerror("Error", "Camera offline")

        faces = FA.get(frame)
        if len(faces) == 0: return messagebox.showerror("Error", "No face detected.")
        if len(faces) > 1: return messagebox.showerror("Error", "Ensure exactly ONE face is in the frame.")
        
        new_emb = faces[0].embedding; is_twin = 0
        sims = face_similarity(KNOWN_EMBEDDINGS, new_emb)
        if sims.size > 0 and sims.max() >= SIMILARITY_THRESHOLD:
            ans = messagebox.askyesnocancel("Duplicate Scan", f"DUPLICATE DETECTED!\nMatches: {KNOWN_LABELS[int(sims.argmax())]} ({sims.max()*100:.1f}%)\nIs this a Twin? Override?")
            if not ans: return
            is_twin = 1

        photo_path, qr_path = os.path.join("photos", f"{reg}.jpg"), os.path.join("qrcodes", f"{reg}.png")
        cv2.imwrite(photo_path, frame); qrcode.make(reg).save(qr_path)
        
        with get_conn() as conn:
            conn.cursor().execute("INSERT INTO students (reg_no, name, course, mobile, photo_path, qr_path, embedding, is_twin) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                                  (reg, name, crs, mob, photo_path, qr_path, new_emb.tobytes(), is_twin))
            student_id = conn.cursor().execute("SELECT id FROM students WHERE reg_no=?", (reg,)).fetchone()[0]
            conn.cursor().execute("""INSERT OR IGNORE INTO people
                (person_type, external_ref, reg_no, name, course, mobile, photo_path, qr_path, embedding, is_twin)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                ("student", str(student_id), reg, name, crs, mob, photo_path, qr_path, new_emb.tobytes(), is_twin))
            conn.commit()
        load_server_memory()
        messagebox.showinfo("Success", f"{name} Enrolled.")
        for e in vars_map.values(): e.delete(0, tk.END)

    tk.Button(form, text="📷 Scan & Enroll", command=save_student, bg="#4CAF50", fg="white", font=("Segoe UI", 11, "bold"), cursor="hand2").pack(pady=10, fill="x")

# ---------- 2. CENTRAL ATTENDANCE (4 TABS) ----------
def open_attendance1():
    win_name = "attendance"
    if win_name in open_windows and open_windows[win_name].winfo_exists():
        open_windows[win_name].deiconify()
        open_windows[win_name].lift()
        return

    win = tk.Toplevel(root)
    win.title("Central Attendance Operations Command")
    win.geometry("1350x840")
    open_windows[win_name] = win
    win.configure(bg=THEMES[current_theme]["bg"])
    
    cam_slot_name = "attendance_hub_stream"
    add_window_toolbar(win, win_name, stop_camera_instance=cam_slot_name)

    # Core Execution & Metric Trackers
    fut = None
    proc_ctr = 0
    group_marked = set()
    course_marked = set()
    cctv_memory = {}
    visual_flashes = {}
    target_emb_s = [None]
    target_sid_s = [None]
    
    # Advanced System Diagnostics States
    session_start_time = time.time()
    fps_last_time = time.time()
    fps_frame_count = 0
    live_fps_metric = tk.StringVar(value="Stream: -- FPS")
    ai_latency_metric = tk.StringVar(value="AI Latency: -- ms")
    uptime_metric = tk.StringVar(value="Uptime: 00:00")
    known_session_counter = tk.IntVar(value=0)
    unknown_session_counter = tk.IntVar(value=0)

    # Clean UI Styles
    style = ttk.Style()
    style.configure("Treeview", 
                    background=THEMES[current_theme]["tree_bg"], 
                    foreground=THEMES[current_theme]["tree_fg"], 
                    fieldbackground=THEMES[current_theme]["tree_bg"],
                    font=("Segoe UI", 10), rowheight=28)
    style.configure("Treeview.Heading", 
                    background=THEMES[current_theme]["tree_header_bg"], 
                    foreground=THEMES[current_theme]["tree_header_fg"],
                    font=("Segoe UI", 10, "bold"))

    # Unified Master Control Frame (FIXED BUG: Removed invalid 'text=' argument)
    control_strip = tk.Frame(win, bg=THEMES[current_theme]["card_bg"], bd=1, relief="solid")
    control_strip.pack(fill="x", padx=15, pady=10, ipady=5)
    
    tk.Label(control_strip, text=" Active Capture Node Selector: ", 
             bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"], 
             font=("Segoe UI", 10, "bold")).pack(side="left", padx=10)
             
    selected_source = tk.StringVar()
    camera_combo = create_camera_device_selector(control_strip, selected_source)
    camera_combo.pack(side="left", padx=5)

    # Professional Telemetry Diagnostics HUD Bar
    telemetry_bar = tk.Frame(control_strip, bg=THEMES[current_theme]["card_bg"])
    telemetry_bar.pack(side="right", padx=15)
    tk.Label(telemetry_bar, textvariable=uptime_metric, fg="#ffffff", bg=THEMES[current_theme]["card_bg"], font=("Consolas", 10, "bold")).pack(side="left", padx=10)
    tk.Label(telemetry_bar, textvariable=live_fps_metric, fg="#00e5ff", bg=THEMES[current_theme]["card_bg"], font=("Consolas", 10, "bold")).pack(side="left", padx=10)
    tk.Label(telemetry_bar, textvariable=ai_latency_metric, fg="#ffeb3b", bg=THEMES[current_theme]["card_bg"], font=("Consolas", 10, "bold")).pack(side="left", padx=10)

    # Master Tabbed Workspace
    nb = ttk.Notebook(win)
    nb.pack(fill="both", expand=True, padx=15, pady=(0, 15))

    def build_dashboard_pane(parent_notebook, tab_title):
        pane = tk.Frame(parent_notebook, bg=THEMES[current_theme]["bg"])
        parent_notebook.add(pane, text=f"  {tab_title}  ")
        
        inner_container = tk.Frame(pane, bg=THEMES[current_theme]["bg"])
        inner_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        viewport_frame = tk.Frame(inner_container, bg="black", bd=1, relief="solid")
        viewport_frame.pack_propagate(False)
        viewport_frame.pack(side="left", fill="both", expand=True)
        
        lbl_canvas = tk.Label(viewport_frame, bg="black")
        lbl_canvas.place(relx=0.5, rely=0.5, anchor="center")
        
        data_card = tk.LabelFrame(inner_container, text=" Operational Metrics & Live Analytics ", 
                                  bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"],
                                  font=("Segoe UI", 11, "bold"), padx=15, pady=12, bd=1, relief="solid", width=440)
        data_card.pack_propagate(False)
        data_card.pack(side="right", fill="y", padx=(12, 0))
        
        # Real-time Summary Counters inside Data Card
        counters_box = tk.Frame(data_card, bg=THEMES[current_theme]["card_bg"])
        counters_box.pack(fill="x", pady=(0, 10))
        tk.Label(counters_box, text="Known Processed:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).grid(row=0, column=0, sticky="w")
        tk.Label(counters_box, textvariable=known_session_counter, fg="#00ff00", bg=THEMES[current_theme]["card_bg"], font=("Segoe UI", 11, "bold")).grid(row=0, column=1, sticky="w", padx=5)
        tk.Label(counters_box, text="Security Alerts:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).grid(row=0, column=2, sticky="w", padx=(20, 0))
        tk.Label(counters_box, textvariable=unknown_session_counter, fg="#ff4444", bg=THEMES[current_theme]["card_bg"], font=("Segoe UI", 11, "bold")).grid(row=0, column=3, sticky="w", padx=5)
        
        return lbl_canvas, data_card, viewport_frame

    # --- DESK 1: SINGLE SCAN WORKSPACE ---
    lbl_cam_s, side_s, canvas_frame_s = build_dashboard_pane(nb, "Single Desk Checkpoint")
    
    reg_var_s, name_var_s = tk.StringVar(), tk.StringVar()
    tk.Label(side_s, text="Target Registration Signature / Reg No:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w", pady=(5,0))
    tk.Entry(side_s, textvariable=reg_var_s, font=("Segoe UI", 12), bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"], bd=1, relief="solid").pack(fill="x", pady=4, ipady=3)
    
    tk.Label(side_s, text="Verified Profile Name Identity:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w", pady=(10,0))
    tk.Entry(side_s, textvariable=name_var_s, state="readonly", font=("Segoe UI", 12, "bold")).pack(fill="x", pady=4, ipady=3)
    
    s_status = tk.Label(side_s, text="SYSTEM STATUS: IDLE", fg="#00ffcc", bg=THEMES[current_theme]["card_bg"], font=("Segoe UI", 10, "bold"), bd=1, relief="solid", pady=6)
    s_status.pack(fill="x", pady=15)

    def async_fetch_profile_signature():
        regno = reg_var_s.get().strip()
        if not regno: return
        with get_conn() as conn: 
            rec = conn.cursor().execute("SELECT id, name, embedding FROM students WHERE reg_no=?", (regno,)).fetchone()
        if not rec: 
            s_status.config(text="STATUS FAILURE: SIGNAL REFUSED / NOT FOUND", fg="#ff4444")
            return
        name_var_s.set(rec[1])
        target_sid_s[0] = rec[0]
        if rec[2] is None:
            messagebox.showerror("Biometric Fault", "Profile lacks a biometric registration array. Complete enrollment first.")
            return
        target_emb_s[0] = np.frombuffer(rec[2], dtype=np.float32)
        
        today = datetime.now().strftime("%Y-%m-%d")
        with get_conn() as conn:
            already = conn.cursor().execute("SELECT time FROM attendance WHERE student_id=? AND date=?", (target_sid_s[0], today)).fetchone()
        if already:
            s_status.config(text=f"VERIFIED: ALREADY RECORDED AT {already[0]}", fg="#00e5ff")
        else:
            s_status.config(text="BIOMETRICS ENGAGED: SCANNING VIEWPORT FEED...", fg="yellow")

    tk.Button(side_s, text="🔍 Interrogate Signature", command=async_fetch_profile_signature, bg="#14818f", fg="white", font=("Segoe UI", 10, "bold"), bd=0, cursor="hand2", pady=6).pack(fill="x", pady=4)
    tk.Button(side_s, text="🗂 Open Full Profile Details", command=lambda: open_student_detail_window(reg_no=reg_var_s.get().strip()), bg="#444", fg="white", font=("Segoe UI", 10), bd=0, cursor="hand2", pady=6).pack(fill="x", pady=4)
    
    def inline_qr_scaffolder():
        s_status.config(text="DECODING MATRIX PATTERNS FROM VIDEO CORE...", fg="magenta")
        ret, frame = cam_manager.read_camera(cam_slot_name)
        if ret and frame is not None:
            codes = decode(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if codes: 
                reg = codes[0].data.decode("utf-8").strip()
                reg_var_s.set(reg)
                async_fetch_profile_signature()

    tk.Button(side_s, text="𔖔 Scan Physical Matrix QR Pass", command=inline_qr_scaffolder, bg="#7b1fa2", fg="white", font=("Segoe UI", 10, "bold"), bd=0, cursor="hand2", pady=6).pack(fill="x", pady=(15, 0))

    # --- DESK 2: BATCH GROUP WORKSPACE ---
    lbl_cam_g, side_g, canvas_frame_g = build_dashboard_pane(nb, "Mass Batch Processing")
    
    tv_g = ttk.Treeview(side_g, columns=("Reg No", "Name", "Time Signature"), show="headings")
    for col in ("Reg No", "Name", "Time Signature"): 
        tv_g.heading(col, text=col)
        tv_g.column(col, anchor="center")
    tv_g.pack(fill="both", expand=True, pady=5)
    
    g_status = tk.Label(side_g, text="BATCH LOG READY", fg="#00ffcc", bg=THEMES[current_theme]["card_bg"], font=("Segoe UI", 9, "bold"))
    g_status.pack(fill="x", pady=4)

    def load_group_logs_async():
        tv_g.delete(*tv_g.get_children())
        today = datetime.now().strftime("%Y-%m-%d")
        with get_conn() as conn:
            rows = conn.cursor().execute("""SELECT s.id, s.reg_no, s.name, a.time FROM attendance a 
                                            JOIN students s ON s.id=a.student_id WHERE a.date=? ORDER BY a.time DESC""", (today,)).fetchall()
        for sid, reg, name, marked_time in rows:
            tv_g.insert("", tk.END, iid=f"sid:{sid}", values=(reg, name, marked_time))
        g_status.config(text=f"METRICS: {len(rows)} RECORDED IN SECURE LOG", fg="#00ffcc")

    tk.Button(side_g, text="🔄 Synchronize Real-Time Log", command=load_group_logs_async, bg="#444", fg="white", font=("Segoe UI", 10), bd=0, cursor="hand2", pady=6).pack(fill="x", pady=4)
    load_group_logs_async()

    # --- DESK 3: COURSE RESTRICTED ROUTINE ---
    lbl_cam_cr, side_cr, canvas_frame_cr = build_dashboard_pane(nb, "Academic Course Node")
    
    tk.Label(side_cr, text="Filter Target Course Profile Architecture:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w")
    crs_var = tk.StringVar()
    crs_combo = ttk.Combobox(side_cr, textvariable=crs_var, values=list(set(KNOWN_COURSES)), state="readonly", font=("Segoe UI", 11))
    crs_combo.pack(fill="x", pady=6)
    
    tv_cr = ttk.Treeview(side_cr, columns=("Reg No", "Name", "Logged At"), show="headings")
    for col in ("Reg No", "Name", "Logged At"): 
        tv_cr.heading(col, text=col)
        tv_cr.column(col, anchor="center")
    tv_cr.pack(fill="both", expand=True, pady=5)

    active_course_embs, active_course_ids, active_course_labels = [], [], []
    active_course_person_ids, active_course_person_types = [], []
    
    def apply_course_matrix_filter(e=None):
        active_course_embs.clear(); active_course_ids.clear(); active_course_labels.clear()
        active_course_person_ids.clear(); active_course_person_types.clear(); tv_cr.delete(*tv_cr.get_children())
        sel_c = crs_var.get()
        for i in range(len(KNOWN_COURSES)):
            if KNOWN_COURSES[i] == sel_c:
                active_course_embs.append(KNOWN_EMBEDDINGS[i]); active_course_ids.append(KNOWN_IDS[i]); active_course_labels.append(KNOWN_LABELS[i])
                active_course_person_ids.append(KNOWN_PERSON_IDS[i]); active_course_person_types.append(KNOWN_PERSON_TYPES[i])
    crs_combo.bind("<<ComboboxSelected>>", apply_course_matrix_filter)

    # --- DESK 4: MISSION CRITICAL CCTV SURVEILLANCE ---
    lbl_cam_c, side_c, canvas_frame_c = build_dashboard_pane(nb, "CCTV Intelligence Scanner")
    
    tk.Label(side_c, text="Assign CCTV Hardware Descriptor Signature:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w")
    cctv_name_var = tk.StringVar(value="SECURE_CAM_NODE_01")
    tk.Entry(side_c, textvariable=cctv_name_var, font=("Segoe UI", 11), bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"], bd=1, relief="solid").pack(fill="x", pady=6, ipady=3)

    tv_c = ttk.Treeview(side_c, columns=("Entity Identity", "Hardware Node", "Intercept Time"), show="headings")
    for col in ("Entity Identity", "Hardware Node", "Intercept Time"): 
        tv_c.heading(col, text=col)
        tv_c.column(col, anchor="center")
    tv_c.pack(fill="both", expand=True, pady=5)
    
    cctv_status = tk.Label(side_c, text="CCTV SURVEILLANCE PASSIVE DETECTION EYE ACTIVE", fg="#00ffcc", bg=THEMES[current_theme]["card_bg"], font=("Segoe UI", 9, "bold"), wraplength=380)
    cctv_status.pack(fill="x", pady=4)

    # --- DETACHED BACKGROUND AI THREAD POOL WORKER ---
    def ai_worker(frame_copy, embs, ids, labels):
        t_start = time.time()
        try: faces = FA.get(cv2.resize(frame_copy, (0, 0), fx=0.5, fy=0.5))
        except Exception: return [], 0
        res = []
        latency = int((time.time() - t_start) * 1000)
        for face in faces:
            sims = face_similarity(embs, face.embedding)
            if sims.size > 0 and sims.max() >= SIMILARITY_THRESHOLD:
                res.append({"box": face.bbox*2, "idx": int(sims.argmax()), "sim": float(sims.max())})
            else:
                best = float(sims.max()) if sims.size > 0 else 0.0
                res.append({"box": face.bbox*2, "unknown": True, "sim": best})
        return res, latency

    def async_db_injector(query, params, reload_callback=None):
        def task():
            with get_conn() as conn:
                conn.cursor().execute(query, params)
                conn.commit()
            if reload_callback: win.after(10, reload_callback)
        threading.Thread(target=task, daemon=True).start()

    def hot_swap_stream_route(*args):
        source = selected_source.get()
        if not source: return
        actual_src = int(source) if source.isdigit() else source
        cam_manager.start_camera(cam_slot_name, actual_src)

    selected_source.trace_add("write", hot_swap_stream_route)
    win.after(400, hot_swap_stream_route)

    # --- SYSTEM COMMAND RENDERING ENGINE LOOP ---
    def core_render_pipeline_loop():
        nonlocal fut, proc_ctr, group_marked, course_marked, visual_flashes, fps_frame_count, fps_last_time
        ret, frame = cam_manager.read_camera(cam_slot_name)
        cur_tab = nb.tab(nb.select(), "text").strip()

        # Update Telemetry Dashboards
        now_time = time.time()
        uptime_seconds = int(now_time - session_start_time)
        uptime_metric.set(f"Uptime: {uptime_seconds // 60:02d}:{uptime_seconds % 60:02d}")
        
        fps_frame_count += 1
        if now_time - fps_last_time >= 1.0:
            live_fps_metric.set(f"Stream: {fps_frame_count} FPS")
            fps_frame_count = 0
            fps_last_time = now_time

        if ret and frame is not None:
            now_dt = datetime.now()
            dt_s, tm_s = now_dt.strftime("%Y-%m-%d"), now_dt.strftime("%H:%M:%S")
            display = frame.copy()
            proc_ctr = (proc_ctr + 1) % PROCESS_EVERY_N
            visual_flashes = {k: v for k, v in visual_flashes.items() if (now_dt.timestamp() - v) < 2.0}

            if cur_tab == "Single Desk Checkpoint":
                display = draw_hud(display, "LOCAL SCAN NODE")
                w, h = canvas_frame_s.winfo_width(), canvas_frame_s.winfo_height()
                if target_emb_s[0] is not None and proc_ctr == 0:
                    faces = FA.get(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
                    if faces:
                        sim = float(face_similarity([target_emb_s[0]], faces[0].embedding)[0])
                        if sim >= SIMILARITY_THRESHOLD:
                            s_status.config(text=f"MATCH CONFIRMED: {sim*100:.1f}% — COMMITTING TO STORAGE", fg="#00ff00")
                            play_success_beep()
                            sid = target_sid_s[0]
                            
                            q = "INSERT OR IGNORE INTO attendance (student_id, date, time, match_percentage, camera_location) VALUES (?, ?, ?, ?, ?)"
                            p = (sid, dt_s, tm_s, sim*100, "Single Desk Checkpoint")
                            known_session_counter.set(known_session_counter.get() + 1)
                            async_db_injector(q, p, lambda: s_status.config(text=f"SUCCESS: ENTRY REGISTERED AT {tm_s}", fg="#00ff00"))
                            target_emb_s[0] = None
                
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resize_to_fit(display, w, h), cv2.COLOR_BGR2RGB)))
                lbl_cam_s.imgtk = img; lbl_cam_s.configure(image=img, text="")

            elif cur_tab == "Mass Batch Processing":
                display = draw_hud(display, "BATCH CAPTURE ARRAY")
                w, h = canvas_frame_g.winfo_width(), canvas_frame_g.winfo_height()
                if fut is None or fut.done():
                    if fut and fut.result():
                        results, latency = fut.result()
                        ai_latency_metric.set(f"AI Latency: {latency}ms")
                        for m in results:
                            if m.get("unknown"):
                                x1,y1,x2,y2 = map(int, m["box"])
                                cv2.rectangle(display, (x1,y1), (x2,y2), (0, 165, 255), 2)
                                continue
                            idx = m["idx"]; sid = KNOWN_IDS[idx]
                            if sid is None: continue
                            mark_key = f"student:{sid}|{dt_s}"
                            if mark_key not in group_marked:
                                visual_flashes[mark_key] = now_dt.timestamp()
                                group_marked.add(mark_key)
                                r, n = KNOWN_LABELS[idx].split("|")
                                
                                q = "INSERT OR IGNORE INTO attendance (student_id, date, time, match_percentage, camera_location) VALUES (?, ?, ?, ?, ?)"
                                p = (sid, dt_s, tm_s, m["sim"]*100, "Batch Processing Matrix")
                                known_session_counter.set(known_session_counter.get() + 1)
                                async_db_injector(q, p, load_group_logs_async)
                                play_success_beep()
                            
                            x1,y1,x2,y2 = map(int, m["box"])
                            color = (0, 255, 0) if mark_key in visual_flashes else (0, 184, 212)
                            cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                            cv2.putText(display, f"{KNOWN_LABELS[idx].split('|')[1].strip()}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    fut = ai_executor.submit(ai_worker, frame.copy(), KNOWN_EMBEDDINGS, KNOWN_IDS, KNOWN_LABELS)
                
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resize_to_fit(display, w, h), cv2.COLOR_BGR2RGB)))
                lbl_cam_g.imgtk = img; lbl_cam_g.configure(image=img, text="")

            elif cur_tab == "Academic Course Node":
                display = draw_hud(display, f"COURSE: {crs_var.get()}")
                w, h = canvas_frame_cr.winfo_width(), canvas_frame_cr.winfo_height()
                if active_course_embs and (fut is None or fut.done()):
                    if fut and fut.result():
                        results, latency = fut.result()
                        ai_latency_metric.set(f"AI Latency: {latency}ms")
                        for m in results:
                            if m.get("unknown"):
                                x1,y1,x2,y2 = map(int, m["box"])
                                cv2.rectangle(display, (x1,y1), (x2,y2), (0, 105, 217), 2)
                                continue
                            
                            global_idx = m["idx"]
                            sid = KNOWN_IDS[global_idx]
                            person_id = KNOWN_PERSON_IDS[global_idx]
                            person_type = KNOWN_PERSON_TYPES[global_idx]
                            label_str = KNOWN_LABELS[global_idx]
                            
                            if label_str not in active_course_labels: continue
                                
                            mark_key = f"person:{person_id}" if person_id is not None else f"student:{sid}"
                            if mark_key not in course_marked:
                                play_success_beep()
                                visual_flashes[mark_key] = now_dt.timestamp()
                                course_marked.add(mark_key)
                                known_session_counter.set(known_session_counter.get() + 1)
                                
                                r, n = label_str.split("|")
                                def inject_course(s_id=sid, p_id=person_id, p_type=person_type, lbl_r=r, lbl_n=n):
                                    with get_conn() as conn:
                                        if s_id is not None:
                                            conn.cursor().execute("INSERT OR IGNORE INTO attendance (student_id, date, time, match_percentage, camera_location) VALUES (?, ?, ?, ?, ?)", (s_id, dt_s, tm_s, m["sim"]*100, f"Course: {crs_var.get()}"))
                                        conn.cursor().execute("INSERT INTO attendance_logs (person_id, legacy_student_id, person_type, date, time, camera_name, camera_location, match_percentage, status, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (p_id, s_id, p_type, dt_s, tm_s, f"Course: {crs_var.get()}", "Class Terminal", m["sim"]*100, "official", "Dynamic Course Node Processing"))
                                        conn.commit()
                                    win.after(10, lambda: tv_cr.insert("", 0, values=(lbl_r.strip(), lbl_n.strip(), tm_s)))
                                threading.Thread(target=inject_course, daemon=True).start()
                            
                            x1,y1,x2,y2 = map(int, m["box"])
                            color = (0, 255, 0) if mark_key in visual_flashes else (224, 224, 224)
                            cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                            cv2.putText(display, f"{label_str.split('|')[1].strip()}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    fut = ai_executor.submit(ai_worker, frame.copy(), KNOWN_EMBEDDINGS, KNOWN_IDS, KNOWN_LABELS)
                
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resize_to_fit(display, w, h), cv2.COLOR_BGR2RGB)))
                lbl_cam_cr.imgtk = img; lbl_cam_cr.configure(image=img, text="")

            elif cur_tab == "CCTV Intelligence Scanner":
                cam_name = cctv_name_var.get().strip()
                display = draw_hud(display, cam_name)
                w, h = canvas_frame_c.winfo_width(), canvas_frame_c.winfo_height()
                
                if fut is None or fut.done():
                    if fut and fut.result():
                        results, latency = fut.result()
                        ai_latency_metric.set(f"AI Latency: {latency}ms")
                        for m in results:
                            if m.get("unknown"):
                                unknown_session_counter.set(unknown_session_counter.get() + 1)
                                threading.Thread(target=log_unknown_event, args=(cam_name, "cctv_surveillance", frame, "alert", "security_alert", "Unknown structural profile intercept", m["sim"]*100), daemon=True).start()
                                x1,y1,x2,y2 = map(int, m["box"])
                                cv2.rectangle(display, (x1,y1), (x2,y2), (0, 0, 255), 2)
                                continue
                            idx = m["idx"]; sid, name = KNOWN_IDS[idx], KNOWN_LABELS[idx].split("|")[1]
                            person_id = KNOWN_PERSON_IDS[idx]
                            mark_key = f"person:{person_id}" if person_id is not None else f"student:{sid}"
                            if mark_key not in cctv_memory or (now_dt.timestamp() - cctv_memory[mark_key]) > CCTV_COOLDOWN_SECONDS:
                                play_success_beep()
                                visual_flashes[mark_key] = now_dt.timestamp()
                                cctv_memory[mark_key] = now_dt.timestamp()
                                known_session_counter.set(known_session_counter.get() + 1)
                                
                                def inject_surveillance(s_id=sid, p_id=person_id, g_idx=idx, name_str=name):
                                    with get_conn() as conn:
                                        if s_id is not None:
                                            conn.cursor().execute("INSERT INTO attendance (student_id, date, time, match_percentage, camera_location) VALUES (?, ?, ?, ?, ?)", (s_id, dt_s, tm_s, m["sim"]*100, cam_name))
                                        conn.cursor().execute("INSERT INTO camera_events (camera_name, event_type, mode, person_id, legacy_student_id, date, time, severity, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (cam_name, "known_face", "cctv_surveillance", p_id, s_id, dt_s, tm_s, "info", f"Intercept match matrix at {m['sim']*100:.1f}%"))
                                        conn.commit()
                                    log_surveillance_track(cam_name, "cctv_surveillance", "known_face", frame, p_id, s_id, KNOWN_PERSON_TYPES[g_idx], KNOWN_LABELS[g_idx], m["sim"]*100, "info", "recorded", "CCTV Engine Capture")
                                    win.after(10, lambda: tv_c.insert("", 0, values=(name_str.strip(), cam_name, tm_s)))
                                threading.Thread(target=inject_surveillance, daemon=True).start()
                            
                            x1,y1,x2,y2 = map(int, m["box"])
                            color = (0, 255, 255) if mark_key in visual_flashes else (158, 158, 158)
                            cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                            cv2.putText(display, f"{name.strip()}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    fut = ai_executor.submit(ai_worker, frame.copy(), KNOWN_EMBEDDINGS, KNOWN_IDS, KNOWN_LABELS)
                
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resize_to_fit(display, w, h), cv2.COLOR_BGR2RGB)))
                lbl_cam_c.imgtk = img; lbl_cam_c.configure(image=img, text="")

        if win.winfo_exists():
            win.after(33, core_render_pipeline_loop)
            
    core_render_pipeline_loop()

def open_attendance():
    win_name = "attendance"
    
    # BUG FIX: Singleton Window Enforcement
    if win_name in open_windows and open_windows[win_name].winfo_exists():
        open_windows[win_name].deiconify()
        open_windows[win_name].lift()
        return

    win = tk.Toplevel(root)
    win.title("Enterprise Attendance Command & Telemetry Center")
    win.geometry("1400x880")
    open_windows[win_name] = win
    win.configure(bg=THEMES[current_theme]["bg"])
    
    cam_slot_name = "attendance_hub_stream"
    add_window_toolbar(win, win_name, stop_camera_instance=cam_slot_name)

    # --- Core Execution & Metric Trackers ---
    fut = None
    proc_ctr = 0
    group_marked = set()
    course_marked = set()
    cctv_memory = {}
    visual_flashes = {}
    target_emb_s = [None]
    target_sid_s = [None]
    
    # Analytics Memory Arrays for Live Plotting
    live_confidence_scores = []
    
    # Advanced System Diagnostics States
    session_start_time = time.time()
    fps_last_time = time.time()
    fps_frame_count = 0
    live_fps_metric = tk.StringVar(value="Stream: -- FPS")
    ai_latency_metric = tk.StringVar(value="AI Latency: -- ms")
    uptime_metric = tk.StringVar(value="Uptime: 00:00")
    known_session_counter = tk.IntVar(value=0)
    unknown_session_counter = tk.IntVar(value=0)

    # Clean UI Styles
    style = ttk.Style()
    style.configure("Treeview", 
                    background=THEMES[current_theme]["tree_bg"], 
                    foreground=THEMES[current_theme]["tree_fg"], 
                    fieldbackground=THEMES[current_theme]["tree_bg"],
                    font=("Segoe UI", 10), rowheight=28)
    style.configure("Treeview.Heading", 
                    background=THEMES[current_theme]["tree_header_bg"], 
                    foreground=THEMES[current_theme]["tree_header_fg"],
                    font=("Segoe UI", 10, "bold"))
    style.configure("TNotebook", background=THEMES[current_theme]["bg"])
    style.configure("TNotebook.Tab", padding=[15, 5], font=("Segoe UI", 10, "bold"))

    # --- Unified Master Control Strip ---
    control_strip = tk.Frame(win, bg=THEMES[current_theme]["card_bg"], bd=1, relief="solid")
    control_strip.pack(fill="x", padx=20, pady=15, ipady=5)
    
    tk.Label(control_strip, text=" 🖧 Active Hardware Node: ", 
             bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"], 
             font=("Segoe UI", 11, "bold")).pack(side="left", padx=10)
             
    selected_source = tk.StringVar()
    camera_combo = create_camera_device_selector(control_strip, selected_source)
    camera_combo.config(width=40)
    camera_combo.pack(side="left", padx=5)

    # Professional Telemetry Diagnostics HUD Bar
    telemetry_bar = tk.Frame(control_strip, bg=THEMES[current_theme]["card_bg"])
    telemetry_bar.pack(side="right", padx=15)
    tk.Label(telemetry_bar, textvariable=uptime_metric, fg="#ffffff", bg=THEMES[current_theme]["card_bg"], font=("Consolas", 11, "bold")).pack(side="left", padx=15)
    tk.Label(telemetry_bar, textvariable=live_fps_metric, fg="#00e5ff", bg=THEMES[current_theme]["card_bg"], font=("Consolas", 11, "bold")).pack(side="left", padx=15)
    tk.Label(telemetry_bar, textvariable=ai_latency_metric, fg="#ffeb3b", bg=THEMES[current_theme]["card_bg"], font=("Consolas", 11, "bold")).pack(side="left", padx=15)

    # --- Master Tabbed Workspace ---
    nb = ttk.Notebook(win)
    nb.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    def build_dashboard_pane(parent_notebook, tab_title):
        pane = tk.Frame(parent_notebook, bg=THEMES[current_theme]["bg"])
        parent_notebook.add(pane, text=f"  {tab_title}  ")
        
        inner_container = tk.Frame(pane, bg=THEMES[current_theme]["bg"])
        inner_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        viewport_frame = tk.Frame(inner_container, bg="black", bd=1, relief="solid")
        viewport_frame.pack_propagate(False)
        viewport_frame.pack(side="left", fill="both", expand=True)
        
        lbl_canvas = tk.Label(viewport_frame, bg="black")
        lbl_canvas.place(relx=0.5, rely=0.5, anchor="center")
        
        data_card = tk.LabelFrame(inner_container, text=" Operational Metrics & Live Analytics ", 
                                  bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"],
                                  font=("Segoe UI", 11, "bold"), padx=15, pady=12, bd=1, relief="solid", width=500)
        data_card.pack_propagate(False)
        data_card.pack(side="right", fill="y", padx=(15, 0))
        
        counters_box = tk.Frame(data_card, bg=THEMES[current_theme]["card_bg"])
        counters_box.pack(fill="x", pady=(0, 10))
        tk.Label(counters_box, text="Verified Scans:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).grid(row=0, column=0, sticky="w")
        tk.Label(counters_box, textvariable=known_session_counter, fg="#00ff00", bg=THEMES[current_theme]["card_bg"], font=("Segoe UI", 11, "bold")).grid(row=0, column=1, sticky="w", padx=5)
        tk.Label(counters_box, text="Security Alerts:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).grid(row=0, column=2, sticky="w", padx=(20, 0))
        tk.Label(counters_box, textvariable=unknown_session_counter, fg="#ff4444", bg=THEMES[current_theme]["card_bg"], font=("Segoe UI", 11, "bold")).grid(row=0, column=3, sticky="w", padx=5)
        
        return lbl_canvas, data_card, viewport_frame

    # --- DESK 1: SINGLE SCAN WORKSPACE ---
    lbl_cam_s, side_s, canvas_frame_s = build_dashboard_pane(nb, "Single Desk Checkpoint")
    
    reg_var_s, name_var_s = tk.StringVar(), tk.StringVar()
    tk.Label(side_s, text="Target Registration Signature / Reg No:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w", pady=(5,0))
    tk.Entry(side_s, textvariable=reg_var_s, font=("Segoe UI", 12), bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"], bd=1, relief="solid").pack(fill="x", pady=4, ipady=3)
    
    tk.Label(side_s, text="Verified Profile Name Identity:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w", pady=(10,0))
    tk.Entry(side_s, textvariable=name_var_s, state="readonly", font=("Segoe UI", 12, "bold")).pack(fill="x", pady=4, ipady=3)
    
    s_status = tk.Label(side_s, text="SYSTEM STATUS: IDLE", fg="#00ffcc", bg=THEMES[current_theme]["card_bg"], font=("Segoe UI", 10, "bold"), bd=1, relief="solid", pady=6)
    s_status.pack(fill="x", pady=15)

    def async_fetch_profile_signature():
        regno = reg_var_s.get().strip()
        if not regno: return
        with get_conn() as conn: 
            rec = conn.cursor().execute("SELECT id, name, embedding FROM students WHERE reg_no=?", (regno,)).fetchone()
        if not rec: 
            s_status.config(text="STATUS FAILURE: SIGNAL REFUSED / NOT FOUND", fg="#ff4444")
            return
        name_var_s.set(rec[1])
        target_sid_s[0] = rec[0]
        if rec[2] is None:
            messagebox.showerror("Biometric Fault", "Profile lacks a biometric registration array. Complete enrollment first.")
            return
        target_emb_s[0] = np.frombuffer(rec[2], dtype=np.float32)
        
        today = datetime.now().strftime("%Y-%m-%d")
        with get_conn() as conn:
            already = conn.cursor().execute("SELECT time FROM attendance WHERE student_id=? AND date=?", (target_sid_s[0], today)).fetchone()
        if already:
            s_status.config(text=f"VERIFIED: ALREADY RECORDED AT {already[0]}", fg="#00e5ff")
        else:
            s_status.config(text="BIOMETRICS ENGAGED: SCANNING VIEWPORT FEED...", fg="yellow")

    tk.Button(side_s, text="🔍 Interrogate Signature", command=async_fetch_profile_signature, bg="#14818f", fg="white", font=("Segoe UI", 10, "bold"), bd=0, cursor="hand2", pady=8).pack(fill="x", pady=4)
    tk.Button(side_s, text="🗂 Open Full Profile Details", command=lambda: open_student_detail_window(reg_no=reg_var_s.get().strip()), bg="#444", fg="white", font=("Segoe UI", 10), bd=0, cursor="hand2", pady=8).pack(fill="x", pady=4)
    
    def inline_qr_scaffolder():
        s_status.config(text="DECODING MATRIX PATTERNS FROM VIDEO CORE...", fg="magenta")
        ret, frame = cam_manager.read_camera(cam_slot_name)
        if ret and frame is not None:
            codes = decode(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if codes: 
                reg = codes[0].data.decode("utf-8").strip()
                reg_var_s.set(reg)
                async_fetch_profile_signature()

    tk.Button(side_s, text="𔖔 Scan Physical Matrix QR Pass", command=inline_qr_scaffolder, bg="#7b1fa2", fg="white", font=("Segoe UI", 10, "bold"), bd=0, cursor="hand2", pady=8).pack(fill="x", pady=(15, 0))

    # --- DESK 2: BATCH GROUP WORKSPACE ---
    lbl_cam_g, side_g, canvas_frame_g = build_dashboard_pane(nb, "Mass Batch Processing")
    
    batch_filter_frame = tk.Frame(side_g, bg=THEMES[current_theme]["card_bg"])
    batch_filter_frame.pack(fill="x", pady=5)
    tk.Label(batch_filter_frame, text="Filter Log:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(side="left")
    batch_filter_var = tk.StringVar()
    tk.Entry(batch_filter_frame, textvariable=batch_filter_var, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(side="left", fill="x", expand=True, padx=5)
    
    tv_g = ttk.Treeview(side_g, columns=("Reg No", "Name", "Time Signature"), show="headings")
    for col in ("Reg No", "Name", "Time Signature"): 
        tv_g.heading(col, text=col)
        tv_g.column(col, anchor="center")
    tv_g.pack(fill="both", expand=True, pady=5)
    
    g_status = tk.Label(side_g, text="BATCH LOG READY", fg="#00ffcc", bg=THEMES[current_theme]["card_bg"], font=("Segoe UI", 9, "bold"))
    g_status.pack(fill="x", pady=4)

    def load_group_logs_async(*args):
        tv_g.delete(*tv_g.get_children())
        today = datetime.now().strftime("%Y-%m-%d")
        filter_text = batch_filter_var.get().strip().lower()
        with get_conn() as conn:
            rows = conn.cursor().execute("""SELECT s.id, s.reg_no, s.name, a.time FROM attendance a 
                                            JOIN students s ON s.id=a.student_id WHERE a.date=? ORDER BY a.time DESC""", (today,)).fetchall()
        for sid, reg, name, marked_time in rows:
            if filter_text and filter_text not in str(reg).lower() and filter_text not in str(name).lower():
                continue
            tv_g.insert("", tk.END, iid=f"sid:{sid}", values=(reg, name, marked_time))
        g_status.config(text=f"METRICS: {len(rows)} TOTAL SECURE LOGS TODAY", fg="#00ffcc")

    batch_filter_var.trace_add("write", load_group_logs_async)
    tk.Button(side_g, text="🔄 Synchronize Real-Time Log", command=load_group_logs_async, bg="#444", fg="white", font=("Segoe UI", 10), bd=0, cursor="hand2", pady=8).pack(fill="x", pady=4)
    load_group_logs_async()

    # --- DESK 3: COURSE RESTRICTED ROUTINE ---
    lbl_cam_cr, side_cr, canvas_frame_cr = build_dashboard_pane(nb, "Academic Course Node")
    
    tk.Label(side_cr, text="Filter Target Course Profile Architecture:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w")
    crs_var = tk.StringVar()
    crs_combo = ttk.Combobox(side_cr, textvariable=crs_var, values=list(set(KNOWN_COURSES)), state="readonly", font=("Segoe UI", 11))
    crs_combo.pack(fill="x", pady=6)
    
    tv_cr = ttk.Treeview(side_cr, columns=("Reg No", "Name", "Logged At", "Match %"), show="headings")
    for col in ("Reg No", "Name", "Logged At", "Match %"): 
        tv_cr.heading(col, text=col)
        if col == "Name": tv_cr.column(col, anchor="w", width=150)
        else: tv_cr.column(col, anchor="center", width=80)
    tv_cr.pack(fill="both", expand=True, pady=5)

    active_course_embs, active_course_ids, active_course_labels = [], [], []
    active_course_person_ids, active_course_person_types = [], []
    
    def apply_course_matrix_filter(e=None):
        active_course_embs.clear(); active_course_ids.clear(); active_course_labels.clear()
        active_course_person_ids.clear(); active_course_person_types.clear(); tv_cr.delete(*tv_cr.get_children())
        sel_c = crs_var.get()
        for i in range(len(KNOWN_COURSES)):
            if KNOWN_COURSES[i] == sel_c:
                active_course_embs.append(KNOWN_EMBEDDINGS[i]); active_course_ids.append(KNOWN_IDS[i]); active_course_labels.append(KNOWN_LABELS[i])
                active_course_person_ids.append(KNOWN_PERSON_IDS[i]); active_course_person_types.append(KNOWN_PERSON_TYPES[i])
    crs_combo.bind("<<ComboboxSelected>>", apply_course_matrix_filter)

    # --- DESK 4: MISSION CRITICAL CCTV SURVEILLANCE ---
    lbl_cam_c, side_c, canvas_frame_c = build_dashboard_pane(nb, "CCTV Intelligence Scanner")
    
    tk.Label(side_c, text="Assign CCTV Hardware Descriptor Signature:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w")
    cctv_name_var = tk.StringVar(value="SECURE_CAM_NODE_01")
    tk.Entry(side_c, textvariable=cctv_name_var, font=("Segoe UI", 11), bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"], bd=1, relief="solid").pack(fill="x", pady=6, ipady=3)

    tv_c = ttk.Treeview(side_c, columns=("Entity Identity", "Hardware Node", "Intercept Time"), show="headings")
    for col in ("Entity Identity", "Hardware Node", "Intercept Time"): 
        tv_c.heading(col, text=col)
        tv_c.column(col, anchor="center")
    tv_c.pack(fill="both", expand=True, pady=5)
    
    cctv_status = tk.Label(side_c, text="CCTV SURVEILLANCE PASSIVE DETECTION EYE ACTIVE", fg="#00ffcc", bg=THEMES[current_theme]["card_bg"], font=("Segoe UI", 9, "bold"), wraplength=380)
    cctv_status.pack(fill="x", pady=4)

    # --- DESK 5: LIVE TELEMETRY & PLOTS (NEW UI) ---
    plot_tab = tk.Frame(nb, bg=THEMES[current_theme]["bg"])
    nb.add(plot_tab, text=" Live Telemetry & Plots ")
    
    tk.Label(plot_tab, text="AI Confidence Matrix Distribution", font=("Segoe UI", 16, "bold"), bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(pady=10)
    
    # Initialize Matplotlib Figure
    fig, ax = plt.subplots(figsize=(8, 4), dpi=100)
    fig.patch.set_facecolor(THEMES[current_theme]["card_bg"])
    ax.set_facecolor(THEMES[current_theme]["entry_bg"])
    ax.tick_params(colors=THEMES[current_theme]["fg"])
    for spine in ax.spines.values():
        spine.set_color(THEMES[current_theme]["fg"])
    
    plot_canvas = FigureCanvasTkAgg(fig, master=plot_tab)
    plot_canvas.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=10)

    def update_live_plot():
        if win.winfo_exists() and nb.tab(nb.select(), "text").strip() == "Live Telemetry & Plots":
            ax.clear()
            if live_confidence_scores:
                # Plot Histogram of confidence scores
                ax.hist(live_confidence_scores, bins=15, range=(40, 100), color='#14818f', edgecolor='white', alpha=0.8)
                ax.set_title("Real-Time Face Match Confidence (%)", color=THEMES[current_theme]["fg"], pad=15)
                ax.set_xlabel("Confidence Percentage", color=THEMES[current_theme]["fg"])
                ax.set_ylabel("Frequency", color=THEMES[current_theme]["fg"])
                ax.grid(True, linestyle='--', alpha=0.3, color=THEMES[current_theme]["fg"])
            else:
                ax.text(0.5, 0.5, "Awaiting Biometric Data...", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, color=THEMES[current_theme]["fg"], fontsize=12)
            
            plot_canvas.draw()
        if win.winfo_exists():
            win.after(2000, update_live_plot) # Refresh plot every 2 seconds
            
    update_live_plot()


    # --- DETACHED BACKGROUND AI THREAD POOL WORKER ---
    def ai_worker(frame_copy, embs, ids, labels):
        t_start = time.time()
        try: faces = FA.get(cv2.resize(frame_copy, (0, 0), fx=0.5, fy=0.5))
        except Exception: return [], 0
        res = []
        latency = int((time.time() - t_start) * 1000)
        for face in faces:
            sims = face_similarity(embs, face.embedding)
            if sims.size > 0 and sims.max() >= SIMILARITY_THRESHOLD:
                res.append({"box": face.bbox*2, "idx": int(sims.argmax()), "sim": float(sims.max())})
            else:
                best = float(sims.max()) if sims.size > 0 else 0.0
                res.append({"box": face.bbox*2, "unknown": True, "sim": best})
        return res, latency

    def async_db_injector(query, params, reload_callback=None):
        def task():
            with get_conn() as conn:
                conn.cursor().execute(query, params)
                conn.commit()
            if reload_callback: win.after(10, reload_callback)
        threading.Thread(target=task, daemon=True).start()

    def hot_swap_stream_route(*args):
        source = selected_source.get()
        if not source: return
        actual_src = int(source) if source.isdigit() else source
        cam_manager.start_camera(cam_slot_name, actual_src)

    selected_source.trace_add("write", hot_swap_stream_route)
    win.after(400, hot_swap_stream_route)

    # --- SYSTEM COMMAND RENDERING ENGINE LOOP ---
    def core_render_pipeline_loop():
        nonlocal fut, proc_ctr, group_marked, course_marked, visual_flashes, fps_frame_count, fps_last_time
        ret, frame = cam_manager.read_camera(cam_slot_name)
        cur_tab = nb.tab(nb.select(), "text").strip()

        # Update Telemetry Dashboards
        now_time = time.time()
        uptime_seconds = int(now_time - session_start_time)
        uptime_metric.set(f"Uptime: {uptime_seconds // 60:02d}:{uptime_seconds % 60:02d}")
        
        fps_frame_count += 1
        if now_time - fps_last_time >= 1.0:
            live_fps_metric.set(f"Stream: {fps_frame_count} FPS")
            fps_frame_count = 0
            fps_last_time = now_time

        if ret and frame is not None:
            now_dt = datetime.now()
            dt_s, tm_s = now_dt.strftime("%Y-%m-%d"), now_dt.strftime("%H:%M:%S")
            display = frame.copy()
            proc_ctr = (proc_ctr + 1) % PROCESS_EVERY_N
            visual_flashes = {k: v for k, v in visual_flashes.items() if (now_dt.timestamp() - v) < 2.0}

            if cur_tab == "Single Desk Checkpoint":
                display = draw_hud(display, "LOCAL SCAN NODE")
                w, h = canvas_frame_s.winfo_width(), canvas_frame_s.winfo_height()
                if target_emb_s[0] is not None and proc_ctr == 0:
                    faces = FA.get(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
                    if faces:
                        sim = float(face_similarity([target_emb_s[0]], faces[0].embedding)[0])
                        if sim >= SIMILARITY_THRESHOLD:
                            s_status.config(text=f"MATCH CONFIRMED: {sim*100:.1f}% — COMMITTING TO STORAGE", fg="#00ff00")
                            play_success_beep()
                            sid = target_sid_s[0]
                            live_confidence_scores.append(sim*100)
                            
                            q = "INSERT OR IGNORE INTO attendance (student_id, date, time, match_percentage, camera_location) VALUES (?, ?, ?, ?, ?)"
                            p = (sid, dt_s, tm_s, sim*100, "Single Desk Checkpoint")
                            known_session_counter.set(known_session_counter.get() + 1)
                            async_db_injector(q, p, lambda: s_status.config(text=f"SUCCESS: ENTRY REGISTERED AT {tm_s}", fg="#00ff00"))
                            target_emb_s[0] = None
                
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resize_to_fit(display, w, h), cv2.COLOR_BGR2RGB)))
                lbl_cam_s.imgtk = img; lbl_cam_s.configure(image=img, text="")

            elif cur_tab == "Mass Batch Processing":
                display = draw_hud(display, "BATCH CAPTURE ARRAY")
                w, h = canvas_frame_g.winfo_width(), canvas_frame_g.winfo_height()
                if fut is None or fut.done():
                    if fut and fut.result():
                        results, latency = fut.result()
                        ai_latency_metric.set(f"AI Latency: {latency}ms")
                        for m in results:
                            if m.get("unknown"):
                                x1,y1,x2,y2 = map(int, m["box"])
                                cv2.rectangle(display, (x1,y1), (x2,y2), (0, 165, 255), 2)
                                continue
                            idx = m["idx"]; sid = KNOWN_IDS[idx]
                            if sid is None: continue
                            mark_key = f"student:{sid}|{dt_s}"
                            if mark_key not in group_marked:
                                visual_flashes[mark_key] = now_dt.timestamp()
                                group_marked.add(mark_key)
                                live_confidence_scores.append(m["sim"]*100)
                                r, n = KNOWN_LABELS[idx].split("|")
                                
                                q = "INSERT OR IGNORE INTO attendance (student_id, date, time, match_percentage, camera_location) VALUES (?, ?, ?, ?, ?)"
                                p = (sid, dt_s, tm_s, m["sim"]*100, "Batch Processing Matrix")
                                known_session_counter.set(known_session_counter.get() + 1)
                                async_db_injector(q, p, load_group_logs_async)
                                play_success_beep()
                            
                            x1,y1,x2,y2 = map(int, m["box"])
                            color = (0, 255, 0) if mark_key in visual_flashes else (0, 184, 212)
                            cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                            cv2.putText(display, f"{KNOWN_LABELS[idx].split('|')[1].strip()} ({m['sim']*100:.0f}%)", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    fut = ai_executor.submit(ai_worker, frame.copy(), KNOWN_EMBEDDINGS, KNOWN_IDS, KNOWN_LABELS)
                
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resize_to_fit(display, w, h), cv2.COLOR_BGR2RGB)))
                lbl_cam_g.imgtk = img; lbl_cam_g.configure(image=img, text="")

            elif cur_tab == "Academic Course Node":
                display = draw_hud(display, f"COURSE: {crs_var.get()}")
                w, h = canvas_frame_cr.winfo_width(), canvas_frame_cr.winfo_height()
                if active_course_embs and (fut is None or fut.done()):
                    if fut and fut.result():
                        results, latency = fut.result()
                        ai_latency_metric.set(f"AI Latency: {latency}ms")
                        for m in results:
                            if m.get("unknown"):
                                x1,y1,x2,y2 = map(int, m["box"])
                                cv2.rectangle(display, (x1,y1), (x2,y2), (0, 105, 217), 2)
                                continue
                            
                            global_idx = m["idx"]
                            sid = KNOWN_IDS[global_idx]
                            person_id = KNOWN_PERSON_IDS[global_idx]
                            person_type = KNOWN_PERSON_TYPES[global_idx]
                            label_str = KNOWN_LABELS[global_idx]
                            
                            if label_str not in active_course_labels: continue
                                
                            mark_key = f"person:{person_id}" if person_id is not None else f"student:{sid}"
                            if mark_key not in course_marked:
                                play_success_beep()
                                visual_flashes[mark_key] = now_dt.timestamp()
                                course_marked.add(mark_key)
                                live_confidence_scores.append(m["sim"]*100)
                                known_session_counter.set(known_session_counter.get() + 1)
                                
                                r, n = label_str.split("|")
                                def inject_course(s_id=sid, p_id=person_id, p_type=person_type, lbl_r=r, lbl_n=n, match=m["sim"]*100):
                                    with get_conn() as conn:
                                        if s_id is not None:
                                            conn.cursor().execute("INSERT OR IGNORE INTO attendance (student_id, date, time, match_percentage, camera_location) VALUES (?, ?, ?, ?, ?)", (s_id, dt_s, tm_s, match, f"Course: {crs_var.get()}"))
                                        conn.cursor().execute("INSERT INTO attendance_logs (person_id, legacy_student_id, person_type, date, time, camera_name, camera_location, match_percentage, status, notes) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", (p_id, s_id, p_type, dt_s, tm_s, f"Course: {crs_var.get()}", "Class Terminal", match, "official", "Dynamic Course Node Processing"))
                                        conn.commit()
                                    win.after(10, lambda: tv_cr.insert("", 0, values=(lbl_r.strip(), lbl_n.strip(), tm_s, f"{match:.1f}%")))
                                threading.Thread(target=inject_course, daemon=True).start()
                            
                            x1,y1,x2,y2 = map(int, m["box"])
                            color = (0, 255, 0) if mark_key in visual_flashes else (224, 224, 224)
                            cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                            cv2.putText(display, f"{label_str.split('|')[1].strip()}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    fut = ai_executor.submit(ai_worker, frame.copy(), KNOWN_EMBEDDINGS, KNOWN_IDS, KNOWN_LABELS)
                
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resize_to_fit(display, w, h), cv2.COLOR_BGR2RGB)))
                lbl_cam_cr.imgtk = img; lbl_cam_cr.configure(image=img, text="")

            elif cur_tab == "CCTV Intelligence Scanner":
                cam_name = cctv_name_var.get().strip()
                display = draw_hud(display, cam_name)
                w, h = canvas_frame_c.winfo_width(), canvas_frame_c.winfo_height()
                
                if fut is None or fut.done():
                    if fut and fut.result():
                        results, latency = fut.result()
                        ai_latency_metric.set(f"AI Latency: {latency}ms")
                        for m in results:
                            if m.get("unknown"):
                                unknown_session_counter.set(unknown_session_counter.get() + 1)
                                threading.Thread(target=log_unknown_event, args=(cam_name, "cctv_surveillance", frame, "alert", "security_alert", "Unknown structural profile intercept", m["sim"]*100), daemon=True).start()
                                x1,y1,x2,y2 = map(int, m["box"])
                                cv2.rectangle(display, (x1,y1), (x2,y2), (0, 0, 255), 2)
                                continue
                            idx = m["idx"]; sid, name = KNOWN_IDS[idx], KNOWN_LABELS[idx].split("|")[1]
                            person_id = KNOWN_PERSON_IDS[idx]
                            mark_key = f"person:{person_id}" if person_id is not None else f"student:{sid}"
                            if mark_key not in cctv_memory or (now_dt.timestamp() - cctv_memory[mark_key]) > CCTV_COOLDOWN_SECONDS:
                                play_success_beep()
                                visual_flashes[mark_key] = now_dt.timestamp()
                                cctv_memory[mark_key] = now_dt.timestamp()
                                live_confidence_scores.append(m["sim"]*100)
                                known_session_counter.set(known_session_counter.get() + 1)
                                
                                def inject_surveillance(s_id=sid, p_id=person_id, g_idx=idx, name_str=name):
                                    with get_conn() as conn:
                                        if s_id is not None:
                                            conn.cursor().execute("INSERT INTO attendance (student_id, date, time, match_percentage, camera_location) VALUES (?, ?, ?, ?, ?)", (s_id, dt_s, tm_s, m["sim"]*100, cam_name))
                                        conn.cursor().execute("INSERT INTO camera_events (camera_name, event_type, mode, person_id, legacy_student_id, date, time, severity, details) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", (cam_name, "known_face", "cctv_surveillance", p_id, s_id, dt_s, tm_s, "info", f"Intercept match matrix at {m['sim']*100:.1f}%"))
                                        conn.commit()
                                    log_surveillance_track(cam_name, "cctv_surveillance", "known_face", frame, p_id, s_id, KNOWN_PERSON_TYPES[g_idx], KNOWN_LABELS[g_idx], m["sim"]*100, "info", "recorded", "CCTV Engine Capture")
                                    win.after(10, lambda: tv_c.insert("", 0, values=(name_str.strip(), cam_name, tm_s)))
                                threading.Thread(target=inject_surveillance, daemon=True).start()
                            
                            x1,y1,x2,y2 = map(int, m["box"])
                            color = (0, 255, 255) if mark_key in visual_flashes else (158, 158, 158)
                            cv2.rectangle(display, (x1,y1), (x2,y2), color, 2)
                            cv2.putText(display, f"{name.strip()}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    fut = ai_executor.submit(ai_worker, frame.copy(), KNOWN_EMBEDDINGS, KNOWN_IDS, KNOWN_LABELS)
                
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resize_to_fit(display, w, h), cv2.COLOR_BGR2RGB)))
                lbl_cam_c.imgtk = img; lbl_cam_c.configure(image=img, text="")

        if win.winfo_exists():
            win.after(33, core_render_pipeline_loop)
            
    core_render_pipeline_loop()

def open_person_face_capture(parent, person_id, name, on_done=None):
    face_win = tk.Toplevel(parent); face_win.title(f"Face Capture: {name}"); face_win.geometry("760x620"); face_win.configure(bg="#222")
    cam_manager.start_camera("person_capture", CAMERA_SOURCE)
    face_win.protocol("WM_DELETE_WINDOW", lambda: [cam_manager.stop_camera("person_capture"), face_win.destroy()])

    tk.Label(face_win, text=f"Capture Face for {name}", font=("Segoe UI", 14), bg="#222", fg="white").pack(pady=10)
    vf = tk.Frame(face_win, bg="black"); vf.pack(fill="both", expand=True, padx=10, pady=10)
    vf.pack_propagate(False)
    lbl_feed = tk.Label(vf, bg="black"); lbl_feed.place(relx=0.5, rely=0.5, anchor="center")

    def face_loop():
        ret, frame = cam_manager.read_camera("person_capture")
        if ret and frame is not None:
            imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resize_to_fit(frame, vf.winfo_width(), vf.winfo_height()), cv2.COLOR_BGR2RGB)))
            lbl_feed.imgtk = imgtk; lbl_feed.configure(image=imgtk)
        if face_win.winfo_exists():
            lbl_feed.after(30, face_loop)
    face_loop()

    def capture_face():
        ret, frame = cam_manager.read_camera("person_capture")
        if not ret or frame is None: return messagebox.showerror("Error", "Camera offline.")
        faces = FA.get(frame)
        if len(faces) != 1: return messagebox.showerror("Error", "Need exactly 1 face in the frame.")
        photo_path = os.path.join("photos", f"person_{person_id}.jpg")
        cv2.imwrite(photo_path, frame)
        with get_conn() as conn:
            conn.cursor().execute("UPDATE people SET embedding=?, photo_path=? WHERE id=?", (faces[0].embedding.tobytes(), photo_path, person_id))
            conn.commit()
        load_server_memory()
        if on_done: on_done()
        messagebox.showinfo("Saved", "Face profile updated.")
        cam_manager.stop_camera("person_capture"); face_win.destroy()

    tk.Button(face_win, text="Capture Face", command=capture_face, bg="#4CAF50", fg="white", font=("Segoe UI", 13, "bold"), cursor="hand2").pack(pady=10)

def open_person_registration(person_type):
    titles = {
        "faculty": "Faculty Registry & Face Profile",
        "non_faculty": "Staff Registry & Face Profile",
        "guest": "Guest Pass Registry & Face Profile"
    }
    win_name = f"register_{person_type}"
    
    # 1. Prevent duplicate windows from stacking
    if win_name in open_windows and open_windows[win_name].winfo_exists():
        open_windows[win_name].deiconify()
        open_windows[win_name].lift()
        return

    win = tk.Toplevel(root)
    win.title(titles[person_type])
    win.geometry("1100x650")  
    open_windows[win_name] = win
    win.configure(bg=THEMES[current_theme]["bg"])
    
    cam_slot_name = f"reg_preview_{person_type}"
    add_window_toolbar(win, win_name, stop_camera_instance=cam_slot_name)

    # --- MODERN TWO-COLUMN SPLIT DASHBOARD LAYOUT ---
    main_container = tk.Frame(win, bg=THEMES[current_theme]["bg"])
    main_container.pack(fill="both", expand=True, padx=20, pady=15)
    
    # Left Column Form Card
    left_card = tk.LabelFrame(main_container, text=" Identification Profile ", 
                             bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"],
                             font=("Segoe UI", 11, "bold"), padx=20, pady=15, bd=1, relief="solid",
                             width=420)
    left_card.pack_propagate(False)
    left_card.pack(side="left", fill="both", padx=(0, 10))

    # Right Column Video Viewport Card
    right_card = tk.LabelFrame(main_container, text=" Live Feed & Hardware Routing ", 
                              bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"],
                              font=("Segoe UI", 11, "bold"), padx=20, pady=15, bd=1, relief="solid")
    right_card.pack(side="right", fill="both", expand=True, padx=(10, 0))

    # --- POPULATING LEFT COLUMN PROFILE FIELDS ---
    field_sets = {
        "faculty": [("Faculty ID/Ref", "ref"), ("Faculty Full Name", "name"), ("Department", "department"), ("Designation", "designation"), ("Mobile Connection", "mobile")],
        "non_faculty": [("Staff ID/Ref", "ref"), ("Staff Full Name", "name"), ("Department/Unit", "department"), ("Role/Designation", "designation"), ("Mobile Connection", "mobile")],
        "guest": [("Guest ID/Ref", "ref"), ("Guest Full Name", "name"), ("Issuing Organization", "organization"), ("Purpose of Visit", "visitor_purpose"), ("Contact Mobile", "mobile"), ("Pass Valid Until", "valid_until")]
    }
    
    vars_map = {}
    for label, key in field_sets[person_type]:
        lbl_field = tk.Label(left_card, text=label, bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"], font=("Segoe UI", 10))
        lbl_field.pack(anchor="w", pady=(8, 0))
        
        # Pull dynamic department selections directly from your database table
        if key in ("department", "department/unit"):
            try:
                with get_conn() as conn:
                    db_depts = [r[0] for r in conn.cursor().execute("SELECT department_name FROM departments WHERE status='active' ORDER BY department_name").fetchall()]
            except Exception:
                db_depts = []
            
            ent = ttk.Combobox(left_card, values=db_depts, state="readonly", font=("Segoe UI", 11))
            ent.pack(fill="x", pady=4)
        else:
            ent = tk.Entry(left_card, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"], 
                           insertbackground=THEMES[current_theme]["fg"], font=("Segoe UI", 11), bd=1, relief="solid")
            ent.pack(fill="x", pady=4, ipady=3)
            
        vars_map[key] = ent
        
    if "ref" in vars_map and person_type == "guest":
        vars_map["ref"].insert(0, f"GUEST-{datetime.now().strftime('%Y%m%d-%H%M')}")

    # --- POPULATING RIGHT COLUMN VIDEO LAYOUT ---
    selected_source = tk.StringVar()
    camera_combo = create_camera_device_selector(right_card, selected_source)
    
    # Clean Digital HUD Display
    feed_viewport = tk.Frame(right_card, bg="black", bd=1, relief="solid")
    feed_viewport.pack(fill="both", expand=True, pady=12)
    feed_viewport.pack_propagate(False)
    
    lbl_video = tk.Label(feed_viewport, bg="black", text="Initializing camera stream...", fg="#888", font=("Segoe UI", 10))
    lbl_video.place(relx=0.5, rely=0.5, anchor="center")

    # Trace variable write updates to instantly hot-swap the video channel
    def switch_routing_stream(*args):
        source = selected_source.get()
        if not source: return
        actual_src = int(source) if source.isdigit() else source
        cam_manager.start_camera(cam_slot_name, actual_src)

    selected_source.trace_add("write", switch_routing_stream)
    win.after(400, switch_routing_stream)

    def run_viewport_refresh():
        if win.winfo_exists():
            ret, frame = cam_manager.read_camera(cam_slot_name)
            if ret and frame is not None:
                display_frame = draw_hud(frame.copy(), f"PREVIEW MODE | {person_type.upper()}")
                w, h = feed_viewport.winfo_width(), feed_viewport.winfo_height()
                resized = resize_to_fit(display_frame, w, h)
                imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
                lbl_video.imgtk = imgtk
                lbl_video.configure(image=imgtk, text="")
            win.after(33, run_viewport_refresh)
    run_viewport_refresh()

    # --- SUBMISSION AND INTERNAL BIO-CAPTURE PIPELINE ---
    def execute_registration_pipeline():
        name = vars_map["name"].get().strip()
        if not name: 
            messagebox.showerror("Validation Error", "Profile Name field cannot be empty.")
            return

        # Read the current live camera handle bound to this module
        ret, frame = cam_manager.read_camera(cam_slot_name)
        if not ret or frame is None:
            messagebox.showerror("Hardware Fault", "Unable to capture stream data. Verify physical camera connection.")
            return

        # Instant AI check on current frame
        faces = FA.get(frame)
        if len(faces) != 1:
            messagebox.showerror("AI Detection Error", f"Detected {len(faces)} faces. Exactly ONE face must be centered in the view area.")
            return

        ref = vars_map["ref"].get().strip() or f"{person_type.upper()}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        qr_path = os.path.join("qrcodes", f"{safe_name(ref)}.png")
        qrcode.make(ref).save(qr_path)
        data = {k: v.get().strip() for k, v in vars_map.items()}
        
        with get_conn() as conn:
            try:
                cur = conn.cursor()
                cur.execute("""INSERT INTO people
                    (person_type, external_ref, reg_no, name, department, mobile, designation, organization,
                     visitor_purpose, valid_until, qr_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (person_type, ref, ref, name, data.get("department"), data.get("mobile"), data.get("designation"),
                     data.get("organization"), data.get("visitor_purpose"), data.get("valid_until"), qr_path))
                person_id = cur.lastrowid
                
                # Write file out to disk and commit binary embedding arrays
                photo_path = os.path.join("photos", f"person_{person_id}.jpg")
                cv2.imwrite(photo_path, frame)
                cur.execute("UPDATE people SET embedding=?, photo_path=? WHERE id=?", (faces[0].embedding.tobytes(), photo_path, person_id))
                conn.commit()
            except sqlite3.IntegrityError: 
                messagebox.showerror("Database Conflict", "A tracking identifier matching this signature already exists.")
                return
                
        load_server_memory()
        play_success_beep()
        messagebox.showinfo("Success", f"Registration record for '{name}' successfully generated.")
        
        # Reset data form elements while keeping the module open
        for key, widget in vars_map.items(): 
            if isinstance(widget, ttk.Combobox):
                widget.set('')
            else:
                widget.delete(0, tk.END)
                
        if "ref" in vars_map and person_type == "guest":
            vars_map["ref"].insert(0, f"GUEST-{datetime.now().strftime('%Y%m%d-%H%M')}")

    # Action Confirmation Button
    tk.Button(left_card, text="✓ Validate & Register Face", command=execute_registration_pipeline, 
              bg="#14818f", fg="white", font=("Segoe UI", 11, "bold"), bd=0, cursor="hand2", pady=8).pack(fill="x", side="bottom", pady=5)   

def open_academic_setup():
    win_name = "academic_setup"
    
    # BUG FIX: Prevent duplicate window spawning
    if win_name in open_windows and open_windows[win_name].winfo_exists():
        open_windows[win_name].deiconify()
        open_windows[win_name].lift()
        return

    win = tk.Toplevel(root)
    win.title("Enterprise Academic Setup Command")
    win.geometry("1200x800")
    open_windows[win_name] = win
    win.configure(bg=THEMES[current_theme]["bg"])
    add_window_toolbar(win, win_name, stop_camera_instance=None)
    
    tk.Label(win, text="Academic Architecture Setup", font=("Segoe UI", 18, "bold"),
             bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(pady=15)

    style = ttk.Style()
    style.configure("TNotebook", background=THEMES[current_theme]["bg"])
    style.configure("TNotebook.Tab", padding=[15, 5], font=("Segoe UI", 10, "bold"))
    
    nb = ttk.Notebook(win)
    nb.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    # Tab Generation
    dept_tab = tk.Frame(nb, bg=THEMES[current_theme]["bg"]); nb.add(dept_tab, text=" Departments ")
    class_tab = tk.Frame(nb, bg=THEMES[current_theme]["bg"]); nb.add(class_tab, text=" Classes ")
    subject_tab = tk.Frame(nb, bg=THEMES[current_theme]["bg"]); nb.add(subject_tab, text=" Subjects ")
    assign_tab = tk.Frame(nb, bg=THEMES[current_theme]["bg"]); nb.add(assign_tab, text=" Faculty Links ")
    session_tab = tk.Frame(nb, bg=THEMES[current_theme]["bg"]); nb.add(session_tab, text=" Class Sessions ")

    # --- Shared Utility Functions ---
    def department_names():
        with get_conn() as conn:
            return [r[0] for r in conn.cursor().execute("SELECT department_name FROM departments WHERE status='active' ORDER BY department_name").fetchall()]

    def attach_delete_menu(tree, table_name):
        menu = tk.Menu(tree, tearoff=0, bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"])
        
        def delete_record():
            sel = tree.selection()
            if not sel: return
            record_id = tree.item(sel[0])["values"][0]
            if messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete this record from {table_name}?"):
                with get_conn() as conn:
                    # In a true enterprise system, consider soft-deletes (status='inactive') instead of hard drops
                    conn.cursor().execute(f"DELETE FROM {table_name} WHERE id=?", (record_id,))
                    conn.commit()
                tree.delete(sel[0])
                
        menu.add_command(label="Delete Selected Record", command=delete_record)
        
        def show_menu(event):
            item = tree.identify_row(event.y)
            if item:
                tree.selection_set(item)
                menu.post(event.x_root, event.y_root)
                
        tree.bind("<Button-3>", show_menu) # Right-click bind

    # --- 1. DEPARTMENTS ---
    dept_card = tk.LabelFrame(dept_tab, text=" Register New Department ", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], font=("Segoe UI", 10, "bold"), pady=10)
    dept_card.pack(fill="x", padx=15, pady=10)
    
    dept_code_var, dept_name_var = tk.StringVar(), tk.StringVar()
    tk.Label(dept_card, text="Dept Code:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 5))
    tk.Entry(dept_card, textvariable=dept_code_var, width=15, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(side="left", padx=5)
    tk.Label(dept_card, text="Department Name:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 5))
    tk.Entry(dept_card, textvariable=dept_name_var, width=35, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(side="left", padx=5)
    
    dept_tree = ttk.Treeview(dept_tab, columns=("ID", "Code", "Name", "Status"), show="headings", height=15)
    for c in dept_tree["columns"]: dept_tree.heading(c, text=c); dept_tree.column(c, anchor="center")
    dept_tree.pack(fill="both", expand=True, padx=15, pady=5)
    attach_delete_menu(dept_tree, "departments")

    def load_departments():
        dept_tree.delete(*dept_tree.get_children())
        with get_conn() as conn:
            df = pd.read_sql_query("SELECT id, department_code, department_name, status FROM departments ORDER BY department_name", conn)
        for _, r in df.iterrows(): dept_tree.insert("", tk.END, values=(r["id"], r["department_code"], r["department_name"], r["status"]))

    def save_department():
        name = dept_name_var.get().strip()
        code = dept_code_var.get().strip()
        if not name or not code: return messagebox.showerror("Validation Error", "Both Code and Department Name are required.")
        with get_conn() as conn:
            try:
                conn.cursor().execute("INSERT INTO departments (department_code, department_name) VALUES (?, ?)", (code, name))
                conn.commit()
            except sqlite3.IntegrityError:
                return messagebox.showerror("Database Conflict", "This Department Code or Name already exists.")
        load_departments(); dept_code_var.set(""); dept_name_var.set("")
        try:
            class_dept_combo["values"] = department_names()
            subject_dept_combo["values"] = department_names()
        except Exception: pass

    tk.Button(dept_card, text="✚ Add Department", command=save_department, bg="#14818f", fg="white", font=("Segoe UI", 9, "bold"), cursor="hand2", padx=10).pack(side="right", padx=15)
    load_departments()

    # --- 2. CLASSES ---
    class_card = tk.LabelFrame(class_tab, text=" Register New Class Matrix ", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], font=("Segoe UI", 10, "bold"), pady=10)
    class_card.pack(fill="x", padx=15, pady=10)
    
    class_vars = {"name": tk.StringVar(), "department": tk.StringVar(), "section": tk.StringVar()}
    tk.Label(class_card, text="Class Name:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 5))
    tk.Entry(class_card, textvariable=class_vars["name"], width=20, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(side="left", padx=5)
    
    tk.Label(class_card, text="Linked Department:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 5))
    class_dept_combo = ttk.Combobox(class_card, textvariable=class_vars["department"], values=department_names(), width=25, state="readonly")
    class_dept_combo.pack(side="left", padx=5)
    
    tk.Label(class_card, text="Section:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 5))
    tk.Entry(class_card, textvariable=class_vars["section"], width=10, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(side="left", padx=5)
    
    class_tree = ttk.Treeview(class_tab, columns=("ID", "Class", "Department", "Section", "Status"), show="headings", height=15)
    for c in class_tree["columns"]: class_tree.heading(c, text=c); class_tree.column(c, anchor="center")
    class_tree.pack(fill="both", expand=True, padx=15, pady=5)
    attach_delete_menu(class_tree, "classes")

    def load_classes():
        class_tree.delete(*class_tree.get_children())
        with get_conn() as conn:
            df = pd.read_sql_query("SELECT id, class_name, department, section, status FROM classes ORDER BY class_name, section", conn)
        for _, r in df.iterrows(): class_tree.insert("", tk.END, values=(r["id"], r["class_name"], r["department"], r["section"], r["status"]))

    def save_class():
        name = class_vars["name"].get().strip()
        dept = class_vars["department"].get().strip()
        if not name or not dept: return messagebox.showerror("Validation Error", "Class Name and Department are strictly required.")
        with get_conn() as conn:
            try:
                conn.cursor().execute("INSERT INTO classes (class_name, department, section) VALUES (?, ?, ?)",
                                      (name, dept, class_vars["section"].get().strip()))
                conn.commit()
            except sqlite3.IntegrityError: return messagebox.showerror("Conflict", "Class name collision detected.")
        load_classes()
        for var in class_vars.values(): var.set("")

    tk.Button(class_card, text="✚ Establish Class", command=save_class, bg="#14818f", fg="white", font=("Segoe UI", 9, "bold"), cursor="hand2", padx=10).pack(side="right", padx=15)
    load_classes()

    # --- 3. SUBJECTS ---
    subject_card = tk.LabelFrame(subject_tab, text=" Register Academic Subject ", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], font=("Segoe UI", 10, "bold"), pady=10)
    subject_card.pack(fill="x", padx=15, pady=10)
    
    subject_vars = {"code": tk.StringVar(), "name": tk.StringVar(), "department": tk.StringVar()}
    tk.Label(subject_card, text="Sub Code:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 5))
    tk.Entry(subject_card, textvariable=subject_vars["code"], width=12, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(side="left", padx=5)
    tk.Label(subject_card, text="Subject Name:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 5))
    tk.Entry(subject_card, textvariable=subject_vars["name"], width=25, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(side="left", padx=5)
    tk.Label(subject_card, text="Department:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 5))
    subject_dept_combo = ttk.Combobox(subject_card, textvariable=subject_vars["department"], values=department_names(), width=25, state="readonly")
    subject_dept_combo.pack(side="left", padx=5)
    
    subject_tree = ttk.Treeview(subject_tab, columns=("ID", "Code", "Subject", "Department", "Status"), show="headings", height=15)
    for c in subject_tree["columns"]: subject_tree.heading(c, text=c); subject_tree.column(c, anchor="center")
    subject_tree.pack(fill="both", expand=True, padx=15, pady=5)
    attach_delete_menu(subject_tree, "subjects")

    def load_subjects():
        subject_tree.delete(*subject_tree.get_children())
        with get_conn() as conn:
            df = pd.read_sql_query("SELECT id, subject_code, subject_name, department, status FROM subjects ORDER BY subject_code, subject_name", conn)
        for _, r in df.iterrows(): subject_tree.insert("", tk.END, values=(r["id"], r["subject_code"], r["subject_name"], r["department"], r["status"]))

    def save_subject():
        name = subject_vars["name"].get().strip()
        code = subject_vars["code"].get().strip()
        if not name or not code: return messagebox.showerror("Validation Error", "Subject Code and Name are required.")
        with get_conn() as conn:
            try:
                conn.cursor().execute("INSERT INTO subjects (subject_code, subject_name, department) VALUES (?, ?, ?)",
                                      (code, name, subject_vars["department"].get().strip()))
                conn.commit()
            except sqlite3.IntegrityError: return messagebox.showerror("Conflict", "This subject already exists.")
        load_subjects()
        for var in subject_vars.values(): var.set("")
        
    tk.Button(subject_card, text="✚ Add Subject", command=save_subject, bg="#14818f", fg="white", font=("Segoe UI", 9, "bold"), cursor="hand2", padx=10).pack(side="right", padx=15)
    load_subjects()

    # --- 4. FACULTY LINKS ---
    assign_card = tk.LabelFrame(assign_tab, text=" Node Assignment (Link Class + Subject + Faculty) ", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], font=("Segoe UI", 10, "bold"), pady=10)
    assign_card.pack(fill="x", padx=15, pady=10)
    
    assign_class, assign_subject, assign_faculty = tk.StringVar(), tk.StringVar(), tk.StringVar()
    tk.Label(assign_card, text="Class:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 2))
    class_combo = ttk.Combobox(assign_card, textvariable=assign_class, width=25, state="readonly"); class_combo.pack(side="left", padx=5)
    tk.Label(assign_card, text="Subject:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 2))
    subject_combo = ttk.Combobox(assign_card, textvariable=assign_subject, width=25, state="readonly"); subject_combo.pack(side="left", padx=5)
    tk.Label(assign_card, text="Faculty Lead:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 2))
    faculty_combo = ttk.Combobox(assign_card, textvariable=assign_faculty, width=30, state="readonly"); faculty_combo.pack(side="left", padx=5)
    
    assign_tree = ttk.Treeview(assign_tab, columns=("ID", "Class", "Subject", "Faculty", "Status"), show="headings", height=15)
    for c in assign_tree["columns"]: assign_tree.heading(c, text=c); assign_tree.column(c, anchor="center")
    assign_tree.pack(fill="both", expand=True, padx=15, pady=5)
    attach_delete_menu(assign_tree, "class_subjects")

    def refresh_assign_options():
        class_combo["values"] = fetch_named_options("classes", "class_name || ' - ' || COALESCE(section, '')")
        subject_combo["values"] = fetch_named_options("subjects", "COALESCE(subject_code, '') || ' - ' || subject_name")
        faculty_combo["values"] = fetch_named_options("people", "COALESCE(external_ref, '') || ' - ' || name", "person_type='faculty' AND status='active'")

    def load_assignments():
        assign_tree.delete(*assign_tree.get_children())
        with get_conn() as conn:
            df = pd.read_sql_query("""SELECT cs.id, c.class_name || ' ' || COALESCE(c.section, ''),
                                      COALESCE(s.subject_code, '') || ' - ' || s.subject_name,
                                      COALESCE(p.external_ref, '') || ' - ' || p.name, cs.status
                                      FROM class_subjects cs
                                      JOIN classes c ON c.id=cs.class_id
                                      JOIN subjects s ON s.id=cs.subject_id
                                      LEFT JOIN people p ON p.id=cs.faculty_person_id
                                      ORDER BY c.class_name, s.subject_name""", conn)
        for _, r in df.iterrows(): assign_tree.insert("", tk.END, values=tuple(r))

    def save_assignment():
        class_id, subject_id, faculty_id = option_id(assign_class.get()), option_id(assign_subject.get()), option_id(assign_faculty.get())
        if not all([class_id, subject_id, faculty_id]): return messagebox.showerror("Validation Error", "All nodes must be selected to create a link.")
        with get_conn() as conn:
            try:
                conn.cursor().execute("INSERT INTO class_subjects (class_id, subject_id, faculty_person_id) VALUES (?, ?, ?)", (class_id, subject_id, faculty_id))
                conn.commit()
            except sqlite3.IntegrityError: return messagebox.showerror("Conflict", "This routing link already exists.")
        load_assignments()
        
    tk.Button(assign_card, text="🔗 Link Matrix", command=save_assignment, bg="#14818f", fg="white", font=("Segoe UI", 9, "bold"), cursor="hand2", padx=10).pack(side="right", padx=15)
    tk.Button(assign_card, text="↻ Refresh", command=lambda: [refresh_assign_options(), load_assignments()], bg="#555", fg="white", cursor="hand2").pack(side="right", padx=5)
    refresh_assign_options(); load_assignments()

    # --- 5. CLASS SESSIONS ---
    session_card = tk.LabelFrame(session_tab, text=" Initialize Live Session ", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], font=("Segoe UI", 10, "bold"), pady=10)
    session_card.pack(fill="x", padx=15, pady=10)
    
    sess_code = tk.StringVar(value=datetime.now().strftime("CLS-%Y%m%d-%H%M"))
    sess_class, sess_subject, sess_faculty = tk.StringVar(), tk.StringVar(), tk.StringVar()
    
    tk.Label(session_card, text="Sess Code:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 2))
    tk.Entry(session_card, textvariable=sess_code, width=18, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(side="left", padx=4)
    tk.Label(session_card, text="Class Node:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 2))
    sess_class_combo = ttk.Combobox(session_card, textvariable=sess_class, width=22, state="readonly"); sess_class_combo.pack(side="left", padx=4)
    tk.Label(session_card, text="Subject Node:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 2))
    sess_subject_combo = ttk.Combobox(session_card, textvariable=sess_subject, width=22, state="readonly"); sess_subject_combo.pack(side="left", padx=4)
    tk.Label(session_card, text="Faculty:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 2))
    sess_faculty_combo = ttk.Combobox(session_card, textvariable=sess_faculty, width=22, state="readonly"); sess_faculty_combo.pack(side="left", padx=4)
    
    session_tree = ttk.Treeview(session_tab, columns=("ID", "Code", "Date", "Class", "Subject", "Faculty", "Status"), show="headings", height=15)
    for c in session_tree["columns"]: session_tree.heading(c, text=c); session_tree.column(c, anchor="center")
    session_tree.pack(fill="both", expand=True, padx=15, pady=5)
    attach_delete_menu(session_tree, "class_sessions")

    def refresh_session_options():
        sess_class_combo["values"] = fetch_named_options("classes", "class_name || ' - ' || COALESCE(section, '')")
        sess_subject_combo["values"] = fetch_named_options("subjects", "COALESCE(subject_code, '') || ' - ' || subject_name")
        sess_faculty_combo["values"] = fetch_named_options("people", "COALESCE(external_ref, '') || ' - ' || name", "person_type='faculty' AND status='active'")

    def load_sessions():
        session_tree.delete(*session_tree.get_children())
        with get_conn() as conn:
            df = pd.read_sql_query("""SELECT ses.id, ses.session_code, ses.session_date, COALESCE(c.class_name, ''),
                                      COALESCE(s.subject_name, ''), COALESCE(p.name, ''), ses.status
                                      FROM class_sessions ses
                                      LEFT JOIN classes c ON c.id=ses.class_id
                                      LEFT JOIN subjects s ON s.id=ses.subject_id
                                      LEFT JOIN people p ON p.id=ses.faculty_person_id
                                      ORDER BY ses.id DESC LIMIT 200""", conn)
        for _, r in df.iterrows(): session_tree.insert("", tk.END, values=tuple(r))

    def save_session():
        cid, sid, fid = option_id(sess_class.get()), option_id(sess_subject.get()), option_id(sess_faculty.get())
        if not all([cid, sid, fid]): return messagebox.showerror("Validation Fault", "Class, Subject, and Faculty must be defined to init a session.")
        create_or_get_class_session(sess_code.get(), cid, sid, fid, "Enterprise Class Session")
        load_sessions()
        sess_code.set(datetime.now().strftime("CLS-%Y%m%d-%H%M"))
        
    tk.Button(session_card, text="▶ Init Session", command=save_session, bg="#14818f", fg="white", font=("Segoe UI", 9, "bold"), cursor="hand2", padx=10).pack(side="right", padx=15)
    tk.Button(session_card, text="↻ Refresh", command=lambda: [refresh_session_options(), load_sessions()], bg="#555", fg="white", cursor="hand2").pack(side="right", padx=5)
    refresh_session_options(); load_sessions()

def open_qr_access_control():
    win = tk.Toplevel(root); win.title("QR Access Control Center"); win.geometry("1220x780")
    open_windows["qr_access"] = win; win.configure(bg=THEMES[current_theme]["bg"])
    add_window_toolbar(win, "qr_access", stop_camera_instance="qr_access")

    scanning = tk.BooleanVar(value=False)
    video_enabled = tk.BooleanVar(value=True)
    sound_enabled = tk.BooleanVar(value=True)
    device_type = tk.StringVar(value="camera")
    area_type = tk.StringVar(value="room")
    event_type = tk.StringVar(value="in")
    area_code = tk.StringVar(value="ROOM-101")
    area_name = tk.StringVar(value="Room 101")
    device_source = tk.StringVar(value=str(CAMERA_SOURCE))
    sensor_value = tk.StringVar()
    status_text = tk.StringVar(value="Ready")
    last_scan_memory = {}

    tk.Label(win, text="QR Access Control Center", font=("Segoe UI", 18, "bold"),
             bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(pady=10)

    main = tk.Frame(win, bg=THEMES[current_theme]["bg"]); main.pack(fill="both", expand=True, padx=14, pady=8)
    left = tk.Frame(main, bg=THEMES[current_theme]["bg"], width=380); left.pack(side="left", fill="y", padx=(0, 12))
    left.pack_propagate(False)
    right = tk.Frame(main, bg=THEMES[current_theme]["bg"]); right.pack(side="right", fill="both", expand=True)

    setup = tk.LabelFrame(left, text="Access Point Setup", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], padx=10, pady=10)
    setup.pack(fill="x", pady=6)

    quick = tk.Frame(setup, bg=THEMES[current_theme]["bg"]); quick.pack(fill="x", pady=4)
    def preset(code, name, typ, evt):
        area_code.set(code); area_name.set(name); area_type.set(typ); event_type.set(evt)
    for text, args in (
        ("Room In", ("ROOM-101", "Room 101", "room", "in")),
        ("Hall In", ("HALL-A", "Hall A", "hall", "in")),
        ("Library Out", ("LIB-RR", "Reading Room", "library", "out")),
        ("Gate In", ("GATE-1", "Main Gate", "gate", "in")),
    ):
        tk.Button(quick, text=text, command=lambda a=args: preset(*a), bg="#333", fg="white", cursor="hand2").pack(side="left", padx=3, pady=3)

    for label, var in (("Area Code", area_code), ("Room/Hall/Area Name", area_name), ("Device Source", device_source)):
        tk.Label(setup, text=label, bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w", pady=(8, 0))
        tk.Entry(setup, textvariable=var, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(fill="x", pady=3)

    row1 = tk.Frame(setup, bg=THEMES[current_theme]["bg"]); row1.pack(fill="x", pady=6)
    tk.Label(row1, text="Area Type", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left")
    ttk.Combobox(row1, textvariable=area_type, values=["room", "hall", "library", "gate", "lab", "office", "hostel"], width=12, state="readonly").pack(side="right")
    row2 = tk.Frame(setup, bg=THEMES[current_theme]["bg"]); row2.pack(fill="x", pady=6)
    tk.Label(row2, text="Event", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left")
    ttk.Combobox(row2, textvariable=event_type, values=["in", "out", "auto", "entry", "exit"], width=12, state="readonly").pack(side="right")
    row3 = tk.Frame(setup, bg=THEMES[current_theme]["bg"]); row3.pack(fill="x", pady=6)
    tk.Label(row3, text="Device", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left")
    ttk.Combobox(row3, textvariable=device_type, values=["camera", "qr_sensor", "manual_reader"], width=12, state="readonly").pack(side="right")

    source_choice = tk.StringVar()
    source_row = tk.Frame(setup, bg=THEMES[current_theme]["bg"]); source_row.pack(fill="x", pady=6)
    tk.Label(source_row, text="Saved Camera / Device", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w")
    source_combo = ttk.Combobox(source_row, textvariable=source_choice, width=42, state="readonly")
    source_combo.pack(fill="x", pady=3)

    def refresh_device_sources():
        with get_conn() as conn:
            cams = conn.cursor().execute("SELECT camera_name, source, location FROM cameras WHERE status='active' ORDER BY camera_name").fetchall()
            devs = conn.cursor().execute("SELECT device_name, device_type, location FROM devices WHERE status='active' ORDER BY device_name").fetchall()
        values = [f"camera | {name} | {src} | {loc or ''}" for name, src, loc in cams]
        values.extend([f"device | {name} | {typ} | {loc or ''}" for name, typ, loc in devs])
        source_combo["values"] = values

    def apply_device_source(_event=None):
        parts = [p.strip() for p in source_choice.get().split("|")]
        if len(parts) >= 3 and parts[0] == "camera":
            device_type.set("camera")
            device_source.set(parts[2])
        elif len(parts) >= 3:
            device_type.set("qr_sensor")
            device_source.set(parts[1])
    source_combo.bind("<<ComboboxSelected>>", apply_device_source)
    refresh_device_sources()

    ops = tk.Frame(setup, bg=THEMES[current_theme]["bg"]); ops.pack(fill="x", pady=6)
    tk.Button(ops, text="IN", command=lambda: event_type.set("in"), bg="#1f7a4d", fg="white", cursor="hand2").pack(side="left", expand=True, fill="x", padx=2)
    tk.Button(ops, text="OUT", command=lambda: event_type.set("out"), bg="#9c3b3b", fg="white", cursor="hand2").pack(side="left", expand=True, fill="x", padx=2)
    tk.Button(ops, text="Camera", command=lambda: device_type.set("camera"), bg="#333", fg="white", cursor="hand2").pack(side="left", expand=True, fill="x", padx=2)
    tk.Button(ops, text="Sensor", command=lambda: device_type.set("qr_sensor"), bg="#333", fg="white", cursor="hand2").pack(side="left", expand=True, fill="x", padx=2)

    def save_access_point():
        if not area_code.get().strip() or not area_name.get().strip():
            return messagebox.showerror("Error", "Area code and name are required.")
        with get_conn() as conn:
            conn.cursor().execute("""INSERT OR REPLACE INTO qr_access_points
                (area_code, area_name, area_type, default_event, device_type, device_source, status, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (area_code.get().strip(), area_name.get().strip(), area_type.get(), event_type.get(),
                 device_type.get(), device_source.get().strip(), "active", "Created from QR Access Control Center"))
            conn.commit()
        messagebox.showinfo("Saved", "Access point saved.")
        load_access_points()

    tk.Button(setup, text="Save Access Point", command=save_access_point, bg="#4CAF50", fg="white", cursor="hand2").pack(fill="x", pady=8)

    controls = tk.LabelFrame(left, text="Scanner Controls", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], padx=10, pady=10)
    controls.pack(fill="x", pady=6)
    tk.Checkbutton(controls, text="Video Preview", variable=video_enabled, bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"],
                   selectcolor=THEMES[current_theme]["entry_bg"]).pack(anchor="w")
    tk.Checkbutton(controls, text="Sound After Scan", variable=sound_enabled, bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"],
                   selectcolor=THEMES[current_theme]["entry_bg"]).pack(anchor="w")
    quick_controls = tk.Frame(controls, bg=THEMES[current_theme]["bg"]); quick_controls.pack(fill="x", pady=5)
    tk.Button(quick_controls, text="Video On", command=lambda: video_enabled.set(True), bg="#333", fg="white", cursor="hand2").pack(side="left", expand=True, fill="x", padx=2)
    tk.Button(quick_controls, text="Video Off", command=lambda: video_enabled.set(False), bg="#333", fg="white", cursor="hand2").pack(side="left", expand=True, fill="x", padx=2)
    tk.Button(quick_controls, text="Sound On", command=lambda: sound_enabled.set(True), bg="#333", fg="white", cursor="hand2").pack(side="left", expand=True, fill="x", padx=2)
    tk.Button(quick_controls, text="Sound Off", command=lambda: sound_enabled.set(False), bg="#333", fg="white", cursor="hand2").pack(side="left", expand=True, fill="x", padx=2)

    def start_scanner():
        source = device_source.get().strip()
        if device_type.get() == "camera":
            actual_src = int(source) if source.isdigit() else source
            cam_manager.start_camera("qr_access", actual_src)
        scanning.set(True)
        status_text.set("Scanning running")

    def stop_scanner():
        scanning.set(False)
        cam_manager.stop_camera("qr_access")
        status_text.set("Scanner stopped")

    tk.Button(controls, text="Start Scanning", command=start_scanner, bg="#14818f", fg="white", cursor="hand2").pack(fill="x", pady=4)
    tk.Button(controls, text="Stop Scanning", command=stop_scanner, bg="#d9534f", fg="white", cursor="hand2").pack(fill="x", pady=4)

    sensor_box = tk.LabelFrame(left, text="Sensor / Manual Reader", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], padx=10, pady=10)
    sensor_box.pack(fill="x", pady=6)
    tk.Entry(sensor_box, textvariable=sensor_value, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(fill="x", pady=4)

    def process_qr(qr_value):
        qr_value = (qr_value or "").strip()
        if not qr_value: return
        now_ts = time.time()
        key = f"{qr_value}|{area_code.get()}|{event_type.get()}"
        if now_ts - last_scan_memory.get(key, 0) < 4: return
        last_scan_memory[key] = now_ts
        status, label, dt_s, tm_s = log_qr_access(qr_value, area_name.get().strip(), event_type.get(), area_code.get().strip(), "qr",
                                                  f"{area_type.get()} access using {device_type.get()}")
        if sound_enabled.get(): play_success_beep()
        status_text.set(f"{status}: {label} {event_type.get().upper()} at {tm_s}")
        log_tree.insert("", 0, values=(dt_s, tm_s, qr_value, label, area_name.get(), event_type.get(), status))

    def manual_scan():
        process_qr(sensor_value.get())
        sensor_value.set("")
    tk.Button(sensor_box, text="Submit Sensor Scan", command=manual_scan, bg="#884EA0", fg="white", cursor="hand2").pack(fill="x", pady=4)

    access_box = tk.LabelFrame(left, text="Saved Access Points", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], padx=8, pady=8)
    access_box.pack(fill="both", expand=True, pady=6)
    access_tree = ttk.Treeview(access_box, columns=("Code", "Name", "Type", "Event"), show="headings", height=5)
    for c in access_tree["columns"]: access_tree.heading(c, text=c); access_tree.column(c, width=80, anchor="center")
    access_tree.pack(fill="both", expand=True)

    def load_access_points():
        access_tree.delete(*access_tree.get_children())
        with get_conn() as conn:
            df = pd.read_sql_query("SELECT area_code, area_name, area_type, default_event, device_type, device_source FROM qr_access_points ORDER BY area_name", conn)
        for _, r in df.iterrows(): access_tree.insert("", tk.END, values=(r["area_code"], r["area_name"], r["area_type"], r["default_event"]))

    def select_access_point(_event=None):
        sel = access_tree.selection()
        if not sel: return
        code, name, typ, evt = access_tree.item(sel[0])["values"]
        with get_conn() as conn:
            row = conn.cursor().execute("SELECT device_type, device_source FROM qr_access_points WHERE area_code=?", (code,)).fetchone()
        area_code.set(code); area_name.set(name); area_type.set(typ); event_type.set(evt)
        if row: device_type.set(row[0] or "camera"); device_source.set(row[1] or str(CAMERA_SOURCE))
    access_tree.bind("<<TreeviewSelect>>", select_access_point)
    load_access_points()

    video_frame = tk.Frame(right, bg="black", height=420); video_frame.pack(fill="both", expand=True)
    video_frame.pack_propagate(False)
    lbl_video = tk.Label(video_frame, bg="black", fg="white", text="Scanner preview")
    lbl_video.place(relx=0.5, rely=0.5, anchor="center")

    tk.Label(right, textvariable=status_text, font=("Segoe UI", 12, "bold"), bg=THEMES[current_theme]["bg"], fg="#00ffcc").pack(fill="x", pady=6)

    log_cols = ("Date", "Time", "QR", "Name", "Area", "Event", "Status")
    log_tree = ttk.Treeview(right, columns=log_cols, show="headings", height=9)
    for c in log_cols: log_tree.heading(c, text=c); log_tree.column(c, anchor="center", width=110)
    log_tree.pack(fill="both", expand=True, pady=6)

    def load_recent_qr_logs():
        log_tree.delete(*log_tree.get_children())
        with get_conn() as conn:
            df = pd.read_sql_query("""SELECT q.date, q.time, q.qr_value, COALESCE(p.name, s.name, 'Unknown QR') as name,
                                      q.area_name, q.event_type, q.status
                                      FROM qr_logs q
                                      LEFT JOIN people p ON p.id=q.person_id
                                      LEFT JOIN students s ON s.id=q.legacy_student_id
                                      ORDER BY q.id DESC LIMIT 80""", conn)
        for _, r in df.iterrows(): log_tree.insert("", tk.END, values=(r["date"], r["time"], r["qr_value"], r["name"], r["area_name"], r["event_type"], r["status"]))
    load_recent_qr_logs()

    def scan_loop():
        if scanning.get() and device_type.get() == "camera":
            ret, frame = cam_manager.read_camera("qr_access")
            if ret and frame is not None:
                codes = decode(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                for code in codes:
                    try: process_qr(code.data.decode("utf-8"))
                    except Exception: pass
                if video_enabled.get():
                    display = draw_hud(frame.copy(), f"QR {area_code.get()} {event_type.get().upper()}")
                    resized = resize_to_fit(display, video_frame.winfo_width(), video_frame.winfo_height())
                    imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
                    lbl_video.imgtk = imgtk; lbl_video.configure(image=imgtk, text="")
                else: lbl_video.configure(image="", text="Video preview off - scanner still running")
        if win.winfo_exists():
            win.after(120, scan_loop)
    scan_loop()

def open_session_attendance():
    win = tk.Toplevel(root); win.title("Session Attendance"); win.geometry("1280x780")
    open_windows["session_attendance"] = win; win.configure(bg=THEMES[current_theme]["bg"])
    add_window_toolbar(win, "session_attendance", stop_camera_instance="session_attendance")

    running = tk.BooleanVar(value=False)
    session_var = tk.StringVar()
    status_var = tk.StringVar(value="Select an academic session.")
    session_meta = {"id": None, "code": None, "class_id": None, "subject_id": None, "faculty_id": None}
    marked_keys = set()

    tk.Label(win, text="Session Attendance", font=("Segoe UI", 18, "bold"),
             bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(pady=8)

    top = tk.Frame(win, bg=THEMES[current_theme]["bg"]); top.pack(fill="x", padx=12, pady=6)
    tk.Label(top, text="Session", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(4, 4))
    session_combo = ttk.Combobox(top, textvariable=session_var, width=62, state="readonly")
    session_combo.pack(side="left", padx=4)

    main = tk.Frame(win, bg=THEMES[current_theme]["bg"]); main.pack(fill="both", expand=True, padx=12, pady=8)
    video_frame = tk.Frame(main, bg="black"); video_frame.pack(side="left", fill="both", expand=True)
    video_frame.pack_propagate(False)
    lbl_video = tk.Label(video_frame, bg="black", fg="white", text="Session camera preview")
    lbl_video.place(relx=0.5, rely=0.5, anchor="center")

    side = tk.Frame(main, width=430, bg=THEMES[current_theme]["bg"]); side.pack(side="right", fill="y", padx=(12, 0))
    side.pack_propagate(False)
    tk.Label(side, textvariable=status_var, wraplength=390, font=("Segoe UI", 11, "bold"),
             bg=THEMES[current_theme]["bg"], fg="#00ffcc").pack(fill="x", pady=6)
    tree = ttk.Treeview(side, columns=("Reg No", "Name", "Time", "Match"), show="headings", height=18)
    for c in tree["columns"]: tree.heading(c, text=c); tree.column(c, anchor="center", width=90)
    tree.pack(fill="both", expand=True, pady=6)

    def load_sessions():
        today = datetime.now().strftime("%Y-%m-%d")
        with get_conn() as conn:
            rows = conn.cursor().execute("""SELECT ses.id, ses.session_code, ses.session_date,
                                            COALESCE(c.class_name || ' ' || COALESCE(c.section, ''), '') as class_label,
                                            COALESCE(s.subject_name, '') as subject_label,
                                            COALESCE(p.name, '') as faculty_name, ses.status
                                            FROM class_sessions ses
                                            LEFT JOIN classes c ON c.id=ses.class_id
                                            LEFT JOIN subjects s ON s.id=ses.subject_id
                                            LEFT JOIN people p ON p.id=ses.faculty_person_id
                                            WHERE ses.status NOT IN ('cancelled', 'holiday')
                                            ORDER BY CASE WHEN ses.session_date=? THEN 0 ELSE 1 END, ses.session_date DESC, ses.id DESC
                                            LIMIT 200""", (today,)).fetchall()
        session_combo["values"] = [f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]}" for r in rows]

    def load_marked_for_session():
        tree.delete(*tree.get_children()); marked_keys.clear()
        sid = session_meta["id"]
        if not sid: return
        with get_conn() as conn:
            rows = conn.cursor().execute("""SELECT COALESCE(s.id, al.legacy_student_id), COALESCE(s.reg_no, ''),
                                            COALESCE(s.name, p.name, ''), al.time, al.match_percentage
                                            FROM attendance_logs al
                                            LEFT JOIN students s ON s.id=al.legacy_student_id
                                            LEFT JOIN people p ON p.id=al.person_id
                                            WHERE al.class_session_id=? ORDER BY al.time DESC""", (sid,)).fetchall()
        for student_id, reg, name, marked_time, match in rows:
            if student_id: marked_keys.add(f"student:{student_id}")
            tree.insert("", tk.END, iid=f"sid:{student_id}" if student_id else "", values=(reg, name, marked_time, "" if match is None else f"{match:.1f}%"))
        status_var.set(f"{len(rows)} people already marked for this session.")

    def select_session(_event=None):
        if not session_var.get(): return
        session_id = option_id(session_var.get())
        with get_conn() as conn:
            row = conn.cursor().execute("""SELECT id, session_code, class_id, subject_id, faculty_person_id, status
                                           FROM class_sessions WHERE id=?""", (session_id,)).fetchone()
        if not row: return
        session_meta.update({"id": row[0], "code": row[1], "class_id": row[2], "subject_id": row[3], "faculty_id": row[4]})
        status_var.set(f"Loaded session {row[1]} ({row[5]}).")
        load_marked_for_session()
    session_combo.bind("<<ComboboxSelected>>", select_session)

    def start_session_scan():
        if not session_meta["id"]: return messagebox.showerror("Session", "Select a session first.")
        cam_manager.start_camera("session_attendance", CAMERA_SOURCE)
        running.set(True)
        status_var.set(f"Scanning session {session_meta['code']}.")

    def stop_session_scan():
        running.set(False); cam_manager.stop_camera("session_attendance")
        status_var.set("Session scanner stopped.")

    def update_session_status(new_status):
        if not session_meta["id"]: return messagebox.showerror("Session", "Select a session first.")
        with get_conn() as conn:
            conn.cursor().execute("UPDATE class_sessions SET status=?, notes=? WHERE id=?",
                                  (new_status, f"Marked {new_status} on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", session_meta["id"]))
            conn.commit()
        stop_session_scan(); load_sessions()
        status_var.set(f"Session {session_meta['code']} marked as {new_status}.")

    def show_selected_session_detail():
        sel = tree.selection()
        if not sel: return messagebox.showerror("Details", "Select a marked student first.")
        iid = str(sel[0])
        if iid.startswith("sid:") and iid.split(":", 1)[1] != "None":
            open_student_detail_window(student_id=int(iid.split(":", 1)[1]))

    tk.Button(top, text="Refresh Sessions", command=load_sessions, bg="#555", fg="white", cursor="hand2").pack(side="left", padx=5)
    tk.Button(top, text="Start Scan", command=start_session_scan, bg="#14818f", fg="white", cursor="hand2").pack(side="left", padx=5)
    tk.Button(top, text="Stop", command=stop_session_scan, bg="#d9534f", fg="white", cursor="hand2").pack(side="left", padx=5)
    tk.Button(top, text="Cancel Today", command=lambda: update_session_status("cancelled"), bg="#9c3b3b", fg="white", cursor="hand2").pack(side="left", padx=5)
    tk.Button(top, text="Holiday", command=lambda: update_session_status("holiday"), bg="#884EA0", fg="white", cursor="hand2").pack(side="left", padx=5)
    tk.Button(side, text="Show Response Details", command=show_selected_session_detail, bg="#14818f", fg="white", cursor="hand2").pack(fill="x", pady=3)
    tk.Button(side, text="Refresh Marked List", command=load_marked_for_session, bg="#555", fg="white", cursor="hand2").pack(fill="x", pady=3)

    fut = None; proc_ctr = 0
    student_indices = [i for i, sid in enumerate(KNOWN_IDS) if sid is not None and KNOWN_PERSON_TYPES[i] == "student"]
    student_embs = [KNOWN_EMBEDDINGS[i] for i in student_indices]

    def session_ai_worker(frame_copy):
        try: faces = FA.get(cv2.resize(frame_copy, (0, 0), fx=0.5, fy=0.5))
        except Exception: return []
        results = []
        for face in faces:
            sims = face_similarity(student_embs, face.embedding)
            if sims.size > 0 and sims.max() >= SIMILARITY_THRESHOLD:
                global_idx = student_indices[int(sims.argmax())]
                results.append({"box": face.bbox*2, "idx": global_idx, "sim": float(sims.max())})
        return results

    def loop():
        nonlocal fut, proc_ctr
        ret, frame = cam_manager.read_camera("session_attendance")
        if ret and frame is not None:
            display = draw_hud(frame.copy(), f"Session {session_meta['code'] or ''}")
            proc_ctr = (proc_ctr + 1) % PROCESS_EVERY_N
            if running.get() and session_meta["id"] and (fut is None or fut.done()):
                if fut and fut.result():
                    for m in fut.result():
                        idx = m["idx"]; student_id = KNOWN_IDS[idx]
                        mark_key = f"student:{student_id}"
                        if mark_key not in marked_keys:
                            now_dt = datetime.now()
                            with get_conn() as conn:
                                existing = conn.cursor().execute("SELECT id FROM attendance_logs WHERE class_session_id=? AND legacy_student_id=?",
                                                                 (session_meta["id"], student_id)).fetchone()
                                if not existing:
                                    conn.cursor().execute("""INSERT INTO attendance_logs
                                        (class_session_id, session_code, person_id, legacy_student_id, person_type, date, time,
                                         camera_name, camera_location, class_id, subject_id, match_percentage, verification_method, status, notes)
                                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                                        (session_meta["id"], session_meta["code"], KNOWN_PERSON_IDS[idx], student_id, "student",
                                         now_dt.strftime("%Y-%m-%d"), now_dt.strftime("%H:%M:%S"), "Session Camera", "Class Session",
                                         session_meta["class_id"], session_meta["subject_id"], m["sim"]*100, "face", "official",
                                         "Session attendance recognition"))
                                    conn.commit()
                                    marked_keys.add(mark_key)
                                    play_success_beep()
                                    r, n = KNOWN_LABELS[idx].split("|")
                                    tree.insert("", 0, iid=f"sid:{student_id}", values=(r.strip(), n.strip(), now_dt.strftime("%H:%M:%S"), f"{m['sim']*100:.1f}%"))
                                    status_var.set(f"Marked {r.strip()} for session {session_meta['code']}.")
                        x1,y1,x2,y2 = map(int, m["box"])
                        cv2.rectangle(display, (x1,y1), (x2,y2), (0,255,0), 3)
                        cv2.putText(display, KNOWN_LABELS[idx], (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                fut = ai_executor.submit(session_ai_worker, frame.copy())
            imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resize_to_fit(display, video_frame.winfo_width(), video_frame.winfo_height()), cv2.COLOR_BGR2RGB)))
            lbl_video.imgtk = imgtk; lbl_video.configure(image=imgtk, text="")
        if win.winfo_exists():
            win.after(40, loop)
    load_sessions(); loop()

# ---------- 3. REPORTS ----------
def open_reports1():
    win = tk.Toplevel(root); win.title("Global Reports"); win.geometry("1060x680")
    open_windows["reports"] = win; win.configure(bg=THEMES[current_theme]["bg"])
    add_window_toolbar(win, "reports", stop_camera_instance=None)
    
    top = tk.Frame(win, bg=THEMES[current_theme]["bg"]); top.pack(fill="x", padx=12, pady=10)
    search_var = tk.StringVar()
    tk.Label(top, text="Search (Reg/Name/Cam):", fg=THEMES[current_theme]["fg"], bg=THEMES[current_theme]["bg"]).pack(side="left")
    tk.Entry(top, textvariable=search_var, width=20, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(side="left", padx=5)
    
    cols = ("Reg No", "Name", "Date", "Time", "Location", "Match %")
    tree = ttk.Treeview(win, columns=cols, show="headings")
    for c in cols: tree.heading(c, text=c); tree.column(c, anchor="center")
    tree.pack(fill="both", expand=True, padx=12, pady=12)
    
    def update_table():
        tree.delete(*tree.get_children())
        with get_conn() as conn:
            q = "SELECT s.reg_no, s.name, a.date, a.time, a.camera_location, a.match_percentage FROM attendance a JOIN students s ON a.student_id = s.id"
            if search_var.get(): q += f" WHERE s.reg_no LIKE '%{search_var.get()}%' OR s.name LIKE '%{search_var.get()}%' OR a.camera_location LIKE '%{search_var.get()}%'"
            df = pd.read_sql_query(q + " ORDER BY a.date DESC, a.time DESC", conn)
        for _, r in df.iterrows(): tree.insert("", tk.END, values=(r["reg_no"], r["name"], r["date"], r["time"], r["camera_location"], f"{r['match_percentage']:.1f}%"))
    
    def export_excel():
        with get_conn() as conn: df = pd.read_sql_query("SELECT s.reg_no, s.name, a.date, a.time, a.camera_location FROM attendance a JOIN students s ON a.student_id = s.id", conn)
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if path: df.to_excel(path, index=False); messagebox.showinfo("Export", "Saved!")

    tk.Button(top, text="Filter", command=update_table, bg="#4CAF50", fg="white", cursor="hand2").pack(side="left", padx=5)
    tk.Button(top, text="Export Excel", command=export_excel, bg="#2196F3", fg="white", cursor="hand2").pack(side="right", padx=5)
    update_table()

def open_reports():
    win_name = "edge_attendance_reports"
    
    # Singleton Window Enforcement
    if win_name in open_windows and open_windows[win_name].winfo_exists():
        open_windows[win_name].deiconify()
        open_windows[win_name].lift()
        return

    win = tk.Toplevel(root)
    win.title("Edge Analytics: Attendance & Biometric Performance")
    win.geometry("1350x850")
    open_windows[win_name] = win
    win.configure(bg=THEMES[current_theme]["bg"])
    add_window_toolbar(win, win_name, stop_camera_instance=None)
    
    # --- Header & KPIs ---
    header_frame = tk.Frame(win, bg=THEMES[current_theme]["bg"])
    header_frame.pack(fill="x", padx=20, pady=10)
    
    tk.Label(header_frame, text="Attendance Intelligence & Reporting", font=("Segoe UI", 18, "bold"),
             bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left")
             
    kpi_frame = tk.Frame(header_frame, bg=THEMES[current_theme]["bg"])
    kpi_frame.pack(side="right")
    
    kpi_total = tk.StringVar(value="Total Scans: --")
    kpi_avg_conf = tk.StringVar(value="Avg Confidence: --")
    
    tk.Label(kpi_frame, textvariable=kpi_total, font=("Consolas", 12, "bold"), fg="#00ffcc", bg=THEMES[current_theme]["card_bg"], padx=10, pady=5, bd=1, relief="solid").pack(side="left", padx=5)
    tk.Label(kpi_frame, textvariable=kpi_avg_conf, font=("Consolas", 12, "bold"), fg="#ffeb3b", bg=THEMES[current_theme]["card_bg"], padx=10, pady=5, bd=1, relief="solid").pack(side="left", padx=5)

    # --- Styles & Notebook ---
    style = ttk.Style()
    style.configure("Treeview", background=THEMES[current_theme]["tree_bg"], foreground=THEMES[current_theme]["tree_fg"], rowheight=28, font=("Segoe UI", 10))
    style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background=THEMES[current_theme]["tree_header_bg"])
    style.configure("TNotebook", background=THEMES[current_theme]["bg"])
    style.configure("TNotebook.Tab", padding=[15, 5], font=("Segoe UI", 10, "bold"))

    nb = ttk.Notebook(win)
    nb.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    # ==========================================
    # TAB 1: ADVANCED FILTERABLE MATRIX LEDGER
    # ==========================================
    tab_ledger = tk.Frame(nb, bg=THEMES[current_theme]["bg"])
    nb.add(tab_ledger, text=" 🗃️ Master Data Ledger ")
    
    filter_bar = tk.LabelFrame(tab_ledger, text=" Deep Filtering Engine ", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"], font=("Segoe UI", 10, "bold"), pady=10, padx=10)
    filter_bar.pack(fill="x", padx=10, pady=10)
    
    tk.Label(filter_bar, text="Query (Reg/Name):", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(side="left")
    search_var = tk.StringVar()
    tk.Entry(filter_bar, textvariable=search_var, width=20, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(side="left", padx=5)
    
    tk.Label(filter_bar, text="Timeframe:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 0))
    timeframe_var = tk.StringVar(value="All Time")
    ttk.Combobox(filter_bar, textvariable=timeframe_var, values=["All Time", "Today", "Last 7 Days", "This Month"], state="readonly", width=15).pack(side="left", padx=5)
    
    tk.Label(filter_bar, text="Min Confidence %:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(15, 0))
    conf_scale = tk.Scale(filter_bar, from_=0, to_=100, orient="horizontal", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"], highlightthickness=0, length=120)
    conf_scale.set(0)
    conf_scale.pack(side="left", padx=5)

    tv_ledger = ttk.Treeview(tab_ledger, columns=("Date", "Time", "Reg No", "Name", "Location", "Match %", "Status"), show="headings", height=18)
    for c in tv_ledger["columns"]: 
        tv_ledger.heading(c, text=c)
        tv_ledger.column(c, anchor="center", width=120 if c != "Name" else 200)
    tv_ledger.pack(fill="both", expand=True, padx=10, pady=5)

    def fetch_filtered_data():
        query = search_var.get().strip().lower()
        time_filter = timeframe_var.get()
        min_conf = conf_scale.get()
        
        where_clauses = [f"al.match_percentage >= {min_conf}"]
        if time_filter == "Today": where_clauses.append(f"al.date = '{datetime.now().strftime('%Y-%m-%d')}'")
        elif time_filter == "Last 7 Days": where_clauses.append("al.date >= date('now', '-7 days')")
        elif time_filter == "This Month": where_clauses.append("al.date >= date('now', 'start of month')")
        
        where_sql = "WHERE " + " AND ".join(where_clauses)
        
        with get_conn() as conn:
            return pd.read_sql_query(f"""
                SELECT al.date, al.time, COALESCE(st.reg_no, p.reg_no, 'Unknown') as reg_no, 
                       COALESCE(st.name, p.name, 'Unknown') as name, al.camera_location, 
                       al.match_percentage, al.status
                FROM attendance_logs al
                LEFT JOIN students st ON st.id = al.legacy_student_id
                LEFT JOIN people p ON p.id = al.person_id
                {where_sql}
                ORDER BY al.date DESC, al.time DESC
            """, conn)

    def update_ledger(*args):
        tv_ledger.delete(*tv_ledger.get_children())
        df = fetch_filtered_data()
        
        query = search_var.get().strip().lower()
        count = 0
        conf_sum = 0
        
        for _, r in df.iterrows():
            if query and not any(query in str(val).lower() for val in r.values): continue
            count += 1
            if pd.notna(r['match_percentage']): conf_sum += float(r['match_percentage'])
            match_str = f"{float(r['match_percentage']):.1f}%" if pd.notna(r['match_percentage']) else "N/A"
            tv_ledger.insert("", tk.END, values=(r["date"], r["time"], r["reg_no"], r["name"], r["camera_location"], match_str, r["status"].upper()))
            
        kpi_total.set(f"Total Scans: {count}")
        kpi_avg_conf.set(f"Avg Confidence: {conf_sum/count:.1f}%" if count > 0 else "Avg Confidence: --")

    # Bind filters to update the ledger automatically
    search_var.trace_add("write", update_ledger)
    timeframe_var.trace_add("write", update_ledger)
    conf_scale.bind("<ButtonRelease-1>", update_ledger)

    def export_ledger():
        df = fetch_filtered_data()
        path = filedialog.asksaveasfilename(initialfile=f"Attendance_Report_{datetime.now().strftime('%Y%m%d')}.xlsx", defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if path:
            df.to_excel(path, index=False)
            messagebox.showinfo("Export Success", "Enterprise Ledger Exported Successfully.")

    tk.Button(filter_bar, text="💾 Export to Excel", command=export_ledger, bg="#2196F3", fg="white", font=("Segoe UI", 9, "bold"), cursor="hand2").pack(side="right", padx=10)

    # ==========================================
    # TAB 2: DYNAMIC GRAPHICS & PLOTS
    # ==========================================
    tab_graphics = tk.Frame(nb, bg=THEMES[current_theme]["bg"])
    nb.add(tab_graphics, text=" 📈 Edge Analytics Visualizer ")
    
    # Create Matplotlib Figure with 3 Subplots (1 top, 2 bottom)
    fig = Figure(figsize=(10, 6), dpi=100)
    fig.patch.set_facecolor(THEMES[current_theme]["bg"])
    
    ax_timeline = fig.add_subplot(211) # Top wide
    ax_hist = fig.add_subplot(223)     # Bottom left
    ax_pie = fig.add_subplot(224)      # Bottom right
    
    for ax in [ax_timeline, ax_hist]:
        ax.set_facecolor(THEMES[current_theme]["entry_bg"])
        ax.tick_params(colors=THEMES[current_theme]["fg"])
        for spine in ax.spines.values(): spine.set_color(THEMES[current_theme]["fg"])

    canvas_graphics = FigureCanvasTkAgg(fig, master=tab_graphics)
    canvas_graphics.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    def refresh_graphics():
        df = fetch_filtered_data()
        if df.empty: return
        
        ax_timeline.clear()
        ax_hist.clear()
        ax_pie.clear()
        
        # 1. Timeline Plot (Scans per day)
        daily_counts = df.groupby('date').size()
        if not daily_counts.empty:
            ax_timeline.plot(daily_counts.index, daily_counts.values, marker='o', color='#00e5ff', linewidth=2)
            ax_timeline.set_title("Attendance Traffic Volume Over Time", color=THEMES[current_theme]["fg"])
            ax_timeline.grid(True, linestyle='--', alpha=0.3, color=THEMES[current_theme]["fg"])
            ax_timeline.tick_params(axis='x', rotation=45)
            
        # 2. Histogram (Confidence Distribution)
        conf_scores = df['match_percentage'].dropna()
        if not conf_scores.empty:
            ax_hist.hist(conf_scores, bins=20, color='#9c27b0', edgecolor='white', alpha=0.8)
            ax_hist.set_title("Biometric Confidence Distribution", color=THEMES[current_theme]["fg"])
            ax_hist.set_xlabel("Match %", color=THEMES[current_theme]["fg"])
            ax_hist.grid(True, linestyle=':', alpha=0.2, color=THEMES[current_theme]["fg"])
            
        # 3. Pie Chart (Official vs Unofficial/Unknown status)
        status_counts = df['status'].value_counts()
        if not status_counts.empty:
            colors = ['#4CAF50', '#ff9800', '#f44336']
            wedges, texts, autotexts = ax_pie.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=colors, startangle=90)
            ax_pie.set_title("Capture Status Breakdown", color=THEMES[current_theme]["fg"])
            for text in texts: text.set_color(THEMES[current_theme]["fg"])
            for autotext in autotexts: autotext.set_color('white')

        fig.tight_layout()
        canvas_graphics.draw()

    graphics_ctrl = tk.Frame(tab_graphics, bg=THEMES[current_theme]["bg"])
    graphics_ctrl.pack(fill="x", pady=5)
    tk.Button(graphics_ctrl, text="↻ Re-Render Visualizations based on Filters", command=refresh_graphics, bg="#14818f", fg="white", font=("Segoe UI", 10, "bold"), cursor="hand2", pady=5).pack()

    # Initial Load
    update_ledger()
    refresh_graphics()

def open_analytics_reports():
    win_name = "analytics_reports"
    
    # BUG FIX: Singleton Window Enforcement
    if win_name in open_windows and open_windows[win_name].winfo_exists():
        open_windows[win_name].deiconify()
        open_windows[win_name].lift()
        return

    win = tk.Toplevel(root)
    win.title("Enterprise Analytics & Automated Insights Engine")
    win.geometry("1400x880")
    open_windows[win_name] = win
    win.configure(bg=THEMES[current_theme]["bg"])
    add_window_toolbar(win, win_name, stop_camera_instance=None)

    tk.Label(win, text="Advanced Analytics & Diagnostics Lab", font=("Segoe UI", 18, "bold"),
             bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(pady=10)

    # --- UI STYLING ---
    style = ttk.Style()
    style.configure("Treeview", background=THEMES[current_theme]["tree_bg"], foreground=THEMES[current_theme]["tree_fg"], rowheight=28)
    style.configure("Treeview.Heading", font=("Segoe UI", 10, "bold"), background=THEMES[current_theme]["tree_header_bg"])
    style.configure("TNotebook", background=THEMES[current_theme]["bg"])
    style.configure("TNotebook.Tab", padding=[15, 5], font=("Segoe UI", 10, "bold"))

    nb = ttk.Notebook(win)
    nb.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    # ==========================================
    # TAB 1: AUTOMATED INSIGHTS & EXECUTIVE HUD
    # ==========================================
    tab_exec = tk.Frame(nb, bg=THEMES[current_theme]["bg"])
    nb.add(tab_exec, text=" 🧠 AI Insights Engine ")

    exec_split = tk.Frame(tab_exec, bg=THEMES[current_theme]["bg"])
    exec_split.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Left: Insights Text Box
    insights_frame = tk.LabelFrame(exec_split, text=" Automated System Diagnostics ", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"], font=("Segoe UI", 12, "bold"), padx=15, pady=15)
    insights_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
    
    insights_text = tk.Text(insights_frame, bg=THEMES[current_theme]["entry_bg"], fg="#00e5ff", font=("Consolas", 12), wrap="word", bd=0, state="disabled")
    insights_text.pack(fill="both", expand=True)

    # Right: Executive Graphs
    exec_fig = Figure(figsize=(6, 5), dpi=100)
    exec_fig.patch.set_facecolor(THEMES[current_theme]["bg"])
    ax_status = exec_fig.add_subplot(211)
    ax_trend = exec_fig.add_subplot(212)
    
    for ax in [ax_status, ax_trend]:
        ax.set_facecolor(THEMES[current_theme]["entry_bg"])
        ax.tick_params(colors=THEMES[current_theme]["fg"])
        for spine in ax.spines.values(): spine.set_color(THEMES[current_theme]["fg"])
        
    canvas_exec = FigureCanvasTkAgg(exec_fig, master=exec_split)
    canvas_exec.get_tk_widget().pack(side="right", fill="both", expand=True)

    # ==========================================
    # TAB 2: HARDWARE & BIOMETRIC HEALTH
    # ==========================================
    tab_hardware = tk.Frame(nb, bg=THEMES[current_theme]["bg"])
    nb.add(tab_hardware, text=" 📷 Hardware Health ")
    
    hw_fig = Figure(figsize=(10, 6), dpi=100)
    hw_fig.patch.set_facecolor(THEMES[current_theme]["bg"])
    ax_cam_perf = hw_fig.add_subplot(111)
    ax_cam_perf.set_facecolor(THEMES[current_theme]["entry_bg"])
    ax_cam_perf.tick_params(colors=THEMES[current_theme]["fg"])
    for spine in ax_cam_perf.spines.values(): spine.set_color(THEMES[current_theme]["fg"])
    
    canvas_hw = FigureCanvasTkAgg(hw_fig, master=tab_hardware)
    canvas_hw.get_tk_widget().pack(fill="both", expand=True, padx=20, pady=20)

    # ==========================================
    # TAB 3: ACADEMIC RISK & REVIEW MATRIX
    # ==========================================
    tab_risk = tk.Frame(nb, bg=THEMES[current_theme]["bg"])
    nb.add(tab_risk, text=" ⚠️ Academic Risk Analysis ")
    
    tk.Label(tab_risk, text="Students/Personnel with Critical Absence Rates or Biometric Failures", font=("Segoe UI", 12, "bold"), bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w", padx=10, pady=10)

    risk_tree = ttk.Treeview(tab_risk, columns=("Identity", "Type", "Course/Dept", "Sessions Logged", "Avg Match %", "Risk Factor"), show="headings", height=15)
    for c in risk_tree["columns"]: risk_tree.heading(c, text=c); risk_tree.column(c, anchor="center")
    risk_tree.pack(fill="both", expand=True, padx=10, pady=5)

    # ==========================================
    # DATA PROCESSING & RENDERING ENGINE
    # ==========================================
    def write_insight(text, tag="info"):
        insights_text.config(state="normal")
        insights_text.insert(tk.END, f"[{datetime.now().strftime('%H:%M:%S')}] {text}\n\n", tag)
        insights_text.config(state="disabled")

    def analyze_and_render_all():
        # 1. Fetch Global Enterprise Data
        with get_conn() as conn:
            df_logs = pd.read_sql_query("""
                SELECT al.date, al.time, al.status, al.match_percentage, al.camera_location,
                       COALESCE(st.name, p.name, 'Unknown') as name, al.person_type,
                       COALESCE(c.class_name, p.department, 'Unassigned') as department
                FROM attendance_logs al
                LEFT JOIN students st ON st.id = al.legacy_student_id
                LEFT JOIN people p ON p.id = al.person_id
                LEFT JOIN classes c ON c.id = al.class_id
            """, conn)

        insights_text.config(state="normal")
        insights_text.delete(1.0, tk.END)
        insights_text.config(state="disabled")
        
        insights_text.tag_config("critical", foreground="#ff4444", font=("Consolas", 12, "bold"))
        insights_text.tag_config("warn", foreground="#ffeb3b", font=("Consolas", 12))
        insights_text.tag_config("good", foreground="#00ff00", font=("Consolas", 12))
        
        if df_logs.empty:
            write_insight("System Standby: Insufficient data to generate enterprise insights.", "warn")
            return

        # --- A. EXECUTE AUTOMATED INSIGHTS ---
        # 1. Hardware Check
        cam_stats = df_logs.dropna(subset=['match_percentage']).groupby('camera_location')['match_percentage'].mean()
        failing_cams = cam_stats[cam_stats < 55.0]
        for cam, avg in failing_cams.items():
            write_insight(f"CRITICAL HARDWARE DEGRADATION: Node '{cam}' average biometric confidence dropped to {avg:.1f}%. Recommend lens cleaning or lighting adjustment.", "critical")
        if failing_cams.empty:
            write_insight(f"All {len(cam_stats)} hardware capture nodes are operating within optimal biometric thresholds (>55% average similarity).", "good")

        # 2. Daily Trend Check
        recent_days = df_logs['date'].value_counts().sort_index().tail(2)
        if len(recent_days) == 2:
            prev, curr = recent_days.iloc[0], recent_days.iloc[1]
            if curr < (prev * 0.5):
                write_insight(f"TRAFFIC ANOMALY: Attendance volume today ({curr} scans) is significantly lower than the previous active day ({prev} scans).", "warn")
            else:
                write_insight(f"Traffic volume is stable. Processed {curr} verified biometric events today.", "good")

        # 3. Status Breakdown
        anomalies = df_logs[df_logs['status'] != 'official']
        anomaly_rate = (len(anomalies) / len(df_logs)) * 100
        if anomaly_rate > 15:
            write_insight(f"SECURITY ALERT: {anomaly_rate:.1f}% of recent scans are marked unofficial/unknown. Check surveillance parameters.", "critical")

        # --- B. RENDER EXECUTIVE GRAPHS ---
        ax_status.clear()
        ax_trend.clear()
        
        # Pie Chart
        status_counts = df_logs['status'].value_counts()
        ax_status.pie(status_counts, labels=status_counts.index, autopct='%1.1f%%', colors=['#4CAF50', '#ff9800', '#f44336'], textprops={'color': THEMES[current_theme]["fg"]})
        ax_status.set_title("System Scan Resolutions", color=THEMES[current_theme]["fg"])
        
        # Traffic Trend (Last 14 days)
        trend_data = df_logs.groupby('date').size().tail(14)
        ax_trend.plot(trend_data.index, trend_data.values, marker='o', color='#00e5ff', linewidth=2)
        ax_trend.set_title("14-Day Global Traffic Volume", color=THEMES[current_theme]["fg"])
        ax_trend.tick_params(axis='x', rotation=45)
        ax_trend.grid(True, linestyle='--', alpha=0.3)
        
        exec_fig.tight_layout()
        canvas_exec.draw()

        # --- C. RENDER HARDWARE HEALTH ---
        ax_cam_perf.clear()
        if not cam_stats.empty:
            bars = ax_cam_perf.bar(cam_stats.index, cam_stats.values, color='#9c27b0', edgecolor='white')
            ax_cam_perf.axhline(y=55.0, color='#ff4444', linestyle='dashed', linewidth=2, label='Critical Threshold (55%)')
            ax_cam_perf.set_title("Biometric Confidence per Hardware Node", color=THEMES[current_theme]["fg"], pad=15)
            ax_cam_perf.set_ylabel("Average Cosine Match (%)", color=THEMES[current_theme]["fg"])
            ax_cam_perf.tick_params(axis='x', rotation=15)
            ax_cam_perf.legend(facecolor=THEMES[current_theme]["card_bg"], labelcolor=THEMES[current_theme]["fg"])
            ax_cam_perf.set_ylim(0, 100)
            
            # Auto-label bars
            for bar in bars:
                yval = bar.get_height()
                ax_cam_perf.text(bar.get_x() + bar.get_width()/2.0, yval, f"{yval:.1f}%", ha='center', va='bottom', color=THEMES[current_theme]["fg"], fontsize=9)
                
        hw_fig.tight_layout()
        canvas_hw.draw()

        # --- D. POPULATE ACADEMIC RISK MATRIX ---
        risk_tree.delete(*risk_tree.get_children())
        
        # Group by person to find those with lowest match rates or frequent anomalies
        risk_df = df_logs.groupby(['name', 'person_type', 'department']).agg(
            sessions_logged=('time', 'count'),
            avg_match=('match_percentage', 'mean'),
            unofficial_count=('status', lambda x: (x != 'official').sum())
        ).reset_index()
        
        # Define a risk factor: High risk if avg match < 60 OR high anomaly ratio
        risk_df['risk_score'] = np.where(risk_df['avg_match'] < 60, 2, 0) + np.where((risk_df['unofficial_count'] / risk_df['sessions_logged']) > 0.3, 2, 0)
        
        critical_cases = risk_df[risk_df['risk_score'] >= 2].sort_values(by='avg_match')
        
        for _, r in critical_cases.iterrows():
            match_str = f"{float(r['avg_match']):.1f}%" if pd.notna(r['avg_match']) else "N/A"
            risk_label = "HIGH RISK (Degraded Biometrics)" if r['avg_match'] < 60 else "MODERATE RISK (Anomalous Activity)"
            risk_tree.insert("", tk.END, values=(r["name"], r["person_type"].upper(), r["department"], r["sessions_logged"], match_str, risk_label))

    # Control Bar at bottom of window
    ctrl_frame = tk.Frame(win, bg=THEMES[current_theme]["card_bg"], bd=1, relief="solid", pady=10)
    ctrl_frame.pack(fill="x", side="bottom")
    
    tk.Button(ctrl_frame, text="⚡ RUN DEEP DIAGNOSTIC ANALYSIS", command=analyze_and_render_all, bg="#884EA0", fg="white", font=("Segoe UI", 12, "bold"), cursor="hand2", padx=20, pady=5).pack()

    # Initial Run
    win.after(500, analyze_and_render_all)


def open_surveillance_reports():
    win = tk.Toplevel(root); win.title("Surveillance Reports"); win.geometry("1180x720")
    open_windows["surveillance_reports"] = win; win.configure(bg=THEMES[current_theme]["bg"])
    add_window_toolbar(win, "surveillance_reports", stop_camera_instance=None)

    tk.Label(win, text="Surveillance Tracks & Security Alerts", font=("Segoe UI", 16, "bold"),
             fg=THEMES[current_theme]["fg"], bg=THEMES[current_theme]["bg"]).pack(pady=10)

    top = tk.Frame(win, bg=THEMES[current_theme]["bg"]); top.pack(fill="x", padx=12, pady=6)
    search_var = tk.StringVar()
    severity_var = tk.StringVar(value="All")
    tk.Label(top, text="Search:", fg=THEMES[current_theme]["fg"], bg=THEMES[current_theme]["bg"]).pack(side="left")
    tk.Entry(top, textvariable=search_var, width=24, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(side="left", padx=5)
    tk.Label(top, text="Severity:", fg=THEMES[current_theme]["fg"], bg=THEMES[current_theme]["bg"]).pack(side="left", padx=(12, 4))
    ttk.Combobox(top, textvariable=severity_var, values=["All", "info", "log", "alert"], width=10, state="readonly").pack(side="left")

    cols = ("Date", "Time", "Camera", "Mode", "Event", "Label", "Match %", "Severity", "Action")
    tree = ttk.Treeview(win, columns=cols, show="headings", height=16)
    for c in cols: tree.heading(c, text=c); tree.column(c, anchor="center", width=120)
    tree.pack(fill="both", expand=True, padx=12, pady=8)

    summary_frame = tk.LabelFrame(win, text="Daily Surveillance Summary", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], padx=8, pady=8)
    summary_frame.pack(fill="x", padx=12, pady=6)
    summary_tree = ttk.Treeview(summary_frame, columns=("Date", "Known", "Unknown", "Alerts", "Updated"), show="headings", height=4)
    for c in summary_tree["columns"]: summary_tree.heading(c, text=c); summary_tree.column(c, anchor="center")
    summary_tree.pack(fill="x")

    def build_filter():
        where, params = [], []
        if search_var.get().strip():
            s = f"%{search_var.get().strip()}%"
            where.append("(camera_name LIKE ? OR mode LIKE ? OR event_type LIKE ? OR label LIKE ? OR action_taken LIKE ? OR details LIKE ?)")
            params.extend([s, s, s, s, s, s])
        if severity_var.get() != "All":
            where.append("severity=?"); params.append(severity_var.get())
        clause = " WHERE " + " AND ".join(where) if where else ""
        return clause, params

    def update_table():
        tree.delete(*tree.get_children())
        clause, params = build_filter()
        with get_surveillance_conn() as conn:
            df = pd.read_sql_query(f"""SELECT date, time, camera_name, mode, event_type, label,
                                       match_percentage, severity, action_taken
                                       FROM surveillance_tracks {clause}
                                       ORDER BY id DESC LIMIT 500""", conn, params=params)
        for _, r in df.iterrows():
            match = "" if pd.isna(r["match_percentage"]) else f"{float(r['match_percentage']):.1f}%"
            tree.insert("", tk.END, values=(r["date"], r["time"], r["camera_name"], r["mode"], r["event_type"],
                                            r["label"], match, r["severity"], r["action_taken"]))
        load_summary()

    def load_summary():
        summary_tree.delete(*summary_tree.get_children())
        with get_surveillance_conn() as conn:
            df = pd.read_sql_query("SELECT report_date, known_count, unknown_count, alert_count, last_updated FROM surveillance_summary ORDER BY report_date DESC LIMIT 30", conn)
        for _, r in df.iterrows(): summary_tree.insert("", tk.END, values=(r["report_date"], r["known_count"], r["unknown_count"], r["alert_count"], r["last_updated"]))

    def export_excel():
        clause, params = build_filter()
        with get_surveillance_conn() as conn:
            df = pd.read_sql_query(f"SELECT * FROM surveillance_tracks {clause} ORDER BY id DESC", conn, params=params)
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if path:
            df.to_excel(path, index=False)
            messagebox.showinfo("Export", "Surveillance report saved.")

    def backup_surveillance_db():
        backup_database_file(SURVEILLANCE_DB_FILE, "Backup Surveillance Database")

    tk.Button(top, text="Refresh", command=update_table, bg="#4CAF50", fg="white", cursor="hand2").pack(side="left", padx=8)
    tk.Button(top, text="Export Excel", command=export_excel, bg="#2196F3", fg="white", cursor="hand2").pack(side="right", padx=5)
    tk.Button(top, text="Backup Surveillance DB", command=backup_surveillance_db, bg="#555", fg="white", cursor="hand2").pack(side="right", padx=5)
    update_table()


def open_cctv_surveillance_soc():
    win_name = "cctv_surveillance_soc"
    
    # Singleton Window Enforcement
    if win_name in open_windows and open_windows[win_name].winfo_exists():
        open_windows[win_name].deiconify()
        open_windows[win_name].lift()
        return

    win = tk.Toplevel(root)
    win.title("Security Operations Center (SOC) & Threat Matrix")
    win.geometry("1450x880")
    open_windows[win_name] = win
    win.configure(bg="#0a0a0a") # Force dark mode for SOC feel
    add_window_toolbar(win, win_name, stop_camera_instance=None)

    # --- Header UI ---
    header_frame = tk.Frame(win, bg="#0a0a0a")
    header_frame.pack(fill="x", padx=20, pady=10)
    
    tk.Label(header_frame, text="🛡️ CCTV Intelligence & Active Threat Matrix", font=("Segoe UI", 18, "bold"),
             bg="#0a0a0a", fg="#00ffcc").pack(side="left")
             
    kpi_frame = tk.Frame(header_frame, bg="#0a0a0a")
    kpi_frame.pack(side="right")
    
    kpi_alerts = tk.StringVar(value="Active Alerts: --")
    kpi_unknowns = tk.StringVar(value="Unknown Intercepts: --")
    
    tk.Label(kpi_frame, textvariable=kpi_alerts, font=("Consolas", 12, "bold"), fg="#ff4444", bg="#1a1a1a", padx=15, pady=5, bd=1, relief="solid").pack(side="left", padx=5)
    tk.Label(kpi_frame, textvariable=kpi_unknowns, font=("Consolas", 12, "bold"), fg="#ffeb3b", bg="#1a1a1a", padx=15, pady=5, bd=1, relief="solid").pack(side="left", padx=5)

    # --- Custom Notebook Style for SOC ---
    style = ttk.Style()
    style.theme_use('default')
    style.configure("SOCTreeview.Treeview", background="#121212", foreground="white", fieldbackground="#121212", rowheight=28, font=("Segoe UI", 10))
    style.configure("SOCTreeview.Treeview.Heading", font=("Segoe UI", 10, "bold"), background="#1f1f1f", foreground="#00e5ff")
    style.map("SOCTreeview.Treeview", background=[('selected', '#14818f')])
    
    style.configure("SOC.TNotebook", background="#0a0a0a")
    style.configure("SOC.TNotebook.Tab", padding=[15, 5], font=("Segoe UI", 10, "bold"), background="#1a1a1a", foreground="white")
    style.map("SOC.TNotebook.Tab", background=[('selected', '#14818f')])

    nb = ttk.Notebook(win, style="SOC.TNotebook")
    nb.pack(fill="both", expand=True, padx=20, pady=(0, 20))

    # ==========================================
    # TAB 1: LIVE THREAT DASHBOARD & PLOTS
    # ==========================================
    tab_dashboard = tk.Frame(nb, bg="#0a0a0a")
    nb.add(tab_dashboard, text=" 👁️ Live Threat Dashboard ")
    
    dash_split = tk.Frame(tab_dashboard, bg="#0a0a0a")
    dash_split.pack(fill="both", expand=True, padx=10, pady=10)
    
    # Left: Critical Alerts Feed
    alerts_frame = tk.LabelFrame(dash_split, text=" CRITICAL ALERT FEED ", bg="#121212", fg="#ff4444", font=("Consolas", 12, "bold"), padx=10, pady=10)
    alerts_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
    
    tv_alerts = ttk.Treeview(alerts_frame, columns=("Time", "Node", "Action Required"), show="headings", height=15, style="SOCTreeview.Treeview")
    for c in tv_alerts["columns"]: tv_alerts.heading(c, text=c); tv_alerts.column(c, anchor="center")
    tv_alerts.pack(fill="both", expand=True)

    # Right: Surveillance Analytics Plot
    plot_frame = tk.Frame(dash_split, bg="#121212", bd=1, relief="solid")
    plot_frame.pack(side="right", fill="both", expand=True)
    
    soc_fig = Figure(figsize=(6, 5), dpi=100)
    soc_fig.patch.set_facecolor('#121212')
    ax_threats = soc_fig.add_subplot(111)
    ax_threats.set_facecolor('#1a1a1a')
    ax_threats.tick_params(colors='white')
    for spine in ax_threats.spines.values(): spine.set_color('#555')
    
    canvas_soc = FigureCanvasTkAgg(soc_fig, master=plot_frame)
    canvas_soc.get_tk_widget().pack(fill="both", expand=True, padx=10, pady=10)

    # ==========================================
    # TAB 2: MASTER SURVEILLANCE LEDGER
    # ==========================================
    tab_ledger = tk.Frame(nb, bg="#0a0a0a")
    nb.add(tab_ledger, text=" 🗃️ Global Camera Ledger ")
    
    filter_bar = tk.Frame(tab_ledger, bg="#121212", bd=1, relief="solid")
    filter_bar.pack(fill="x", padx=10, pady=10, ipady=8)
    
    tk.Label(filter_bar, text="Severity Filter:", bg="#121212", fg="white", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(15, 5))
    severity_var = tk.StringVar(value="All")
    ttk.Combobox(filter_bar, textvariable=severity_var, values=["All", "INFO", "ALERT", "CRITICAL"], state="readonly", width=15).pack(side="left", padx=5)
    
    tk.Label(filter_bar, text="Search Identity/Node:", bg="#121212", fg="white", font=("Segoe UI", 10, "bold")).pack(side="left", padx=(20, 5))
    search_var = tk.StringVar()
    tk.Entry(filter_bar, textvariable=search_var, width=30, bg="#1a1a1a", fg="#00e5ff", insertbackground="white").pack(side="left", padx=5)

    tv_ledger = ttk.Treeview(tab_ledger, columns=("Date", "Time", "Hardware Node", "Event", "Identity", "Match %", "Severity"), show="headings", height=18, style="SOCTreeview.Treeview")
    for c in tv_ledger["columns"]: 
        tv_ledger.heading(c, text=c)
        tv_ledger.column(c, anchor="center" if c != "Identity" else "w")
    tv_ledger.pack(fill="both", expand=True, padx=10, pady=5)

    # ==========================================
    # DATA AGGREGATION & RENDERING
    # ==========================================
    def refresh_soc_data(*args):
        # 1. Update Ledger
        tv_ledger.delete(*tv_ledger.get_children())
        tv_alerts.delete(*tv_alerts.get_children())
        
        sev_filter = severity_var.get().lower()
        search_query = search_var.get().strip().lower()
        
        where_clauses = []
        if sev_filter != "all": where_clauses.append(f"severity = '{sev_filter}'")
        where_sql = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        with get_surveillance_conn() as conn:
            df = pd.read_sql_query(f"""
                SELECT date, time, camera_name, event_type, label, match_percentage, severity
                FROM surveillance_tracks
                {where_sql}
                ORDER BY id DESC LIMIT 1000
            """, conn)
            
        alerts_count = 0
        unknowns_count = 0
            
        for _, r in df.iterrows():
            if search_query and not any(search_query in str(val).lower() for val in r.values): continue
            
            match_str = f"{float(r['match_percentage']):.1f}%" if pd.notna(r['match_percentage']) else "N/A"
            sev = str(r["severity"]).upper()
            
            # Count KPIs
            if sev == "ALERT" or sev == "CRITICAL": alerts_count += 1
            if r["event_type"] == "unknown_face": unknowns_count += 1
            
            tv_ledger.insert("", tk.END, values=(r["date"], r["time"], r["camera_name"], str(r["event_type"]).upper(), r["label"], match_str, sev))
            
            # Populate Critical Feed
            if sev == "ALERT" or sev == "CRITICAL":
                tv_alerts.insert("", tk.END, values=(r["time"], r["camera_name"], f"Investigate {r['label']}"))

        kpi_alerts.set(f"Active Alerts: {alerts_count}")
        kpi_unknowns.set(f"Unknown Intercepts: {unknowns_count}")

        # 2. Update Matplotlib Chart (Hourly Threat Distribution)
        ax_threats.clear()
        if not df.empty:
            # Convert time to hour
            df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S', errors='coerce').dt.hour
            alert_df = df[df['severity'].isin(['alert', 'critical'])]
            
            if not alert_df.empty:
                hourly_alerts = alert_df.groupby('hour').size()
                ax_threats.plot(hourly_alerts.index, hourly_alerts.values, marker='o', color='#ff4444', linewidth=2, label="Security Alerts")
                ax_threats.fill_between(hourly_alerts.index, hourly_alerts.values, color='#ff4444', alpha=0.2)
                
            ax_threats.set_title("Intrusion / Alert Velocity (24h Window)", color="white", pad=10)
            ax_threats.set_xlabel("Hour of Day", color="#aaa")
            ax_threats.set_ylabel("Alert Frequency", color="#aaa")
            ax_threats.set_xticks(range(0, 24, 2))
            ax_threats.grid(True, linestyle=':', alpha=0.2, color="white")
            if not alert_df.empty: ax_threats.legend(facecolor='#121212', labelcolor='white')
        else:
            ax_threats.text(0.5, 0.5, "No Tracking Data Available", horizontalalignment='center', verticalalignment='center', transform=ax_threats.transAxes, color="#aaa")
            
        soc_fig.tight_layout()
        canvas_soc.draw()

    severity_var.trace_add("write", refresh_soc_data)
    search_var.trace_add("write", refresh_soc_data)

    # --- Control & Export ---
    ctrl_frame = tk.Frame(win, bg="#121212", bd=1, relief="solid", pady=10)
    ctrl_frame.pack(fill="x", side="bottom")
    
    def export_soc_ledger():
        path = filedialog.asksaveasfilename(initialfile=f"SOC_Report_{datetime.now().strftime('%Y%m%d')}.xlsx", defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if path:
            with get_surveillance_conn() as conn:
                df = pd.read_sql_query("SELECT * FROM surveillance_tracks ORDER BY id DESC", conn)
                df.to_excel(path, index=False)
            messagebox.showinfo("Export Complete", "SOC Database successfully exported to Excel.")
            
    tk.Button(ctrl_frame, text="↻ Synchronize Feeds", command=refresh_soc_data, bg="#14818f", fg="white", font=("Segoe UI", 10, "bold"), cursor="hand2", padx=15).pack(side="left", padx=20)
    tk.Button(ctrl_frame, text="💾 Export Secure Ledger", command=export_soc_ledger, bg="#4CAF50", fg="white", font=("Segoe UI", 10, "bold"), cursor="hand2", padx=15).pack(side="right", padx=20)

    # Boot the interface
    refresh_soc_data()


# ---------- 4. TOOLS & STUDENT MANAGEMENT ----------
def open_tools_window():
    win = tk.Toplevel(root); win.title("Tools & Management"); win.geometry("1100x750")
    open_windows["tools"] = win; win.configure(bg=THEMES[current_theme]["bg"])
    add_window_toolbar(win, "tools", stop_camera_instance=None)
    tk.Label(win, text="System Tools & Database Management", font=("Segoe UI", 16, "bold"), fg=THEMES[current_theme]["fg"], bg=THEMES[current_theme]["bg"]).pack(pady=10)

    frm_people = tk.LabelFrame(win, text="People Registry (Faculty, Non-Faculty, Guests)", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], padx=10, pady=8)
    frm_people.pack(fill="x", padx=12, pady=5)

    people_form = tk.Frame(frm_people, bg=THEMES[current_theme]["bg"]); people_form.pack(fill="x")
    p_type = tk.StringVar(value="guest"); p_ref = tk.StringVar(); p_name = tk.StringVar(); p_dept = tk.StringVar(); p_mobile = tk.StringVar()
    for label, var, width in (("Type", p_type, 12), ("Ref/ID", p_ref, 14), ("Name", p_name, 22), ("Dept/Course", p_dept, 18), ("Mobile", p_mobile, 14)):
        tk.Label(people_form, text=label, bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(side="left", padx=(8, 2))
        if label == "Type":
            ttk.Combobox(people_form, textvariable=var, values=["faculty", "non_faculty", "guest"], width=width, state="readonly").pack(side="left")
        else:
            tk.Entry(people_form, textvariable=var, width=width, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(side="left")

    people_tree = ttk.Treeview(frm_people, columns=("ID", "Type", "Ref", "Name", "Dept", "Mobile", "Face"), show="headings", height=4)
    for c in people_tree["columns"]: people_tree.heading(c, text=c); people_tree.column(c, anchor="center", width=120)
    people_tree.pack(fill="x", pady=6)

    def load_people():
        people_tree.delete(*people_tree.get_children())
        with get_conn() as conn:
            df = pd.read_sql_query("""SELECT id, person_type, COALESCE(external_ref, reg_no, '') as ref,
                                      name, COALESCE(department, course, '') as dept, COALESCE(mobile, '') as mobile,
                                      CASE WHEN embedding IS NULL THEN 'No' ELSE 'Yes' END as face
                                      FROM people ORDER BY created_at DESC, id DESC LIMIT 40""", conn)
        for _, r in df.iterrows(): people_tree.insert("", tk.END, values=(r["id"], r["person_type"], r["ref"], r["name"], r["dept"], r["mobile"], r["face"]))

    def add_person():
        name = p_name.get().strip()
        if not name: return messagebox.showerror("Error", "Name is required.")
        person_type = p_type.get().strip()
        ref = p_ref.get().strip() or f"{person_type.upper()}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
        qr_path = os.path.join("qrcodes", f"{safe_name(ref)}.png")
        qrcode.make(ref).save(qr_path)
        with get_conn() as conn:
            try:
                conn.cursor().execute("""INSERT INTO people
                    (person_type, external_ref, reg_no, name, department, mobile, qr_path)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (person_type, ref, ref, name, p_dept.get().strip(), p_mobile.get().strip(), qr_path))
                conn.commit()
            except sqlite3.IntegrityError: return messagebox.showerror("Duplicate", "This type + reference already exists.")
        p_ref.set(""); p_name.set(""); p_dept.set(""); p_mobile.set("")
        load_people(); load_server_memory()
        messagebox.showinfo("Saved", f"{person_type.replace('_', ' ').title()} registered.")

    def update_person_face():
        sel = people_tree.selection()
        if not sel: return messagebox.showerror("Error", "Select a person first.")
        person_id, name = people_tree.item(sel[0])["values"][0], people_tree.item(sel[0])["values"][3]
        face_win = tk.Toplevel(win); face_win.title(f"Face Enrollment: {name}"); face_win.geometry("760x620"); face_win.configure(bg="#222")
        cam_manager.start_camera("person_registry_update", CAMERA_SOURCE)
        face_win.protocol("WM_DELETE_WINDOW", lambda: [cam_manager.stop_camera("person_registry_update"), face_win.destroy()])

        tk.Label(face_win, text=f"Capture Face for {name}", font=("Segoe UI", 14), bg="#222", fg="white").pack(pady=10)
        vf = tk.Frame(face_win, bg="black"); vf.pack(fill="both", expand=True, padx=10, pady=10)
        vf.pack_propagate(False)
        lbl_feed = tk.Label(vf, bg="black"); lbl_feed.place(relx=0.5, rely=0.5, anchor="center")

        def face_loop():
            ret, frame = cam_manager.read_camera("person_registry_update")
            if ret and frame is not None:
                imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resize_to_fit(frame, vf.winfo_width(), vf.winfo_height()), cv2.COLOR_BGR2RGB)))
                lbl_feed.imgtk = imgtk; lbl_feed.configure(image=imgtk)
            if face_win.winfo_exists():
                lbl_feed.after(30, face_loop)
        face_loop()

        def capture_face():
            ret, frame = cam_manager.read_camera("person_registry_update")
            if not ret or frame is None: return messagebox.showerror("Error", "Camera offline.")
            faces = FA.get(frame)
            if len(faces) != 1: return messagebox.showerror("Error", "Need exactly 1 face in the frame.")
            photo_path = os.path.join("photos", f"person_{person_id}.jpg")
            cv2.imwrite(photo_path, frame)
            with get_conn() as conn:
                conn.cursor().execute("UPDATE people SET embedding=?, photo_path=? WHERE id=?", (faces[0].embedding.tobytes(), photo_path, person_id))
                conn.commit()
            load_people(); load_server_memory()
            messagebox.showinfo("Saved", "Face profile updated.")
            cam_manager.stop_camera("person_registry_update"); face_win.destroy()

        tk.Button(face_win, text="Capture Face", command=capture_face, bg="#4CAF50", fg="white", font=("Segoe UI", 13, "bold"), cursor="hand2").pack(pady=10)

    tk.Button(people_form, text="Add Person", command=add_person, bg="#4CAF50", fg="white", cursor="hand2").pack(side="left", padx=8)
    tk.Button(people_form, text="Capture Selected Face", command=update_person_face, bg="#f39c12", fg="white", cursor="hand2").pack(side="left")
    load_people()

    frm_mgr = tk.LabelFrame(win, text="Student Management (Update & Delete)", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], padx=10, pady=10)
    frm_mgr.pack(fill="both", expand=True, padx=12, pady=5)
    
    tv_mgr = ttk.Treeview(frm_mgr, columns=("ID", "Reg No", "Name", "Course", "Mobile"), show="headings", height=8)
    for c in tv_mgr["columns"]: tv_mgr.heading(c, text=c); tv_mgr.column(c, anchor="center")
    tv_mgr.pack(fill="both", expand=True)

    def load_students():
        tv_mgr.delete(*tv_mgr.get_children())
        with get_conn() as conn: df = pd.read_sql_query("SELECT id, reg_no, name, course, mobile FROM students", conn)
        for _, r in df.iterrows(): tv_mgr.insert("", tk.END, values=(r["id"], r["reg_no"], r["name"], r["course"], r["mobile"]))
    load_students()

    btn_row = tk.Frame(frm_mgr, bg=THEMES[current_theme]["bg"]); btn_row.pack(fill="x", pady=5)

    def delete_student():
        sel = tv_mgr.selection()
        if not sel: return messagebox.showerror("Error", "Select a student first.")
        sid, reg = tv_mgr.item(sel[0])["values"][0], tv_mgr.item(sel[0])["values"][1]
        if messagebox.askyesno("Confirm", f"Delete {reg} completely?"):
            with get_conn() as conn:
                conn.cursor().execute("DELETE FROM students WHERE id=?", (sid,))
                conn.cursor().execute("DELETE FROM attendance WHERE student_id=?", (sid,))
                conn.cursor().execute("DELETE FROM people WHERE person_type='student' AND external_ref=?", (str(sid),))
                conn.commit()
            load_server_memory(); load_students()
            messagebox.showinfo("Deleted", "Student and attendance records deleted.")

    
    def open_student_detail_editor():
        sel = tv_mgr.selection()
        if not sel: return messagebox.showerror("Error", "Select a student to update.")
        sid, reg, name, course, mobile = tv_mgr.item(sel[0])["values"]
        edit_win = tk.Toplevel(win); edit_win.title(f"Edit Student: {reg}"); edit_win.geometry("520x390")
        edit_win.configure(bg=THEMES[current_theme]["bg"])
        vars_map = {"Reg No": tk.StringVar(value=reg), "Name": tk.StringVar(value=name), "Course": tk.StringVar(value=course), "Mobile": tk.StringVar(value=mobile)}
        tk.Label(edit_win, text="Update Student Details", font=("Segoe UI", 16, "bold"),
                 bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(pady=12)
        form = tk.Frame(edit_win, bg=THEMES[current_theme]["bg"]); form.pack(fill="both", expand=True, padx=24)
        for label, var in vars_map.items():
            tk.Label(form, text=label, bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w", pady=(8, 0))
            tk.Entry(form, textvariable=var, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(fill="x", pady=3)

        def save_details():
            new_reg, new_name, new_course, new_mobile = (vars_map[k].get().strip() for k in ["Reg No", "Name", "Course", "Mobile"])
            if not all([new_reg, new_name, new_course]): return messagebox.showerror("Error", "Reg No, Name and Course are required.")
            with get_conn() as conn:
                try:
                    conn.cursor().execute("UPDATE students SET reg_no=?, name=?, course=?, mobile=? WHERE id=?", (new_reg, new_name, new_course, new_mobile, sid))
                    conn.cursor().execute("UPDATE people SET reg_no=?, name=?, course=?, mobile=? WHERE person_type='student' AND external_ref=?", (new_reg, new_name, new_course, new_mobile, str(sid)))
                    conn.commit()
                except sqlite3.IntegrityError: return messagebox.showerror("Duplicate", "This registration number already exists.")
            load_students(); load_server_memory()
            messagebox.showinfo("Saved", "Student details updated.")
            edit_win.destroy()

        tk.Button(edit_win, text="Save Details", command=save_details, bg="#4CAF50", fg="white", font=("Segoe UI", 12, "bold"), cursor="hand2").pack(pady=12)
    

    tk.Button(btn_row, text="Open Advanced Identity Hub", command=lambda: [win.destroy(), open_identity_management_hub()], bg="#f39c12", fg="white", cursor="hand2").pack(side="left", padx=5)
    tk.Button(btn_row, text="Update Selected Details", command=open_student_detail_editor, bg="#14818f", fg="white", cursor="hand2").pack(side="left", padx=5)
    #tk.Button(btn_row, text="Update Selected Face (Live Preview)", command=open_live_updater, bg="#f39c12", fg="white", cursor="hand2").pack(side="left", padx=5)
    tk.Button(btn_row, text="Delete Selected Student", command=delete_student, bg="#d9534f", fg="white", cursor="hand2").pack(side="left", padx=5)

    frm_db = tk.LabelFrame(win, text="Data Management", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], padx=10, pady=10)
    frm_db.pack(fill="x", padx=12, pady=5)
    qr_reg = tk.StringVar()
    tk.Entry(frm_db, textvariable=qr_reg, width=20, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(side="left", padx=10)
    tk.Button(frm_db, text="Generate QR", command=lambda: messagebox.showinfo("QR", "Generated.") if qrcode.make(qr_reg.get().strip()).save(os.path.join("qrcodes", f"{qr_reg.get().strip()}.png")) is None else None, bg="#4CAF50", fg="white", cursor="hand2").pack(side="left")
    tk.Button(frm_db, text="Backup Main DB", command=lambda: backup_database_file(DB_FILE, "Backup Main Database"), bg="#2196F3", fg="white", cursor="hand2").pack(side="left", padx=15)
    tk.Button(frm_db, text="Backup Surveillance DB", command=lambda: backup_database_file(SURVEILLANCE_DB_FILE, "Backup Surveillance Database"), bg="#555", fg="white", cursor="hand2").pack(side="left", padx=5)
    tk.Button(frm_db, text="📷 Camera Management", command=open_camera_management, bg="#555", fg="white", cursor="hand2").pack(side="left", padx=5)

    frm_unknown = tk.LabelFrame(win, text="Unknown Face Alerts & Unofficial Camera Tracks", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], padx=10, pady=8)
    frm_unknown.pack(fill="both", expand=True, padx=12, pady=5)
    tree_u = ttk.Treeview(frm_unknown, columns=("Date", "Time", "Camera", "Mode", "Severity", "Action"), show="headings", height=5)
    for c in tree_u["columns"]: tree_u.heading(c, text=c); tree_u.column(c, anchor="center")
    tree_u.pack(fill="both", expand=True)

    def load_unknown_events():
        tree_u.delete(*tree_u.get_children())
        with get_surveillance_conn() as conn:
            df = pd.read_sql_query("SELECT date, time, camera_name, mode, severity, action_taken FROM surveillance_unknowns ORDER BY id DESC LIMIT 50", conn)
        for _, r in df.iterrows(): tree_u.insert("", tk.END, values=(r["date"], r["time"], r["camera_name"], r["mode"], r["severity"], r["action_taken"]))

    tk.Button(frm_unknown, text="Refresh Events", command=load_unknown_events, bg="#555", fg="white", cursor="hand2").pack(anchor="e", pady=4)
    load_unknown_events()

    frm_sum = tk.LabelFrame(win, text="Daily Summary", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], padx=10, pady=10)
    frm_sum.pack(fill="both", expand=True, padx=12, pady=5)
    tree_s = ttk.Treeview(frm_sum, columns=("Date", "Count"), show="headings", height=5)
    for c in ("Date", "Count"): tree_s.heading(c, text=c); tree_s.column(c, anchor="center")
    tree_s.pack(fill="both", expand=True)
    with get_conn() as conn: df = pd.read_sql_query("SELECT date, COUNT(*) as c FROM attendance GROUP BY date ORDER BY date DESC", conn)
    for _, r in df.iterrows(): tree_s.insert("", tk.END, values=(r["date"], r["c"]))


def open_identity_management_hub():
    win_name = "identity_hub"
    if win_name in open_windows and open_windows[win_name].winfo_exists():
        open_windows[win_name].deiconify()
        open_windows[win_name].lift()
        return

    win = tk.Toplevel(root)
    win.title("Unified Biometric Progressive Stitching Hub — Enterprise Multi-Registry")
    win.geometry("1350x860")
    open_windows[win_name] = win
    win.configure(bg=THEMES[current_theme]["bg"])
    
    cam_slot_name = "identity_updater_stream"
    add_window_toolbar(win, win_name, stop_camera_instance=cam_slot_name)

    # State Machine Variables
    selected_person = {"id": None, "reg": None, "name": None, "type": None}
    target_historical_embedding = [None] 
    
    stitching_state = tk.IntVar(value=0)
    stitching_progress = tk.DoubleVar(value=0.0)
    hud_instruction = tk.StringVar(value="Select an identity profile from registry to engage.")
    live_verification_score = tk.StringVar(value="Verification Score: --")
    
    # Volatile buffers
    compiled_center_vector = [None]
    compiled_left_vector = [None]
    compiled_right_vector = [None]
    last_frame_biometrics = [None]
    is_processing_finalize = [False] 

    # --- MODERN SPLIT DASHBOARD LAYOUT ---
    main_container = tk.Frame(win, bg=THEMES[current_theme]["bg"])
    main_container.pack(fill="both", expand=True, padx=20, pady=15)
    
    left_card = tk.LabelFrame(main_container, text=" Global Identity Registry ", 
                             bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"],
                             font=("Segoe UI", 11, "bold"), padx=15, pady=10, bd=1, relief="solid", width=540)
    left_card.pack_propagate(False)
    left_card.pack(side="left", fill="both", padx=(0, 10))

    right_card = tk.LabelFrame(main_container, text=" Guided Biometric Stitching Viewport ", 
                              bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"],
                              font=("Segoe UI", 11, "bold"), padx=15, pady=10, bd=1, relief="solid")
    right_card.pack(side="right", fill="both", expand=True, padx=(10, 0))

    # --- LEFT COLUMN: SEARCH & TREEVIEW POPULATOR ---
    search_frame = tk.Frame(left_card, bg=THEMES[current_theme]["card_bg"])
    search_frame.pack(fill="x", pady=(0, 10))
    tk.Label(search_frame, text="Search Identity:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(side="left")
    search_var = tk.StringVar()
    search_entry = tk.Entry(search_frame, textvariable=search_var, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"], font=("Segoe UI", 11), bd=1, relief="solid")
    search_entry.pack(side="left", fill="x", expand=True, padx=(5, 0), ipady=3)

    cols = ("ID", "Reg No", "Name", "Type")
    tv_registry = ttk.Treeview(left_card, columns=cols, show="headings", height=18)
    tv_registry.heading("ID", text="DB ID"); tv_registry.column("ID", width=50, anchor="center")
    tv_registry.heading("Reg No", text="Registration"); tv_registry.column("Reg No", width=120, anchor="center")
    tv_registry.heading("Name", text="Full Name"); tv_registry.column("Name", width=200, anchor="w")
    tv_registry.heading("Type", text="Type"); tv_registry.column("Type", width=100, anchor="center")
    tv_registry.pack(fill="both", expand=True, pady=5)

    def load_registry(event=None):
        tv_registry.delete(*tv_registry.get_children())
        query = search_var.get().strip().lower()
        with get_conn() as conn:
            df = pd.read_sql_query("""
                SELECT id, reg_no, name, 'student' as type FROM students
                UNION ALL
                SELECT id, COALESCE(external_ref, reg_no) as reg_no, name, person_type as type FROM people WHERE person_type != 'student'
                ORDER BY name
            """, conn)
        for _, r in df.iterrows():
            if query and query not in str(r["reg_no"]).lower() and query not in str(r["name"]).lower():
                continue
            tv_registry.insert("", tk.END, values=(r["id"], r["reg_no"], r["name"], r["type"]))
            
    search_var.trace_add("write", lambda *args: load_registry())
    load_registry()

    target_frame = tk.Frame(left_card, bg="#1a1a1a", bd=1, relief="solid", pady=10)
    target_frame.pack(fill="x", pady=10)
    lbl_target = tk.Label(target_frame, text="TARGET NODE: NONE SELECTED", font=("Segoe UI", 11, "bold"), bg="#1a1a1a", fg="#ff4444")
    lbl_target.pack()

    def reset_stitching_matrix():
        stitching_state.set(0)
        stitching_progress.set(0.0)
        compiled_center_vector[0] = None
        compiled_left_vector[0] = None
        compiled_right_vector[0] = None
        last_frame_biometrics[0] = None
        is_processing_finalize[0] = False

    def on_person_select(event):
        sel = tv_registry.selection()
        if not sel: return
        vals = tv_registry.item(sel[0])["values"]
        
        # Unified alignment format across all record categories
        selected_person.update({
            "id": int(vals[0]), 
            "reg": str(vals[1]).strip(), 
            "name": str(vals[2]).strip(), 
            "type": str(vals[3]).strip()
        })
        
        lbl_target.config(text=f"TARGET: [{selected_person['type'].upper()}] {selected_person['reg']} | {selected_person['name']}", fg="#00ff00")
        reset_stitching_matrix()
        hud_instruction.set("Target locked. Align face to center bracket and click 'Engage Matrix Scanner'.")
        
        target_historical_embedding[0] = None
        with get_conn() as conn:
            if selected_person["type"] == "student":
                row = conn.cursor().execute("SELECT embedding FROM students WHERE id=?", (selected_person["id"],)).fetchone()
            else:
                row = conn.cursor().execute("SELECT embedding FROM people WHERE id=?", (selected_person["id"],)).fetchone()
        if row and row[0] is not None:
            target_historical_embedding[0] = np.frombuffer(row[0], dtype=np.float32)
        
    tv_registry.bind("<<TreeviewSelect>>", on_person_select)

    # --- RIGHT COLUMN: PROGRESSIVE STITCHING VIEWPORT ---
    routing_frame = tk.Frame(right_card, bg=THEMES[current_theme]["card_bg"])
    routing_frame.pack(fill="x", pady=(0, 5))
    tk.Label(routing_frame, text="Viewport Link Source:", bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(side="left")
    
    selected_source = tk.StringVar()
    camera_combo = create_camera_device_selector(routing_frame, selected_source)
    camera_combo.pack(side="left", fill="x", expand=True, padx=(10, 0))

    anim_progress_frame = tk.Frame(right_card, bg=THEMES[current_theme]["card_bg"])
    anim_progress_frame.pack(fill="x", pady=5)
    tk.Label(anim_progress_frame, text="Biometric Stitching Accumulation Matrix:", font=("Segoe UI", 9), bg=THEMES[current_theme]["card_bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w")
    bar_anim = ttk.Progressbar(anim_progress_frame, variable=stitching_progress, maximum=100)
    bar_anim.pack(fill="x", pady=2, ipady=3)

    feed_viewport = tk.Frame(right_card, bg="black", bd=1, relief="solid")
    feed_viewport.pack(fill="both", expand=True, pady=8)
    feed_viewport.pack_propagate(False)
    
    lbl_video = tk.Label(feed_viewport, bg="black", text="Initializing Secure Streaming Array...", fg="#888", font=("Segoe UI", 10))
    lbl_video.place(relx=0.5, rely=0.5, anchor="center")

    lbl_hud_msg = tk.Label(right_card, textvariable=hud_instruction, font=("Segoe UI", 11, "bold"), bg="#1a1a1a", fg="#00ffff", pady=8, bd=1, relief="solid")
    lbl_hud_msg.pack(fill="x", pady=2)
    
    lbl_verify_score = tk.Label(right_card, textvariable=live_verification_score, font=("Consolas", 10, "bold"), bg="#1a1a1a", fg="yellow", pady=4)
    lbl_verify_score.pack(fill="x")

    def switch_routing_stream(*args):
        source = selected_source.get()
        if not source: return
        actual_src = int(source) if source.isdigit() else source
        cam_manager.start_camera(cam_slot_name, actual_src)

    selected_source.trace_add("write", switch_routing_stream)
    win.after(400, switch_routing_stream)

    def draw_alignment_reticle(frame, state):
        h, w = frame.shape[:2]
        cx, cy = w // 2, h // 2
        bw, bh = 240, 320
        x1, y1 = cx - bw // 2, cy - bh // 2
        x2, y2 = cx + bw // 2, cy + bh // 2
        
        colors = {0: (0, 255, 255), 1: (255, 0, 255), 2: (255, 255, 0), 3: (0, 255, 0)}
        color = colors.get(state, (255, 255, 255))
        
        thick = 3; length = 35
        cv2.line(frame, (x1, y1), (x1 + length, y1), color, thick)
        cv2.line(frame, (x1, y1), (x1, y1 + length), color, thick)
        cv2.line(frame, (x2, y1), (x2 - length, y1), color, thick)
        cv2.line(frame, (x2, y1), (x2, y1 + length), color, thick)
        cv2.line(frame, (x1, y2), (x1 + length, y2), color, thick)
        cv2.line(frame, (x1, y2), (x1, y2 - length), color, thick)
        cv2.line(frame, (x2, y2), (x2 - length, y2), color, thick)
        cv2.line(frame, (x2, y2), (x2, y2 - length), color, thick)
        
        if state == 1:
            cv2.arrowedLine(frame, (cx + 60, cy), (cx - 60, cy), (255, 0, 255), 4, tipLength=0.3)
        elif state == 2:
            cv2.arrowedLine(frame, (cx - 60, cy), (cx + 60, cy), (255, 255, 0), 4, tipLength=0.3)
            
        return frame

    def run_viewport_refresh():
        if win.winfo_exists() and not is_processing_finalize[0]:
            ret, frame = cam_manager.read_camera(cam_slot_name)
            current_state = stitching_state.get()
            
            if ret and frame is not None:
                # --- CRITICAL FIX: Safe fallback if no person is selected yet ---
                display_type = selected_person['type'].upper() if selected_person['type'] else "AWAITING TARGET"
                display_frame = draw_hud(frame.copy(), f"STITCHING LAYER | {display_type}")
                display_frame = draw_alignment_reticle(display_frame, current_state)
                
                faces = FA.get(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
                
                if len(faces) == 1:
                    fx1, fy1, fx2, fy2 = map(int, faces[0].bbox * 2)
                    current_emb = faces[0].embedding
                    
                    if target_historical_embedding[0] is not None:
                        sim_old = float(face_similarity([target_historical_embedding[0]], current_emb)[0])
                        live_verification_score.set(f"Target Authentication Match: {sim_old*100:.1f}%")
                        if sim_old < SIMILARITY_THRESHOLD and current_state > 0:
                            stitching_state.set(0)
                            stitching_progress.set(0.0)
                            hud_instruction.set("❌ EXCLUSION BLOCKED: SUBJECT SWAP DETECTED! Resetting...")
                            lbl_hud_msg.config(fg="#ff4444")
                            last_frame_biometrics[0] = None
                            return
                    
                    is_consecutive_match = True
                    if last_frame_biometrics[0] is not None:
                        sim_bridge = float(face_similarity([last_frame_biometrics[0]], current_emb)[0])
                        if sim_bridge < 0.75:
                            is_consecutive_match = False
                    
                    last_frame_biometrics[0] = current_emb
                    
                    if current_state == 1 and is_consecutive_match:
                        if compiled_left_vector[0] is None:
                            compiled_left_vector[0] = current_emb
                            stitching_progress.set(40.0)
                            hud_instruction.set("✓ Left profile cached! Now slowly turn head to the RIGHT...")
                            lbl_hud_msg.config(fg="#ffeb3b")
                            stitching_state.set(2)
                            play_success_beep()
                            
                    elif current_state == 2 and is_consecutive_match:
                        if compiled_right_vector[0] is None:
                            compiled_right_vector[0] = current_emb
                            stitching_progress.set(80.0)
                            hud_instruction.set("✓ Right profile cached! Finalizing biometric calculation arrays...")
                            lbl_hud_msg.config(fg="#00ff00")
                            stitching_state.set(3)
                            play_success_beep()
                            
                            is_processing_finalize[0] = True 
                            win.after(10, finalize_and_save_biometric_matrix)
                            
                    cv2.rectangle(display_frame, (fx1, fy1), (fx2, fy2), (0, 255, 255) if current_state < 3 else (0, 255, 0), 2)
                else:
                    if selected_person["id"] and current_state > 0 and current_state < 3:
                        live_verification_score.set("Target alignment lost from tracking view area...")
                
                w, h = feed_viewport.winfo_width(), feed_viewport.winfo_height()
                resized = resize_to_fit(display_frame, w, h)
                imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
                lbl_video.imgtk = imgtk
                lbl_video.configure(image=imgtk, text="")
                
            win.after(33, run_viewport_refresh)
            
    run_viewport_refresh()

    def initiate_progressive_stitching_pipeline():
        if not selected_person["id"]:
            messagebox.showerror("Selection Required", "Select an active user profile from the database grid to map biometrics.")
            return
            
        ret, frame = cam_manager.read_camera(cam_slot_name)
        if not ret or frame is None:
            messagebox.showerror("Stream Error", "Hardware streaming nodes are offline.")
            return
            
        faces = FA.get(frame)
        if len(faces) != 1:
            messagebox.showerror("Alignment Error", "Precisely ONE face must be positioned inside the targeting matrix.")
            return
            
        play_success_beep()
        compiled_center_vector[0] = faces[0].embedding
        last_frame_biometrics[0] = faces[0].embedding
        
        stitching_progress.set(15.0)
        stitching_state.set(1)
        hud_instruction.set("▶ Anchor Locked! Now look slightly to your LEFT profile angle...")
        lbl_hud_msg.config(fg="#ff00ff")

    # --- ENHANCED SYSTEM TRANSACTIONS FOR ALL REGISTRY TYPES ---
    def finalize_and_save_biometric_matrix():
        # Disconnect streaming array synchronously before committing SQL updates
        cam_manager.stop_camera(cam_slot_name)
        lbl_video.configure(image="", text="Processing and compiling multi-angle centralized arrays...")
        win.update_idletasks()

        if compiled_center_vector[0] is None or compiled_left_vector[0] is None or compiled_right_vector[0] is None:
            reset_stitching_matrix()
            switch_routing_stream()
            hud_instruction.set("❌ Progressive stitching timed out or tracking lost. Restart scan.")
            lbl_hud_msg.config(fg="#ff4444")
            return
            
        stitching_progress.set(100.0)
        
        # Max-Pooling consolidation
        super_embedding = np.maximum(
            compiled_center_vector[0], 
            np.maximum(compiled_left_vector[0], compiled_right_vector[0])
        )
        
        # Perfect unit normalization to secure compatibility check rates
        norm = np.linalg.norm(super_embedding)
        if norm > 0:
            super_embedding = super_embedding / norm
        
        # Crucial cast step: Force precision allocation to float32 architecture explicitly
        super_embedding = super_embedding.astype(np.float32)
        embedding_bytes = super_embedding.tobytes()

        cropped_img = np.zeros((300, 300, 3), dtype=np.uint8)

        p_id = selected_person["id"]
        reg = selected_person["reg"]
        p_type = selected_person["type"]
        name_str = selected_person["name"]
        
        # Atomic Transaction for both tables
        with get_conn() as conn:
            cur = conn.cursor()
            if p_type == "student":
                photo_path = os.path.join("photos", f"{reg}.jpg")
                cv2.imwrite(photo_path, cropped_img)
                cur.execute("UPDATE students SET embedding=?, photo_path=? WHERE id=?", (embedding_bytes, photo_path, p_id))
                cur.execute("UPDATE people SET embedding=?, photo_path=? WHERE person_type='student' AND external_ref=?", (embedding_bytes, photo_path, str(p_id)))
            else:
                # FIXED: Force absolute updates across the generic categories in people table
                photo_path = os.path.join("photos", f"person_{p_id}.jpg")
                cv2.imwrite(photo_path, cropped_img)
                cur.execute("UPDATE people SET embedding=?, photo_path=? WHERE id=? AND person_type=?", (embedding_bytes, photo_path, p_id, p_type))
            conn.commit()

        # Flash variables completely to eliminate verification latency drops
        load_server_memory() 
        play_success_beep()
        
        messagebox.showinfo("Centralized Matrix Stitched", f"Success! Fresh full-angle centralized vector mapped onto [{p_type.upper()}] profile: '{name_str}'.")
        
        # Reset control channels cleanly
        reset_stitching_matrix()
        switch_routing_stream()
        hud_instruction.set("Centralized multi-angle matrix committed. System ready.")
        lbl_hud_msg.config(fg="#00ff00")

    # Command UI Action Launchers
    tk.Button(right_card, text="⚡ ENGAGE PROGRESSIVE SCANNER TERMINAL", command=initiate_progressive_stitching_pipeline, 
              bg="#14818f", fg="white", font=("Segoe UI", 11, "bold"), bd=0, cursor="hand2", pady=10).pack(fill="x", side="bottom", pady=2)
              
    tk.Button(right_card, text="🔄 Reset Active Stitching Grid Buffer", command=reset_stitching_matrix, 
              bg="#444", fg="white", font=("Segoe UI", 10), bd=0, cursor="hand2", pady=6).pack(fill="x", side="bottom", pady=4)


def open_ghost_tracking_matrix():
    win_name = "ghost_tracker"
    
    # Singleton Window Enforcement
    if win_name in open_windows and open_windows[win_name].winfo_exists():
        open_windows[win_name].deiconify()
        open_windows[win_name].lift()
        return

    win = tk.Toplevel(root)
    win.title("Ghost Tracking Matrix (Autonomous Clustering)")
    win.geometry("1300x800")
    open_windows[win_name] = win
    win.configure(bg="#050505")
    add_window_toolbar(win, win_name, stop_camera_instance=None)

    # --- Header UI ---
    header = tk.Frame(win, bg="#050505")
    header.pack(fill="x", padx=20, pady=15)
    
    tk.Label(header, text="👻 Phantom Entity Tracking (DBSCAN)", font=("Segoe UI", 18, "bold"), bg="#050505", fg="#00e5ff").pack(side="left")
    
    status_var = tk.StringVar(value="Status: Awaiting Matrix Compilation...")
    tk.Label(header, textvariable=status_var, font=("Consolas", 12), bg="#050505", fg="#ffeb3b").pack(side="right", padx=10)

    # --- Main Workspace ---
    main_frame = tk.Frame(win, bg="#050505")
    main_frame.pack(fill="both", expand=True, padx=20, pady=10)

    # Left: The Cluster Roster
    roster_frame = tk.LabelFrame(main_frame, text=" Unidentified Recurring Entities ", bg="#121212", fg="#ff4444", font=("Segoe UI", 11, "bold"), padx=10, pady=10, width=400)
    roster_frame.pack(side="left", fill="y", expand=False)
    roster_frame.pack_propagate(False)

    tv_clusters = ttk.Treeview(roster_frame, columns=("Entity ID", "Sightings", "Risk"), show="headings", height=20, style="SOCTreeview.Treeview")
    tv_clusters.heading("Entity ID", text="Entity ID")
    tv_clusters.heading("Sightings", text="Sightings")
    tv_clusters.heading("Risk", text="Risk Profile")
    tv_clusters.column("Entity ID", width=120, anchor="center")
    tv_clusters.column("Sightings", width=100, anchor="center")
    tv_clusters.column("Risk", width=120, anchor="center")
    tv_clusters.pack(fill="both", expand=True, pady=5)

    # Right: Entity Investigation Panel
    investigate_frame = tk.LabelFrame(main_frame, text=" Entity Investigation Node ", bg="#121212", fg="#00e5ff", font=("Segoe UI", 11, "bold"), padx=15, pady=15)
    investigate_frame.pack(side="right", fill="both", expand=True, padx=(15, 0))
    
    lbl_snapshot = tk.Label(investigate_frame, bg="black", text="SELECT ENTITY", fg="#555", font=("Segoe UI", 14))
    lbl_snapshot.pack(pady=10)
    
    details_var = tk.StringVar(value="Select an entity from the roster to view their spatial tracks.")
    tk.Label(investigate_frame, textvariable=details_var, bg="#121212", fg="white", font=("Segoe UI", 12), justify="left", wraplength=700).pack(pady=15, anchor="w")

    # --- AI CLUSTERING ENGINE ---
    def compile_ghost_matrix():
        status_var.set("Status: Querying Surveillance Sub-Systems...")
        win.update()
        
        with get_surveillance_conn() as conn:
            # Fetch all unknown records that have a saved embedding BLOB
            df = pd.read_sql_query("SELECT id, date, time, camera_name, snapshot_path, embedding FROM surveillance_unknowns WHERE embedding IS NOT NULL", conn)
            
        if df.empty or len(df) < 2:
            status_var.set("Status: Insufficient vector data for clustering.")
            return

        status_var.set(f"Status: Clustering {len(df)} unknown vectors using DBSCAN...")
        win.update()

        # Convert BLOBs back to numpy arrays
        embeddings = []
        valid_indices = []
        for idx, row in df.iterrows():
            try:
                emb = np.frombuffer(row['embedding'], dtype=np.float32)
                embeddings.append(emb)
                valid_indices.append(idx)
            except Exception:
                continue

        X = np.array(embeddings)
        
        # Run DBSCAN (Density-Based Spatial Clustering)
        # eps is the maximum distance between two samples for one to be considered as in the neighborhood of the other.
        # min_samples is the number of samples in a neighborhood for a point to be considered as a core point.
        db = DBSCAN(eps=0.45, min_samples=2, metric='cosine').fit(X)
        labels = db.labels_
        
        df_valid = df.iloc[valid_indices].copy()
        df_valid['Cluster'] = labels
        
        # Filter out noise (DBSCAN labels noise as -1)
        clusters = df_valid[df_valid['Cluster'] != -1]
        
        tv_clusters.delete(*tv_clusters.get_children())
        
        if clusters.empty:
            status_var.set("Status: No recurring entities detected. All intercepts are isolated noise.")
            return

        # Group by the cluster ID to find recurring ghosts
        grouped = clusters.groupby('Cluster')
        global ghost_memory
        ghost_memory = {}
        
        for cluster_id, group in grouped:
            sightings = len(group)
            risk = "CRITICAL" if sightings >= 5 else "ELEVATED"
            entity_name = f"GHOST_{cluster_id:04d}"
            
            # Save group data to memory for the UI to read
            ghost_memory[entity_name] = group
            
            tv_clusters.insert("", tk.END, values=(entity_name, sightings, risk))
            
        status_var.set(f"Status: Matrix Compiled. {len(grouped)} Unique Entities Identified.")

    # --- UI INTERACTION ---
    def on_entity_select(event):
        sel = tv_clusters.selection()
        if not sel: return
        entity_id = tv_clusters.item(sel[0])["values"][0]
        
        group = ghost_memory.get(entity_id)
        if group is None: return
        
        # Grab the most recent snapshot
        latest_record = group.sort_values(by=['date', 'time'], ascending=False).iloc[0]
        snap_path = latest_record['snapshot_path']
        
        if snap_path and os.path.exists(snap_path):
            try:
                img = Image.open(snap_path).resize((300, 300), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(img)
                lbl_snapshot.imgtk = imgtk
                lbl_snapshot.configure(image=imgtk, text="")
            except Exception:
                lbl_snapshot.configure(image="", text="IMAGE CORRUPTED")
        else:
            lbl_snapshot.configure(image="", text="IMAGE NOT FOUND ON DISK")

        # Build Intel Report
        first_seen = group.sort_values(by=['date', 'time']).iloc[0]
        cameras = ", ".join(group['camera_name'].unique())
        
        report = (
            f"IDENTIFIER: {entity_id}\n"
            f"TOTAL INTERCEPTS: {len(group)}\n\n"
            f"FIRST SPOTTED: {first_seen['date']} at {first_seen['time']}\n"
            f"LAST SPOTTED: {latest_record['date']} at {latest_record['time']}\n\n"
            f"KNOWN ROUTES (Cameras): {cameras}"
        )
        details_var.set(report)

    tv_clusters.bind("<<TreeviewSelect>>", on_entity_select)

    # --- Action Buttons ---
    action_frame = tk.Frame(investigate_frame, bg="#121212")
    action_frame.pack(fill="x", side="bottom", pady=10)
    
    def register_entity(person_type):
        sel = tv_clusters.selection()
        if not sel: return messagebox.showwarning("Select Entity", "Select a Phantom Entity first.")
        entity_id = tv_clusters.item(sel[0])["values"][0]
        # In a full implementation, this would pass the embedding and snapshot to the enrollment module
        messagebox.showinfo("Matrix Action", f"Routing {entity_id} to {person_type.upper()} Registration Pipeline...")

    tk.Button(action_frame, text="⚠️ Register as Blacklist Threat", command=lambda: register_entity('blacklist'), bg="#d9534f", fg="white", font=("Segoe UI", 11, "bold"), cursor="hand2", padx=10, pady=5).pack(side="left", padx=10)
    tk.Button(action_frame, text="✅ Register as Known Guest", command=lambda: register_entity('guest'), bg="#4CAF50", fg="white", font=("Segoe UI", 11, "bold"), cursor="hand2", padx=10, pady=5).pack(side="left", padx=10)
    tk.Button(action_frame, text="🗑️ Purge Entity Data", bg="#555", fg="white", font=("Segoe UI", 11), cursor="hand2", padx=10, pady=5).pack(side="right", padx=10)

    # Boot Sequence
    tk.Button(roster_frame, text="⚡ RECOMPILE SPATIAL MATRIX", command=compile_ghost_matrix, bg="#884EA0", fg="white", font=("Segoe UI", 11, "bold"), cursor="hand2", pady=8).pack(fill="x", side="bottom", pady=5)
    
    win.after(500, compile_ghost_matrix)

def open_master_soc_matrix():
    win_name = "master_soc_matrix"
    
    if win_name in open_windows and open_windows[win_name].winfo_exists():
        open_windows[win_name].deiconify()
        open_windows[win_name].lift()
        return

    win = tk.Toplevel(root)
    win.title("Global SOC - Multi-Node Threat Matrix")
    win.geometry("1450x880")
    open_windows[win_name] = win
    win.configure(bg="#050505")
    add_window_toolbar(win, win_name, stop_camera_instance=None)

    # --- Header UI ---
    header = tk.Frame(win, bg="#050505")
    header.pack(fill="x", padx=20, pady=10)
    tk.Label(header, text="🌐 Global Multi-Node Security Command", font=("Segoe UI", 18, "bold"), bg="#050505", fg="#00e5ff").pack(side="left")
    
    kpi_frame = tk.Frame(header, bg="#050505")
    kpi_frame.pack(side="right")
    lbl_active_nodes = tk.Label(kpi_frame, text="Active Nodes: 0", font=("Consolas", 12, "bold"), fg="#00ff00", bg="#121212", padx=10, pady=5, bd=1, relief="solid")
    lbl_active_nodes.pack(side="left", padx=5)

    # --- Main Workspace Layout ---
    main_workspace = tk.Frame(win, bg="#050505")
    main_workspace.pack(fill="both", expand=True, padx=15, pady=5)

    # LEFT: The Focus Camera Viewport (Large)
    focus_frame = tk.LabelFrame(main_workspace, text=" PRIMARY INTERCEPT VIEWPORT ", bg="#0a0a0a", fg="#ffeb3b", font=("Consolas", 12, "bold"), padx=10, pady=10)
    focus_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))
    focus_frame.pack_propagate(False)

    # BUG FIX 1: Pack the bottom threat feed BEFORE the expanding video label to prevent layout squishing
    threat_frame = tk.Frame(focus_frame, bg="#121212", height=180)
    threat_frame.pack(fill="x", side="bottom", pady=(10, 0))
    threat_frame.pack_propagate(False)
    
    tv_threats = ttk.Treeview(threat_frame, columns=("Time", "Node", "Identity", "Threat Level"), show="headings", style="SOCTreeview.Treeview")
    for c in tv_threats["columns"]: 
        tv_threats.heading(c, text=c)
        tv_threats.column(c, anchor="center")
    tv_threats.pack(fill="both", expand=True)

    lbl_focus_video = tk.Label(focus_frame, bg="black", text="AWAITING NODE SELECTION...", fg="#555", font=("Segoe UI", 14))
    lbl_focus_video.pack(fill="both", expand=True)

    # RIGHT: The Multi-Camera Grid (Scrollable)
    grid_frame = tk.LabelFrame(main_workspace, text=" ACTIVE NODE MATRIX ", bg="#0a0a0a", fg="#00e5ff", font=("Consolas", 12, "bold"), width=420)
    grid_frame.pack(side="right", fill="y")
    grid_frame.pack_propagate(False)

    canvas_grid = tk.Canvas(grid_frame, bg="#0a0a0a", highlightthickness=0)
    scrollbar = ttk.Scrollbar(grid_frame, orient="vertical", command=canvas_grid.yview)
    scrollable_grid = tk.Frame(canvas_grid, bg="#0a0a0a")
    
    scrollable_grid.bind("<Configure>", lambda e: canvas_grid.configure(scrollregion=canvas_grid.bbox("all")))
    canvas_grid.create_window((0, 0), window=scrollable_grid, anchor="nw")
    canvas_grid.configure(yscrollcommand=scrollbar.set)
    canvas_grid.pack(side="left", fill="both", expand=True, padx=5, pady=5)
    scrollbar.pack(side="right", fill="y")

    # --- AUTONOMOUS MULTI-CAMERA DAEMON ---
    cctv_cameras = {}
    grid_widgets = {}
    known_threat_memory = {} # Prevent UI from spamming the same threat every single frame

    def fetch_assigned_cctv_nodes():
        with get_conn() as conn:
            return conn.cursor().execute("SELECT camera_name, source FROM cameras WHERE status='active' AND can_surveillance=1").fetchall()

    def boot_surveillance_network():
        nodes = fetch_assigned_cctv_nodes()
        if not nodes:
            messagebox.showwarning("No Nodes", "No cameras in the database are assigned for CCTV Surveillance.")
            return

        lbl_active_nodes.config(text=f"Active Nodes: {len(nodes)}")
        cctv_daemon_running[0] = True

        for idx, (cam_name, source) in enumerate(nodes):
            actual_src = int(source) if str(source).isdigit() else source
            cam_manager.start_camera(f"soc_{cam_name}", actual_src)
            cctv_cameras[cam_name] = f"soc_{cam_name}"
            
            cam_box = tk.Frame(scrollable_grid, bg="#121212", bd=1, relief="solid", pady=5, padx=5)
            cam_box.grid(row=idx, column=0, pady=5, padx=5, sticky="ew")
            
            lbl_title = tk.Label(cam_box, text=cam_name, bg="#121212", fg="white", font=("Segoe UI", 10, "bold"))
            lbl_title.pack(anchor="w")
            
            lbl_feed = tk.Label(cam_box, bg="black", width=45, height=12) 
            lbl_feed.pack(pady=5)
            
            def make_focus(event, name=cam_name):
                focused_cctv_camera[0] = name
                focus_frame.config(text=f" PRIMARY INTERCEPT VIEWPORT: [ {name} ] ")
                
            lbl_feed.bind("<Button-1>", make_focus)
            lbl_title.bind("<Button-1>", make_focus)
            
            grid_widgets[cam_name] = {"box": cam_box, "feed": lbl_feed, "title": lbl_title}
            
            if focused_cctv_camera[0] is None:
                make_focus(None, cam_name) 

        threading.Thread(target=autonomous_ai_inference_daemon, daemon=True).start()
        win.after(33, ui_render_loop)

    def autonomous_ai_inference_daemon():
        """ BACKGROUND HEADLESS AI ENGINE. Runs independently of UI. """
        while cctv_daemon_running[0]:
            for cam_name, cam_slot in cctv_cameras.items():
                ret, frame = cam_manager.read_camera(cam_slot)
                if not ret or frame is None: continue

                dt_s, tm_s = datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H:%M:%S")
                display_frame = frame.copy()
                has_critical_threat = False

                faces = FA.get(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
                
                for face in faces:
                    x1, y1, x2, y2 = map(int, face.bbox * 2)
                    sims = face_similarity(KNOWN_EMBEDDINGS, face.embedding)
                    
                    if sims.size > 0 and sims.max() >= SIMILARITY_THRESHOLD:
                        idx = int(sims.argmax())
                        p_type = KNOWN_PERSON_TYPES[idx]
                        name = KNOWN_LABELS[idx].split("|")[1].strip()
                        
                        # THE BLOCKLIST LOGIC YOU REQUESTED
                        if p_type == "blacklist":
                            has_critical_threat = True
                            color = (0, 0, 255) # RED
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                            cv2.putText(display_frame, f"THREAT: {name}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                            
                            threat_key = f"{cam_name}_{name}"
                            if time.time() - known_threat_memory.get(threat_key, 0) > 10:
                                known_threat_memory[threat_key] = time.time()
                                focused_cctv_camera[0] = cam_name # Auto-Focus UI hook
                                win.after(10, lambda n=name, c=cam_name, t=tm_s: tv_threats.insert("", 0, values=(t, c, n, "CRITICAL")))
                                play_success_beep() 
                        else:
                            color = (0, 255, 0) # GREEN for known staff/students
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display_frame, name, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                    else:
                        color = (0, 165, 255) # ORANGE for unknown ghosts
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(display_frame, "UNKNOWN ENTITY", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

                display_frame = draw_hud(display_frame, cam_name)
                shared_cctv_frames[cam_name] = (display_frame, has_critical_threat)

            time.sleep(0.05) 

    def ui_render_loop():
        """ TKINTER UI LOOP. Only reads processed frames, does NOT run AI. """
        if not win.winfo_exists() or not cctv_daemon_running[0]: return

        # BUG FIX 3: Thread-safe dictionary iteration
        for cam_name, (frame, is_threat) in list(shared_cctv_frames.items()):
            if frame is None: continue
            
            if cam_name in grid_widgets:
                gw = grid_widgets[cam_name]
                if is_threat: gw["box"].config(bg="#ff0000")
                else: gw["box"].config(bg="#121212")
                
                thumb_img = resize_to_fit(frame, 320, 240)
                imgtk_small = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(thumb_img, cv2.COLOR_BGR2RGB)))
                gw["feed"].imgtk = imgtk_small
                gw["feed"].configure(image=imgtk_small)

            if cam_name == focused_cctv_camera[0]:
                if is_threat: focus_frame.config(fg="#ff4444")
                else: focus_frame.config(fg="#ffeb3b")
                
                # BUG FIX 2: Resize to the LABEL's bounds, not the parent frame's bounds
                w, h = lbl_focus_video.winfo_width(), lbl_focus_video.winfo_height()
                if w > 10 and h > 10:
                    focus_img = resize_to_fit(frame, w, h)
                    imgtk_large = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(focus_img, cv2.COLOR_BGR2RGB)))
                    lbl_focus_video.imgtk = imgtk_large
                    lbl_focus_video.configure(image=imgtk_large, text="")

        win.after(33, ui_render_loop) 

    def on_close():
        cctv_daemon_running[0] = False
        for slot in cctv_cameras.values(): cam_manager.stop_camera(slot)
        shared_cctv_frames.clear()
        show_dashboard(win, win_name)
        
    win.protocol("WM_DELETE_WINDOW", on_close)

    ctrl_frame = tk.Frame(grid_frame, bg="#121212")
    ctrl_frame.pack(fill="x", side="bottom")
    tk.Button(ctrl_frame, text="⚡ INITIALIZE GLOBAL NETWORK", command=boot_surveillance_network, bg="#884EA0", fg="white", font=("Segoe UI", 11, "bold"), cursor="hand2", pady=10).pack(fill="x")


# MODULE: ENTERPRISE YOLOv8 + FAISS SOC (TRACK-THEN-RECOGNIZE)
def open_enterprise_yolo_soc():
    win_name = "enterprise_yolo_soc"
    
    if win_name in open_windows and open_windows[win_name].winfo_exists():
        open_windows[win_name].deiconify()
        open_windows[win_name].lift()
        return

    win = tk.Toplevel(root)
    win.title("Enterprise SOC: YOLOv8 ByteTrack + FAISS Engine")
    win.geometry("1350x880")
    open_windows[win_name] = win
    win.configure(bg="#050505")
    add_window_toolbar(win, win_name, stop_camera_instance="yolo_tracker")

    tk.Label(win, text="🛡️ Enterprise SOC: Body Tracking + AI Sniper", font=("Segoe UI", 18, "bold"), bg="#050505", fg="#00e5ff").pack(pady=10)
    
    status_var = tk.StringVar(value="Status: Booting Neural Engines...")
    tk.Label(win, textvariable=status_var, font=("Consolas", 12), bg="#050505", fg="#ffeb3b").pack()

    # --- HARDWARE ROUTING SELECTOR ---
    routing_frame = tk.Frame(win, bg="#121212", bd=1, relief="solid", pady=5)
    routing_frame.pack(fill="x", padx=20, pady=10)
    
    tk.Label(routing_frame, text="Active Hardware Source:", bg="#121212", fg="#00e5ff", font=("Segoe UI", 11, "bold")).pack(side="left", padx=10)
    
    selected_source = tk.StringVar()
    camera_combo = create_camera_device_selector(routing_frame, selected_source)
    camera_combo.pack(side="left", fill="x", expand=True, padx=10)

    viewport_frame = tk.Frame(win, bg="black", bd=1, relief="solid")
    viewport_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
    viewport_frame.pack_propagate(False)
    
    lbl_video = tk.Label(viewport_frame, bg="black")
    lbl_video.pack(fill="both", expand=True)

    tracking_running = [True]
    shared_ai_data = {"frame": None, "status": "Booting..."}

    def switch_tracking_stream(*args):
        source = selected_source.get()
        if not source: return
        actual_src = int(source) if source.isdigit() else source
        cam_manager.start_camera("yolo_tracker", actual_src)

    selected_source.trace_add("write", switch_tracking_stream)
    win.after(400, switch_tracking_stream)

    # --- 1. BOOT YOLOv8 & FAISS ---
    def initialize_engines():
        global yolo_model, faiss_index
        shared_ai_data["status"] = "Status: Downloading/Loading YOLOv8 Nano..."
        
        try:
            yolo_model = YOLO("yolov8n.pt") 
        except Exception as e:
            shared_ai_data["status"] = f"YOLO Load Failed: {str(e)}"
            return

        shared_ai_data["status"] = "Status: Building FAISS Vector Index..."
        
        dimension = 512
        faiss_index = faiss.IndexFlatIP(dimension) 
        
        if len(KNOWN_EMBEDDINGS) > 0:
            emb_matrix = np.array(KNOWN_EMBEDDINGS).astype('float32')
            faiss.normalize_L2(emb_matrix) 
            faiss_index.add(emb_matrix)
            
        shared_ai_data["status"] = f"Status: Neural Engines Online. FAISS Index Size: {faiss_index.ntotal} vectors."
        
        threading.Thread(target=ai_inference_daemon, daemon=True).start()
        win.after(100, ui_render_loop)

    # --- 2. DECOUPLED AI DAEMON (BACKGROUND THREAD) ---
    def ai_inference_daemon():
        # ID Recycling Pool Memory (Added frame_counter for Cooldown Logic)
        active_tracks = {} 
        
        while tracking_running[0]:
            ret, frame = cam_manager.read_camera("yolo_tracker")
            if not ret or frame is None:
                time.sleep(0.01)
                continue
                
            display_frame = frame.copy()
            current_time = time.time()
            current_frame_ids = []
            
            # PHASE 1: YOLOv8 BYTETRACK 
            # conf bumped to 0.45 to prevent ghost tracks on furniture
            results = yolo_model.track(frame, persist=True, classes=[0], conf=0.45, tracker="bytetrack.yaml", verbose=False)
            
            if results and results[0].boxes and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                
                for box, track_id in zip(boxes, track_ids):
                    current_frame_ids.append(track_id)
                    x1, y1, x2, y2 = box
                    
                    # Initialize or Update the Track
                    if track_id not in active_tracks:
                        active_tracks[track_id] = {"name": None, "sim": 0.0, "frame_counter": 0, "last_seen": current_time}
                    else:
                        active_tracks[track_id]["last_seen"] = current_time
                    
                    track_data = active_tracks[track_id]
                    track_data["frame_counter"] += 1
                    
                    # PHASE 2: INFINITE PATIENCE FACE SNIPER
                    # If we don't know who this is, try to scan them ONLY once every 10 frames (Cooldown)
                    if track_data["name"] is None or track_data["name"] == "UNKNOWN ENTITY":
                        if track_data["frame_counter"] % 10 == 0:
                            shared_ai_data["status"] = f"Status: Target [{track_id}] locked. Sniping Face Vector..."
                            
                            # Padded Crop: Send the whole body box, but expand it by 15 pixels 
                            # so if their face is on the very edge of the box, it doesn't get cut in half.
                            pad = 15
                            body_crop = frame[max(0, y1-pad):min(frame.shape[0], y2+pad), max(0, x1-pad):min(frame.shape[1], x2+pad)]
                            
                            if body_crop.size > 0:
                                faces = FA.get(body_crop)
                                if len(faces) > 0:
                                    # Face Found! Extract and FAISS match
                                    query_vector = np.array([faces[0].embedding]).astype('float32')
                                    faiss.normalize_L2(query_vector)
                                    
                                    if faiss_index.ntotal > 0:
                                        distances, indices = faiss_index.search(query_vector, 1)
                                        best_sim = distances[0][0]
                                        best_idx = indices[0][0]
                                        
                                        if best_sim >= SIMILARITY_THRESHOLD:
                                            name = KNOWN_LABELS[best_idx].split("|")[1].strip()
                                            track_data["name"] = name
                                            track_data["sim"] = best_sim * 100
                                            shared_ai_data["status"] = f"Status: Identity Confirmed: {name}"
                                            play_success_beep()
                                        else:
                                            # Face found, but doesn't match database
                                            track_data["name"] = "UNKNOWN ENTITY"
                                            track_data["sim"] = best_sim * 100
                                else:
                                    # No face found (e.g. back of head). Do nothing. 
                                    # We will automatically try again in 10 frames!
                                    pass
                                    
                    # PHASE 3: HUD RENDERING
                    if track_data["name"] and "UNKNOWN" not in track_data["name"]:
                        color = (0, 255, 0)
                        text = f"ID:{track_id} {track_data['name']} ({track_data['sim']:.0f}%)"
                    elif track_data["name"] == "UNKNOWN ENTITY":
                        color = (0, 0, 255)
                        text = f"ID:{track_id} {track_data['name']} [ALERT]"
                    else:
                        color = (0, 165, 255)
                        text = f"ID:{track_id} WAITING FOR FACE..."
                        
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(display_frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # PHASE 4: 5-SECOND ID RECYCLING CLEANUP
            expired_ids = []
            for tid, data in active_tracks.items():
                if tid not in current_frame_ids:
                    if (current_time - data["last_seen"]) > 5.0:
                        expired_ids.append(tid)
            for tid in expired_ids:
                del active_tracks[tid]

            display_frame = draw_hud(display_frame, "ENTERPRISE YOLO NODE")
            
            # Push to shared memory
            shared_ai_data["frame"] = display_frame
            
            # CRITICAL FIX: Sleep for 30ms to sync with the camera's 30FPS. 
            # Prevents the AI from processing the exact same frame twice and burning the CPU.
            time.sleep(0.03) 

    # --- 3. UI RENDER LOOP (Main Thread) ---
    def ui_render_loop():
        if not win.winfo_exists() or not tracking_running[0]: return

        status_var.set(shared_ai_data["status"])
        frame = shared_ai_data.get("frame")
        
        if frame is not None:
            w, h = viewport_frame.winfo_width(), viewport_frame.winfo_height()
            if w > 10 and h > 10:
                resized = resize_to_fit(frame, w, h)
                imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
                lbl_video.imgtk = imgtk
                lbl_video.configure(image=imgtk, text="")
                
        win.after(33, ui_render_loop)
        
    def on_close():
        tracking_running[0] = False
        cam_manager.stop_camera("yolo_tracker")
        show_dashboard(win, win_name)
        
    win.protocol("WM_DELETE_WINDOW", on_close)
    threading.Thread(target=initialize_engines, daemon=True).start()

# MODULE: PURE PYTHON SPATIAL CENTROID TRACKER (CPU OPTIMIZED)

class CentroidTracker:
    def __init__(self, max_disappeared=15):
        self.nextObjectID = 0
        self.objects = OrderedDict() 
        self.identities = {}         
        self.attempts = {}           # NEW: CPU Armor - Tracks failed recognition attempts
        self.disappeared = OrderedDict()
        self.maxDisappeared = max_disappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.identities[self.nextObjectID] = None 
        self.attempts[self.nextObjectID] = 0 
        self.nextObjectID += 1
        return self.nextObjectID - 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]
        if objectID in self.identities: del self.identities[objectID]
        if objectID in self.attempts: del self.attempts[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects, self.identities, self.attempts

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())

            D = dist.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows, usedCols = set(), set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols: continue
                if D[row, col] > 100: continue 

                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects, self.identities, self.attempts



# MODULE: ADVANCED TRACKING SOC (TRACK-THEN-RECOGNIZE)

def open_advanced_tracking_soc():
    win_name = "advanced_tracking_soc"
    
    if win_name in open_windows and open_windows[win_name].winfo_exists():
        open_windows[win_name].deiconify()
        open_windows[win_name].lift()
        return

    win = tk.Toplevel(root)
    win.title("Next-Gen Tracker: Spatial Compute Reduction Engine")
    win.geometry("1200x800")
    open_windows[win_name] = win
    win.configure(bg="#050505")
    add_window_toolbar(win, win_name, stop_camera_instance="adv_tracker")

    tk.Label(win, text="🎯 Advanced AI Spatial Tracking (Compute Saver)", font=("Segoe UI", 18, "bold"), bg="#050505", fg="#00e5ff").pack(pady=10)
    
    status_var = tk.StringVar(value="Status: Spatial Tracker Offline")
    tk.Label(win, textvariable=status_var, font=("Consolas", 12), bg="#050505", fg="#ffeb3b").pack()

    routing_frame = tk.Frame(win, bg="#121212", bd=1, relief="solid", pady=5)
    routing_frame.pack(fill="x", padx=20, pady=10)
    
    tk.Label(routing_frame, text="Active Hardware Source:", bg="#121212", fg="#00e5ff", font=("Segoe UI", 11, "bold")).pack(side="left", padx=10)
    
    selected_source = tk.StringVar()
    camera_combo = create_camera_device_selector(routing_frame, selected_source)
    camera_combo.pack(side="left", fill="x", expand=True, padx=10)

    viewport_frame = tk.Frame(win, bg="black", bd=1, relief="solid")
    viewport_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
    viewport_frame.pack_propagate(False)
    
    lbl_video = tk.Label(viewport_frame, bg="black")
    lbl_video.pack(fill="both", expand=True)

    ct = CentroidTracker(max_disappeared=15)
    tracking_running = [True]

    def switch_tracking_stream(*args):
        source = selected_source.get()
        if not source: return
        actual_src = int(source) if source.isdigit() else source
        cam_manager.start_camera("adv_tracker", actual_src)

    selected_source.trace_add("write", switch_tracking_stream)
    win.after(400, switch_tracking_stream)

    def tracking_loop():
        if not win.winfo_exists() or not tracking_running[0]: return

        ret, frame = cam_manager.read_camera("adv_tracker")
        if ret and frame is not None:
            display_frame = frame.copy()
            
            # --- PHASE 1: DETECTION (FAST DOWNSCALE) ---
            # Shrink the frame by 50% to heavily reduce baseline CPU usage
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

            try:
                bboxes, _ = FA.models['detection'].detect(small_frame, max_num=10, metric='default')
            except Exception:
                bboxes = []

            rects = []
            if bboxes is not None:
                for bbox in bboxes:
                    # Multiply by 2 to map the small frame coordinates back to the original HD frame
                    rects.append((int(bbox[0]*2), int(bbox[1]*2), int(bbox[2]*2), int(bbox[3]*2)))

            objects, identities, attempts = ct.update(rects)

            # --- PHASE 2: SAFE SELECTIVE RECOGNITION (NO LOOPS) ---
            needs_recognition = [objID for objID, ident in identities.items() if ident is None]

            if len(needs_recognition) > 0:
                status_var.set("Status: Resolving Matrix Identity...")
                
                # Run full extraction safely on the downscaled frame
                faces = FA.get(small_frame)

                for face in faces:
                    fx1, fy1, fx2, fy2 = map(int, face.bbox * 2)
                    fcX, fcY = (fx1 + fx2) / 2.0, (fy1 + fy2) / 2.0

                    for objID in needs_recognition:
                        if identities[objID] is not None: continue 

                        centroid = objects[objID]
                        if dist.euclidean((fcX, fcY), centroid) < 80:
                            sims = face_similarity(KNOWN_EMBEDDINGS, face.embedding)
                            if sims.size > 0 and sims.max() >= SIMILARITY_THRESHOLD:
                                idx = int(sims.argmax())
                                name = KNOWN_LABELS[idx].split("|")[1].strip()
                                identities[objID] = {"name": name, "sim": sims.max() * 100, "color": (0,255,0)}
                            else:
                                # Found face, but vector not in DB -> Mark as UNKNOWN permanently
                                identities[objID] = {"name": "UNKNOWN", "sim": 0.0, "color": (0,165,255)}

                # CPU ARMOR: If the face extraction failed entirely, increment attempt counter
                for objID in needs_recognition:
                    if identities[objID] is None:
                        attempts[objID] += 1
                        if attempts[objID] >= 5:
                            # Stop trying to recognize after 5 failed frames to prevent CPU melting
                            identities[objID] = {"name": "UNRECOGNIZED", "sim": 0.0, "color": (150,150,150)}
                            status_var.set("Status: Target Unrecognized. Aborting heavy AI loop.")
                        else:
                            status_var.set(f"Status: Recognition Attempt {attempts[objID]}/5...")
            else:
                status_var.set("Status: Spatial Tracking Matrix Active (Compute Saver ON)")

            # --- PHASE 3: RENDER HUD ---
            if bboxes is not None:
                for (startX, startY, endX, endY) in rects:
                    cX, cY = int((startX + endX) / 2.0), int((startY + endY) / 2.0)
                    
                    matched_id = None
                    for objID, centroid in objects.items():
                        if dist.euclidean((cX, cY), centroid) < 50:
                            matched_id = objID
                            break
                    
                    if matched_id is not None:
                        identity = identities.get(matched_id)
                        if identity:
                            color = identity["color"]
                            text = f"[{matched_id}] {identity['name']} ({identity['sim']:.0f}%)"
                        else:
                            color = (255,255,0)
                            text = f"[{matched_id}] ANALYZING..."
                            
                        cv2.rectangle(display_frame, (startX, startY), (endX, endY), color, 2)
                        cv2.putText(display_frame, text, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        cv2.circle(display_frame, (cX, cY), 4, color, -1)

            display_frame = draw_hud(display_frame, "SPATIAL TRACKING NODE")
            
            w, h = viewport_frame.winfo_width(), viewport_frame.winfo_height()
            if w > 10 and h > 10:
                resized = resize_to_fit(display_frame, w, h)
                imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
                lbl_video.imgtk = imgtk
                lbl_video.configure(image=imgtk, text="")
                
        # Thermal Throttling: Runs at ~16 FPS to let the CPU breathe
        win.after(60, tracking_loop)
        
    def on_close():
        tracking_running[0] = False
        cam_manager.stop_camera("adv_tracker")
        show_dashboard(win, win_name)
        
    win.protocol("WM_DELETE_WINDOW", on_close)
    win.after(500, tracking_loop)

# ---------- DASHBOARD ----------
root = tk.Tk(); root.title("CCTV & Attendance Master")
root.geometry("1200x780"); root.minsize(1000, 650); set_zoomed(root)

def hex_to_rgba(hex_str, alpha=255):
    h = hex_str.lstrip("#")
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)

def make_outer_shadow_image(card_w, card_h, card_hex="#ffffff", radius=18, shadow_hex="#000000", blur_radius=18, shadow_opacity=110):
    pad = blur_radius
    img = Image.new("RGBA", (card_w + pad * 2, card_h + pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([pad, pad, pad + card_w, pad + card_h], radius=radius, fill=hex_to_rgba(shadow_hex, shadow_opacity))
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    ImageDraw.Draw(img).rounded_rectangle([pad, pad, pad + card_w, pad + card_h], radius=radius, fill=hex_to_rgba(card_hex, 255))
    return img


top = tk.Frame(root, pady=12); top.pack(fill="x")
tk.Label(top, text="Master Attendance Server", font=("Helvetica", 28, "bold")).pack(side="left", padx=20)
tk.Button(top, text="Minimize", command=root.iconify, bg="#555", fg="white", font=("Segoe UI", 10, "bold"), bd=0, padx=10, pady=5).pack(side="right", padx=10)
tk.Button(top, text="Maximize / Restore", command=lambda: toggle_zoom(root), bg="#555", fg="white", font=("Segoe UI", 10, "bold"), bd=0, padx=10, pady=5).pack(side="right", padx=5)
center = tk.Frame(root); center.pack(expand=True)
grid = tk.Frame(center); grid.pack()

cards = []
def make_card(parent, icon_text, label_text, command, card_w=300, card_h=170, icon_image_path=None):
    theme = THEMES[current_theme]
    normal_tk = ImageTk.PhotoImage(make_outer_shadow_image(card_w, card_h, card_hex=theme["card_bg"], shadow_hex=theme["shadow"], blur_radius=theme["shadow_blur"], shadow_opacity=theme["shadow_opacity"]))
    
    shadow_lbl = tk.Label(parent, image=normal_tk, bd=0, bg=theme["bg"], cursor="hand2")
    shadow_lbl.image = normal_tk; shadow_lbl.pack_propagate(False)

    inner = tk.Frame(shadow_lbl, bg=theme["card_bg"], cursor="hand2")
    inner.place(relx=0.5, rely=0.5, anchor="center")
    
    if icon_image_path and os.path.exists(icon_image_path):
        try:
            pil_img = Image.open(icon_image_path).resize((48, 48), Image.Resampling.LANCZOS)
            icon_img = ImageTk.PhotoImage(pil_img)
            icon_widget = tk.Label(inner, image=icon_img, bg=theme["card_bg"])
            icon_widget.image = icon_img
        except Exception:
            icon_widget = tk.Label(inner, text=icon_text, font=("Segoe UI Emoji", 36), bg=theme["card_bg"], fg=theme["card_fg"])
    else:
        icon_widget = tk.Label(inner, text=icon_text, font=("Segoe UI Emoji", 36), bg=theme["card_bg"], fg=theme["card_fg"])
        
    icon_widget.pack(pady=(0, 6))
    tk.Label(inner, text=label_text, font=("Segoe UI", 14, "bold"), bg=theme["card_bg"], fg=theme["card_fg"]).pack()

    for w in (shadow_lbl, inner, icon_widget, inner.winfo_children()[-1]): w.bind("<Button-1>", lambda ev: command())
    cards.append({"shadow_lbl": shadow_lbl, "inner": inner, "w": card_w, "h": card_h}); return shadow_lbl

def apply_theme():
    theme = THEMES[current_theme]
    root.configure(bg=theme["bg"]); top.configure(bg=theme["bg"]); center.configure(bg=theme["bg"]); grid.configure(bg=theme["bg"]); bot.configure(bg=theme["bg"])
    for widget in top.winfo_children(): widget.configure(bg=theme["bg"], fg=theme["fg"])
    bot_label.configure(bg=theme["bg"], fg=theme["fg"])
    for cinfo in cards:
        normal_tk = ImageTk.PhotoImage(make_outer_shadow_image(cinfo["w"], cinfo["h"], card_hex=theme["card_bg"], shadow_hex=theme.get("shadow", "#000"), blur_radius=theme.get("shadow_blur", 28), shadow_opacity=theme.get("shadow_opacity", 110)))
        cinfo["shadow_lbl"].configure(image=normal_tk, bg=theme["bg"]); cinfo["shadow_lbl"].image = normal_tk
        cinfo["inner"].configure(bg=theme["card_bg"])
        for child in cinfo["inner"].winfo_children(): child.configure(bg=theme["card_bg"], fg=theme["card_fg"])

make_card(grid, "🎓", "Student Face Enrollment", open_enrollment, 250, 138, icon_image_path="icons/student.png").grid(row=0, column=0, padx=10, pady=10)
make_card(grid, "👨‍🏫", "Faculty Registry", lambda: open_person_registration("faculty"), 250, 138).grid(row=0, column=1, padx=10, pady=10)
make_card(grid, "🧑‍💼", "Staff Registry", lambda: open_person_registration("non_faculty"), 250, 138).grid(row=0, column=2, padx=10, pady=10)
make_card(grid, "🎫", "Guest Pass Registry", lambda: open_person_registration("guest"), 250, 138).grid(row=0, column=3, padx=10, pady=10)
make_card(grid, "☑", "Attendance Operations", open_attendance, 250, 138, icon_image_path=f"icons/face-id.png").grid(row=1, column=0, padx=10, pady=10)
make_card(grid, "SES", "Session Attendance", open_session_attendance, 250, 138, icon_image_path="icons/training.png").grid(row=1, column=1, padx=10, pady=10)
make_card(grid, "QR", "QR Access Control", open_qr_access_control, 250, 138, icon_image_path="icons/qr-code-pay.png").grid(row=1, column=2, padx=10, pady=10)
make_card(grid, "🏛", "Academic Setup", open_academic_setup, 250, 138, icon_image_path="icons/acadamic-setup.png").grid(row=1, column=3, padx=10, pady=10)
make_card(grid, "📊", "Main Attendance Reports", open_reports, 250, 138).grid(row=2, column=0, padx=10, pady=10)
make_card(grid, "📈", "Analytics Lab", open_analytics_reports, 250, 138).grid(row=2, column=1, padx=10, pady=10)
#make_card(grid, "◉", "CCTV Intelligence Reports", open_surveillance_reports, 250, 138).grid(row=2, column=2, padx=10, pady=10)
make_card(grid, "🛡️", "Security Operations Center", open_cctv_surveillance_soc, 250, 138).grid(row=2, column=2, padx=10, pady=10)
make_card(grid, "⌘", "System Tools", open_tools_window, 250, 138).grid(row=2, column=3, padx=10, pady=10)
#make_card(grid, "⌘", "Identity & Biometric Hub", open_identity_management_hub, 250, 138).grid(row=3, column=1, padx=10, pady=10)
make_card(grid, "⌘", "Tracking", open_ghost_tracking_matrix, 250, 138).grid(row=3, column=1, padx=10, pady=10)
make_card(grid, "🛡️", "Master SOC Command", open_master_soc_matrix, 250, 138).grid(row=3, column=0, padx=10, pady=10)
make_card(grid, "🎯", "Advanced Tracking SOC", open_advanced_tracking_soc, 250, 138).grid(row=3, column=2, padx=10, pady=10)
make_card(grid, "👁️", "YOLOv8 Enterprise SOC", open_enterprise_yolo_soc, 250, 138).grid(row=3, column=3, padx=10, pady=10)

bot = tk.Frame(root, pady=10); bot.pack(fill="x", side="bottom")

def toggle_theme():
    global current_theme
    current_theme = "dark" if current_theme == "light" else "light"
    apply_theme()

bot_label = tk.Label(bot, text="Global Config", font=("Segoe UI", 11))
bot_label.pack(side="left", padx=20)
tk.Button(bot, text="Toggle Theme", command=toggle_theme, bg="#555", fg="white", font=("Segoe UI", 11, "bold"), bd=0, padx=10, pady=5).pack(side="right", padx=15)
tk.Button(bot, text="Exit Server", command=lambda: [cam_manager.stop_all(), root.destroy()], bg="#d32f2f", fg="white", font=("Segoe UI", 11, "bold"), bd=0, padx=10, pady=5).pack(side="right", padx=15)

apply_theme(); root.mainloop()