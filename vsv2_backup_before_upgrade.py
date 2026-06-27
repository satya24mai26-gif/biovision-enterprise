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

try: import winsound; HAS_SOUND = True
except ImportError: HAS_SOUND = False

# ---------- InsightFace & Threading ----------
_MODEL_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(_MODEL_ROOT, exist_ok=True)
FA = FaceAnalysis(name="buffalo_sc", root=_MODEL_ROOT, providers=["CPUExecutionProvider"])
FA.prepare(ctx_id=0, det_size=(640, 480))

ai_executor = ThreadPoolExecutor(max_workers=2)

# ---------- GLOBALS & CONFIG ----------
DB_FILE = "students.db"
CAMERA_SOURCE = 2 
SIMILARITY_THRESHOLD = 0.45
CCTV_COOLDOWN_SECONDS = 300 
PROCESS_EVERY_N = 3

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
current_theme = "dark"
open_windows = {}

# ---------- DATABASE ENGINE & AUTO-MIGRATOR ----------
def get_conn():
    conn = sqlite3.connect(DB_FILE)
    conn.execute("PRAGMA journal_mode=WAL;") 
    return conn

def init_db():
    os.makedirs("photos", exist_ok=True); os.makedirs("qrcodes", exist_ok=True)
    with get_conn() as conn:
        c = conn.cursor()
        c.execute("CREATE TABLE IF NOT EXISTS students (id INTEGER PRIMARY KEY AUTOINCREMENT, reg_no TEXT UNIQUE, name TEXT, course TEXT, mobile TEXT, photo_path TEXT, qr_path TEXT)")
        c.execute("CREATE TABLE IF NOT EXISTS attendance (id INTEGER PRIMARY KEY AUTOINCREMENT, student_id INTEGER, date TEXT, time TEXT, match_percentage REAL)")
        
        c.execute("PRAGMA table_info(students)")
        student_cols = [row[1] for row in c.fetchall()]
        if 'embedding' not in student_cols: c.execute("ALTER TABLE students ADD COLUMN embedding BLOB")
        if 'is_twin' not in student_cols: c.execute("ALTER TABLE students ADD COLUMN is_twin INTEGER DEFAULT 0")

        c.execute("PRAGMA table_info(attendance)")
        att_cols = [row[1] for row in c.fetchall()]
        if 'camera_location' not in att_cols: c.execute("ALTER TABLE attendance ADD COLUMN camera_location TEXT DEFAULT 'Main Server'")
        conn.commit()
init_db()

# ---------- SERVER MEMORY ----------
KNOWN_EMBEDDINGS, KNOWN_IDS, KNOWN_LABELS, KNOWN_COURSES = [], [], [], []

def load_server_memory():
    global KNOWN_EMBEDDINGS, KNOWN_IDS, KNOWN_LABELS, KNOWN_COURSES
    KNOWN_EMBEDDINGS.clear(); KNOWN_IDS.clear(); KNOWN_LABELS.clear(); KNOWN_COURSES.clear()
    with get_conn() as conn:
        df = pd.read_sql_query("SELECT id, reg_no, name, course, embedding FROM students WHERE embedding IS NOT NULL", conn)
    for _, row in df.iterrows():
        KNOWN_EMBEDDINGS.append(np.frombuffer(row["embedding"], dtype=np.float32))
        KNOWN_IDS.append(int(row["id"]))
        KNOWN_LABELS.append(f"{row['reg_no']} | {row['name']}")
        KNOWN_COURSES.append(str(row["course"]).strip())
load_server_memory()

def face_similarity(known_encodings, probe_enc):
    if not known_encodings: return np.array([])
    k, p = np.array(known_encodings, dtype=np.float32), np.array(probe_enc, dtype=np.float32)
    return (k / (np.linalg.norm(k, axis=1, keepdims=True) + 1e-9)) @ (p / (np.linalg.norm(p) + 1e-9))

# ---------- GLOBAL CAMERA MANAGER ----------
class AsyncCamera:
    def __init__(self, source):
        self.source = source
        self.cap = None; self.frame = None; self.ret = False; self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            if isinstance(self.source, int) or (isinstance(self.source, str) and self.source.isdigit()):
                self.cap = cv2.VideoCapture(int(self.source), cv2.CAP_DSHOW)
            else: 
                self.cap = cv2.VideoCapture(self.source)
            
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            time.sleep(1) 
            
            while self.running:
                if self.cap.isOpened():
                    try:
                        self.ret, f = self.cap.read()
                        if self.ret: self.frame = f.copy()
                    except cv2.error:
                        self.ret = False
                time.sleep(0.01)
        except Exception:
            self.ret = False

    def read(self): return self.ret, self.frame
    
    def stop(self): 
        self.running = False
        if self.thread.is_alive(): self.thread.join(timeout=1.0) 
        if self.cap: 
            self.cap.release() 
            self.cap = None

class CameraManager:
    def __init__(self): self.cam = None
    def start(self, source):
        self.stop(); time.sleep(0.5); self.cam = AsyncCamera(source)
    def stop(self):
        if self.cam: self.cam.stop(); self.cam = None
    def read(self):
        if self.cam: return self.cam.read()
        return False, None

cam_manager = CameraManager()

def switch_camera_instance():
    val = simpledialog.askstring("Switch Camera", "Enter Camera Source (0, 1, 2...):")
    if val:
        global CAMERA_SOURCE
        CAMERA_SOURCE = int(val.strip()) if val.strip().isdigit() else val.strip()
        cam_manager.start(CAMERA_SOURCE)
        messagebox.showinfo("Camera", f"Switched to Camera {CAMERA_SOURCE}")

def close_all_modules():
    cam_manager.stop()
    for win_name in list(open_windows.keys()):
        try: open_windows[win_name].destroy()
        except Exception: pass
        open_windows.pop(win_name, None)

def play_success_beep():
    if HAS_SOUND: threading.Thread(target=lambda: winsound.Beep(1200, 150), daemon=True).start()

# Helper: Keep Aspect Ratio securely
def resize_to_fit(image, target_w, target_h):
    if target_w < 10 or target_h < 10: return image
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    return cv2.resize(image, (int(w * scale), int(h * scale)))

def draw_hud(display, cam_name):
    # Professional CCTV overlay
    tm = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    blink = int(time.time() * 2) % 2 == 0
    if blink: cv2.circle(display, (30, 30), 8, (0, 0, 255), -1)
    cv2.putText(display, f"REC | {cam_name}", (50, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(display, tm, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    return display

# ---------- 1. ENROLLMENT ----------
def open_enrollment():
    close_all_modules()
    win = tk.Toplevel(root); win.title("Enrollment Server"); win.geometry("1100x650")
    open_windows["enrollment"] = win; win.configure(bg=THEMES[current_theme]["bg"])
    
    cam_manager.start(CAMERA_SOURCE)
    win.protocol("WM_DELETE_WINDOW", lambda: [cam_manager.stop(), open_windows.pop("enrollment", None), win.destroy()])

    # Left Side: Fixed Form with HARD LOCK
    form = tk.Frame(win, bg=THEMES[current_theme]["bg"], width=350)
    form.pack_propagate(False) # Prevents expanding camera from squeezing this out!
    form.pack(side="left", padx=20, pady=20, fill="y")
    
    tk.Label(form, text="Register Student", font=("Segoe UI", 18, "bold"), bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(pady=(0, 10))
    tk.Button(form, text="⚙️ Switch Camera Feed", command=switch_camera_instance, bg="#555", fg="white", cursor="hand2").pack(fill="x", pady=(0,15))

    vars_map = {}
    with get_conn() as conn: courses = [r[0] for r in conn.cursor().execute("SELECT DISTINCT course FROM students").fetchall()]
    for label in ["Reg No", "Name", "Course", "Mobile"]:
        tk.Label(form, text=f"{label}:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w")
        ent = ttk.Combobox(form, values=courses) if label == "Course" else tk.Entry(form, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"])
        ent.pack(fill="x", pady=5); vars_map[label] = ent

    # Right Side: Safe Video Canvas
    video_frame = tk.Frame(win, bg="black"); video_frame.pack(side="right", padx=20, pady=20, fill="both", expand=True)
    video_frame.pack_propagate(False)
    lbl_video = tk.Label(video_frame, bg="black")
    lbl_video.place(relx=0.5, rely=0.5, anchor="center") # Floating perfectly inside frame

    def update_cam():
        ret, frame = cam_manager.read()
        if ret and frame is not None:
            w, h = video_frame.winfo_width(), video_frame.winfo_height()
            resized = resize_to_fit(draw_hud(frame.copy(), "Enrollment Node"), w, h)
            imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)))
            lbl_video.imgtk = imgtk; lbl_video.configure(image=imgtk)
        lbl_video.after(30, update_cam)
    update_cam()

    def save_student():
        reg, name, crs, mob = (vars_map[k].get().strip() for k in ["Reg No", "Name", "Course", "Mobile"])
        if not all([reg, name, crs, mob]): return messagebox.showerror("Error", "All fields required")
        with get_conn() as conn:
            if conn.cursor().execute("SELECT 1 FROM students WHERE reg_no=?", (reg,)).fetchone():
                return messagebox.showerror("Error", "Student exists!")

        ret, frame = cam_manager.read()
        if not ret: return messagebox.showerror("Error", "Camera offline")

        faces = FA.get(frame)
        if len(faces) != 1: return messagebox.showerror("Error", "Ensure exactly ONE face is in the frame.")
        
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
            conn.commit()
        load_server_memory()
        messagebox.showinfo("Success", f"{name} Enrolled.")
        for e in vars_map.values(): e.delete(0, tk.END)

    tk.Button(form, text="📷 Scan & Enroll", command=save_student, bg="#4CAF50", fg="white", font=("Segoe UI", 11, "bold"), cursor="hand2").pack(pady=10, fill="x")

# ---------- 2. CENTRAL ATTENDANCE (4 TABS) ----------
def open_attendance():
    close_all_modules()
    win = tk.Toplevel(root); win.title("Central Attendance Hub"); win.geometry("1280x800")
    open_windows["attendance"] = win; win.configure(bg=THEMES[current_theme]["bg"])
    
    cam_manager.start(CAMERA_SOURCE)
    win.protocol("WM_DELETE_WINDOW", lambda: [cam_manager.stop(), open_windows.pop("attendance", None), win.destroy()])

    style = ttk.Style()
    style.configure("Treeview", background=THEMES[current_theme]["tree_bg"], foreground=THEMES[current_theme]["tree_fg"], fieldbackground=THEMES[current_theme]["tree_bg"])
    style.configure("Treeview.Heading", background=THEMES[current_theme]["tree_header_bg"], foreground=THEMES[current_theme]["tree_header_fg"])

    nb = ttk.Notebook(win); nb.pack(fill="both", expand=True)

    def build_tab(parent, title):
        t = tk.Frame(parent, bg=THEMES[current_theme]["bg"]); t.pack(fill="x", padx=10, pady=5)
        tk.Label(t, text=title, font=("Segoe UI", 16, "bold"), fg="#00ffcc", bg=THEMES[current_theme]["bg"]).pack(side="left")
        tk.Button(t, text="⚙️ Switch Camera", command=switch_camera_instance, bg="#555", fg="white", cursor="hand2").pack(side="left", padx=15)
        
        b = tk.Frame(parent, bg=THEMES[current_theme]["bg"]); b.pack(fill="both", expand=True, padx=10, pady=5)
        
        # PACK SIDEBAR FIRST & HARD-LOCK IT TO 380px WIDTH
        s = tk.Frame(b, width=380, bg=THEMES[current_theme]["bg"])
        s.pack_propagate(False) 
        s.pack(side="right", fill="y", padx=10)

        # PACK CAMERA FRAME SECOND
        cf = tk.Frame(b, bg="black")
        cf.pack_propagate(False)
        cf.pack(side="left", fill="both", expand=True)
        
        lc = tk.Label(cf, bg="black")
        lc.place(relx=0.5, rely=0.5, anchor="center") # Free-floating to prevent infinite expansion loops
        return t, lc, s, cf

    # --- TAB 1: SINGLE STUDENT ---
    single_tab = tk.Frame(nb, bg=THEMES[current_theme]["bg"]); nb.add(single_tab, text="Single Scan")
    top_s, lbl_cam_s, side_s, cam_fr_s = build_tab(single_tab, "Single Student Checkpoint")
    
    reg_var_s, name_var_s = tk.StringVar(), tk.StringVar()
    tk.Label(side_s, text="Reg No:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w")
    tk.Entry(side_s, textvariable=reg_var_s, font=("Segoe UI", 14), bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(fill="x", pady=5)
    tk.Label(side_s, text="Name:", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"]).pack(anchor="w")
    tk.Entry(side_s, textvariable=name_var_s, state="readonly", font=("Segoe UI", 14)).pack(fill="x", pady=5)
    
    s_status = tk.Label(side_s, text="Waiting...", fg="#00ffcc", bg=THEMES[current_theme]["bg"], font=("Segoe UI", 12))
    s_status.pack(pady=10)
    target_emb_s = [None]

    def fetch_s(regno):
        with get_conn() as conn: rec = conn.cursor().execute("SELECT name, embedding FROM students WHERE reg_no=?", (regno,)).fetchone()
        if not rec: s_status.config(text="Not Found", fg="red"); return
        name_var_s.set(rec[0]); target_emb_s[0] = np.frombuffer(rec[1], dtype=np.float32)
        s_status.config(text="Loaded. Looking for face...", fg="yellow")

    tk.Button(side_s, text="Fetch Reg No", command=lambda: fetch_s(reg_var_s.get().strip()), bg="#14818f", fg="white", cursor="hand2").pack(fill="x", pady=5)
    
    def qr_scan_s():
        s_status.config(text="Scanning QR...")
        ret, frame = cam_manager.read()
        if ret:
            codes = decode(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            if codes: reg = codes[0].data.decode("utf-8").strip(); reg_var_s.set(reg); fetch_s(reg)
    tk.Button(side_s, text="Scan QR Code", command=qr_scan_s, bg="#884EA0", fg="white", cursor="hand2").pack(fill="x", pady=5)

    # --- TAB 2: GROUP / CLASS ---
    grp_tab = tk.Frame(nb, bg=THEMES[current_theme]["bg"]); nb.add(grp_tab, text="Group")
    top_g, lbl_cam_g, side_g, cam_fr_g = build_tab(grp_tab, "Class Attendance")
    
    force_var = tk.StringVar()
    ttk.Combobox(top_g, textvariable=force_var, values=KNOWN_LABELS, state="readonly").pack(side="right", padx=5)
    
    tv_g = ttk.Treeview(side_g, columns=("Reg No", "Name", "Time"), show="headings", height=20)
    for c in ("Reg No", "Name", "Time"): tv_g.heading(c, text=c); tv_g.column(c, width=100)
    tv_g.pack(fill="both", expand=True)

    def force_scan():
        if not force_var.get(): return
        reg = force_var.get().split("|")[0].strip()
        with get_conn() as conn:
            sid = conn.cursor().execute("SELECT id FROM students WHERE reg_no=?", (reg,)).fetchone()[0]
            conn.cursor().execute("INSERT OR IGNORE INTO attendance (student_id, date, time, match_percentage, camera_location) VALUES (?, ?, ?, ?, ?)", (sid, datetime.now().strftime("%Y-%m-%d"), datetime.now().strftime("%H:%M:%S"), 100.0, "Manual Group Override"))
            conn.commit()
        play_success_beep(); messagebox.showinfo("Forced", f"Manually marked {reg} present.")
    tk.Button(top_g, text="Force Mark Present", command=force_scan, bg="#d9534f", fg="white", cursor="hand2").pack(side="right")

    # --- TAB 3: COURSE ---
    crs_tab = tk.Frame(nb, bg=THEMES[current_theme]["bg"]); nb.add(crs_tab, text="Course")
    top_cr, lbl_cam_cr, side_cr, cam_fr_cr = build_tab(crs_tab, "Course-Wise Filter")
    
    crs_var = tk.StringVar()
    crs_combo = ttk.Combobox(top_cr, textvariable=crs_var, values=list(set(KNOWN_COURSES)), state="readonly")
    crs_combo.pack(side="left", padx=10)
    
    tv_cr = ttk.Treeview(side_cr, columns=("Reg No", "Name", "Time"), show="headings", height=20)
    for c in ("Reg No", "Name", "Time"): tv_cr.heading(c, text=c); tv_cr.column(c, width=100)
    tv_cr.pack(fill="both", expand=True)

    active_course_embs, active_course_ids, active_course_labels = [], [], []
    def apply_course_filter(e=None):
        active_course_embs.clear(); active_course_ids.clear(); active_course_labels.clear(); tv_cr.delete(*tv_cr.get_children())
        sel_c = crs_var.get()
        for i in range(len(KNOWN_COURSES)):
            if KNOWN_COURSES[i] == sel_c:
                active_course_embs.append(KNOWN_EMBEDDINGS[i]); active_course_ids.append(KNOWN_IDS[i]); active_course_labels.append(KNOWN_LABELS[i])
    crs_combo.bind("<<ComboboxSelected>>", apply_course_filter)

    # --- TAB 4: CCTV SURVEILLANCE ---
    cctv_tab = tk.Frame(nb, bg=THEMES[current_theme]["bg"]); nb.add(cctv_tab, text="CCTV")
    top_c, lbl_cam_c, side_c, cam_fr_c = build_tab(cctv_tab, "CCTV Node Server")
    
    tk.Label(top_c, text="Camera Tag:", bg=THEMES[current_theme]["bg"], fg="white").pack(side="left", padx=10)
    cctv_name_var = tk.StringVar(value="Main Gate Cam 1")
    tk.Entry(top_c, textvariable=cctv_name_var, width=20, font=("Segoe UI", 12)).pack(side="left")

    tv_c = ttk.Treeview(side_c, columns=("Name", "Location", "Time"), show="headings", height=20)
    for c in ("Name", "Location", "Time"): tv_c.heading(c, text=c); tv_c.column(c, width=100)
    tv_c.pack(fill="both", expand=True)

    # --- MASTER CAMERA ROUTER LOOP ---
    fut = None; cctv_memory = {}; proc_ctr = 0; group_marked = set(); course_marked = set()
    visual_flashes = {} # For the green success box

    def ai_worker(frame_copy, embs, ids, labels):
        faces = FA.get(cv2.resize(frame_copy, (0, 0), fx=0.5, fy=0.5))
        res = []
        for face in faces:
            sims = face_similarity(embs, face.embedding)
            if sims.size > 0 and sims.max() >= SIMILARITY_THRESHOLD:
                res.append({"box": face.bbox*2, "idx": int(sims.argmax()), "sim": float(sims.max())})
        return res

    def master_loop():
        nonlocal fut, proc_ctr, group_marked, course_marked, visual_flashes
        ret, frame = cam_manager.read()
        cur_tab = nb.tab(nb.select(), "text")

        if ret and frame is not None:
            now_dt = datetime.now()
            dt_s, tm_s = now_dt.strftime("%Y-%m-%d"), now_dt.strftime("%H:%M:%S")
            display = frame.copy()
            proc_ctr = (proc_ctr + 1) % PROCESS_EVERY_N

            # Clear old flashes
            visual_flashes = {k: v for k, v in visual_flashes.items() if (now_dt.timestamp() - v) < 2.0}

            if cur_tab == "Single Scan":
                display = draw_hud(display, "Single Checkpoint")
                w, h = cam_fr_s.winfo_width(), cam_fr_s.winfo_height()
                if target_emb_s[0] is not None and proc_ctr == 0:
                    faces = FA.get(cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
                    if faces:
                        sim = float(face_similarity([target_emb_s[0]], faces[0].embedding)[0])
                        if sim >= SIMILARITY_THRESHOLD:
                            s_status.config(text=f"MATCH: {sim*100:.1f}% - Saved!", fg="#00ff00")
                            play_success_beep()
                            with get_conn() as conn:
                                sid = conn.cursor().execute("SELECT id FROM students WHERE reg_no=?", (reg_var_s.get(),)).fetchone()[0]
                                conn.cursor().execute("INSERT OR IGNORE INTO attendance (student_id, date, time, match_percentage, camera_location) VALUES (?, ?, ?, ?, ?)", (sid, dt_s, tm_s, sim*100, "Single Desk"))
                                conn.commit()
                            target_emb_s[0] = None
                
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resize_to_fit(display, w, h), cv2.COLOR_BGR2RGB)))
                lbl_cam_s.imgtk = img; lbl_cam_s.configure(image=img)

            elif cur_tab == "Group":
                display = draw_hud(display, "Classroom Node")
                w, h = cam_fr_g.winfo_width(), cam_fr_g.winfo_height()
                if fut is None or fut.done():
                    if fut and fut.result():
                        for m in fut.result():
                            sid = KNOWN_IDS[m["idx"]]
                            if sid not in group_marked:
                                play_success_beep()
                                visual_flashes[sid] = now_dt.timestamp()
                                with get_conn() as conn:
                                    conn.cursor().execute("INSERT OR IGNORE INTO attendance (student_id, date, time, match_percentage, camera_location) VALUES (?, ?, ?, ?, ?)", (sid, dt_s, tm_s, m["sim"]*100, "Group Node"))
                                    conn.commit()
                                group_marked.add(sid)
                                r, n = KNOWN_LABELS[m["idx"]].split("|")
                                tv_g.insert("", 0, values=(r.strip(), n.strip(), tm_s))
                            
                            x1,y1,x2,y2 = map(int, m["box"])
                            # Visual Feedback Thick Green Box if recently marked
                            color = (0, 255, 0) if sid in visual_flashes else (200, 200, 200)
                            thick = 4 if sid in visual_flashes else 2
                            cv2.rectangle(display, (x1,y1), (x2,y2), color, thick)
                            cv2.putText(display, f"{KNOWN_LABELS[m['idx']]}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    fut = ai_executor.submit(ai_worker, frame.copy(), KNOWN_EMBEDDINGS, KNOWN_IDS, KNOWN_LABELS)
                
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resize_to_fit(display, w, h), cv2.COLOR_BGR2RGB)))
                lbl_cam_g.imgtk = img; lbl_cam_g.configure(image=img)

            elif cur_tab == "Course":
                display = draw_hud(display, f"Course: {crs_var.get()}")
                w, h = cam_fr_cr.winfo_width(), cam_fr_cr.winfo_height()
                if active_course_embs:
                    if fut is None or fut.done():
                        if fut and fut.result():
                            for m in fut.result():
                                sid = active_course_ids[m["idx"]]
                                if sid not in course_marked:
                                    play_success_beep()
                                    visual_flashes[sid] = now_dt.timestamp()
                                    with get_conn() as conn:
                                        conn.cursor().execute("INSERT OR IGNORE INTO attendance (student_id, date, time, match_percentage, camera_location) VALUES (?, ?, ?, ?, ?)", (sid, dt_s, tm_s, m["sim"]*100, f"Course: {crs_var.get()}"))
                                        conn.commit()
                                    course_marked.add(sid)
                                    r, n = active_course_labels[m["idx"]].split("|")
                                    tv_cr.insert("", 0, values=(r.strip(), n.strip(), tm_s))
                                
                                x1,y1,x2,y2 = map(int, m["box"])
                                color = (255, 255, 0) if sid in visual_flashes else (200, 200, 200)
                                thick = 4 if sid in visual_flashes else 2
                                cv2.rectangle(display, (x1,y1), (x2,y2), color, thick)
                                cv2.putText(display, f"{active_course_labels[m['idx']]}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        fut = ai_executor.submit(ai_worker, frame.copy(), active_course_embs, active_course_ids, active_course_labels)
                
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resize_to_fit(display, w, h), cv2.COLOR_BGR2RGB)))
                lbl_cam_cr.imgtk = img; lbl_cam_cr.configure(image=img)

            elif cur_tab == "CCTV":
                cam_name = cctv_name_var.get().strip()
                display = draw_hud(display, cam_name)
                w, h = cam_fr_c.winfo_width(), cam_fr_c.winfo_height()
                
                if fut is None or fut.done():
                    if fut and fut.result():
                        for m in fut.result():
                            sid, name = KNOWN_IDS[m["idx"]], KNOWN_LABELS[m["idx"]].split("|")[1]
                            if sid not in cctv_memory or (now_dt.timestamp() - cctv_memory[sid]) > CCTV_COOLDOWN_SECONDS:
                                play_success_beep()
                                visual_flashes[sid] = now_dt.timestamp()
                                with get_conn() as conn:
                                    conn.cursor().execute("INSERT INTO attendance (student_id, date, time, match_percentage, camera_location) VALUES (?, ?, ?, ?, ?)", (sid, dt_s, tm_s, m["sim"]*100, cam_name))
                                    conn.commit()
                                cctv_memory[sid] = now_dt.timestamp()
                                tv_c.insert("", 0, values=(name, cam_name, tm_s))
                            
                            x1,y1,x2,y2 = map(int, m["box"])
                            color = (0, 255, 255) if sid in visual_flashes else (100, 100, 100)
                            thick = 4 if sid in visual_flashes else 1
                            cv2.rectangle(display, (x1,y1), (x2,y2), color, thick)
                            cv2.putText(display, f"{name}", (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    fut = ai_executor.submit(ai_worker, frame.copy(), KNOWN_EMBEDDINGS, KNOWN_IDS, KNOWN_LABELS)
                
                img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resize_to_fit(display, w, h), cv2.COLOR_BGR2RGB)))
                lbl_cam_c.imgtk = img; lbl_cam_c.configure(image=img)

        win.after(30, master_loop)
    master_loop()

# ---------- 3. REPORTS ----------
def open_reports():
    close_all_modules()
    win = tk.Toplevel(root); win.title("Global Reports"); win.geometry("1060x680")
    open_windows["reports"] = win; win.configure(bg=THEMES[current_theme]["bg"])
    
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

# ---------- 4. TOOLS & STUDENT MANAGEMENT ----------
def open_tools_window():
    close_all_modules()
    win = tk.Toplevel(root); win.title("Tools & Management"); win.geometry("1100x750")
    open_windows["tools"] = win; win.configure(bg=THEMES[current_theme]["bg"])
    tk.Label(win, text="System Tools & Database Management", font=("Segoe UI", 16, "bold"), fg=THEMES[current_theme]["fg"], bg=THEMES[current_theme]["bg"]).pack(pady=10)

    # 1. STUDENT MANAGEMENT DASHBOARD
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
                conn.commit()
            load_server_memory(); load_students()
            messagebox.showinfo("Deleted", "Student and attendance records deleted.")

    def open_live_updater():
        sel = tv_mgr.selection()
        if not sel: return messagebox.showerror("Error", "Select a student to update.")
        reg, name = tv_mgr.item(sel[0])["values"][1], tv_mgr.item(sel[0])["values"][2]
        
        upd_win = tk.Toplevel(win); upd_win.title(f"Live Face Updater: {name}"); upd_win.geometry("700x600"); upd_win.configure(bg="#222")
        cam_manager.start(CAMERA_SOURCE)
        upd_win.protocol("WM_DELETE_WINDOW", lambda: [cam_manager.stop(), upd_win.destroy()])
        
        tk.Label(upd_win, text=f"Updating Face for {name} ({reg})", font=("Segoe UI", 14), bg="#222", fg="white").pack(pady=10)
        
        vf = tk.Frame(upd_win, bg="black"); vf.pack(fill="both", expand=True)
        vf.pack_propagate(False)
        lbl_feed = tk.Label(vf, bg="black"); lbl_feed.place(relx=0.5, rely=0.5, anchor="center")

        def loop_updater():
            ret, frame = cam_manager.read()
            if ret and frame is not None:
                w, h = vf.winfo_width(), vf.winfo_height()
                imgtk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(resize_to_fit(frame, w, h), cv2.COLOR_BGR2RGB)))
                lbl_feed.imgtk = imgtk; lbl_feed.configure(image=imgtk)
            lbl_feed.after(30, loop_updater)
        loop_updater()

        def capture_new_face():
            ret, frame = cam_manager.read()
            if not ret: return messagebox.showerror("Error", "Camera offline.")
            faces = FA.get(frame)
            if len(faces) != 1: return messagebox.showerror("Error", "Need exactly 1 face in the frame!")
            
            with get_conn() as conn:
                conn.cursor().execute("UPDATE students SET embedding = ? WHERE reg_no = ?", (faces[0].embedding.tobytes(), reg))
                conn.commit()
            cv2.imwrite(os.path.join("photos", f"{reg}.jpg"), frame)
            load_server_memory()
            messagebox.showinfo("Success", "Face Updated in Database!")
            cam_manager.stop(); upd_win.destroy()

        tk.Button(upd_win, text="📸 Capture & Overwrite Face", command=capture_new_face, bg="#4CAF50", fg="white", font=("Segoe UI", 14), cursor="hand2").pack(pady=10)

    tk.Button(btn_row, text="Update Selected Face (Live Preview)", command=open_live_updater, bg="#f39c12", fg="white", cursor="hand2").pack(side="left", padx=5)
    tk.Button(btn_row, text="Delete Selected Student", command=delete_student, bg="#d9534f", fg="white", cursor="hand2").pack(side="left", padx=5)

    # 2. QR & DB TOOLS
    frm_db = tk.LabelFrame(win, text="Data Management", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], padx=10, pady=10)
    frm_db.pack(fill="x", padx=12, pady=5)
    qr_reg = tk.StringVar()
    tk.Entry(frm_db, textvariable=qr_reg, width=20, bg=THEMES[current_theme]["entry_bg"], fg=THEMES[current_theme]["entry_fg"]).pack(side="left", padx=10)
    tk.Button(frm_db, text="Generate QR", command=lambda: messagebox.showinfo("QR", "Generated.") if qrcode.make(qr_reg.get().strip()).save(os.path.join("qrcodes", f"{qr_reg.get().strip()}.png")) is None else None, bg="#4CAF50", fg="white", cursor="hand2").pack(side="left")
    tk.Button(frm_db, text="Backup DB", command=lambda: shutil.copy2(DB_FILE, filedialog.asksaveasfilename(defaultextension=".db")), bg="#2196F3", fg="white", cursor="hand2").pack(side="left", padx=15)

    # 3. SUMMARY
    frm_sum = tk.LabelFrame(win, text="Daily Summary", bg=THEMES[current_theme]["bg"], fg=THEMES[current_theme]["fg"], padx=10, pady=10)
    frm_sum.pack(fill="both", expand=True, padx=12, pady=5)
    tree_s = ttk.Treeview(frm_sum, columns=("Date", "Count"), show="headings", height=5)
    for c in ("Date", "Count"): tree_s.heading(c, text=c); tree_s.column(c, anchor="center")
    tree_s.pack(fill="both", expand=True)
    with get_conn() as conn: df = pd.read_sql_query("SELECT date, COUNT(*) as c FROM attendance GROUP BY date ORDER BY date DESC", conn)
    for _, r in df.iterrows(): tree_s.insert("", tk.END, values=(r["date"], r["c"]))

# ---------- DASHBOARD ----------
root = tk.Tk(); root.title("CCTV & Attendance Master"); root.attributes("-fullscreen", True)

def hex_to_rgba(hex_str, alpha=255):
    h = hex_str.lstrip("#")
    r, g, b = (int(h[i:i+2], 16) for i in (0, 2, 4))
    return (r, g, b, alpha)

def make_outer_shadow_image(card_w, card_h, card_hex="#ffffff", radius=18, shadow_hex="#000000", blur_radius=18, shadow_opacity=110):
    pad = blur_radius
    img = Image.new("RGBA", (card_w + pad * 2, card_h + pad * 2), (0, 0, 0, 0))
    ImageDraw.Draw(img).rounded_rectangle([pad, pad, pad + card_w, pad + card_h], radius=radius, fill=hex_to_rgba(shadow_hex, shadow_opacity))
    img = img.filter(ImageFilter.GaussianBlur(blur_radius))
    ImageDraw.Draw(img).rounded_rectangle([pad, pad, pad + card_w, pad + card_h], radius=radius, fill=hex_to_rgba(card_hex, 255))
    return img

top = tk.Frame(root, pady=12); top.pack(fill="x")
tk.Label(top, text="Master Attendance Server", font=("Helvetica", 28, "bold")).pack(side="left", padx=20)
center = tk.Frame(root); center.pack(expand=True)
grid = tk.Frame(center); grid.pack()

cards = []
def make_card(parent, icon_text, label_text, command, card_w=300, card_h=170):
    theme = THEMES[current_theme]
    normal_tk = ImageTk.PhotoImage(make_outer_shadow_image(card_w, card_h, card_hex=theme["card_bg"], shadow_hex=theme["shadow"], blur_radius=theme["shadow_blur"], shadow_opacity=theme["shadow_opacity"]))
    
    shadow_lbl = tk.Label(parent, image=normal_tk, bd=0, bg=theme["bg"], cursor="hand2")
    shadow_lbl.image = normal_tk; shadow_lbl.pack_propagate(False)

    inner = tk.Frame(shadow_lbl, bg=theme["card_bg"], cursor="hand2")
    inner.place(relx=0.5, rely=0.5, anchor="center")
    tk.Label(inner, text=icon_text, font=("Segoe UI Emoji", 36), bg=theme["card_bg"], fg=theme["card_fg"]).pack(pady=(0, 6))
    tk.Label(inner, text=label_text, font=("Segoe UI", 14, "bold"), bg=theme["card_bg"], fg=theme["card_fg"]).pack()

    for w in (shadow_lbl, inner, inner.winfo_children()[0], inner.winfo_children()[1]): w.bind("<Button-1>", lambda ev: command())
    cards.append({"shadow_lbl": shadow_lbl, "inner": inner, "w": card_w, "h": card_h}); return shadow_lbl

def apply_theme():
    theme = THEMES[current_theme]
    root.configure(bg=theme["bg"]); top.configure(bg=theme["bg"]); center.configure(bg=theme["bg"]); grid.configure(bg=theme["bg"]); bot.configure(bg=theme["bg"])
    for widget in top.winfo_children(): widget.configure(bg=theme["bg"], fg=theme["fg"])
    for cinfo in cards:
        normal_tk = ImageTk.PhotoImage(make_outer_shadow_image(cinfo["w"], cinfo["h"], card_hex=theme["card_bg"], shadow_hex=theme.get("shadow", "#000"), blur_radius=theme.get("shadow_blur", 28), shadow_opacity=theme.get("shadow_opacity", 110)))
        cinfo["shadow_lbl"].configure(image=normal_tk, bg=theme["bg"]); cinfo["shadow_lbl"].image = normal_tk
        cinfo["inner"].configure(bg=theme["card_bg"])
        for child in cinfo["inner"].winfo_children(): child.configure(bg=theme["card_bg"], fg=theme["card_fg"])

make_card(grid, "📝", "Enroll Student", open_enrollment).grid(row=0, column=0, padx=15, pady=15)
make_card(grid, "👥", "Attendance Hub", open_attendance).grid(row=0, column=1, padx=15, pady=15)
make_card(grid, "📊", "Global Reports", open_reports).grid(   row=1, column=0, padx=15, pady=15)
make_card(grid, "🛠", "Tools & Update", open_tools_window).grid(row=1, column=1, padx=15, pady=15)

bot = tk.Frame(root, pady=10); bot.pack(fill="x", side="bottom")

def toggle_theme():
    global current_theme
    current_theme = "light" if current_theme == "dark" else "dark"
    apply_theme()

tk.Label(bot, text="Global Config", fg="#B0B0B0", font=("Segoe UI", 11), bg="#121212").pack(side="left", padx=20)
tk.Button(bot, text="Toggle Theme", command=toggle_theme, bg="#555", fg="white", font=("Segoe UI", 11, "bold"), bd=0, padx=10, pady=5).pack(side="right", padx=15)
tk.Button(bot, text="Exit Server", command=lambda: [cam_manager.stop(), root.destroy()], bg="#d32f2f", fg="white", font=("Segoe UI", 11, "bold"), bd=0, padx=10, pady=5).pack(side="right", padx=15)

apply_theme(); root.mainloop()