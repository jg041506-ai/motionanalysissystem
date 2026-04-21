import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
import threading
import queue
from scipy.signal import savgol_filter
import time
import os
import sys
import shutil
import urllib.request
import uuid
import json
from datetime import datetime, timedelta
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.gridspec import GridSpec
from PIL import Image, ImageTk
import textwrap
import math
import platform
import webbrowser
import matplotlib.font_manager as _fm

def _register_noto_fonts():
    if getattr(sys, 'frozen', False):
        _base = os.path.dirname(sys.executable)
    else:
        _base = os.path.dirname(os.path.abspath(__file__))
    _font_files = {
        'NotoSansCJK-Regular.otf': ['Noto Sans CJK KR', 'Noto Sans CJK JP'],
        'NotoSans-Regular.ttf':    ['Noto Sans'],
    }
    _loaded = []
    for _fname, _names in _font_files.items():
        _fpath = os.path.join(_base, 'fonts', _fname)
        if os.path.exists(_fpath):
            _fm.fontManager.addfont(_fpath)
            _loaded.extend(_names)
    return _loaded

_loaded_fonts = _register_noto_fonts()

plt.rcParams['pdf.fonttype'] = 42   
plt.rcParams['ps.fonttype']  = 42
plt.rcParams['font.family']  = 'sans-serif'

if _loaded_fonts:
    if platform.system() == 'Windows':
        plt.rcParams['font.sans-serif'] = _loaded_fonts + ['Malgun Gothic', 'Microsoft YaHei', 'Meiryo', 'sans-serif']
    elif platform.system() == 'Darwin':
        plt.rcParams['font.sans-serif'] = _loaded_fonts + ['AppleGothic', 'PingFang SC', 'sans-serif']
    else:
        plt.rcParams['font.sans-serif'] = _loaded_fonts + ['sans-serif']
else:
    if platform.system() == 'Windows':
        plt.rcParams['font.sans-serif'] = ['Malgun Gothic', 'Microsoft YaHei', 'SimHei', 'Meiryo', 'MS Gothic', 'sans-serif']
    elif platform.system() == 'Darwin':
        plt.rcParams['font.sans-serif'] = ['AppleGothic', 'PingFang SC', 'STHeiti', 'Hiragino Sans', 'sans-serif']

plt.rcParams['axes.unicode_minus'] = False

def get_base_dir():
    if getattr(sys, 'frozen', False):
        return os.path.dirname(sys.executable)
    return os.path.dirname(os.path.abspath(__file__))

def calculate_angle(p1, p2, p3):
    if p1 is None or p2 is None or p3 is None: return np.nan
    if np.isnan(p1).any() or np.isnan(p2).any() or np.isnan(p3).any(): return np.nan
    v1, v2 = p1 - p2, p3 - p2
    l1, l2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if l1 == 0 or l2 == 0: return np.nan
    return np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (l1 * l2), -1.0, 1.0)))

def get_midpoint(p1_x, p1_y, p2_x, p2_y):
    if p1_x is None or p2_x is None or np.isnan(p1_x) or np.isnan(p2_x): return np.nan, np.nan
    return (p1_x + p2_x) / 2.0, (p1_y + p2_y) / 2.0

# ==========================================
# 메인 분석 클래스
# ==========================================
class BiomechanicsAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Markerless Dynamic Motion Analysis System")
        self.root.geometry("650x900")

        self.base_dir = get_base_dir()
        self.video_dir = os.path.join(self.base_dir, "Video")
        self.result_dir = os.path.join(self.base_dir, "Result")
        self.excel_dir = os.path.join(self.base_dir, "Result_excel")
        self.pdf_dir = os.path.join(self.base_dir, "Result_pdf")
        self.cache_dir = os.path.join(self.base_dir, "Cache")
        self.image_dir = os.path.join(self.base_dir, "Image")
        self.setting_dir = os.path.join(self.base_dir, "Setting")
        self.lang_dir = os.path.join(self.setting_dir, "Language")
        self.config_file = os.path.join(self.cache_dir, "config.json")
        
        # Individual 셋팅 폴더 생성
        self.indiv_dir = os.path.join(self.setting_dir, "Individual")

        for d in [self.video_dir, self.result_dir, self.excel_dir, self.pdf_dir, self.cache_dir, self.image_dir, self.lang_dir, self.indiv_dir]:
            os.makedirs(d, exist_ok=True)
            
        for tier in ["High", "Mid", "Low"]:
            os.makedirs(os.path.join(self.setting_dir, tier), exist_ok=True)

        self.custom_analyses_data = {}
        self.load_custom_analyses()

        if not os.listdir(self.lang_dir):
            template = {
                "System": {
                    "tab_client_info": "Personal Info",
                    "tab_body_landmarks": "Body Landmarks",
                    "tab_tools_tracking": "Tools Tracking",
                    "tab_sport_analysis": "Sport Analysis",
                    "tab_compare_data": "Compare Data",
                    "tab_setting": "Setting",
                    "lbl_client_name": "Name:",
                    "lbl_date": "Date:",
                    "lbl_examiner": "Examiner:",
                    "btn_save": "Save",
                    "btn_delete": "Delete",
                    "btn_run": "Run Program (Select Video & Analyze)",
                    "btn_apply": "Apply Settings"
                },
                "Output": {
                    "pdf_main_title": "Biomechanics Analysis Report",
                    "pdf_compare_title": "Pre-Test vs Post-Test Comparison Report",
                    "lbl_name": "Name:",
                    "lbl_date": "Date:",
                    "lbl_examiner": "Examiner:",
                    "lbl_signature": "Signature:",
                    "status_pass": "PASS (Good Posture)",
                    "status_fail": "FAIL (Error Detected)"
                }
            }
            with open(os.path.join(self.lang_dir, "en.json"), "w", encoding="utf-8") as f:
                json.dump(template, f, indent=4)
            readme_text = "To add a new language, create a .json file here (e.g., ko.json). Format:\n{\n  \"System\": {\"key\": \"value\"},\n  \"Output\": {\"key\": \"value\"}\n}"
            with open(os.path.join(self.lang_dir, "README_Language.txt"), "w", encoding="utf-8") as f:
                f.write(readme_text)

        self.config = {"perf": "High", "sys_lang": "en", "out_lang": "en", "input_mode": "Video"}
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, "r", encoding="utf-8") as f:
                    self.config.update(json.load(f))
            except (json.JSONDecodeError, IOError, OSError):
                pass

        self.sys_lang_data = {}
        self.out_lang_data = {}
        self.load_all_language_files()

        self.status_var = tk.StringVar(value="Waiting to load models...")
        tk.Label(root, textvariable=self.status_var, fg="blue", font=("Helvetica", 10, "bold")).pack(pady=10)
        self.progress = ttk.Progressbar(root, orient='horizontal', length=550, mode='determinate')
        self.progress.pack(pady=5)

        self.checkbox_vars = {}
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.tab_info = ttk.Frame(self.notebook)
        self.tab_body = ttk.Frame(self.notebook)
        self.tab_tools = ttk.Frame(self.notebook)
        self.tab_analysis = ttk.Frame(self.notebook)
        self.tab_compare = ttk.Frame(self.notebook)
        self.tab_setting = ttk.Frame(self.notebook)

        self.notebook.add(self.tab_info, text="  Personal Info  ")
        self.notebook.add(self.tab_body, text="  Body Landmarks  ")
        self.notebook.add(self.tab_tools, text="  Tools Tracking  ")
        self.notebook.add(self.tab_analysis, text="  Sport Analysis  ")
        self.notebook.add(self.tab_compare, text="  Compare Data  ")
        self.notebook.add(self.tab_setting, text="  Setting  ")

        info_frame = tk.Frame(self.tab_info, padx=20, pady=20)
        info_frame.pack(fill='both', expand=True)

        self.lbl_client_name = tk.Label(info_frame, text="Name:", font=("Helvetica", 10, "bold"))
        self.lbl_client_name.grid(row=0, column=0, sticky='w', pady=10)
        self.clients_file = os.path.join(self.cache_dir, "clients.txt")
        self.clients_list = self.load_clients()
        self.client_name_var = tk.StringVar(value="None")

        self.client_combo = ttk.Combobox(info_frame, textvariable=self.client_name_var, values=["None"] + self.clients_list, width=27)
        self.client_combo.grid(row=0, column=1, pady=10, sticky='w')

        btn_frame_client = tk.Frame(info_frame)
        btn_frame_client.grid(row=0, column=2, padx=10)
        self.btn_save_client = tk.Button(btn_frame_client, text="Save", command=self.add_client, bg="lightblue")
        self.btn_save_client.pack(side='left', padx=2)
        self.btn_delete_client = tk.Button(btn_frame_client, text="Delete", command=self.delete_client, bg="lightblue")
        self.btn_delete_client.pack(side='left', padx=2)

        self.lbl_date = tk.Label(info_frame, text="Date:", font=("Helvetica", 10, "bold"))
        self.lbl_date.grid(row=1, column=0, sticky='w', pady=10)
        date_frame = tk.Frame(info_frame)
        date_frame.grid(row=1, column=1, pady=10, sticky='w')

        self.date_yy_var = tk.StringVar(value="00")
        self.date_mm_var = tk.StringVar(value="00")
        self.date_dd_var = tk.StringVar(value="00")

        tk.Entry(date_frame, textvariable=self.date_yy_var, width=4, justify='center').pack(side='left')
        tk.Label(date_frame, text=".", font=("Helvetica", 11, "bold")).pack(side='left', padx=2)
        tk.Entry(date_frame, textvariable=self.date_mm_var, width=3, justify='center').pack(side='left')
        tk.Label(date_frame, text=".", font=("Helvetica", 11, "bold")).pack(side='left', padx=2)
        tk.Entry(date_frame, textvariable=self.date_dd_var, width=3, justify='center').pack(side='left')

        self.lbl_examiner = tk.Label(info_frame, text="Examiner:", font=("Helvetica", 10, "bold"))
        self.lbl_examiner.grid(row=2, column=0, sticky='w', pady=10)
        self.examiners_file = os.path.join(self.cache_dir, "examiners.txt")
        self.examiners_list = self.load_examiners()
        self.examiner_var = tk.StringVar(value="None")

        self.examiner_combo = ttk.Combobox(info_frame, textvariable=self.examiner_var, values=["None"] + self.examiners_list, width=27)
        self.examiner_combo.grid(row=2, column=1, pady=10, sticky='w')

        btn_frame_ex = tk.Frame(info_frame)
        btn_frame_ex.grid(row=2, column=2, padx=10)
        self.btn_save_ex = tk.Button(btn_frame_ex, text="Save", command=self.add_examiner, bg="lightblue")
        self.btn_save_ex.pack(side='left', padx=2)
        self.btn_delete_ex = tk.Button(btn_frame_ex, text="Delete", command=self.delete_examiner, bg="lightblue")
        self.btn_delete_ex.pack(side='left', padx=2)

        self.body_options_frame = tk.Frame(self.tab_body)
        self.body_options_frame.pack(side='top', fill='x', padx=5, pady=5)
        
        self.use_mediapipe = tk.BooleanVar(value=True)
        self.use_yolo_seg = tk.BooleanVar(value=False)
        self.use_yolo_pose = tk.BooleanVar(value=False)
        
        ttk.Checkbutton(self.body_options_frame, text="MediaPipe Pose", variable=self.use_mediapipe).pack(side='left', padx=5)
        ttk.Checkbutton(self.body_options_frame, text="YOLO Segmentation", variable=self.use_yolo_seg).pack(side='left', padx=5)
        ttk.Checkbutton(self.body_options_frame, text="YOLO Pose", variable=self.use_yolo_pose).pack(side='left', padx=5)

        self.landmark_names = {
            0: "0. Nose", 1: "1. L Eye Inner", 2: "2. L Eye", 3: "3. L Eye Outer",
            4: "4. R Eye Inner", 5: "5. R Eye", 6: "6. R Eye Outer", 7: "7. L Ear", 8: "8. R Ear",
            9: "9. Mouth L", 10: "10. Mouth R", 11: "11. L Shoulder", 12: "12. R Shoulder",
            13: "13. L Elbow", 14: "14. R Elbow", 15: "15. L Wrist", 16: "16. R Wrist",
            17: "17. L Pinky", 18: "18. R Pinky", 19: "19. L Index", 20: "20. R Index",
            21: "21. L Thumb", 22: "22. R Thumb", 23: "23. L Hip", 24: "24. R Hip",
            25: "25. L Knee", 26: "26. R Knee", 27: "27. L Ankle", 28: "28. R Ankle",
            29: "29. L Heel", 30: "30. R Heel", 31: "31. L Foot Index", 32: "32. R Foot Index"
        }

        canvas = tk.Canvas(self.tab_body, borderwidth=0, relief='flat')
        scrollbar = ttk.Scrollbar(self.tab_body, orient='vertical', command=canvas.yview)
        self.scrollable_frame = ttk.Frame(canvas)
        self.scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        scrollbar.pack(side='right', fill='y')

        for key, name in self.landmark_names.items():
            self.checkbox_vars[key] = tk.BooleanVar(value=False)
            tk.Checkbutton(self.scrollable_frame, text=name, variable=self.checkbox_vars[key]).pack(anchor='w', padx=5, pady=2)

        tk.Label(self.tab_tools, text="Select Object(s) for Tracking:", font=("Helvetica", 11, "bold"), fg="blue").pack(pady=10)

        self.tool_vars = {}
        tool_items = [
            ("Ball", "Ball"), ("Baseball_Bat", "Baseball Bat"),
            ("Tennis_Racket", "Tennis Racket"), ("Golf_Club", "Golf Club")
        ]
        for key, label in tool_items:
            var = tk.BooleanVar(value=False)
            self.tool_vars[key] = var
            tk.Checkbutton(self.tab_tools, text=label, variable=var, font=("Helvetica", 10)).pack(anchor='w', padx=40, pady=2)

        tk.Label(self.tab_analysis, text="1. Select Sport:", font=("Helvetica", 11, "bold")).pack(anchor='w', padx=20, pady=(10, 5))
        
        self.sport_var = tk.StringVar(value="Workout")
        self.sport_combo = ttk.Combobox(self.tab_analysis, textvariable=self.sport_var, values=["Golf", "Baseball", "Workout", "Individual"], state="readonly")
        self.sport_combo.pack(fill='x', padx=20)

        tk.Label(self.tab_analysis, text="2. Select Movement/Fault to Track:", font=("Helvetica", 11, "bold")).pack(anchor='w', padx=20, pady=(10, 5))
        
        self.golf_analyses = [
            "C-Posture", "Chicken Wing", "Early Extension", "Flat Shoulder Plane", 
            "Hanging Back", "Loss of Posture", "Reverse Spine Angle", 
            "S-Posture", "Slide", "Sway", "Free"
        ]
        self.baseball_analyses = ["Dead Hand", "Drifting", "Flying Elbow", "Hanging Back", "Loss of Posture", "Push", "Sway", "Free"]
        self.workout_analyses = ["Squat", "Sit to Stand", "Gait", "Timed Up and Go", "Static Balance", "ROM", "Spinal Alignment", "Free"]

        self.analysis_var = tk.StringVar(value="Free")
        self.analysis_combo = ttk.Combobox(self.tab_analysis, textvariable=self.analysis_var, values=self.workout_analyses, state="readonly")
        self.analysis_combo.pack(fill='x', padx=20)

        self.img_preview_frame = tk.Frame(self.tab_analysis, bg="white", relief="sunken", borderwidth=1)
        self.img_preview_frame.pack(fill='both', expand=True, padx=20, pady=15)
        
        self.img_label = tk.Label(self.img_preview_frame, text="Select an analysis to view reference posture", bg="white", fg="gray")
        self.img_label.pack(expand=True)

        def update_analysis_options(event):
            if self.sport_var.get() == "Golf":
                self.analysis_combo['values'] = self.golf_analyses
            elif self.sport_var.get() == "Baseball":
                self.analysis_combo['values'] = self.baseball_analyses
            elif self.sport_var.get() == "Workout":
                self.analysis_combo['values'] = self.workout_analyses
            elif self.sport_var.get() == "Individual":
                self.load_custom_analyses() 
                custom_names = list(self.custom_analyses_data.keys())
                self.analysis_combo['values'] = custom_names if custom_names else ["No Custom Files"]
            
            self.analysis_var.set("Free")
            self.update_reference_image()

        self.sport_combo.bind("<<ComboboxSelected>>", update_analysis_options)
        self.analysis_combo.bind("<<ComboboxSelected>>", self.update_reference_image)

        tk.Label(self.tab_compare, text="Compare Pre/Post Excel Data", font=("Helvetica", 11, "bold"), fg="blue").pack(pady=(15, 5))

        file_frame = tk.Frame(self.tab_compare)
        file_frame.pack(fill='x', padx=20, pady=5)

        self.file1_var = tk.StringVar()
        self.file2_var = tk.StringVar()

        tk.Label(file_frame, text="Pre-Test Excel:").grid(row=0, column=0, sticky='w', pady=5)
        tk.Entry(file_frame, textvariable=self.file1_var, width=35).grid(row=0, column=1, padx=5, pady=5)
        tk.Button(file_frame, text="Browse", command=lambda: self.file1_var.set(filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")]))).grid(row=0, column=2, pady=5)

        tk.Label(file_frame, text="Post-Test Excel:").grid(row=1, column=0, sticky='w', pady=5)
        tk.Entry(file_frame, textvariable=self.file2_var, width=35).grid(row=1, column=1, padx=5, pady=5)
        tk.Button(file_frame, text="Browse", command=lambda: self.file2_var.set(filedialog.askopenfilename(filetypes=[("Excel", "*.xlsx")]))).grid(row=1, column=2, pady=5)

        # --- Subtitle Input Field Added Here ---
        subtitle_frame = tk.Frame(self.tab_compare)
        subtitle_frame.pack(fill='x', padx=20, pady=0)
        self.compare_subtitle_var = tk.StringVar()
        tk.Label(subtitle_frame, text="Custom Subtitle (Optional):").grid(row=0, column=0, sticky='w', pady=5)
        tk.Entry(subtitle_frame, textvariable=self.compare_subtitle_var, width=35).grid(row=0, column=1, padx=5, pady=5)
        # ---------------------------------------

        tk.Button(self.tab_compare, text="Generate Comparison PDF", command=self.generate_comparison_pdf, height=2, bg='lightblue', font=("Helvetica", 10, "bold")).pack(pady=10, fill='x', padx=40)

        tk.Label(self.tab_compare, text="Select Metrics to Compare:", font=("Helvetica", 10, "bold")).pack(anchor='w', padx=20, pady=(5, 5))
        
        canvas_comp = tk.Canvas(self.tab_compare, borderwidth=0, relief='flat')
        scrollbar_comp = ttk.Scrollbar(self.tab_compare, orient='vertical', command=canvas_comp.yview)
        self.scrollable_comp = ttk.Frame(canvas_comp)
        self.scrollable_comp.bind("<Configure>", lambda e: canvas_comp.configure(scrollregion=canvas_comp.bbox("all")))
        canvas_comp.create_window((0, 0), window=self.scrollable_comp, anchor='nw')
        canvas_comp.configure(yscrollcommand=scrollbar_comp.set)
        
        canvas_comp.pack(side='left', fill='both', expand=True, padx=20, pady=5)
        scrollbar_comp.pack(side='right', fill='y')

        self.compare_vars = {}

        header_frame1 = tk.Frame(self.scrollable_comp)
        header_frame1.grid(row=0, column=0, columnspan=2, sticky='w', pady=(5, 2))
        tk.Label(header_frame1, text="[ Basic Metrics ]", font=("Helvetica", 9, "bold")).pack(side='left')
        basic_metrics = ["Total Time (sec)", "Rep Count", "Max Body Speed (px/s)", "Max Measured Parameter", "Avg Step Width (px)", "Step Count"]
        tk.Button(header_frame1, text="Select All", command=lambda: self.set_metrics(basic_metrics, True)).pack(side='left', padx=5)
        tk.Button(header_frame1, text="Deselect All", command=lambda: self.set_metrics(basic_metrics, False)).pack(side='left', padx=2)
        
        for i, m in enumerate(basic_metrics):
            var = tk.BooleanVar(value=(i==0))
            self.compare_vars[m] = var
            tk.Checkbutton(self.scrollable_comp, text=m, variable=var).grid(row=1+(i//2), column=i%2, sticky='w', padx=10, pady=2)

        cur_row = 1 + (len(basic_metrics)//2) + 1
        header_frame2 = tk.Frame(self.scrollable_comp)
        header_frame2.grid(row=cur_row, column=0, columnspan=2, sticky='w', pady=(10, 2))
        tk.Label(header_frame2, text="[ Joint ROM (Range of Motion) ]", font=("Helvetica", 9, "bold")).pack(side='left')
        rom_metrics = ["Neck ROM", "Trunk ROM", "L Shoulder ROM", "R Shoulder ROM", "L Elbow ROM", "R Elbow ROM", 
                       "L Wrist ROM", "R Wrist ROM", "L Hip ROM", "R Hip ROM", "L Knee ROM", "R Knee ROM", "L Ankle ROM", "R Ankle ROM"]
        tk.Button(header_frame2, text="Select All", command=lambda: self.set_metrics(rom_metrics, True)).pack(side='left', padx=5)
        tk.Button(header_frame2, text="Deselect All", command=lambda: self.set_metrics(rom_metrics, False)).pack(side='left', padx=2)

        for i, m in enumerate(rom_metrics):
            var = tk.BooleanVar(value=False)
            self.compare_vars[m] = var
            tk.Checkbutton(self.scrollable_comp, text=m, variable=var).grid(row=cur_row+1+(i//2), column=i%2, sticky='w', padx=10, pady=2)

        cur_row += 1 + (len(rom_metrics)//2) + 1
        header_frame3 = tk.Frame(self.scrollable_comp)
        header_frame3.grid(row=cur_row, column=0, columnspan=2, sticky='w', pady=(10, 2))
        tk.Label(header_frame3, text="[ Max Joint Speed (px/s) ]", font=("Helvetica", 9, "bold")).pack(side='left')
        speed_metrics = ["Neck Max Speed", "Trunk Max Speed", "L Shoulder Max Speed", "R Shoulder Max Speed", "L Elbow Max Speed", "R Elbow Max Speed", 
                         "L Wrist Max Speed", "R Wrist Max Speed", "L Hip Max Speed", "R Hip Max Speed", "L Knee Max Speed", "R Knee Max Speed", "L Ankle Max Speed", "R Ankle Max Speed"]
        tk.Button(header_frame3, text="Select All", command=lambda: self.set_metrics(speed_metrics, True)).pack(side='left', padx=5)
        tk.Button(header_frame3, text="Deselect All", command=lambda: self.set_metrics(speed_metrics, False)).pack(side='left', padx=2)

        for i, m in enumerate(speed_metrics):
            var = tk.BooleanVar(value=False)
            self.compare_vars[m] = var
            tk.Checkbutton(self.scrollable_comp, text=m.replace(" Max Speed", " Speed"), variable=var).grid(row=cur_row+1+(i//2), column=i%2, sticky='w', padx=10, pady=2)

        cur_row += 1 + (len(speed_metrics)//2) + 1
        header_frame4 = tk.Frame(self.scrollable_comp)
        header_frame4.grid(row=cur_row, column=0, columnspan=2, sticky='w', pady=(10, 2))
        tk.Label(header_frame4, text="[ Avg Joint Speed (px/s) ]", font=("Helvetica", 9, "bold")).pack(side='left')
        avg_speed_metrics = ["Neck Avg Speed", "Trunk Avg Speed", "L Shoulder Avg Speed", "R Shoulder Avg Speed", "L Elbow Avg Speed", "R Elbow Avg Speed", 
                             "L Wrist Avg Speed", "R Wrist Avg Speed", "L Hip Avg Speed", "R Hip Avg Speed", "L Knee Avg Speed", "R Knee Avg Speed", "L Ankle Avg Speed", "R Ankle Avg Speed"]
        tk.Button(header_frame4, text="Select All", command=lambda: self.set_metrics(avg_speed_metrics, True)).pack(side='left', padx=5)
        tk.Button(header_frame4, text="Deselect All", command=lambda: self.set_metrics(avg_speed_metrics, False)).pack(side='left', padx=2)

        for i, m in enumerate(avg_speed_metrics):
            var = tk.BooleanVar(value=False)
            self.compare_vars[m] = var
            tk.Checkbutton(self.scrollable_comp, text=m, variable=var).grid(row=cur_row+1+(i//2), column=i%2, sticky='w', padx=10, pady=2)

        self.current_st_var = tk.StringVar()
        tk.Label(self.tab_setting, textvariable=self.current_st_var, font=("Helvetica", 10, "bold"), fg="darkgreen").pack(pady=(20, 10))
        ttk.Separator(self.tab_setting, orient='horizontal').pack(fill='x', padx=20, pady=10)

        tk.Label(self.tab_setting, text="Performance Tier:", font=("Helvetica", 10, "bold")).pack()
        self.perf_var = tk.StringVar(value=self.config["perf"])
        ttk.Combobox(self.tab_setting, textvariable=self.perf_var, values=["High", "Mid", "Low"], state="readonly", width=27).pack(pady=5)

        lang_list = [f.replace('.json', '') for f in os.listdir(self.lang_dir) if f.endswith('.json')]
        if not lang_list: lang_list = ["en"]

        tk.Label(self.tab_setting, text="System Language (UI):", font=("Helvetica", 10, "bold")).pack(pady=(15, 5))
        self.sys_lang_var = tk.StringVar(value=self.config["sys_lang"])
        ttk.Combobox(self.tab_setting, textvariable=self.sys_lang_var, values=lang_list, state="readonly", width=27).pack(pady=5)

        tk.Label(self.tab_setting, text="Output Language (PDF):", font=("Helvetica", 10, "bold")).pack(pady=(15, 5))
        self.out_lang_var = tk.StringVar(value=self.config.get("out_lang", "en"))
        ttk.Combobox(self.tab_setting, textvariable=self.out_lang_var, values=lang_list, state="readonly", width=27).pack(pady=5)

        tk.Label(self.tab_setting, text="Input Source:", font=("Helvetica", 10, "bold")).pack(pady=(15, 5))
        self.input_mode_var = tk.StringVar(value=self.config.get("input_mode", "Video File"))
        
        # 위에서 만든 함수를 통해 현재 연결된 카메라 목록을 동적으로 가져옵니다.
        detected_inputs = self.get_available_cameras()
        
        # values에 검색된 목록(detected_inputs)을 넣어줍니다.
        self.input_combo = ttk.Combobox(self.tab_setting, textvariable=self.input_mode_var, values=detected_inputs, state="readonly", width=27)
        self.input_combo.pack(pady=5)

        self.btn_apply = tk.Button(self.tab_setting, text="Apply Settings", command=self.apply_settings, bg="lightblue", font=("Helvetica", 10, "bold"))
        self.btn_apply.pack(pady=20)

        bottom_frame = tk.Frame(root)
        bottom_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=10, padx=20)

        tk.Button(bottom_frame, text="Select All Body", command=lambda: [self.checkbox_vars[k].set(True) for k in self.landmark_names.keys()], height=1).pack(side=tk.TOP, fill=tk.X, pady=2)
        tk.Button(bottom_frame, text="Deselect All Body", command=lambda: [self.checkbox_vars[k].set(False) for k in self.landmark_names.keys()], height=1).pack(side=tk.TOP, fill=tk.X, pady=2)

        self.btn_run = tk.Button(bottom_frame, text="Run Program (Select Video & Analyze)", command=lambda: self.start_analysis(mode="video"), height=2, bg='lightblue', font=("Helvetica", 10, "bold"))
        self.btn_run.pack(side=tk.TOP, fill=tk.X, pady=4)

        tk.Label(bottom_frame, text="Developer: Kim Seongjung | E-mail: jg041506@gmail.com", font=("Helvetica", 9), fg="gray").pack(side=tk.TOP, pady=(5, 0))

        ask_donate_lbl = tk.Label(bottom_frame, text="Ask & Donate", font=("Helvetica", 9, "underline"), fg="blue", cursor="hand2")
        ask_donate_lbl.pack(side=tk.TOP, pady=(2, 5))
        ask_donate_lbl.bind("<Button-1>", lambda e: webbrowser.open_new("http://ko-fi.com/jg041506"))

        disclaimer_lbl = tk.Label(bottom_frame, text="Disclaimer: Not intended for medical diagnosis.", font=("Helvetica", 8), fg="black")
        disclaimer_lbl.pack(side=tk.TOP, pady=(0, 5))

        self.frame_queue = queue.Queue(maxsize=256)
        self.is_running = False
        self.raw_data = []
        self.last_timestamp_ms = -1
        self.prev_crop_box = None
        self.init_address_coords = None 
        
        self.rep_count = 0
        self.step_count = 0       # [추가됨]
        self.gait_state = None    # [추가됨]
        self.workout_stage = "UP"
        self.is_moving = 0
        self.last_hip_x = None
        self.idle_frames = 0

        self.update_ui_text()
        self.root.after(100, self.create_detector)

    def load_custom_analyses(self):
        self.custom_analyses_data = {}
        if not os.path.exists(self.indiv_dir): return
        for filename in os.listdir(self.indiv_dir):
            if filename.endswith(".json"):
                try:
                    with open(os.path.join(self.indiv_dir, filename), "r", encoding="utf-8") as f:
                        data = json.load(f)
                        file_key = filename.replace(".json", "")
                        
                        # JSON이 배열(리스트) 형태인 경우
                        if isinstance(data, list):
                            self.custom_analyses_data[file_key] = data
                        # JSON이 딕셔너리 형태인 경우 (기존 호환성)
                        elif isinstance(data, dict):
                            name = data.get("analysis_name", file_key)
                            self.custom_analyses_data[name] = [data]
                except Exception as e:
                    print(f"Error loading custom analysis {filename}: {e}")

    def load_all_language_files(self):
        def get_lang(code):
            path = os.path.join(self.lang_dir, f"{code}.json")
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f: return json.load(f)
            return {}
        self.sys_lang_data = get_lang(self.config.get("sys_lang", "en"))
        self.out_lang_data = get_lang(self.config.get("out_lang", "en"))

    def update_ui_text(self):
        s = self.sys_lang_data.get("System", {})
        if not s: return
        self.notebook.tab(0, text=s.get("tab_client_info", "  Client Info  "))
        self.notebook.tab(1, text=s.get("tab_body_landmarks", "  Body Landmarks  "))
        self.notebook.tab(2, text=s.get("tab_tools_tracking", "  Tools Tracking  "))
        self.notebook.tab(3, text=s.get("tab_sport_analysis", "  Sport Analysis  "))
        self.notebook.tab(4, text=s.get("tab_compare_data", "  Compare Data  "))
        self.notebook.tab(5, text=s.get("tab_setting", "  Setting  "))
        
        self.lbl_client_name.config(text=s.get("lbl_client_name", "Client Name:"))
        self.lbl_date.config(text=s.get("lbl_date", "Date:"))
        self.lbl_examiner.config(text=s.get("lbl_examiner", "Examiner:"))
        self.btn_save_client.config(text=s.get("btn_save", "Save"))
        self.btn_delete_client.config(text=s.get("btn_delete", "Delete"))
        self.btn_save_ex.config(text=s.get("btn_save", "Save"))
        self.btn_delete_ex.config(text=s.get("btn_delete", "Delete"))
        
        self.btn_run.config(text=s.get("btn_run", "Run Program (Select Video & Analyze)"))
        self.btn_apply.config(text=s.get("btn_apply", "Apply Settings"))
        
        self.current_st_var.set(f"Perf: {self.config.get('perf', 'High')} | Sys Lang: {self.config.get('sys_lang', 'en')} | PDF Lang: {self.config.get('out_lang', 'en')}")

    def get_available_cameras(self):
        # 기본 옵션인 Video File을 먼저 넣습니다.
        inputs = ["Video File"]
        
        # 0~3번(총 4개)까지 체크하도록 range(4)로 수정했습니다.
        for i in range(4):
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                inputs.append(f"Webcam ({i})")
                cap.release()
                
        return inputs

    def apply_settings(self):
        new_perf = self.perf_var.get()
        new_input = self.input_mode_var.get() # 새로 추가됨
        
        # ---------- [여기에 새로 추가] ----------
        if "Webcam" in new_input and new_perf != "Low":
            if messagebox.askyesno("Performance Warning", "For Real-time Webcam tracking, 'Low' Performance Tier is strongly recommended to prevent lag.\n\nChange to 'Low' automatically?"):
                new_perf = "Low"
                self.perf_var.set("Low")
        # ----------------------------------------
        
        perf_changed = (self.config.get("perf") != new_perf)
        
        # 설정 저장 시 input_mode도 같이 저장되도록 교체
        self.config.update({
            "perf": new_perf, 
            "sys_lang": self.sys_lang_var.get(), 
            "out_lang": self.out_lang_var.get(),
            "input_mode": new_input
        })
        
        with open(self.config_file, "w", encoding="utf-8") as f: 
            json.dump(self.config, f)
            
        self.load_all_language_files()
        self.update_ui_text()
        
        if perf_changed:
            self.reload_models()
        else:
            messagebox.showinfo("Success", "Settings applied successfully.")

    # ---------- 팝업창 제어용 메서드 추가 ----------
    def start_webcam_cmd(self):
        self.is_recording = True
        if hasattr(self, 'btn_webcam_start'):
            self.btn_webcam_start.config(state=tk.DISABLED, text="▶ 녹화 및 분석 중... (Recording)")
        if hasattr(self, 'btn_webcam_stop'):
            self.btn_webcam_stop.config(state=tk.NORMAL)

    def stop_webcam_cmd(self):
        self.is_running = False
        self.is_recording = False

    # 함수 정의를 아래와 같이 변경합니다.
    def start_analysis(self, mode="video"):
        # [수정] 중복 실행 방지 및 이전 상태 완전 초기화
        if self.is_running:
            return

        # [추가] 다음 분석을 위해 주요 변수들 리셋
        self.is_webcam = False
        self.frames_processed = 0
        self.last_timestamp_ms = -1
        self.raw_data_chunk = []
        self.frame_queue = queue.Queue(maxsize=256) # 큐 초기화

        self.selected_keys = [k for k, v in self.checkbox_vars.items() if k in self.landmark_names and v.get()]

        # ... (중략: 기존 required_keys 추가 로직 유지) ...
                    
        if self.analysis_var.get() == "Spinal Alignment":
            for k in [11, 12, 23, 24]: 
                if k not in self.selected_keys: self.selected_keys.append(k)

        # ---------- [비디오 탐색기 띄우는 부분부터 캡처 초기화 전까지 아래 코드로 교체] ----------
        input_mode_setting = self.config.get("input_mode", "Video File")
        
        # mode가 "webcam"이거나 Setting에서 Webcam이 선택된 경우
        if mode == "webcam" or "Webcam" in input_mode_setting:
            self.is_webcam = True
            
            # 카메라 인덱스 설정 (0번 또는 1번)
            if mode == "webcam":
                self.input_source = 0
            else:
                self.input_source = int(input_mode_setting.split("(")[1].replace(")", ""))
                
            self.video_path = "webcam_capture.mp4" # 더미 경로

            # 팝업(컨트롤 패널) 창 생성
            self.is_recording = False
            self.webcam_ctrl_win = tk.Toplevel(self.root)
            self.webcam_ctrl_win.title("Webcam Control Panel")
            self.webcam_ctrl_win.geometry("350x150")
            self.webcam_ctrl_win.attributes('-topmost', True) 
            self.webcam_ctrl_win.protocol("WM_DELETE_WINDOW", self.stop_webcam_cmd)

            tk.Label(self.webcam_ctrl_win, text="Webcam Preview Active. Ready to analyze.", font=("Helvetica", 10, "bold")).pack(pady=10)

            self.btn_webcam_start = tk.Button(self.webcam_ctrl_win, text="▶ 시작 (Start Analysis)", bg="lightgreen", font=("Helvetica", 11, "bold"), height=2, command=self.start_webcam_cmd)
            self.btn_webcam_start.pack(fill='x', padx=20, pady=5)

            self.btn_webcam_stop = tk.Button(self.webcam_ctrl_win, text="⏹ 저장 및 종료 (Stop & Save)", bg="salmon", font=("Helvetica", 11, "bold"), height=2, state=tk.DISABLED, command=self.stop_webcam_cmd)
            self.btn_webcam_stop.pack(fill='x', padx=20, pady=5)
            
            # 메인 버튼 잠금
            self.btn_run.config(state=tk.DISABLED)
            if hasattr(self, 'btn_webcam_main'):
                self.btn_webcam_main.config(state=tk.DISABLED)
                
            self.total_frames = 0
            self.progress.configure(mode='indeterminate')
            self.progress.start()

        else:
            self.is_webcam = False
            self.is_recording = True
            self.video_path = filedialog.askopenfilename(initialdir=self.video_dir, filetypes=[("Video", "*.mp4 *.avi *.mov")])
            if not self.video_path: return
            
            self.btn_run.config(state=tk.DISABLED)
            if hasattr(self, 'btn_webcam_main'):
                self.btn_webcam_main.config(state=tk.DISABLED)
                
            self.progress.configure(mode='determinate', value=0)
        # ----------------------------------------------------------------------------------------

        self.is_running = True
        self.raw_data = []
        self.prev_crop_box = None
        self.init_address_coords = None
        
        # ... (이하 기존 폴더 생성 로직 등 유지) ...
        
        # ---------- [cap = cv2.VideoCapture 부분 교체] ----------
        if self.is_webcam:
            self.temp_in = self.input_source
            cap = cv2.VideoCapture(self.input_source)
            # 웹캠의 실제 FPS를 가져오거나 기본값 30.0 부여
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.width, self.height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            self.temp_out = os.path.join(self.cache_dir, "temp_out.mp4")
        else:
            self.temp_in = os.path.join(self.cache_dir, "temp_in.mp4")
            self.temp_out = os.path.join(self.cache_dir, "temp_out.mp4")
            shutil.copy2(self.video_path, self.temp_in)

            cap = cv2.VideoCapture(self.temp_in)
            self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 100
            self.width, self.height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

        # ===== 공통 초기화 =====
        self.is_running = True
        self.raw_data = []
        self.prev_crop_box = None
        self.init_address_coords = None
        self.last_hip_pos = None
        self.last_time_s = None

        client_name = self.client_name_var.get().strip()
        folder_name = client_name if client_name and client_name != "None" else "Unknown"

        self.current_result_dir = os.path.join(self.result_dir, folder_name)
        self.current_excel_dir = os.path.join(self.excel_dir, folder_name)
        self.current_pdf_dir = os.path.join(self.pdf_dir, folder_name)

        os.makedirs(self.current_result_dir, exist_ok=True)
        os.makedirs(self.current_excel_dir, exist_ok=True)
        os.makedirs(self.current_pdf_dir, exist_ok=True)

        self.rep_count = 0
        self.step_count = 0
        self.gait_state = None
        if self.analysis_var.get() == "Squat":
            self.workout_stage = "UP"
        elif self.analysis_var.get() == "Timed Up and Go":
            self.workout_stage = "SITTING"
        else:
            self.workout_stage = "STANDING"
            
        if hasattr(self, 'start_move_time'):
            del self.start_move_time

        self.tug_start_time = None       
        self.tug_end_time = None         
        self.tug_prev_knee_ang = None    
        self.tug_completed = False       

        self.is_moving = 0
        self.last_hip_x = None
        self.idle_frames = 0

        self.video_writer = cv2.VideoWriter(self.temp_out, cv2.VideoWriter_fourcc(*'mp4v'), self.fps, (self.width, self.height))

        self.frames_processed = 0          
        self.raw_data_chunk = []           
        self.temp_csv_path = os.path.join(self.cache_dir, "temp_data.csv")
        
        if os.path.exists(self.temp_csv_path):
            os.remove(self.temp_csv_path)  

        threading.Thread(target=self.video_reader_thread, daemon=True).start()
        threading.Thread(target=self.processing_thread, daemon=True).start()
        self.update_progress_loop()

    def reload_models(self, event=None):
        self.btn_run.config(state=tk.DISABLED)
        self.root.after(10, self.create_detector)

    def set_metrics(self, metric_list, state):
        for m in metric_list:
            if m in self.compare_vars:
                self.compare_vars[m].set(state)

    def compute_metric(self, df, metric_name):
        if metric_name == "Total Time (sec)": return df["Time(sec)"].max() if "Time(sec)" in df.columns else 0
        if metric_name == "Rep Count": return df["Rep_Count"].max() if "Rep_Count" in df.columns else 0
        if metric_name == "Max Body Speed (px/s)": return df["Body_Speed_px_s"].max() if "Body_Speed_px_s" in df.columns else 0
        if metric_name == "Max Measured Parameter": return df["Measured_Value"].max() if "Measured_Value" in df.columns else 0
        
        if metric_name == "Step Count":
            if "Step_Count" in df.columns and not df.empty:
                valid_steps = df["Step_Count"].dropna()
                return int(valid_steps.iloc[-1]) if not valid_steps.empty else 0
            return 0
        
        if metric_name == "Avg Step Width (px)":
            if "Step_Width_px" in df.columns:
                return df["Step_Width_px"].mean()
            elif all(c in df.columns for c in ["L_Ankle_X", "R_Ankle_X", "L_Ankle_Y", "R_Ankle_Y"]):
                return np.sqrt((df["L_Ankle_X"] - df["R_Ankle_X"])**2 + (df["L_Ankle_Y"] - df["R_Ankle_Y"])**2).mean()
            return 0.0

        def get_col(df, name, axis):
            if name == "Neck": name = "Nose" 
            if name == "Trunk" or name == "Mid_Shoulder":
                if f"L_Shoulder_{axis}" in df.columns and f"R_Shoulder_{axis}" in df.columns:
                    return (df[f"L_Shoulder_{axis}"] + df[f"R_Shoulder_{axis}"]) / 2
                return pd.Series(np.nan, index=df.index)
            if name == "Mid_Hip":
                if f"L_Hip_{axis}" in df.columns and f"R_Hip_{axis}" in df.columns:
                    return (df[f"L_Hip_{axis}"] + df[f"R_Hip_{axis}"]) / 2
                return pd.Series(np.nan, index=df.index)
            if name == "Mid_Knee":
                if f"L_Knee_{axis}" in df.columns and f"R_Knee_{axis}" in df.columns:
                    return (df[f"L_Knee_{axis}"] + df[f"R_Knee_{axis}"]) / 2
                return pd.Series(np.nan, index=df.index)
            col_name = f"{name}_{axis}"
            return df[col_name] if col_name in df.columns else pd.Series(np.nan, index=df.index)

        if metric_name.endswith(" ROM"):
            joint = metric_name.replace(" ROM", "")
            rom_map = {
                "Neck": ("Nose", "Mid_Shoulder", "Mid_Hip"),
                "Trunk": ("Mid_Shoulder", "Mid_Hip", "Mid_Knee"),
                "L Shoulder": ("L_Hip", "L_Shoulder", "L_Elbow"),
                "R Shoulder": ("R_Hip", "R_Shoulder", "R_Elbow"),
                "L Elbow": ("L_Shoulder", "L_Elbow", "L_Wrist"),
                "R Elbow": ("R_Shoulder", "R_Elbow", "R_Wrist"),
                "L Wrist": ("L_Elbow", "L_Wrist", "L_Index"),
                "R Wrist": ("R_Elbow", "R_Wrist", "R_Index"),
                "L Hip": ("L_Shoulder", "L_Hip", "L_Knee"),
                "R Hip": ("R_Shoulder", "R_Hip", "R_Knee"),
                "L Knee": ("L_Hip", "L_Knee", "L_Ankle"),
                "R Knee": ("R_Hip", "R_Knee", "R_Ankle"),
                "L Ankle": ("L_Knee", "L_Ankle", "L_Foot_Index"),
                "R Ankle": ("R_Knee", "R_Ankle", "R_Foot_Index")
            }
            if joint in rom_map:
                p1, p2, p3 = rom_map[joint]
                x1, y1 = get_col(df, p1, 'X'), get_col(df, p1, 'Y')
                x2, y2 = get_col(df, p2, 'X'), get_col(df, p2, 'Y')
                x3, y3 = get_col(df, p3, 'X'), get_col(df, p3, 'Y')
                
                v1_x, v1_y = x1 - x2, y1 - y2
                v2_x, v2_y = x3 - x2, y3 - y2
                
                dot = v1_x * v2_x + v1_y * v2_y
                mag1 = np.sqrt(v1_x**2 + v1_y**2)
                mag2 = np.sqrt(v2_x**2 + v2_y**2)
                denom = (mag1 * mag2).replace(0, np.nan)
                
                cos_ang = np.clip(dot / denom, -1.0, 1.0)
                ang_deg = np.degrees(np.arccos(cos_ang))
                
                if not ang_deg.isna().all():
                    return float(ang_deg.max() - ang_deg.min())
            return 0.0

        if metric_name.endswith(" Max Speed") or metric_name.endswith(" Avg Speed"):
            is_avg = metric_name.endswith(" Avg Speed")
            joint = metric_name.replace(" Avg Speed", "").replace(" Max Speed", "")
            x = get_col(df, joint, 'X')
            y = get_col(df, joint, 'Y')
            t = df["Time(sec)"] if "Time(sec)" in df.columns else None

            if t is not None and not x.isna().all() and not y.isna().all():
                dx = x.diff().fillna(0)
                dy = y.diff().fillna(0)
                dt = t.diff().fillna(1/30.0)
                dt[dt == 0] = 1/30.0 
                
                speed = np.sqrt(dx**2 + dy**2) / dt
                if not speed.isna().all():
                    return float(speed.mean()) if is_avg else float(speed.max())
            return 0.0
            
        return 0.0

    def generate_comparison_pdf(self):
        f1 = self.file1_var.get()
        f2 = self.file2_var.get()
        
        if not f1 or not f2 or not os.path.exists(f1) or not os.path.exists(f2):
            messagebox.showerror("Error", "Please select valid Pre and Post Excel files.")
            return
            
        try:
            df1 = pd.read_excel(f1)
            df2 = pd.read_excel(f2)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read Excel files: {e}")
            return
            
        selected_metrics = [m for m, var in self.compare_vars.items() if var.get()]
        if not selected_metrics:
            messagebox.showwarning("Warning", "Please select at least one metric to compare.")
            return
            
        results = []
        for m in selected_metrics:
            val1 = self.compute_metric(df1, m)
            val2 = self.compute_metric(df2, m)
            if val1 == 0 and val2 == 0: 
                continue
            diff = val2 - val1
            results.append({"name": m, "pre": val1, "post": val2, "diff": diff})
            
        if not results:
            messagebox.showinfo("Info", "No valid data found for the selected metrics.")
            return
            
        increased = [r for r in results if r["diff"] > 0]
        decreased = [r for r in results if r["diff"] <= 0]
        
        client_name = self.client_name_var.get().strip()
        folder_name = client_name if client_name and client_name != "None" else "Unknown"
        current_comp_pdf_dir = os.path.join(self.pdf_dir, folder_name)
        os.makedirs(current_comp_pdf_dir, exist_ok=True)
        
        out_name = f"Comparison_Report_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        out_path = os.path.join(current_comp_pdf_dir, out_name)
        
        try:
            date_str = f"{self.date_yy_var.get().strip()}.{self.date_mm_var.get().strip()}.{self.date_dd_var.get().strip()}"
        except AttributeError:
            date_str = "00.00.00"
            
        if not client_name or client_name == "None":
            client_name_str = "________________________"
        else:
            client_name_str = client_name
        
        examiner = self.examiner_var.get().strip()
        if not examiner or examiner == "None": examiner = "____________________"
        
        o = self.out_lang_data.get("Output", {})
        pdf_title = o.get("pdf_compare_title", "Pre-Test vs Post-Test Comparison Report")
        lbl_name = o.get("lbl_name", "Name:")
        lbl_date = o.get("lbl_date", "Date:")
        lbl_examiner = o.get("lbl_examiner", "Examiner:")
        lbl_signature = o.get("lbl_signature", "Signature:")

        # --- Subtitle Fetching ---
        custom_subtitle = self.compare_subtitle_var.get().strip()
        # -------------------------

        with PdfPages(out_path) as pdf:
            fig_sum = plt.figure(figsize=(8.27, 11.69))
            
            fig_sum.suptitle(pdf_title, fontsize=18, fontweight='bold', y=0.97)
            
            # --- Render Subtitle on Summary Page ---
            if custom_subtitle:
                fig_sum.text(0.5, 0.93, f"Analysis: {custom_subtitle}", fontsize=13, fontweight='bold', ha='center')
            # ---------------------------------------
            
            # 이름과 날짜를 부제목보다 아래(y=0.89, y=0.87)로 내림
            fig_sum.text(0.95, 0.89, f"{lbl_name} {client_name_str}", fontsize=12, fontweight='bold', ha='right', va='top')
            if date_str not in ["00.00.00", "0.0.0", "..", ""]:
                fig_sum.text(0.95, 0.87, f"{lbl_date} {date_str}", fontsize=10, ha='right', va='top')
                
            y_pos = 0.81 # 텍스트가 겹치지 않게 본문 시작 위치도 아래(0.85 -> 0.81)로 조정
            
            fig_sum.text(0.1, y_pos, "[ Increased Metrics ]", fontsize=13, fontweight='bold', color='darkgreen')
            y_pos -= 0.04
            if increased:
                for m in increased:
                    unit = "px/s" if "Speed" in m['name'] else ("sec" if "Time" in m['name'] else ("reps" if "Count" in m['name'] else "°/px"))
                    fig_sum.text(0.1, y_pos, f"• {m['name']}: {m['pre']:.1f} -> {m['post']:.1f} ( +{m['diff']:.1f} {unit} )", fontsize=11)
                    fig_sum.text(0.7, y_pos, "Positive (   )   Negative (   )", fontsize=10, fontweight='bold')
                    y_pos -= 0.03
            else:
                fig_sum.text(0.1, y_pos, "  None", fontsize=11)
                y_pos -= 0.03
                
            y_pos -= 0.03
            fig_sum.text(0.1, y_pos, "[ Decreased Metrics ]", fontsize=13, fontweight='bold', color='darkred')
            y_pos -= 0.04
            if decreased:
                for m in decreased:
                    unit = "px/s" if "Speed" in m['name'] else ("sec" if "Time" in m['name'] else ("reps" if "Count" in m['name'] else "°/px"))
                    fig_sum.text(0.1, y_pos, f"• {m['name']}: {m['pre']:.1f} -> {m['post']:.1f} ( {m['diff']:.1f} {unit} )", fontsize=11)
                    fig_sum.text(0.7, y_pos, "Positive (   )   Negative (   )", fontsize=10, fontweight='bold')
                    y_pos -= 0.03
            else:
                fig_sum.text(0.1, y_pos, "  None", fontsize=11)
                
            pdf.savefig(fig_sum)
            plt.close(fig_sum)
            
            charts_per_page = 6
            for i in range(0, len(results), charts_per_page):
                fig_c = plt.figure(figsize=(8.27, 11.69))
                
                fig_c.text(0.5, 0.97, f"Chart Data (Page {i//charts_per_page + 1})", fontsize=18, fontweight='bold', ha='center', va='top')
                
                if custom_subtitle:
                    fig_c.text(0.5, 0.93, f"Analysis: {custom_subtitle}", fontsize=13, fontweight='bold', ha='center')
                
                # 차트 시작 위치(top)를 아래로 약간 내려줌 (0.88 -> 0.85)
                gs = GridSpec(3, 2, figure=fig_c, top=0.85, bottom=0.1, wspace=0.3, hspace=0.4)
                
                page_results = results[i:i+charts_per_page]
                for j, m in enumerate(page_results):
                    r, c = j // 2, j % 2
                    ax_chart = fig_c.add_subplot(gs[r, c])
                    bars = ax_chart.bar(['Pre', 'Post'], [m['pre'], m['post']], color=['#4472C4', '#ED7D31'], edgecolor='black', alpha=0.8)
                    ax_chart.set_title(m['name'], fontweight='bold', fontsize=10)
                    ax_chart.grid(axis='y', linestyle='--', alpha=0.7)
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax_chart.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
                
                if i + charts_per_page >= len(results):
                    fig_c.text(0.10, 0.04, f"{lbl_examiner} {examiner}", fontsize=12, fontweight='bold')
                    fig_c.text(0.60, 0.04, f"{lbl_signature} ________________________", fontsize=12, fontweight='bold')
                
                pdf.savefig(fig_c)
                plt.close(fig_c)
                
        messagebox.showinfo("Success", f"Comparison PDF generated successfully!\nSaved at:\n{out_path}")

    def load_clients(self):
        if os.path.exists(self.clients_file):
            with open(self.clients_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        return []

    def save_clients(self):
        with open(self.clients_file, 'w', encoding='utf-8') as f:
            for cl in self.clients_list:
                f.write(cl + "\n")

    def add_client(self):
        name = self.client_name_var.get().strip()
        if name and name != "None" and name not in self.clients_list:
            self.clients_list.append(name)
            self.save_clients()
            self.client_combo['values'] = ["None"] + self.clients_list
            messagebox.showinfo("Saved", f"Client '{name}' saved.")

    def delete_client(self):
        name = self.client_name_var.get().strip()
        if name in self.clients_list:
            self.clients_list.remove(name)
            self.save_clients()
            self.client_combo['values'] = ["None"] + self.clients_list
            self.client_name_var.set("None")
            messagebox.showinfo("Deleted", f"Client '{name}' deleted.")

    def load_examiners(self):
        if os.path.exists(self.examiners_file):
            with open(self.examiners_file, 'r', encoding='utf-8') as f:
                return [line.strip() for line in f if line.strip()]
        return []

    def save_examiners(self):
        with open(self.examiners_file, 'w', encoding='utf-8') as f:
            for ex in self.examiners_list:
                f.write(ex + "\n")

    def add_examiner(self):
        name = self.examiner_var.get().strip()
        if name and name != "None" and name not in self.examiners_list:
            self.examiners_list.append(name)
            self.save_examiners()
            self.examiner_combo['values'] = ["None"] + self.examiners_list
            messagebox.showinfo("Saved", f"Examiner '{name}' saved.")

    def delete_examiner(self):
        name = self.examiner_var.get().strip()
        if name in self.examiners_list:
            self.examiners_list.remove(name)
            self.save_examiners()
            self.examiner_combo['values'] = ["None"] + self.examiners_list
            self.examiner_var.set("None")
            messagebox.showinfo("Deleted", f"Examiner '{name}' deleted.")

    def update_reference_image(self, event=None):
        sport = self.sport_var.get()
        analysis = self.analysis_var.get()
        
        if analysis in ["Free", "None", ""]:
            self.img_label.config(image='', text="Select an analysis to view reference posture")
            return

        safe_analysis = analysis.replace(" ", "").replace("-", "")
        extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG']
        img_path = None
        
        for ext in extensions:
            if sport == "Individual" and analysis in self.custom_analyses_data:
                # 데이터가 리스트 형태로 저장되어 있으므로 첫 번째 요소에서 가져오도록 수정
                analysis_data = self.custom_analyses_data[analysis]
                if isinstance(analysis_data, list) and len(analysis_data) > 0:
                    custom_img_name = analysis_data[0].get("image_file", "")
                else:
                    custom_img_name = ""
                
                # 파일명이 있을 때만 경로 병합
                test_path = os.path.join(self.indiv_dir, custom_img_name) if custom_img_name else ""
            else:
                test_path = os.path.join(self.image_dir, f"{sport}_{safe_analysis}{ext}")
                
            if test_path and os.path.exists(test_path):
                img_path = test_path
                break
                
        if img_path:
            try:
                img = Image.open(img_path)
                img.thumbnail((350, 250)) 
                photo = ImageTk.PhotoImage(img)
                self.img_label.config(image=photo, text="")
                self.img_label.image = photo 
            except Exception as e:
                self.img_label.config(image='', text=f"Failed to load image: {e}")
        else:
            expected_name = f"{sport}_{safe_analysis}.png"
            self.img_label.config(image='', text=f"No image found.\nPlease save an image named\n'{expected_name}'\ninside the 'Image' folder.")

    def create_detector(self):
        try:
            perf = self.config.get("perf", "High")
            yolo_map = {"High": "yolo26x.pt", "Mid": "yolo26m.pt", "Low": "yolo26n.pt"}
            yolo_seg_map = {"High": "yolo26x-seg.pt", "Mid": "yolo26m-seg.pt", "Low": "yolo26n-seg.pt"}
            yolo_pose_map = {"High": "yolo26x-pose.pt", "Mid": "yolo26m-pose.pt", "Low": "yolo26n-pose.pt"}
            mp_map = {"High": "pose_landmarker_heavy.task", "Mid": "pose_landmarker_full.task", "Low": "pose_landmarker_lite.task"}
            
            yolo_path = os.path.join(self.setting_dir, perf, yolo_map[perf])
            yolo_seg_path = os.path.join(self.setting_dir, perf, yolo_seg_map[perf])
            yolo_pose_path = os.path.join(self.setting_dir, perf, yolo_pose_map[perf])
            mp_path = os.path.join(self.setting_dir, perf, mp_map[perf])
            
            if not os.path.exists(mp_path):
                url = f"https://storage.googleapis.com/mediapipe-models/pose_landmarker/{mp_map[perf].split('.')[0]}/float16/1/{mp_map[perf]}"
                self.status_var.set(f"Downloading {mp_map[perf]}...")
                self.root.update()
                urllib.request.urlretrieve(url, mp_path)

            self.status_var.set(f"Loading {perf} YOLO & MediaPipe models...")
            self.root.update()
            
            if not os.path.exists(yolo_path):
                messagebox.showwarning("Missing File", f"{yolo_map[perf]} not found in Setting/{perf}/ folder.\nPlease make sure the file exists.")
                
            self.yolo_model = YOLO(yolo_path) if os.path.exists(yolo_path) else None
            self.yolo_seg_model = YOLO(yolo_seg_path) if os.path.exists(yolo_seg_path) else self.yolo_model
            self.yolo_pose_model = YOLO(yolo_pose_path) if os.path.exists(yolo_pose_path) else self.yolo_model

            with open(mp_path, 'rb') as f:
                model_data = f.read()
            options = vision.PoseLandmarkerOptions(
                base_options=python.BaseOptions(model_asset_buffer=model_data),
                running_mode=vision.RunningMode.IMAGE,  # [수정] VIDEO -> IMAGE 모드로 변경
                min_pose_detection_confidence=0.6,      # [추가] 검출 신뢰도 60% 이상일 때만 인식
                min_pose_presence_confidence=0.6,       # [추가] 존재 신뢰도 60% 이상일 때만 인식
                output_segmentation_masks=False
            )
            self.detector = vision.PoseLandmarker.create_from_options(options)
            self.status_var.set(f"Models ({perf}) loaded successfully!")
            if hasattr(self, 'btn_run'):
                self.btn_run.config(state=tk.NORMAL)
        except Exception as e:
            perf = self.config.get("perf", "High")
            messagebox.showerror("Model Error", f"Failed: {e}\nPlease check if model files exist in Setting/{perf}/ folder.")
            self.status_var.set("Model load failed.")
            if hasattr(self, 'btn_run'):
                self.btn_run.config(state=tk.NORMAL)

    def get_pixel_coords(self, lm, crop_w, crop_h, off_x=0, off_y=0):
        # [수정] 가시성 기준을 0.4에서 0.65로 상향 조정 (가려진 관절은 과감히 버림)
        if lm.visibility < 0.65: return None
        return np.array([lm.x * crop_w + off_x, lm.y * crop_h + off_y])

    def update_progress_loop(self):
        if self.is_running:
            progress_val = (self.frames_processed / max(1, self.total_frames)) * 100
            self.progress.configure(value=progress_val)
            self.root.after(100, self.update_progress_loop) 
        else:
            self.progress.configure(value=100)

    def video_reader_thread(self):
        # 웹캠일 경우 윈도우 다이렉트쇼(DSHOW) 옵션을 사용해 강제로 엽니다
        if getattr(self, 'is_webcam', False):
            cap = cv2.VideoCapture(self.temp_in, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # [추가 1] 카메라 자체의 내부 버퍼를 1로 줄여 지연 최소화
        else:
            cap = cv2.VideoCapture(self.temp_in)
            
        while self.is_running and cap.isOpened():
            # [수정 2] 웹캠과 비디오 파일의 프레임 처리 방식을 분리합니다.
            if getattr(self, 'is_webcam', False):
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)

                # [핵심] 큐가 꽉 차 있으면 가장 오래된 과거 프레임을 꺼내서 버립니다.
                # 이렇게 해야 화면이 밀리지 않고 항상 '가장 최신 프레임'만 분석합니다.
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame)
            else:
                # 비디오 파일은 분석을 건너뛰면 안 되므로, 큐가 꽉 차면 잠시 대기합니다 (기존 로직 유지)
                if not self.frame_queue.full():
                    ret, frame = cap.read()
                    if not ret: break
                    self.frame_queue.put(frame)
                else:
                    time.sleep(0.005)
                    
        cap.release()
        self.frame_queue.put(None)

    def processing_thread(self):
        is_first_chunk = True
        
        while self.is_running:
            try:
                frame = self.frame_queue.get(timeout=1)
            except queue.Empty:
                if not self.is_running: break
                continue

            if frame is None:
                self.is_running = False
                break

            if getattr(self, 'is_webcam', False) and not getattr(self, 'is_recording', True):
                # 텍스트 오버레이 후 화면 출력
                cv2.putText(frame, "PREVIEW: Ready. Click 'Start' in Control Panel", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("Real-Time Tracking (Preview)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False
                continue # 여기서 루프를 다시 처음으로 돌림

            # len(self.raw_data)를 self.frames_processed로 변경
            current_video_time_sec = self.frames_processed / self.fps
            calculated_ms = int(current_video_time_sec * 1000)
            
            # [수정 3] 타임스탬프 엄격 증가 보장: 계산된 시간이 이전 시간과 같거나 작을 경우 강제 프레임 간격(ms) 추가
            step_ms = max(1, int(1000 / self.fps))
            if calculated_ms <= self.last_timestamp_ms:
                timestamp_ms = self.last_timestamp_ms + step_ms
            else:
                timestamp_ms = calculated_ms
                
            self.last_timestamp_ms = timestamp_ms

            use_ball, use_bat, use_racket, use_golf = [self.tool_vars.get(k) and self.tool_vars[k].get() for k in ["Ball", "Baseball_Bat", "Tennis_Racket", "Golf_Club"]]
            target_classes = {0}
            if use_ball: target_classes.add(32)
            if use_bat or use_golf: target_classes.add(34)
            if use_racket or use_golf: target_classes.add(38)

            # [최적화] 웹캠/동영상 상관없이 사용자가 도구 추적이나 YOLO 옵션을 켰을 때만 YOLO 엔진 가동
            results = None
            
            # 1. 도구 추적(공, 배트 등)이 켜져 있는지 확인
            # 2. UI에서 YOLO Segmentation 또는 YOLO Pose 체크박스가 켜져 있는지 확인
            needs_yolo = any([use_ball, use_bat, use_racket, use_golf, 
                              self.use_yolo_seg.get(), self.use_yolo_pose.get()])

            # 사용자가 명시적으로 기능을 켰을 때만 YOLO 모델 실행
            if needs_yolo:
                if self.use_yolo_seg.get() and hasattr(self, 'yolo_seg_model'):
                    results = self.yolo_seg_model(frame, classes=list(target_classes), verbose=False)
                else:
                    results = self.yolo_model(frame, classes=list(target_classes), verbose=False)
                
            best_person = None
            max_p_area = 0

            frame_data = {"Time(sec)": current_video_time_sec, "Rep_Count": self.rep_count, "Is_Active_Phase": self.is_moving}

            for k in ["Ball", "Baseball_Bat", "Tennis_Racket", "Golf_Club"]:
                if self.tool_vars.get(k) and self.tool_vars[k].get():
                    frame_data[f"{k}_X"] = np.nan
                    frame_data[f"{k}_Y"] = np.nan

            tool_max_areas = {"Ball": 0, "Baseball_Bat": 0, "Tennis_Racket": 0, "Golf_Club": 0}

            if results:
                for r in results:
                    for b in r.boxes:
                        cls = int(b.cls[0])
                        x1, y1, x2, y2 = map(int, b.xyxy[0])
                        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                        area = (x2 - x1) * (y2 - y1)

                        if cls == 0:
                            if area > max_p_area: max_p_area, best_person = area, (x1, y1, x2, y2)
                        elif cls == 32 and use_ball:
                            if area > tool_max_areas["Ball"]:
                                tool_max_areas["Ball"] = area
                                frame_data["Ball_X"], frame_data["Ball_Y"] = cx, cy
                                cv2.circle(frame, (int(cx), int(cy)), 6, (0, 165, 255), -1)
                        elif cls == 34 and use_bat:
                            if area > tool_max_areas["Baseball_Bat"]:
                                tool_max_areas["Baseball_Bat"] = area
                                frame_data["Baseball_Bat_X"], frame_data["Baseball_Bat_Y"] = cx, cy
                                cv2.circle(frame, (int(cx), int(cy)), 6, (0, 165, 255), -1)
                        elif cls == 38 and use_racket:
                            if area > tool_max_areas["Tennis_Racket"]:
                                tool_max_areas["Tennis_Racket"] = area
                                frame_data["Tennis_Racket_X"], frame_data["Tennis_Racket_Y"] = cx, cy
                                cv2.circle(frame, (int(cx), int(cy)), 6, (0, 165, 255), -1)
                        elif (cls == 34 or cls == 38) and use_golf:
                            if area > tool_max_areas["Golf_Club"]:
                                tool_max_areas["Golf_Club"] = area
                                frame_data["Golf_Club_X"], frame_data["Golf_Club_Y"] = cx, cy
                                cv2.circle(frame, (int(cx), int(cy)), 6, (0, 165, 255), -1)

            if self.use_mediapipe.get() and self.selected_keys:
                crop_w, crop_h, crop_x1, crop_y1 = self.width, self.height, 0, 0
                if best_person:
                    bx1, by1, bx2, by2 = best_person
                    # [수정 1] 상하좌우 마진 확대: 너비는 40%, 높이는 50%로 늘려 하체 잘림 방지
                    pw, ph = int((bx2 - bx1) * 0.40), int((by2 - by1) * 0.50)
                    target_x1, target_y1 = max(0, bx1 - pw), max(0, by1 - ph)
                    target_x2, target_y2 = min(self.width, bx2 + pw), min(self.height, by2 + ph)

                    if self.prev_crop_box is not None:
                        px1, py1, px2, py2 = self.prev_crop_box
                        # [수정 2] 스무딩 가중치 감소: 0.5 -> 0.2로 변경하여 빠른 움직임에 즉각 반응하게 만듦
                        alpha = 0.2  
                        crop_x1 = int(px1 * alpha + target_x1 * (1 - alpha))
                        crop_y1 = int(py1 * alpha + target_y1 * (1 - alpha))
                        crop_x2 = int(px2 * alpha + target_x2 * (1 - alpha))
                        crop_y2 = int(py2 * alpha + target_y2 * (1 - alpha))
                    else:
                        crop_x1, crop_y1, crop_x2, crop_y2 = target_x1, target_y1, target_x2, target_y2

                    self.prev_crop_box = (crop_x1, crop_y1, crop_x2, crop_y2)
                    crop_w, crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1

                    cropped_img = frame[crop_y1:crop_y2, crop_x1:crop_x2]
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                else:
                    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                # [수정] detect_for_video -> detect 로 변경 (타임스탬프 불필요)
                detection_result = self.detector.detect(mp_img)

                for key in self.selected_keys:
                    name = self.landmark_names.get(key, "").split(". ")[-1].replace(" ", "_")
                    if key < 33: frame_data.update({f"{name}_X": np.nan, f"{name}_Y": np.nan})

                if detection_result and detection_result.pose_landmarks:
                    landmarks = detection_result.pose_landmarks[0]
                    coords = [self.get_pixel_coords(lm, crop_w, crop_h, crop_x1, crop_y1) for lm in landmarks]

                    for key in self.selected_keys:
                        if key < 33 and coords[key] is not None:
                            name = self.landmark_names.get(key, "").split(". ")[-1].replace(" ", "_")
                            cx, cy = coords[key]
                            frame_data[f"{name}_X"], frame_data[f"{name}_Y"] = cx, cy
                            cv2.circle(frame, (int(cx), int(cy)), 2, (0, 255, 0), -1)

            if self.use_yolo_seg.get() and hasattr(self, 'yolo_seg_model') and results and len(results) > 0 and hasattr(results[0], 'masks') and results[0].masks is not None:
                frame = results[0].plot(img=frame, labels=False, boxes=False, conf=False)
                
            if self.use_yolo_pose.get() and hasattr(self, 'yolo_pose_model'):
                pose_results = getattr(self, 'yolo_pose_model', self.yolo_model)(frame, classes=[0], verbose=False)
                if pose_results and len(pose_results) > 0 and hasattr(pose_results[0], 'keypoints') and pose_results[0].keypoints is not None:
                    frame = pose_results[0].plot(img=frame, labels=False, boxes=False, conf=False)

                    YOLO_TO_MP = {
                        0: 0,   1: 2,   2: 5,   3: 7,   4: 8,   5: 11,  6: 12,  7: 13,  
                        8: 14,  9: 15,  10: 16, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28,
                    }
                    try:
                        kp_data = pose_results[0].keypoints.xy[0]   
                        kp_conf = pose_results[0].keypoints.conf[0] if pose_results[0].keypoints.conf is not None else None
                        for yolo_idx, mp_idx in YOLO_TO_MP.items():
                            if mp_idx not in self.selected_keys:
                                continue
                            mp_name = self.landmark_names.get(mp_idx, "").split(". ")[-1].replace(" ", "_")
                            x_key, y_key = f"{mp_name}_X", f"{mp_name}_Y"
                            if x_key in frame_data and not np.isnan(frame_data.get(x_key, np.nan)):
                                continue
                            cx_t = float(kp_data[yolo_idx][0])
                            cy_t = float(kp_data[yolo_idx][1])
                            conf_t = float(kp_conf[yolo_idx]) if kp_conf is not None else 1.0
                            if cx_t > 0 and cy_t > 0 and conf_t >= 0.4:
                                frame_data[x_key] = cx_t
                                frame_data[y_key] = cy_t
                                cv2.circle(frame, (int(cx_t), int(cy_t)), 3, (255, 165, 0), -1) 
                    except Exception:
                        pass

            sport = self.sport_var.get()
            analysis = self.analysis_var.get()
            warning_active = False
            warning_text = ""
            
            frame_data["Measured_Value"] = np.nan
            frame_data["Target_Value"] = np.nan
            frame_data["Analysis_Param_Name"] = ""

            if analysis not in ["Free", "ROM", "Spinal Alignment", "None", ""] and self.init_address_coords is None:
                _l_hip_chk = frame_data.get("L_Hip_X")
                _r_hip_chk = frame_data.get("R_Hip_X")
                _hip_ok = (_l_hip_chk is not None and not np.isnan(_l_hip_chk)) or (_r_hip_chk is not None and not np.isnan(_r_hip_chk))
                if _hip_ok:
                    self.init_address_coords = frame_data.copy()
                    self.init_address_coords["Mid_Hip_X"], self.init_address_coords["Mid_Hip_Y"] = get_midpoint(frame_data.get("L_Hip_X"), frame_data.get("L_Hip_Y"), frame_data.get("R_Hip_X"), frame_data.get("R_Hip_Y"))
                    self.init_address_coords["Mid_Shoulder_X"], self.init_address_coords["Mid_Shoulder_Y"] = get_midpoint(frame_data.get("L_Shoulder_X"), frame_data.get("L_Shoulder_Y"), frame_data.get("R_Shoulder_X"), frame_data.get("R_Shoulder_Y"))
                    self.init_address_coords["Mid_Knee_X"], self.init_address_coords["Mid_Knee_Y"] = get_midpoint(frame_data.get("L_Knee_X"), frame_data.get("L_Knee_Y"), frame_data.get("R_Knee_X"), frame_data.get("R_Knee_Y"))

            cur_mid_hip_x, cur_mid_hip_y = get_midpoint(frame_data.get("L_Hip_X"), frame_data.get("L_Hip_Y"), frame_data.get("R_Hip_X"), frame_data.get("R_Hip_Y"))
            cur_mid_shld_x, cur_mid_shld_y = get_midpoint(frame_data.get("L_Shoulder_X"), frame_data.get("L_Shoulder_Y"), frame_data.get("R_Shoulder_X"), frame_data.get("R_Shoulder_Y"))
            cur_mid_knee_x, cur_mid_knee_y = get_midpoint(frame_data.get("L_Knee_X"), frame_data.get("L_Knee_Y"), frame_data.get("R_Knee_X"), frame_data.get("R_Knee_Y"))
            
            # --- 추가된 부분: 계산된 중심점 좌표를 JSON이 읽을 수 있도록 frame_data에 공식 등록 ---
            frame_data["Mid_Hip_X"], frame_data["Mid_Hip_Y"] = cur_mid_hip_x, cur_mid_hip_y
            frame_data["Mid_Shoulder_X"], frame_data["Mid_Shoulder_Y"] = cur_mid_shld_x, cur_mid_shld_y
            frame_data["Mid_Knee_X"], frame_data["Mid_Knee_Y"] = cur_mid_knee_x, cur_mid_knee_y
            
            rt_speed = 0.0
            cur_time_s = frame_data["Time(sec)"]
            if not np.isnan(cur_mid_hip_x) and not np.isnan(cur_mid_hip_y):
                if getattr(self, 'last_hip_pos', None) is not None and getattr(self, 'last_time_s', None) is not None:
                    dt_val = cur_time_s - self.last_time_s
                    if dt_val > 0:
                        dist = np.linalg.norm([cur_mid_hip_x - self.last_hip_pos[0], cur_mid_hip_y - self.last_hip_pos[1]])
                        rt_speed = dist / dt_val
                self.last_hip_pos = (cur_mid_hip_x, cur_mid_hip_y)
                self.last_time_s = cur_time_s

            if not np.isnan(cur_mid_hip_x):
                if self.last_hip_x is not None:
                    speed_hip_x = abs(cur_mid_hip_x - self.last_hip_x)
                    if speed_hip_x > 1.5:  
                        self.is_moving = 1
                        self.idle_frames = 0
                    else:
                        self.idle_frames += 1
                        if self.idle_frames > 20:  
                            self.is_moving = 0
                self.last_hip_x = cur_mid_hip_x
            frame_data["Is_Active_Phase"] = self.is_moving

            nose_x = self.init_address_coords.get("Nose_X", 0) if self.init_address_coords else np.nan
            r_wrist_x = self.init_address_coords.get("R_Wrist_X", 0) if self.init_address_coords else np.nan
            l_wrist_x = self.init_address_coords.get("L_Wrist_X", 0) if self.init_address_coords else np.nan
            
            is_right_handed = True 
            if not np.isnan(nose_x) and not np.isnan(r_wrist_x) and not np.isnan(l_wrist_x):
                is_right_handed = l_wrist_x > r_wrist_x

            # --------------------- GOLF ---------------------
            if sport == "Golf" and self.init_address_coords is not None:
                if analysis == "Sway":
                    trail_hip = np.array([frame_data.get("R_Hip_X"), frame_data.get("R_Hip_Y")]) if is_right_handed else np.array([frame_data.get("L_Hip_X"), frame_data.get("L_Hip_Y")])
                    trail_knee = np.array([frame_data.get("R_Knee_X"), frame_data.get("R_Knee_Y")]) if is_right_handed else np.array([frame_data.get("L_Knee_X"), frame_data.get("L_Knee_Y")])
                    trail_ankle = np.array([frame_data.get("R_Ankle_X"), frame_data.get("R_Ankle_Y")]) if is_right_handed else np.array([frame_data.get("L_Ankle_X"), frame_data.get("L_Ankle_Y")])

                    init_trail_hip = np.array([self.init_address_coords.get("R_Hip_X"), self.init_address_coords.get("R_Hip_Y")]) if is_right_handed else np.array([self.init_address_coords.get("L_Hip_X"), self.init_address_coords.get("L_Hip_Y")])
                    init_trail_ankle = np.array([self.init_address_coords.get("R_Ankle_X"), self.init_address_coords.get("R_Ankle_Y")]) if is_right_handed else np.array([self.init_address_coords.get("L_Ankle_X"), self.init_address_coords.get("L_Ankle_Y")])

                    if not any(np.isnan(x).any() for x in [trail_hip, trail_knee, trail_ankle, init_trail_hip, init_trail_ankle]):
                        x1, y1 = init_trail_ankle
                        x2, y2 = init_trail_hip
                        A, B = y2 - y1, -(x2 - x1)
                        C = x2 * y1 - y2 * x1
                        norm = np.sqrt(A**2 + B**2)

                        if norm > 0:
                            dist_hip = abs(A * trail_hip[0] + B * trail_hip[1] + C) / norm
                            dist_knee = abs(A * trail_knee[0] + B * trail_knee[1] + C) / norm
                            dist_ankle = abs(A * trail_ankle[0] + B * trail_ankle[1] + C) / norm
                            total_dist = dist_hip + dist_knee + dist_ankle

                            line_x_at_hip = x1 + (trail_hip[1] - y1) * (x2 - x1) / (y2 - y1) if y2 != y1 else x1
                            is_away = trail_hip[0] > line_x_at_hip if is_right_handed else trail_hip[0] < line_x_at_hip

                            if is_away:
                                frame_data["Measured_Value"] = total_dist
                                frame_data["Target_Value"] = 15.0
                                frame_data["Analysis_Param_Name"] = "Sway Distance (px)"

                                if total_dist > 15.0:
                                    warning_active, warning_text = True, f"WARNING: Sway! ({total_dist:.1f}px)"
                                    cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
                                    cv2.line(frame, tuple(trail_hip.astype(int)), tuple(trail_knee.astype(int)), (0, 0, 255), 3)
                                    cv2.line(frame, tuple(trail_knee.astype(int)), tuple(trail_ankle.astype(int)), (0, 0, 255), 3)

                elif analysis == "Slide":
                    lead_hip = np.array([frame_data.get("L_Hip_X"), frame_data.get("L_Hip_Y")]) if is_right_handed else np.array([frame_data.get("R_Hip_X"), frame_data.get("R_Hip_Y")])
                    lead_knee = np.array([frame_data.get("L_Knee_X"), frame_data.get("L_Knee_Y")]) if is_right_handed else np.array([frame_data.get("R_Knee_X"), frame_data.get("R_Knee_Y")])
                    lead_ankle = np.array([frame_data.get("L_Ankle_X"), frame_data.get("L_Ankle_Y")]) if is_right_handed else np.array([frame_data.get("R_Ankle_X"), frame_data.get("R_Ankle_Y")])

                    init_lead_ankle = np.array([self.init_address_coords.get("L_Ankle_X"), self.init_address_coords.get("L_Ankle_Y")]) if is_right_handed else np.array([self.init_address_coords.get("R_Ankle_X"), self.init_address_coords.get("R_Ankle_Y")])

                    if not any(np.isnan(x).any() for x in [lead_hip, lead_knee, lead_ankle, init_lead_ankle]):
                        hip_diff = (init_lead_ankle[0] - lead_hip[0]) if is_right_handed else (lead_hip[0] - init_lead_ankle[0])
                        knee_diff = (init_lead_ankle[0] - lead_knee[0]) if is_right_handed else (lead_knee[0] - init_lead_ankle[0])
                        
                        angle = calculate_angle(lead_ankle, lead_knee, lead_hip)
                        
                        max_diff = max(hip_diff, knee_diff)
                        frame_data["Measured_Value"] = max_diff if max_diff > 0 else 0
                        frame_data["Target_Value"] = 5.0
                        frame_data["Analysis_Param_Name"] = "Slide Shift (px)"

                        is_slide = False
                        if max_diff > 5.0:
                            is_slide = True
                        
                        hip_forward_of_knee = (lead_knee[0] - lead_hip[0]) if is_right_handed else (lead_hip[0] - lead_knee[0])
                        if not np.isnan(angle) and angle < 150 and hip_forward_of_knee > 0:
                            is_slide = True

                        if is_slide:
                            warning_active, warning_text = True, f"WARNING: Slide! (Shift: {max_diff:.1f}px)"
                            cv2.line(frame, (int(init_lead_ankle[0]), 0), (int(init_lead_ankle[0]), self.height), (0, 255, 0), 2)
                            cv2.line(frame, tuple(lead_hip.astype(int)), tuple(lead_knee.astype(int)), (0, 0, 255), 3)
                            cv2.line(frame, tuple(lead_knee.astype(int)), tuple(lead_ankle.astype(int)), (0, 0, 255), 3)

                elif analysis == "Chicken Wing":
                    lead_shld = np.array([frame_data.get("L_Shoulder_X"), frame_data.get("L_Shoulder_Y")]) if is_right_handed else np.array([frame_data.get("R_Shoulder_X"), frame_data.get("R_Shoulder_Y")])
                    lead_elb = np.array([frame_data.get("L_Elbow_X"), frame_data.get("L_Elbow_Y")]) if is_right_handed else np.array([frame_data.get("R_Elbow_X"), frame_data.get("R_Elbow_Y")])
                    lead_wrst = np.array([frame_data.get("L_Wrist_X"), frame_data.get("L_Wrist_Y")]) if is_right_handed else np.array([frame_data.get("R_Wrist_X"), frame_data.get("R_Wrist_Y")])
                    ang = calculate_angle(lead_shld, lead_elb, lead_wrst)
                    if not np.isnan(ang):
                        frame_data["Measured_Value"] = ang
                        frame_data["Target_Value"] = 150.0
                        frame_data["Analysis_Param_Name"] = "Lead Elbow Angle (deg)"
                        if ang < 150.0:
                            warning_active, warning_text = True, "WARNING: Chicken Wing!"
                            cv2.line(frame, tuple(lead_shld.astype(int)), tuple(lead_elb.astype(int)), (0, 0, 255), 3)
                            cv2.line(frame, tuple(lead_wrst.astype(int)), tuple(lead_elb.astype(int)), (0, 0, 255), 3)

                elif analysis == "Early Extension":
                    if not np.isnan(cur_mid_hip_y) and self.init_address_coords.get("Mid_Hip_Y") is not None:
                        y_diff = self.init_address_coords["Mid_Hip_Y"] - cur_mid_hip_y
                        frame_data["Measured_Value"] = y_diff
                        frame_data["Target_Value"] = 20.0
                        frame_data["Analysis_Param_Name"] = "Hip Height Diff (px)"
                        if y_diff > 20:
                            warning_active, warning_text = True, "WARNING: Early Extension!"

                elif analysis == "Flat Shoulder Plane":
                    if not np.isnan(cur_mid_shld_x) and not np.isnan(cur_mid_hip_x):
                        v_spine_init = np.array([self.init_address_coords["Mid_Shoulder_X"] - self.init_address_coords["Mid_Hip_X"], self.init_address_coords["Mid_Shoulder_Y"] - self.init_address_coords["Mid_Hip_Y"]])
                        v_shld_cur = np.array([frame_data.get("R_Shoulder_X") - frame_data.get("L_Shoulder_X"), frame_data.get("R_Shoulder_Y") - frame_data.get("L_Shoulder_Y")])
                        norm_spine = np.linalg.norm(v_spine_init)
                        norm_shld = np.linalg.norm(v_shld_cur)
                        if norm_spine != 0 and norm_shld != 0:
                            cos_ang = np.dot(v_spine_init, v_shld_cur) / (norm_spine * norm_shld)
                            ang = np.degrees(np.arccos(np.clip(cos_ang, -1.0, 1.0)))
                            frame_data["Measured_Value"] = ang
                            frame_data["Target_Value"] = 95.0
                            frame_data["Analysis_Param_Name"] = "Spine-Shoulder Angle (deg)"
                            if ang > 95:
                                warning_active, warning_text = True, f"WARNING: Flat Shoulder ({ang:.1f} deg)"

                elif analysis == "Reverse Spine Angle":
                    if not np.isnan(cur_mid_shld_x) and not np.isnan(cur_mid_hip_x):
                        diff = (cur_mid_hip_x - cur_mid_shld_x) if is_right_handed else (cur_mid_shld_x - cur_mid_hip_x)
                        frame_data["Measured_Value"] = diff
                        frame_data["Target_Value"] = 10.0
                        frame_data["Analysis_Param_Name"] = "Reverse Spine Shift (px)"
                        if diff > 10.0:
                            warning_active, warning_text = True, "WARNING: Reverse Spine Angle!"

                elif analysis == "Hanging Back":
                    lead_hip_x = frame_data.get("L_Hip_X") if is_right_handed else frame_data.get("R_Hip_X")
                    back_ankle_x = frame_data.get("R_Ankle_X") if is_right_handed else frame_data.get("L_Ankle_X")
                    if lead_hip_x is not None and back_ankle_x is not None and not np.isnan(lead_hip_x) and not np.isnan(back_ankle_x):
                        diff = (lead_hip_x - back_ankle_x) if is_right_handed else (back_ankle_x - lead_hip_x)
                        frame_data["Measured_Value"] = diff
                        frame_data["Target_Value"] = 10.0
                        frame_data["Analysis_Param_Name"] = "Hip to Ankle Diff (px)"
                        if diff > 10.0:
                            warning_active, warning_text = True, "WARNING: Hanging Back!"

                elif analysis == "Loss of Posture":
                    if not np.isnan(cur_mid_shld_x) and not np.isnan(cur_mid_hip_x):
                        shld_dist = np.linalg.norm([cur_mid_shld_x - self.init_address_coords["Mid_Shoulder_X"], cur_mid_shld_y - self.init_address_coords["Mid_Shoulder_Y"]])
                        hip_dist = np.linalg.norm([cur_mid_hip_x - self.init_address_coords["Mid_Hip_X"], cur_mid_hip_y - self.init_address_coords["Mid_Hip_Y"]])
                        avg_dist = (shld_dist + hip_dist) / 2
                        frame_data["Measured_Value"] = avg_dist
                        frame_data["Target_Value"] = 25.0
                        frame_data["Analysis_Param_Name"] = "Posture Shift (px)"
                        if avg_dist > 25.0:
                            warning_active, warning_text = True, f"WARNING: Loss of Posture ({avg_dist:.1f}px)"

                elif analysis in ["C-Posture", "S-Posture"]:
                    p_shld = np.array([cur_mid_shld_x, cur_mid_shld_y])
                    p_hip = np.array([cur_mid_hip_x, cur_mid_hip_y])
                    p_knee = np.array([cur_mid_knee_x, cur_mid_knee_y])
                    ang = calculate_angle(p_shld, p_hip, p_knee)
                    if not np.isnan(ang):
                        frame_data["Measured_Value"] = ang
                        frame_data["Analysis_Param_Name"] = "Trunk Flexion Angle (deg)"
                        if analysis == "C-Posture":
                            frame_data["Target_Value"] = 130.0
                            if ang < 130:
                                warning_active, warning_text = True, "WARNING: C-Posture!"
                        elif analysis == "S-Posture":
                            frame_data["Target_Value"] = 175.0
                            if ang > 175:
                                warning_active, warning_text = True, "WARNING: S-Posture!"

            # --------------------- BASEBALL ---------------------
            elif sport == "Baseball" and self.init_address_coords is not None:
                if analysis == "Dead Hand":
                    back_wrist_x = frame_data.get("R_Wrist_X") if is_right_handed else frame_data.get("L_Wrist_X")
                    back_wrist_y = frame_data.get("R_Wrist_Y") if is_right_handed else frame_data.get("L_Wrist_Y")
                    nose_x, nose_y = frame_data.get("Nose_X"), frame_data.get("Nose_Y")
                    
                    init_nose_x, init_nose_y = self.init_address_coords.get("Nose_X"), self.init_address_coords.get("Nose_Y")
                    init_bw_x = self.init_address_coords.get("R_Wrist_X") if is_right_handed else self.init_address_coords.get("L_Wrist_X")
                    init_bw_y = self.init_address_coords.get("R_Wrist_Y") if is_right_handed else self.init_address_coords.get("L_Wrist_Y")

                    if not any(np.isnan(v) for v in [back_wrist_x, back_wrist_y, nose_x, nose_y, init_nose_x, init_nose_y, init_bw_x, init_bw_y]):
                        cur_dist = np.linalg.norm([nose_x - back_wrist_x, nose_y - back_wrist_y])
                        init_dist = np.linalg.norm([init_nose_x - init_bw_x, init_nose_y - init_bw_y])
                        
                        target_dist = init_dist * 0.75 
                        frame_data["Measured_Value"] = cur_dist
                        frame_data["Target_Value"] = target_dist
                        frame_data["Analysis_Param_Name"] = "Nose-to-Wrist Dist (px)"
                        
                        if init_dist > 0 and cur_dist < target_dist:
                            warning_active, warning_text = True, f"WARNING: Dead Hand! ({cur_dist:.1f}px)"
                            cv2.line(frame, (int(nose_x), int(nose_y)), (int(back_wrist_x), int(back_wrist_y)), (0, 0, 255), 3)

                elif analysis == "Drifting":
                    lead_hip_x = frame_data.get("L_Hip_X") if is_right_handed else frame_data.get("R_Hip_X")
                    init_lead_hip_x = self.init_address_coords.get("L_Hip_X") if is_right_handed else self.init_address_coords.get("R_Hip_X")
                    
                    if not np.isnan(lead_hip_x) and init_lead_hip_x is not None:
                        diff = (init_lead_hip_x - lead_hip_x) if is_right_handed else (lead_hip_x - init_lead_hip_x)
                        frame_data["Measured_Value"] = diff
                        frame_data["Target_Value"] = 30.0
                        frame_data["Analysis_Param_Name"] = "Lead Hip Drifting (px)"
                        
                        if diff > 30.0:
                            warning_active, warning_text = True, f"WARNING: Drifting! ({diff:.1f}px)"
                            cv2.line(frame, (int(init_lead_hip_x), 0), (int(init_lead_hip_x), self.height), (0, 0, 255), 2)
                            cv2.circle(frame, (int(lead_hip_x), int(frame_data.get("L_Hip_Y") if is_right_handed else frame_data.get("R_Hip_Y"))), 8, (0, 0, 255), -1)

                elif analysis == "Flying Elbow":
                    back_elb_x, back_elb_y = (frame_data.get("R_Elbow_X"), frame_data.get("R_Elbow_Y")) if is_right_handed else (frame_data.get("L_Elbow_X"), frame_data.get("L_Elbow_Y"))
                    back_wrist_x, back_wrist_y = (frame_data.get("R_Wrist_X"), frame_data.get("R_Wrist_Y")) if is_right_handed else (frame_data.get("L_Wrist_X"), frame_data.get("L_Wrist_Y"))
                    
                    if not any(np.isnan(v) for v in [back_elb_y, back_wrist_y]):
                        diff = back_wrist_y - back_elb_y
                        frame_data["Measured_Value"] = diff
                        frame_data["Target_Value"] = 0.0 
                        frame_data["Analysis_Param_Name"] = "Elbow over Wrist (px)"
                        
                        if diff > 0.0: 
                            warning_active, warning_text = True, f"WARNING: Flying Elbow! (+{diff:.1f}px)"
                            cv2.line(frame, (int(back_elb_x), int(back_elb_y)), (int(back_wrist_x), int(back_wrist_y)), (0, 0, 255), 3)

                elif analysis == "Hanging Back":
                    back_heel_x = frame_data.get("R_Heel_X") if is_right_handed else frame_data.get("L_Heel_X")
                    init_lead_toe_x = self.init_address_coords.get("L_Foot_Index_X") if is_right_handed else self.init_address_coords.get("R_Foot_Index_X")
                    
                    if not np.isnan(back_heel_x) and init_lead_toe_x is not None and not np.isnan(init_lead_toe_x):
                        diff = (back_heel_x - init_lead_toe_x) if is_right_handed else (init_lead_toe_x - back_heel_x)
                        frame_data["Measured_Value"] = diff
                        frame_data["Target_Value"] = 0.0
                        frame_data["Analysis_Param_Name"] = "Heel to Initial Toe (px)"
                        
                        if diff > 0.0:
                            warning_active, warning_text = True, "WARNING: Hanging Back!"
                            cv2.line(frame, (int(init_lead_toe_x), 0), (int(init_lead_toe_x), self.height), (0, 255, 255), 2)
                            cv2.circle(frame, (int(back_heel_x), int(frame_data.get("R_Heel_Y") if is_right_handed else frame_data.get("L_Heel_Y"))), 8, (0, 0, 255), -1)

                elif analysis == "Push":
                    back_elb = np.array([frame_data.get("R_Elbow_X"), frame_data.get("R_Elbow_Y")]) if is_right_handed else np.array([frame_data.get("L_Elbow_X"), frame_data.get("L_Elbow_Y")])
                    back_hip = np.array([frame_data.get("R_Hip_X"), frame_data.get("R_Hip_Y")]) if is_right_handed else np.array([frame_data.get("L_Hip_X"), frame_data.get("L_Hip_Y")])
                    
                    if not any(np.isnan(x).any() for x in [back_elb, back_hip]):
                        dist_elb_hip = np.linalg.norm(back_elb - back_hip)
                        frame_data["Measured_Value"] = dist_elb_hip
                        frame_data["Target_Value"] = 20.0
                        frame_data["Analysis_Param_Name"] = "Elbow to Hip Dist (px)"
                        
                        if dist_elb_hip < 20.0: 
                            warning_active, warning_text = True, f"WARNING: Push / Bat Drag! ({dist_elb_hip:.1f}px)"
                            cv2.line(frame, tuple(back_elb.astype(int)), tuple(back_hip.astype(int)), (0, 0, 255), 3)

                elif analysis == "Sway":
                    back_heel_x, back_heel_y = (frame_data.get("R_Heel_X"), frame_data.get("R_Heel_Y")) if is_right_handed else (frame_data.get("L_Heel_X"), frame_data.get("L_Heel_Y"))
                    back_knee_x, back_knee_y = (frame_data.get("R_Knee_X"), frame_data.get("R_Knee_Y")) if is_right_handed else (frame_data.get("L_Knee_X"), frame_data.get("L_Knee_Y"))
                    
                    init_heel_x, init_heel_y = (self.init_address_coords.get("R_Heel_X"), self.init_address_coords.get("R_Heel_Y")) if is_right_handed else (self.init_address_coords.get("L_Heel_X"), self.init_address_coords.get("L_Heel_Y"))
                    init_knee_x, init_knee_y = (self.init_address_coords.get("R_Knee_X"), self.init_address_coords.get("R_Knee_Y")) if is_right_handed else (self.init_address_coords.get("L_Knee_X"), self.init_address_coords.get("L_Knee_Y"))
                    
                    if not any(np.isnan(v) for v in [back_heel_x, back_heel_y, back_knee_x, back_knee_y, init_heel_x, init_heel_y, init_knee_x, init_knee_y]):
                        init_angle = np.degrees(np.arctan2(init_knee_y - init_heel_y, init_knee_x - init_heel_x))
                        cur_angle = np.degrees(np.arctan2(back_knee_y - back_heel_y, back_knee_x - back_heel_x))
                        
                        angle_diff = cur_angle - init_angle
                        
                        frame_data["Measured_Value"] = angle_diff
                        frame_data["Target_Value"] = 10.0
                        frame_data["Analysis_Param_Name"] = "Heel-Knee Angle Diff (deg)"
                        
                        if abs(angle_diff) > 10.0:
                            warning_active, warning_text = True, f"WARNING: Sway! ({angle_diff:.1f}deg)"
                            cv2.line(frame, (int(back_heel_x), int(back_heel_y)), (int(back_knee_x), int(back_knee_y)), (0, 0, 255), 3)
                            
                elif analysis == "Loss of Posture":
                    eye_x = frame_data.get("R_Eye_X") if is_right_handed else frame_data.get("L_Eye_X")
                    eye_y = frame_data.get("R_Eye_Y") if is_right_handed else frame_data.get("L_Eye_Y")
                    
                    if not np.isnan(eye_x) and not np.isnan(eye_y) and not np.isnan(cur_mid_hip_x) and not np.isnan(cur_mid_hip_y):
                        eye_hip_angle = np.degrees(np.arctan2(cur_mid_hip_y - eye_y, cur_mid_hip_x - eye_x))
                        posture_diff = abs(eye_hip_angle - 90)
                        
                        frame_data["Measured_Value"] = posture_diff
                        frame_data["Target_Value"] = 25.0
                        frame_data["Analysis_Param_Name"] = "Eye-Hip Posture Dev (deg)"
                        
                        if posture_diff > 25.0:
                            warning_active, warning_text = True, f"WARNING: Loss of Posture! ({posture_diff:.1f}deg)"
                            cv2.line(frame, (int(eye_x), int(eye_y)), (int(cur_mid_hip_x), int(cur_mid_hip_y)), (0, 0, 255), 3)

            # --------------------- WORKOUT 통합 ---------------------
            elif sport == "Workout" and self.init_address_coords is not None:
                if analysis == "Spinal Alignment":
                    l_shld_x, l_shld_y = frame_data.get("L_Shoulder_X"), frame_data.get("L_Shoulder_Y")
                    r_shld_x, r_shld_y = frame_data.get("R_Shoulder_X"), frame_data.get("R_Shoulder_Y")
                    l_hip_x, l_hip_y = frame_data.get("L_Hip_X"), frame_data.get("L_Hip_Y")
                    r_hip_x, r_hip_y = frame_data.get("R_Hip_X"), frame_data.get("R_Hip_Y")
                    
                    spine_dev, shld_tilt, hip_tilt = 0.0, 0.0, 0.0
                    
                    if all(v is not None and not np.isnan(v) for v in [cur_mid_shld_x, cur_mid_shld_y, cur_mid_hip_x, cur_mid_hip_y]):
                        spine_dx = cur_mid_hip_x - cur_mid_shld_x
                        spine_dy = cur_mid_hip_y - cur_mid_shld_y
                        spine_angle = np.degrees(np.arctan2(spine_dy, spine_dx))
                        spine_dev = abs(spine_angle - 90.0)
                        
                        line_color = (0, 255, 0) if spine_dev < 5.0 else (0, 0, 255)
                        cv2.line(frame, (int(cur_mid_shld_x), int(cur_mid_shld_y)), (int(cur_mid_hip_x), int(cur_mid_hip_y)), line_color, 3)
                        cv2.line(frame, (int(cur_mid_hip_x), int(cur_mid_hip_y)), (int(cur_mid_hip_x), int(cur_mid_shld_y - 50)), (255, 255, 255), 1, cv2.LINE_AA)

                    if all(v is not None and not np.isnan(v) for v in [l_shld_x, l_shld_y, r_shld_x, r_shld_y]):
                        shld_tilt = np.degrees(np.arctan2(r_shld_y - l_shld_y, r_shld_x - l_shld_x))
                        cv2.line(frame, (int(l_shld_x), int(l_shld_y)), (int(r_shld_x), int(r_shld_y)), (255, 0, 0), 2)

                    if all(v is not None and not np.isnan(v) for v in [l_hip_x, l_hip_y, r_hip_x, r_hip_y]):
                        hip_tilt = np.degrees(np.arctan2(r_hip_y - l_hip_y, r_hip_x - l_hip_x))
                        cv2.line(frame, (int(l_hip_x), int(l_hip_y)), (int(r_hip_x), int(r_hip_y)), (255, 0, 0), 2)

                    max_dev = max(spine_dev, abs(shld_tilt), abs(hip_tilt))
                    
                    frame_data["Measured_Value"] = max_dev
                    frame_data["Target_Value"] = 5.0
                    frame_data["Analysis_Param_Name"] = "Max Postural Tilt (deg)"
                    frame_data["Spine_Dev_Angle"] = spine_dev
                    frame_data["Shoulder_Tilt_Angle"] = shld_tilt
                    frame_data["Hip_Tilt_Angle"] = hip_tilt

                    warnings = []
                    if spine_dev > 5.0: warnings.append(f"Spine({spine_dev:.1f}°)")
                    if abs(shld_tilt) > 5.0: warnings.append(f"Shoulder({abs(shld_tilt):.1f}°)")
                    if abs(hip_tilt) > 5.0: warnings.append(f"Hip({abs(hip_tilt):.1f}°)")

                    if len(warnings) > 0:
                        warning_active = True
                        warning_text = "TILT WARNING: " + " | ".join(warnings)

                elif analysis == "Squat":
                    warnings = []
                    # 자동 뷰 판별 로직
                    l_shld_x, r_shld_x = frame_data.get("L_Shoulder_X"), frame_data.get("R_Shoulder_X")
                    shld_width = abs(l_shld_x - r_shld_x) if pd.notna(l_shld_x) and pd.notna(r_shld_x) else 100
                    torso_height = abs(cur_mid_shld_y - cur_mid_hip_y) if pd.notna(cur_mid_shld_y) and pd.notna(cur_mid_hip_y) else 100
                    is_lateral = (shld_width / torso_height) < 0.55 if torso_height > 0 else False
                    
                    toe_x_r, heel_x_r = frame_data.get("R_Foot_Index_X"), frame_data.get("R_Heel_X")
                    toe_x_l, heel_x_l = frame_data.get("L_Foot_Index_X"), frame_data.get("L_Heel_X")
                    
                    # 방향 판별 (오른쪽/왼쪽 중 데이터가 있는 쪽 우선)
                    if pd.notna(heel_x_r) and pd.notna(toe_x_r):
                        is_facing_right = heel_x_r < toe_x_r
                    elif pd.notna(heel_x_l) and pd.notna(toe_x_l):
                        is_facing_right = heel_x_l < toe_x_l
                    else:
                        is_facing_right = True

                    if is_lateral:
                        # [측면 모드] 데이터 추출
                        if is_facing_right:
                            hip = np.array([frame_data.get("R_Hip_X"), frame_data.get("R_Hip_Y")], dtype=float)
                            knee = np.array([frame_data.get("R_Knee_X"), frame_data.get("R_Knee_Y")], dtype=float)
                            ankle = np.array([frame_data.get("R_Ankle_X"), frame_data.get("R_Ankle_Y")], dtype=float)
                            toe_x = frame_data.get("R_Foot_Index_X")
                            shld = np.array([frame_data.get("R_Shoulder_X"), frame_data.get("R_Shoulder_Y")], dtype=float)
                        else:
                            hip = np.array([frame_data.get("L_Hip_X"), frame_data.get("L_Hip_Y")], dtype=float)
                            knee = np.array([frame_data.get("L_Knee_X"), frame_data.get("L_Knee_Y")], dtype=float)
                            ankle = np.array([frame_data.get("L_Ankle_X"), frame_data.get("L_Ankle_Y")], dtype=float)
                            toe_x = frame_data.get("L_Foot_Index_X")
                            shld = np.array([frame_data.get("L_Shoulder_X"), frame_data.get("L_Shoulder_Y")], dtype=float)

                        squat_val = calculate_angle(hip, knee, ankle)
                        down_thresh, up_thresh = 100.0, 110.0
                        
                        # 1. Rep Count & Stage Logic
                        if not np.isnan(squat_val):
                            if squat_val < down_thresh and self.workout_stage == "UP":
                                self.workout_stage = "DOWN"
                            elif squat_val > up_thresh and self.workout_stage == "DOWN":
                                self.workout_stage = "UP"
                                self.rep_count += 1
                                
                        frame_data["Measured_Value"] = squat_val
                        frame_data["Target_Value"] = down_thresh 
                        frame_data["Analysis_Param_Name"] = "Squat Angle (deg)"

                        # 2. 측면 오버레이 드로잉
                        if not any(np.isnan(v).any() for v in [hip, knee, ankle]):
                            line_color = (0, 0, 255) if self.workout_stage == "DOWN" else (0, 255, 0)
                            cv2.line(frame, tuple(hip.astype(int)), tuple(knee.astype(int)), line_color, 3)
                            cv2.line(frame, tuple(knee.astype(int)), tuple(ankle.astype(int)), line_color, 3)

                            # --- 무릎이 발끝을 초과할 경우 경고 (Knee Over Toe) ---
                            if pd.notna(toe_x):
                                is_over = (knee[0] > toe_x) if is_facing_right else (knee[0] < toe_x)
                                if is_over:
                                    warning_active = True
                                    warning_text = "WARNING: Knee Over Toe!"
                                    cv2.line(frame, (int(toe_x), 0), (int(toe_x), self.height), (0, 165, 255), 2) # 오렌지색 수직 기준선

                            # --- 상체 기울기 경고 (Trunk Lean) ---
                            if not np.isnan(shld).any():
                                trunk_vec = shld - hip
                                vertical_vec = np.array([0, -100]) # 수직 위 방향
                                trunk_lean_angle = calculate_angle(shld, hip, hip + vertical_vec)
                                
                                if trunk_lean_angle > 45.0: # 45도 이상 숙여지면 경고
                                    warning_active = True
                                    warning_text = "WARNING: Excessive Trunk Lean!"
                                    cv2.line(frame, tuple(shld.astype(int)), tuple(hip.astype(int)), (0, 0, 255), 4)

                    else:
                        # [정면 모드] 기존 로직 유지
                        a_x, a_y = get_midpoint(frame_data.get("L_Ankle_X"), frame_data.get("L_Ankle_Y"), frame_data.get("R_Ankle_X"), frame_data.get("R_Ankle_Y"))
                        if pd.notna(a_y) and pd.notna(cur_mid_knee_y) and pd.notna(cur_mid_hip_y):
                            calf_length = max(20, abs(a_y - cur_mid_knee_y))
                            hip_ratio = (cur_mid_knee_y - cur_mid_hip_y) / calf_length
                            down_thresh, up_thresh = 0.55, 0.80
                            if hip_ratio < down_thresh and self.workout_stage == "UP":
                                self.workout_stage = "DOWN"
                            elif hip_ratio > up_thresh and self.workout_stage == "DOWN":
                                self.workout_stage = "UP"
                                self.rep_count += 1
                            frame_data["Measured_Value"] = hip_ratio * 100
                            frame_data["Target_Value"] = down_thresh * 100
                            frame_data["Analysis_Param_Name"] = "Hip Height Ratio (%)"

                    # 공통 UI 표시 (화면 좌상단/우상단)
                    cv2.putText(frame, f"VIEW: {'LATERAL' if is_lateral else 'FRONTAL'}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f"STAGE: {self.workout_stage}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"REPS: {self.rep_count}", (self.width - 180, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

                elif analysis == "Sit to Stand":
                    warnings = []
                    # 1. 자동 뷰 판별 (어깨 너비/몸통 높이 비율)
                    l_shld_x, r_shld_x = frame_data.get("L_Shoulder_X"), frame_data.get("R_Shoulder_X")
                    shld_width = abs(l_shld_x - r_shld_x) if pd.notna(l_shld_x) and pd.notna(r_shld_x) else 100
                    torso_height = abs(cur_mid_shld_y - cur_mid_hip_y) if pd.notna(cur_mid_shld_y) and pd.notna(cur_mid_hip_y) else 100
                    is_lateral = (shld_width / torso_height) < 0.55 if torso_height > 0 else False
                    
                    toe_x_r, heel_x_r = frame_data.get("R_Foot_Index_X"), frame_data.get("R_Heel_X")
                    toe_x_l, heel_x_l = frame_data.get("L_Foot_Index_X"), frame_data.get("L_Heel_X")
                    
                    # 방향 판별
                    if pd.notna(heel_x_r) and pd.notna(toe_x_r):
                        is_facing_right = heel_x_r < toe_x_r
                    elif pd.notna(heel_x_l) and pd.notna(toe_x_l):
                        is_facing_right = heel_x_l < toe_x_l
                    else:
                        is_facing_right = True
    
                    if is_lateral:
                        # [측면 모드] 데이터 추출
                        if is_facing_right:
                            hip = np.array([frame_data.get("R_Hip_X"), frame_data.get("R_Hip_Y")], dtype=float)
                            knee = np.array([frame_data.get("R_Knee_X"), frame_data.get("R_Knee_Y")], dtype=float)
                            ankle = np.array([frame_data.get("R_Ankle_X"), frame_data.get("R_Ankle_Y")], dtype=float)
                            toe_x = frame_data.get("R_Foot_Index_X")
                            shld = np.array([frame_data.get("R_Shoulder_X"), frame_data.get("R_Shoulder_Y")], dtype=float)
                        else:
                            hip = np.array([frame_data.get("L_Hip_X"), frame_data.get("L_Hip_Y")], dtype=float)
                            knee = np.array([frame_data.get("L_Knee_X"), frame_data.get("L_Knee_Y")], dtype=float)
                            ankle = np.array([frame_data.get("L_Ankle_X"), frame_data.get("L_Ankle_Y")], dtype=float)
                            toe_x = frame_data.get("L_Foot_Index_X")
                            shld = np.array([frame_data.get("L_Shoulder_X"), frame_data.get("L_Shoulder_Y")], dtype=float)

                        sts_ang = calculate_angle(hip, knee, ankle)
                        
                        # --- 완화된 기준 적용 ---
                        sit_thresh, stand_thresh = 110.0, 140.0 
                        
                        # 횟수 및 단계 로직
                        if not np.isnan(sts_ang):
                            if sts_ang < sit_thresh and self.workout_stage == "STANDING":
                                self.workout_stage = "SITTING"
                            elif sts_ang > stand_thresh and self.workout_stage == "SITTING":
                                self.workout_stage = "STANDING"
                                self.rep_count += 1
                                
                        frame_data["Measured_Value"] = sts_ang
                        frame_data["Target_Value"] = sit_thresh
                        frame_data["Analysis_Param_Name"] = "Knee Angle (deg)"

                        # --- 측면 오버레이 드로잉 및 경고 처리 (복구됨) ---
                        if not any(np.isnan(v).any() for v in [hip, knee, ankle]):
                            line_color = (0, 165, 255) if self.workout_stage == "SITTING" else (0, 255, 0)
                            cv2.line(frame, tuple(hip.astype(int)), tuple(knee.astype(int)), line_color, 3)
                            cv2.line(frame, tuple(knee.astype(int)), tuple(ankle.astype(int)), line_color, 3)

                            # 1. 무릎 발끝 초과 경고 (Knee Over Toe)
                            if pd.notna(toe_x):
                                is_over = (knee[0] > toe_x) if is_facing_right else (knee[0] < toe_x)
                                if is_over:
                                    warning_active = True
                                    warning_text = "WARNING: Knee Over Toe!"
                                    cv2.line(frame, (int(toe_x), 0), (int(toe_x), self.height), (0, 165, 255), 2)

                            # 2. 상체 과도 기울기 경고 (Trunk Lean)
                            if not np.isnan(shld).any():
                                trunk_vec = shld - hip
                                vertical_ref = np.array([0, -100])
                                trunk_lean = calculate_angle(shld, hip, hip + vertical_ref)
                                
                                if trunk_lean > 50.0:
                                    warning_active = True
                                    warning_text = "WARNING: Excessive Trunk Lean!"
                                    cv2.line(frame, tuple(shld.astype(int)), tuple(hip.astype(int)), (0, 0, 255), 4)

                    else:  # <--- 백스페이스로 4칸 왼쪽으로 당겨서 'if is_lateral:'과 라인 일치시킴
                        # [정면 모드] 사용자가 요청한 정면 무릎 각도 기반 로직으로 수정
                        a_x, a_y = get_midpoint(frame_data.get("L_Ankle_X"), frame_data.get("L_Ankle_Y"), frame_data.get("R_Ankle_X"), frame_data.get("R_Ankle_Y"))
                        
                        if pd.notna(a_x) and pd.notna(cur_mid_knee_x) and pd.notna(cur_mid_hip_x):
                            # 좌우 평균 좌표를 활용하여 정면 기준 관절 벡터 생성
                            hip_arr = np.array([cur_mid_hip_x, cur_mid_hip_y], dtype=float)
                            knee_arr = np.array([cur_mid_knee_x, cur_mid_knee_y], dtype=float)
                            ankle_arr = np.array([a_x, a_y], dtype=float)

                            # 골반-무릎-발목이 이루는 정면 2D 각도 계산
                            frontal_knee_angle = calculate_angle(hip_arr, knee_arr, ankle_arr)
                            
                            sit_thresh_f = 155.0  # 정면 기준 넉넉한 앉음 인식 각도로 수정
                            stand_thresh_f = 165.0 # 일어섬 인식 각도
                            
                            if not np.isnan(frontal_knee_angle):
                                # 각도가 155 미만일 때 down (SITTING) 처리
                                if frontal_knee_angle < sit_thresh_f and self.workout_stage == "STANDING":
                                    self.workout_stage = "SITTING"
                                # 각도가 165 초과하고 현재 down 상태일 때 up (STANDING) 처리 및 카운트 증가
                                elif frontal_knee_angle > stand_thresh_f and self.workout_stage == "SITTING":
                                    self.workout_stage = "STANDING"
                                    self.rep_count += 1
                                    
                            # 측정된 각도 데이터를 엑셀 및 PDF 결과지에 반영
                            frame_data["Measured_Value"] = frontal_knee_angle
                            frame_data["Target_Value"] = sit_thresh_f
                            frame_data["Analysis_Param_Name"] = "Frontal Knee Angle (deg)"

                    # 하단 정보 표시
                    cv2.putText(frame, f"VIEW: {'LATERAL' if is_lateral else 'FRONTAL'}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                    cv2.putText(frame, f"STAGE: {self.workout_stage}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    cv2.putText(frame, f"REPS: {self.rep_count}", (self.width - 180, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

                elif analysis in ["Gait", "Timed Up and Go"]:
                    # ==========================================================
                    # 1. 정면/측면 자동 판별 및 걸음 수(Step Count) 측정 (Gait, TUG 공통)
                    # ==========================================================
                    l_shld_x, r_shld_x = frame_data.get("L_Shoulder_X"), frame_data.get("R_Shoulder_X")
                    shld_width = abs(l_shld_x - r_shld_x) if pd.notna(l_shld_x) and pd.notna(r_shld_x) else 100
                    torso_height = abs(cur_mid_shld_y - cur_mid_hip_y) if pd.notna(cur_mid_shld_y) and pd.notna(cur_mid_hip_y) else 100
                    is_lateral = (shld_width / torso_height) < 0.55 if torso_height > 0 else False

                    l_a_x, l_a_y = frame_data.get("L_Ankle_X"), frame_data.get("L_Ankle_Y")
                    r_a_x, r_a_y = frame_data.get("R_Ankle_X"), frame_data.get("R_Ankle_Y")

                    if pd.notna(l_a_x) and pd.notna(r_a_x) and pd.notna(l_a_y) and pd.notna(r_a_y):
                        # 측면이면 X축 교차, 정면이면 Y축 교차(원근감)를 기준으로 걸음 인식
                        diff = (l_a_x - r_a_x) if is_lateral else (l_a_y - r_a_y)
                        
                        threshold = 20.0 # 노이즈 방지용 최소 교차 거리
                        if abs(diff) > threshold:
                            curr_state = "L_fwd" if diff > 0 else "R_fwd"
                            if getattr(self, 'gait_state', None) is None:
                                self.gait_state = curr_state
                            elif self.gait_state != curr_state:
                                # TUG는 'WALKING' 상태일 때만 카운트, Gait는 상시 카운트
                                if analysis == "Gait" or (analysis == "Timed Up and Go" and self.workout_stage == "WALKING"):
                                    self.step_count = getattr(self, 'step_count', 0) + 1
                                self.gait_state = curr_state
                    
                    frame_data["Step_Count"] = getattr(self, 'step_count', 0)

                    # ==========================================================
                    # 2. 개별 모드 로직 (선생님이 작성하신 기존 코드 유지 + 화면 표시 추가)
                    # ==========================================================
                    if analysis == "Gait":
                        # 1. 좌/우측 관절 좌표 추출 (골반, 무릎, 발목, 발끝)
                        l_hip = np.array([frame_data.get("L_Hip_X"), frame_data.get("L_Hip_Y")], dtype=float)
                        l_knee = np.array([frame_data.get("L_Knee_X"), frame_data.get("L_Knee_Y")], dtype=float)
                        l_ankle = np.array([frame_data.get("L_Ankle_X"), frame_data.get("L_Ankle_Y")], dtype=float)
                        l_toe = np.array([frame_data.get("L_Foot_Index_X"), frame_data.get("L_Foot_Index_Y")], dtype=float)

                        r_hip = np.array([frame_data.get("R_Hip_X"), frame_data.get("R_Hip_Y")], dtype=float)
                        r_knee = np.array([frame_data.get("R_Knee_X"), frame_data.get("R_Knee_Y")], dtype=float)
                        r_ankle = np.array([frame_data.get("R_Ankle_X"), frame_data.get("R_Ankle_Y")], dtype=float)
                        r_toe = np.array([frame_data.get("R_Foot_Index_X"), frame_data.get("R_Foot_Index_Y")], dtype=float)

                        # 2. 발걸음 길이 (Step Length) 계산
                        step_length = np.nan
                        if not np.isnan(l_ankle[0]) and not np.isnan(r_ankle[0]):
                            step_length = np.linalg.norm(l_ankle - r_ankle) # 양 발목 사이의 거리
                            frame_data["Measured_Value"] = step_length
                            frame_data["Target_Value"] = 50.0 
                            frame_data["Analysis_Param_Name"] = "Step Length (px)"

                        # 3. 무릎 각도 (Knee Angle) 계산
                        l_knee_angle = calculate_angle(l_hip, l_knee, l_ankle)
                        r_knee_angle = calculate_angle(r_hip, r_knee, r_ankle)

                        frame_data["L_Knee_Angle"] = l_knee_angle if not np.isnan(l_knee_angle) else np.nan
                        frame_data["R_Knee_Angle"] = r_knee_angle if not np.isnan(r_knee_angle) else np.nan

                        # 4. 발목 각도 (Ankle Angle) 계산 (무릎-발목-발끝 기준)
                        l_ankle_angle = calculate_angle(l_knee, l_ankle, l_toe)
                        r_ankle_angle = calculate_angle(r_knee, r_ankle, r_toe)

                        frame_data["L_Ankle_Angle"] = l_ankle_angle if not np.isnan(l_ankle_angle) else np.nan
                        frame_data["R_Ankle_Angle"] = r_ankle_angle if not np.isnan(r_ankle_angle) else np.nan

                        # 5. 시각화 (하체 스켈레톤 라인 그리기)
                        if not np.isnan(l_hip).any() and not np.isnan(l_knee).any() and not np.isnan(l_ankle).any():
                            cv2.line(frame, tuple(l_hip.astype(int)), tuple(l_knee.astype(int)), (0, 255, 0), 3) # 좌측: 초록색
                            cv2.line(frame, tuple(l_knee.astype(int)), tuple(l_ankle.astype(int)), (0, 255, 0), 3)
                            if not np.isnan(l_toe).any():
                                cv2.line(frame, tuple(l_ankle.astype(int)), tuple(l_toe.astype(int)), (0, 200, 0), 3) # 발등 라인

                        if not np.isnan(r_hip).any() and not np.isnan(r_knee).any() and not np.isnan(r_ankle).any():
                            cv2.line(frame, tuple(r_hip.astype(int)), tuple(r_knee.astype(int)), (0, 0, 255), 3) # 우측: 빨간색
                            cv2.line(frame, tuple(r_knee.astype(int)), tuple(r_ankle.astype(int)), (0, 0, 255), 3)
                            if not np.isnan(r_toe).any():
                                cv2.line(frame, tuple(r_ankle.astype(int)), tuple(r_toe.astype(int)), (0, 0, 200), 3) # 발등 라인

                        # 6. 화면 텍스트 오버레이 (시간, 발걸음 길이, 속도, 걸음 수, 각도)
                        cv2.putText(frame, f"Time: {frame_data['Time(sec)']:.2f}s", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                        
                        if not np.isnan(step_length):
                            cv2.putText(frame, f"Step Length: {step_length:.1f}px", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                        
                        cv2.putText(frame, f"Speed: {rt_speed:.1f}px/s", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 165, 0), 2)
                        cv2.putText(frame, f"Steps: {getattr(self, 'step_count', 0)}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        # 좌/우 무릎 및 발목 각도 표시
                        angle_text_y = 210
                        if not np.isnan(l_knee_angle) and not np.isnan(l_ankle_angle):
                            cv2.putText(frame, f"L Knee: {l_knee_angle:.1f} / Ankle: {l_ankle_angle:.1f}", (20, angle_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            angle_text_y += 30
                        if not np.isnan(r_knee_angle) and not np.isnan(r_ankle_angle):
                            cv2.putText(frame, f"R Knee: {r_knee_angle:.1f} / Ankle: {r_ankle_angle:.1f}", (20, angle_text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    elif analysis == "Timed Up and Go":
                        current_val = np.nan

                        if is_lateral:
                            hip_arr = np.array([cur_mid_hip_x, cur_mid_hip_y], dtype=float)
                            knee_arr = np.array([cur_mid_knee_x, cur_mid_knee_y], dtype=float)
                            a_x, a_y = get_midpoint(l_a_x, l_a_y, r_a_x, r_a_y)
                            ankle_arr = np.array([a_x, a_y], dtype=float)

                            current_val = calculate_angle(hip_arr, knee_arr, ankle_arr)
                            sit_thresh = 110.0   
                            stand_thresh = 145.0 
                            frame_data["Analysis_Param_Name"] = "Knee Angle (deg)"
                        else:
                            a_x, a_y = get_midpoint(l_a_x, l_a_y, r_a_x, r_a_y)
                            if pd.notna(a_y) and pd.notna(cur_mid_knee_y) and pd.notna(cur_mid_hip_y):
                                calf_length = abs(a_y - cur_mid_knee_y)
                                if calf_length < 20: calf_length = 100
                                hip_ratio = (cur_mid_knee_y - cur_mid_hip_y) / calf_length

                                current_val = hip_ratio * 100
                                sit_thresh = 40.0   
                                stand_thresh = 70.0 
                                frame_data["Analysis_Param_Name"] = "Hip Height Ratio (%)"

                        frame_data["Measured_Value"] = current_val if not np.isnan(current_val) else np.nan
                        frame_data["Target_Value"] = sit_thresh
                        frame_data["Knee_Angle"] = current_val  

                        # 2. TUG 상태 변환 및 타이머 시작/종료 트리거
                        if not np.isnan(current_val):
                            if self.workout_stage == "SITTING":
                                if current_val > stand_thresh:
                                    self.workout_stage = "WALKING"  
                                    self.tug_start_time = frame_data["Time(sec)"]
                            elif self.workout_stage == "WALKING":
                                elapsed_since_start = frame_data["Time(sec)"] - (self.tug_start_time if self.tug_start_time else 0)
                                if current_val < sit_thresh and elapsed_since_start > 2.5:
                                    self.workout_stage = "COMPLETED" 
                                    self.tug_end_time = frame_data["Time(sec)"]

                        # 3. 화면 텍스트 오버레이
                        cv2.putText(frame, f"TUG STAGE: {self.workout_stage}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                        
                        # --- [추가] TUG 화면에 실시간 걸음 수 표시 ---
                        cv2.putText(frame, f"Steps: {getattr(self, 'step_count', 0)}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                        if self.tug_start_time is not None:
                            if self.workout_stage == "COMPLETED":
                                elapsed = self.tug_end_time - self.tug_start_time
                                cv2.putText(frame, f"FINAL TIME: {elapsed:.2f}s", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                            else:
                                elapsed = frame_data["Time(sec)"] - self.tug_start_time
                                cv2.putText(frame, f"Timer: {elapsed:.2f}s", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                            
                            frame_data["TUG_Time_sec"] = elapsed

                elif analysis == "Static Balance":
                    init_hip_x = self.init_address_coords.get("Mid_Hip_X")
                    init_hip_y = self.init_address_coords.get("Mid_Hip_Y")
                    
                    if all(v is not None and not np.isnan(v) for v in [cur_mid_hip_x, cur_mid_hip_y, init_hip_x, init_hip_y, cur_mid_shld_y]):
                        # 체간 길이(어깨 중심 ~ 골반 중심) 계산
                        torso_length = abs(cur_mid_shld_y - cur_mid_hip_y)
                        if torso_length < 30: torso_length = 150 # 인식 오류 대비 기본값
                        
                        # 실제 흔들린 픽셀 거리
                        sway_dist_px = np.linalg.norm([cur_mid_hip_x - init_hip_x, cur_mid_hip_y - init_hip_y])
                        
                        # 체간 길이 대비 흔들림 비율 (%)
                        sway_ratio = (sway_dist_px / torso_length) * 100
                        
                        frame_data["Measured_Value"] = sway_ratio
                        frame_data["Target_Value"] = 10.0 # 체간 길이의 10% 이상 흔들리면 에러로 간주
                        frame_data["Analysis_Param_Name"] = "Sway Ratio (%)"
                        
                        if sway_ratio > 10.0:
                            warning_active, warning_text = True, f"WARNING: Loss of Balance! ({sway_ratio:.1f}%)"
                            cv2.circle(frame, (int(cur_mid_hip_x), int(cur_mid_hip_y)), 10, (0, 0, 255), -1)
                        else:
                            cv2.circle(frame, (int(cur_mid_hip_x), int(cur_mid_hip_y)), 6, (0, 255, 0), -1)
                            
                        cv2.drawMarker(frame, (int(init_hip_x), int(init_hip_y)), (255, 165, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

                elif analysis == "ROM":
                    frame_data["Analysis_Param_Name"] = "ROM Assessment"

            # --------------------- INDIVIDUAL (CUSTOM JSON) ---------------------
            elif sport == "Individual" and analysis in self.custom_analyses_data:
                analyses_to_run = self.custom_analyses_data[analysis]
                custom_warnings = []

                for custom_cfg in analyses_to_run:
                    a_name = custom_cfg.get("analysis_name", "Custom")
                    m_type = custom_cfg.get("measure_type", "")
                    thresh = float(custom_cfg.get("threshold", 50.0))
                    condition = custom_cfg.get("condition", "over")
                    
                    param_name = custom_cfg.get("param_name", "Custom Value")
                    frame_data[f"{a_name}_Param_Name"] = param_name
                    frame_data[f"{a_name}_Target_Value"] = thresh
                    frame_data[f"{a_name}_Measured_Value"] = np.nan
                    
                    # 1. 흔들림(sway) 측정: 초기 위치 기준 직선 거리 추적
                    if m_type == "sway" and self.init_address_coords is not None:
                        target_pt = custom_cfg.get("target_point", "Mid_Hip")
                        init_x = self.init_address_coords.get(f"{target_pt}_X")
                        init_y = self.init_address_coords.get(f"{target_pt}_Y")
                        cur_x = frame_data.get(f"{target_pt}_X")
                        cur_y = frame_data.get(f"{target_pt}_Y")
                        
                        if all(v is not None and not np.isnan(v) for v in [cur_x, cur_y, init_x, init_y]):
                            sway_dist = np.linalg.norm([cur_x - init_x, cur_y - init_y])
                            frame_data[f"{a_name}_Measured_Value"] = sway_dist
                            
                            is_error = (condition == "over" and sway_dist > thresh) or (condition == "under" and sway_dist < thresh)
                            if is_error:
                                custom_warnings.append(f"{a_name}({sway_dist:.1f})")
                                cv2.circle(frame, (int(cur_x), int(cur_y)), 10, (0, 0, 255), -1)
                            else:
                                cv2.circle(frame, (int(cur_x), int(cur_y)), 6, (0, 255, 0), -1)
                            cv2.drawMarker(frame, (int(init_x), int(init_y)), (255, 165, 0), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)

                    # 2. 각도(angle) 측정: 3개의 관절이 이루는 각도 계산
                    elif m_type == "angle":
                        target_pts = custom_cfg.get("target_points", [])
                        if len(target_pts) == 3:
                            p1_name, p2_name, p3_name = target_pts
                            p1 = np.array([frame_data.get(f"{p1_name}_X"), frame_data.get(f"{p1_name}_Y")])
                            p2 = np.array([frame_data.get(f"{p2_name}_X"), frame_data.get(f"{p2_name}_Y")])
                            p3 = np.array([frame_data.get(f"{p3_name}_X"), frame_data.get(f"{p3_name}_Y")])
                            
                            ang = calculate_angle(p1, p2, p3)
                            if not np.isnan(ang):
                                frame_data[f"{a_name}_Measured_Value"] = ang
                                
                                is_error = (condition == "over" and ang > thresh) or (condition == "under" and ang < thresh)
                                if is_error:
                                    custom_warnings.append(f"{a_name}({ang:.1f}°)")
                                    cv2.line(frame, tuple(p1.astype(int)), tuple(p2.astype(int)), (0, 0, 255), 3)
                                    cv2.line(frame, tuple(p3.astype(int)), tuple(p2.astype(int)), (0, 0, 255), 3)

                    # 3. 거리(distance_between) 측정: 두 관절 사이의 거리 추적
                    elif m_type == "distance_between":
                        target_pts = custom_cfg.get("target_points", [])
                        if len(target_pts) == 2:
                            p1_name, p2_name = target_pts
                            p1_x, p1_y = frame_data.get(f"{p1_name}_X"), frame_data.get(f"{p1_name}_Y")
                            p2_x, p2_y = frame_data.get(f"{p2_name}_X"), frame_data.get(f"{p2_name}_Y")
                            
                            if all(v is not None and not np.isnan(v) for v in [p1_x, p1_y, p2_x, p2_y]):
                                dist = np.linalg.norm([p1_x - p2_x, p1_y - p2_y])
                                frame_data[f"{a_name}_Measured_Value"] = dist
                                
                                is_error = (condition == "over" and dist > thresh) or (condition == "under" and dist < thresh)
                                if is_error:
                                    custom_warnings.append(f"{a_name}({dist:.1f})")
                                    cv2.line(frame, (int(p1_x), int(p1_y)), (int(p2_x), int(p2_y)), (0, 0, 255), 3)

                if custom_warnings:
                    warning_active = True
                    warning_text = "WARNING: " + " | ".join(custom_warnings)

            h, w = frame.shape[:2]
            progress_ratio = self.frames_processed / max(1, self.total_frames)
            progress_x = int(progress_ratio * w)

            cv2.line(frame, (0, h - 20), (w, h - 20), (200, 200, 200), 10)
            cv2.line(frame, (0, h - 20), (progress_x, h - 20), (0, 255, 0), 10)

            for i, d in enumerate(self.raw_data):
                if d.get("Error_Flag", 0) == 1:
                    ex = int((i / max(1, self.total_frames)) * w)
                    cv2.line(frame, (ex, h - 25), (ex, h - 15), (0, 0, 255), 2)

            if warning_active:
                cv2.circle(frame, (w - 40, 40), 15, (0, 0, 255), -1)
                cv2.putText(frame, warning_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.circle(frame, (progress_x, h - 20), 8, (0, 0, 255), -1)
                frame_data["Error_Flag"] = 1
            else:
                frame_data["Error_Flag"] = 0

            if analysis in ["Squat", "Sit to Stand"]:
                cv2.putText(frame, f"REPS: {self.rep_count}", (w - 220, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 165, 255), 3)

            if getattr(self, 'is_webcam', False):
                # 우측 상단에 빨간색 REC 녹화 표시 추가
                cv2.putText(frame, "REC", (w - 100, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
                cv2.circle(frame, (w - 120, 30), 10, (0, 0, 255), -1)
                
                cv2.imshow("Real-Time Tracking (Recording)", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    self.is_running = False

            if self.video_writer: 
                self.video_writer.write(frame)
                      
            # 메모리에 끝없이 쌓지 않고 chunk 리스트에 담음
            self.raw_data_chunk.append(frame_data)
            self.frames_processed += 1
            
            # 300프레임이 쌓일 때마다 CSV에 쓰고 메모리 비우기 (메모리 폭발 방지)
            if len(self.raw_data_chunk) >= 300: 
                df_chunk = pd.DataFrame(self.raw_data_chunk)
                df_chunk.to_csv(self.temp_csv_path, mode='a', header=is_first_chunk, index=False)
                is_first_chunk = False
                self.raw_data_chunk = []

        # ----------------------------------------------------
        # 수정 1: 루프가 완전히 끝난 뒤 한 번만 실행되도록 깔끔하게 정리
        # 1. 남은 잔여 데이터 저장
        if len(self.raw_data_chunk) > 0:
            df_chunk = pd.DataFrame(self.raw_data_chunk)
            df_chunk.to_csv(self.temp_csv_path, mode='a', header=is_first_chunk, index=False)
            self.raw_data_chunk = []

        # 2. 영상 파일 닫기 (이게 한 번만 제대로 실행되어야 영상이 안 깨집니다)
        if self.video_writer: 
            self.video_writer.release()
            self.video_writer = None

        # 3. 웹캠 창 닫기
        if getattr(self, 'is_webcam', False):
            cv2.destroyAllWindows()
            cv2.waitKey(1) 
            
        # 4. 메인 스레드에 후처리(엑셀/PDF 저장) 지시 (딱 한 번만 호출)
        self.root.after(0, self.finish_analysis)
        # ----------------------------------------------------

    def generate_pdf_report(self, df, name_part, sport, analysis):
        pdf_path = os.path.join(self.current_pdf_dir, f"{name_part}_report.pdf")
        
        o = self.out_lang_data.get("Output", {})
        pdf_title = o.get("pdf_main_title", "Biomechanics Analysis Report")
        lbl_name = o.get("lbl_name", "Name:")
        lbl_date = o.get("lbl_date", "Date:")
        lbl_examiner = o.get("lbl_examiner", "Examiner:")
        lbl_signature = o.get("lbl_signature", "Signature:")
        
        pt_name = self.client_name_var.get().strip()
        try:
            pt_date = f"{self.date_yy_var.get().strip()}.{self.date_mm_var.get().strip()}.{self.date_dd_var.get().strip()}"
        except AttributeError:
            pt_date = "00.00.00"
            
        if not pt_name or pt_name == "None":
            pt_name_str = "________________________"
        else:
            pt_name_str = pt_name

        ex_name = self.examiner_var.get().strip()
        if ex_name and ex_name != "None":
            examiner_str = f"{lbl_examiner} {ex_name}"
        else:
            examiner_str = f"{lbl_examiner} ________________________"

        # Individual과 Workout일 때는 Sport 표시 생략
        header_prefix = "" if sport in ["Individual", "Workout"] else f"Sport: {sport} | "

        if analysis == "Spinal Alignment":
            fig = plt.figure(figsize=(8.27, 11.69))
            gs = GridSpec(2, 2, height_ratios=[1, 1.2], figure=fig)
            
            fig.suptitle(pdf_title, fontsize=18, fontweight='bold', y=0.97)
            fig.text(0.5, 0.93, f"{header_prefix}Analysis: Scoliosis & Tilt Assessment", fontsize=13, fontweight='bold', ha='center')
            fig.text(0.95, 0.89, f"{lbl_name} {pt_name_str}", fontsize=12, fontweight='bold', ha='right', va='top')
            if pt_date not in ["00.00.00", "0.0.0", "..", ""]:
                fig.text(0.95, 0.87, f"{lbl_date} {pt_date}", fontsize=10, ha='right', va='top')

            ax_img = fig.add_subplot(gs[0, 0])
            ax_table = fig.add_subplot(gs[0, 1])
            ax_graph = fig.add_subplot(gs[1, :])

            valid_df = df.dropna(subset=['Measured_Value']).copy()
            if len(valid_df) > 0:
                peak_idx = valid_df['Measured_Value'].idxmax()
                peak_val = valid_df.loc[peak_idx, 'Measured_Value']
                
                max_spine = valid_df['Spine_Dev_Angle'].max()
                max_shld = valid_df['Shoulder_Tilt_Angle'].abs().max()
                max_hip = valid_df['Hip_Tilt_Angle'].abs().max()
            else:
                peak_idx, peak_val, max_spine, max_shld, max_hip = 0, 0, 0, 0, 0

            cap = cv2.VideoCapture(self.temp_out)
            cap.set(cv2.CAP_PROP_POS_FRAMES, peak_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret: ax_img.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else: ax_img.imshow(np.zeros((400, 400, 3), dtype=np.uint8))
            ax_img.axis('off')
            ax_img.set_title(f"Peak Deviation Snapshot (Max {peak_val:.1f}°)", fontsize=12, fontweight='bold', pad=10)

            ax_table.axis('off')
            has_error = df['Error_Flag'].sum() > 0
            status_text = o.get("status_fail", "Attention Needed") if has_error else o.get("status_pass", "Normal Alignment")
            status_color = "red" if has_error else "green"

            table_data = [
                ["Postural Metric", "Peak Deviation"],
                ["Spine Tilt (from Vertical)", f"{max_spine:.1f}°"],
                ["Shoulder Tilt (from Horiz)", f"{max_shld:.1f}°"],
                ["Hip/Pelvis Tilt (from Horiz)", f"{max_hip:.1f}°"],
                ["Tolerance Threshold", "< 5.0°"]
            ]
            
            for tool_name in ["Ball", "Baseball_Bat", "Tennis_Racket", "Golf_Club"]:
                spd_col = f"{tool_name}_Speed_px_s"
                if spd_col in df.columns and not df[spd_col].isna().all():
                    table_data.append([f"Max Speed ({tool_name.replace('_', ' ')})", f"{df[spd_col].max():.1f} px/s"])
                    
            table_data.append(["Overall Status", status_text])
            
            table = ax_table.table(cellText=table_data, loc='center', cellLoc='center', bbox=[0, 0.05, 1, 0.82])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            
            for (row, col), cell in table.get_celld().items():
                if col == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#4472C4')
                if row == len(table_data)-1 and col == 1:
                    cell.set_text_props(weight='bold', color=status_color)

            ax_table.set_title("Postural Deviations Summary", fontsize=12, fontweight='bold', y=0.9)

            if len(valid_df) > 0:
                ax_graph.plot(valid_df['Time(sec)'], valid_df['Spine_Dev_Angle'], label='Spine Tilt', color='green', linewidth=2)
                ax_graph.plot(valid_df['Time(sec)'], valid_df['Shoulder_Tilt_Angle'].abs(), label='Shoulder Tilt', color='blue', linewidth=2)
                ax_graph.plot(valid_df['Time(sec)'], valid_df['Hip_Tilt_Angle'].abs(), label='Hip Tilt', color='purple', linewidth=2)
                
                ax_graph.axhline(y=5.0, color='red', linestyle='--', label='Tolerance Threshold (5°)', linewidth=2)
                
                ax_graph.set_ylabel("Absolute Deviation Angle (deg)", fontsize=12)
                ax_graph.set_title("Postural Tilt Tracker over Time", fontsize=14, fontweight='bold')
                ax_graph.legend(loc='upper right')
                ax_graph.set_xlabel("Time (sec)", fontsize=12)
                ax_graph.grid(True, linestyle=':', alpha=0.7)
                
                if "Is_Active_Phase" in valid_df.columns:
                    ax_graph.fill_between(valid_df['Time(sec)'], ax_graph.get_ylim()[0], ax_graph.get_ylim()[1], 
                                          where=valid_df['Is_Active_Phase']==1, color='yellow', alpha=0.2, label='Active Phase')

            fig.text(0.10, 0.02, examiner_str, fontsize=13, fontweight='bold')
            fig.text(0.60, 0.02, f"{lbl_signature} ________________________", fontsize=13, fontweight='bold')
            plt.tight_layout(rect=[0, 0.06, 1, 0.84])
            
            plt.savefig(pdf_path, format='pdf')
            plt.close(fig)

        elif analysis == "ROM":
            fig = plt.figure(figsize=(8.27, 11.69))
            gs = GridSpec(2, 2, height_ratios=[1, 1.2], figure=fig)
            
            fig.suptitle(pdf_title, fontsize=18, fontweight='bold', y=0.97)
            fig.text(0.5, 0.93, f"{header_prefix}Mode: ROM Assessment", fontsize=13, fontweight='bold', ha='center')
            fig.text(0.95, 0.89, f"{lbl_name} {pt_name_str}", fontsize=12, fontweight='bold', ha='right', va='top')
            if pt_date not in ["00.00.00", "0.0.0", "..", ""]:
                fig.text(0.95, 0.87, f"{lbl_date} {pt_date}", fontsize=10, ha='right', va='top')

            ax_img = fig.add_subplot(gs[0, 0])
            ax_table = fig.add_subplot(gs[0, 1])
            ax_graph = fig.add_subplot(gs[1, :])

            cap = cv2.VideoCapture(self.temp_out)
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(self.total_frames / 2))
            ret, frame = cap.read()
            cap.release()
            if ret: ax_img.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else: ax_img.imshow(np.zeros((400, 400, 3), dtype=np.uint8))
            ax_img.axis('off')
            ax_img.set_title("Assessment Snapshot", fontsize=12, fontweight='bold', pad=10)

            angle_defs = {
                "L Elbow": (11, 13, 15), "R Elbow": (12, 14, 16),
                "L Shoulder": (23, 11, 13), "R Shoulder": (24, 12, 14),
                "L Hip": (11, 23, 25), "R Hip": (12, 24, 26),
                "L Knee": (23, 25, 27), "R Knee": (24, 26, 28)
            }
            
            rom_results = []
            plotted_angles = False
            
            for ang_name, (p1, p2, p3) in angle_defs.items():
                if p1 in self.selected_keys and p2 in self.selected_keys and p3 in self.selected_keys:
                    raw_n1 = self.landmark_names.get(p1, "")
                    raw_n2 = self.landmark_names.get(p2, "")
                    raw_n3 = self.landmark_names.get(p3, "")

                    n1 = raw_n1.split(".", 1)[-1].strip().replace(" ", "_") if raw_n1 else "Unknown1"
                    n2 = raw_n2.split(".", 1)[-1].strip().replace(" ", "_") if raw_n2 else "Unknown2"
                    n3 = raw_n3.split(".", 1)[-1].strip().replace(" ", "_") if raw_n3 else "Unknown3"
                    
                    c1x, c1y = f"{n1}_X", f"{n1}_Y"
                    c2x, c2y = f"{n2}_X", f"{n2}_Y"
                    c3x, c3y = f"{n3}_X", f"{n3}_Y"
                
                    if all(c in df.columns for c in [c1x, c1y, c2x, c2y, c3x, c3y]):
                        v1_x = df[c1x] - df[c2x]
                        v1_y = df[c1y] - df[c2y]
                        v2_x = df[c3x] - df[c2x]
                        v2_y = df[c3y] - df[c2y]
                        
                        dot = v1_x * v2_x + v1_y * v2_y
                        mag1 = np.sqrt(v1_x**2 + v1_y**2)
                        mag2 = np.sqrt(v2_x**2 + v2_y**2)
                        denom = (mag1 * mag2).replace(0, np.nan)
                        
                        ang_deg = np.degrees(np.arccos(np.clip(dot / denom, -1.0, 1.0)))
                        
                        valid_mask = ang_deg.notna()
                        if valid_mask.sum() > 3:
                            win_len = valid_mask.sum()
                            win_len = win_len if win_len % 2 != 0 else win_len - 1
                            win_len = min(11, win_len)
                            
                            if win_len > 2:
                                ang_smoothed = savgol_filter(ang_deg.loc[valid_mask], window_length=win_len, polyorder=2)
                                ang_deg.loc[valid_mask] = ang_smoothed
                            
                            min_ang = ang_deg.min()
                            max_ang = ang_deg.max()
                            rom_val = max_ang - min_ang
                            rom_results.append([ang_name, f"{min_ang:.1f}°", f"{max_ang:.1f}°", f"{rom_val:.1f}°"])
                            
                            ax_graph.plot(df['Time(sec)'], ang_deg, label=f"{ang_name} (ROM: {rom_val:.1f}°)", linewidth=2)
                            plotted_angles = True

            ax_table.axis('off')
            if rom_results:
                table_data = [["Joint", "Min", "Max", "ROM"]] + rom_results[:8]
                table = ax_table.table(cellText=table_data, loc='center', cellLoc='center', bbox=[0, 0.05, 1, 0.82])
                table.auto_set_font_size(False)
                table.set_fontsize(10)
                for (row, col), cell in table.get_celld().items():
                    if row == 0:
                        cell.set_text_props(weight='bold', color='white')
                        cell.set_facecolor('#4472C4')
            else:
                ax_table.text(0.5, 0.5, "No joints selected for ROM.\nPlease select 3 points in Body tab\n(e.g., Hip, Knee, Ankle)", ha='center', va='center')
                
            ax_table.set_title("Calculated ROM Summary", fontsize=12, fontweight='bold', y=0.9)

            if plotted_angles:
                ax_graph.set_ylabel("Joint Angle (deg)", fontsize=12)
                ax_graph.set_title("Continuous Angle Tracking", fontsize=14, fontweight='bold')
                ax_graph.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=9, ncol=4)
            else:
                ax_graph.text(0.5, 0.5, "No angle data available.", ha='center', va='center')
                
            ax_graph.set_xlabel("Time (sec)", fontsize=12)
            ax_graph.grid(True, linestyle=':', alpha=0.7)

            if "Is_Active_Phase" in df.columns:
                ax_graph.fill_between(df['Time(sec)'], ax_graph.get_ylim()[0], ax_graph.get_ylim()[1], 
                                      where=df['Is_Active_Phase']==1, color='yellow', alpha=0.2, label='Active Phase')

            fig.text(0.10, 0.02, examiner_str, fontsize=13, fontweight='bold')
            fig.text(0.60, 0.02, f"{lbl_signature} ________________________", fontsize=13, fontweight='bold')
            plt.tight_layout(rect=[0, 0.06, 1, 0.84])
            
            plt.savefig(pdf_path, format='pdf')
            plt.close(fig)

        elif sport == "Workout" and analysis in ["Squat", "Sit to Stand", "Gait", "Timed Up and Go", "Static Balance"]:
            fig = plt.figure(figsize=(8.27, 11.69))
            gs = GridSpec(3, 2, height_ratios=[1, 1, 1.2], figure=fig)

            fig.suptitle(pdf_title, fontsize=18, fontweight='bold', y=0.97)
            fig.text(0.5, 0.93, f"{header_prefix}Analysis: {analysis}", fontsize=13, fontweight='bold', ha='center')
            fig.text(0.95, 0.89, f"{lbl_name} {pt_name_str}", fontsize=12, fontweight='bold', ha='right', va='top')
            if pt_date not in ["00.00.00", "0.0.0", "..", ""]:
                fig.text(0.95, 0.87, f"{lbl_date} {pt_date}", fontsize=10, ha='right', va='top')

            ax_img = fig.add_subplot(gs[0, 0])
            ax_table = fig.add_subplot(gs[0, 1])
            ax_graph_main = fig.add_subplot(gs[1, :])
            ax_graph_angles = fig.add_subplot(gs[2, :])

            valid_df = df.dropna(subset=['Measured_Value']).copy()
            if len(valid_df) > 0:
                peak_idx = valid_df['Measured_Value'].idxmax()
                peak_val = valid_df.loc[peak_idx, 'Measured_Value']
            else:
                peak_idx, peak_val = 0, 0

            cap = cv2.VideoCapture(self.temp_out)
            cap.set(cv2.CAP_PROP_POS_FRAMES, peak_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret: ax_img.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else: ax_img.imshow(np.zeros((400, 400, 3), dtype=np.uint8))
            ax_img.axis('off')
            ax_img.set_title(f"Analysis Snapshot", fontsize=12, fontweight='bold', pad=10)

            total_reps = 0
            successful_reps = 0
            
            if 'Rep_Count' in df.columns:
                valid_reps = pd.to_numeric(df['Rep_Count'], errors='coerce').dropna()
                if not valid_reps.empty:
                    total_reps = int(valid_reps.max())
            
            if total_reps > 0:
                for r in range(total_reps):
                    rep_df = df[pd.to_numeric(df['Rep_Count'], errors='coerce') == r]
                    if 'Is_Active_Phase' in rep_df.columns and rep_df['Is_Active_Phase'].sum() > 0:
                        active_df = rep_df[rep_df['Is_Active_Phase'] == 1]
                        if active_df['Error_Flag'].sum() == 0:
                            successful_reps += 1
                    else:
                        if rep_df['Error_Flag'].sum() == 0:
                            successful_reps += 1

            active_frames = df['Is_Active_Phase'].sum() if 'Is_Active_Phase' in df.columns else 0
            total_active_time = active_frames / (self.fps if hasattr(self, 'fps') and self.fps > 0 else 30.0)

            table_data = [["Metric", "Value"]]
            video_start_val = df['Time(sec)'].iloc[0]
            video_end_val = df['Time(sec)'].iloc[-1]
            video_duration = video_end_val - video_start_val

            if analysis == "Gait":
                table_data.append(["Video Playback Time", f"{video_duration:.2f} s"])
                if 'Measured_Value' in df.columns:
                    table_data.append(["Avg Step Width", f"{df['Measured_Value'].mean():.1f} px"])
                if 'Step_Count' in df.columns:
                    table_data.append(["Total Steps", f"{int(df['Step_Count'].max())} steps"])
                    
            elif analysis == "Timed Up and Go":
                table_data.append(["Video Playback Time", f"{video_duration:.2f} s"])
                if "TUG_Time_sec" in df.columns:
                    final_tug_time = df["TUG_Time_sec"].max()
                    table_data.append(["Final TUG Time (Record)", f"{final_tug_time:.2f} seconds"])
                if 'Step_Count' in df.columns:
                    table_data.append(["Total Steps", f"{int(df['Step_Count'].max())} steps"])

            elif analysis in ["Squat", "Sit to Stand"]:
                table_data.append(["Total Repetitions", f"{total_reps} reps"])
                
                if total_reps > 0:
                    success_rate_pct = (successful_reps / total_reps) * 100
                    table_data.append(["Success Rate", f"{successful_reps} / {total_reps} ({success_rate_pct:.0f}%)"])
                else:
                    table_data.append(["Success Rate", "0 / 0 (0%)"])
                
                if analysis == "Sit to Stand":
                    table_data.append(["Total Active Time", f"{total_active_time:.2f} s"])
                if 'Measured_Value' in df.columns:
                    table_data.append(["Min Knee Angle (Deepest)", f"{df['Measured_Value'].min():.1f}°"])

            if "Body_Speed_px_s" in df.columns:
                table_data.append(["Avg Speed (Body)", f"{df['Body_Speed_px_s'].mean():.1f} px/s"])
                
            for tool_name in ["Ball", "Baseball_Bat", "Tennis_Racket", "Golf_Club"]:
                spd_col = f"{tool_name}_Speed_px_s"
                if spd_col in df.columns and not df[spd_col].isna().all():
                    table_data.append([f"Max Speed ({tool_name.replace('_', ' ')})", f"{df[spd_col].max():.1f} px/s"])

            ax_table.axis('off')
            table = ax_table.table(cellText=table_data, loc='center', cellLoc='center', bbox=[0, 0.1, 1, 0.8])
            table.auto_set_font_size(False)
            table.set_fontsize(11)
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_facecolor('#4472C4')
                    cell.set_text_props(color='white', weight='bold')
            ax_table.set_title("Performance Summary", fontsize=12, fontweight='bold', y=0.9)

            if 'Measured_Value' in valid_df.columns:
                param_name = df['Analysis_Param_Name'].dropna().iloc[0] if 'Analysis_Param_Name' in df else "Measured Parameter"
                ax_graph_main.plot(valid_df['Time(sec)'], valid_df['Measured_Value'], color='blue', label=param_name)
                ax_graph_main.set_title(f"{analysis} Trend Over Time", fontsize=12, fontweight='bold')
                ax_graph_main.set_ylabel("Value")
                ax_graph_main.grid(True, alpha=0.3)
                ax_graph_main.legend()

            if analysis == "Timed Up and Go" and 'Knee_Angle' in df.columns:
                knee_df = df.dropna(subset=['Knee_Angle'])
                if not knee_df.empty:
                    ax_graph_angles.plot(knee_df['Time(sec)'], knee_df['Knee_Angle'],
                                         color='darkorange', linewidth=2, label='Knee Angle (deg)')
                    ax_graph_angles.axhline(y=150.0, color='green', linestyle='--', linewidth=1.5,
                                            label='Stand Threshold (150°)')
                    ax_graph_angles.axhline(y=120.0, color='red', linestyle='--', linewidth=1.5,
                                            label='Sit Threshold (120°)')

                    ax_graph_angles.set_title("TUG Knee Angle Tracker",
                                              fontsize=12, fontweight='bold')
                    ax_graph_angles.set_xlabel("Time (sec)", fontsize=11)
                    ax_graph_angles.set_ylabel("Knee Angle (deg)", fontsize=11)
                    ax_graph_angles.legend(fontsize=8, loc='upper right')
                    ax_graph_angles.grid(True, alpha=0.3)

            elif "R_Knee_X" in df.columns and "L_Knee_X" in df.columns and "R_Hip_X" in df.columns:
                v1_xl, v1_yl = df['L_Hip_X'] - df['L_Knee_X'], df['L_Hip_Y'] - df['L_Knee_Y']
                v2_xl, v2_yl = df['L_Ankle_X'] - df['L_Knee_X'], df['L_Ankle_Y'] - df['L_Knee_Y']
                l_knee_ang = np.degrees(np.arccos(np.clip((v1_xl*v2_xl + v1_yl*v2_yl) / (np.sqrt(v1_xl**2 + v1_yl**2) * np.sqrt(v2_xl**2 + v2_yl**2)).replace(0, np.nan), -1.0, 1.0)))

                v1_xr, v1_yr = df['R_Hip_X'] - df['R_Knee_X'], df['R_Hip_Y'] - df['R_Knee_Y']
                v2_xr, v2_yr = df['R_Ankle_X'] - df['R_Knee_X'], df['R_Ankle_Y'] - df['R_Knee_Y']
                r_knee_ang = np.degrees(np.arccos(np.clip((v1_xr*v2_xr + v1_yr*v2_yr) / (np.sqrt(v1_xr**2 + v1_yr**2) * np.sqrt(v2_xr**2 + v2_yr**2)).replace(0, np.nan), -1.0, 1.0)))

                ax_graph_angles.plot(df['Time(sec)'], l_knee_ang, label='Left Knee Angle', color='green', alpha=0.8)
                ax_graph_angles.plot(df['Time(sec)'], r_knee_ang, label='Right Knee Angle', color='red', alpha=0.8)

                ax_graph_angles.set_title("Lower Limb Kinematics (Joint Angles)", fontsize=12, fontweight='bold')
                ax_graph_angles.set_xlabel("Time (sec)")
                ax_graph_angles.set_ylabel("Angle (deg)")
                ax_graph_angles.legend()
                ax_graph_angles.grid(True, linestyle='--')

            fig.text(0.10, 0.02, examiner_str, fontsize=13, fontweight='bold')
            fig.text(0.60, 0.02, f"{lbl_signature} ________________________", fontsize=13, fontweight='bold')
            plt.tight_layout(rect=[0, 0.06, 1, 0.84])

            plt.savefig(pdf_path, format='pdf')
            plt.close(fig)

        # 4. 야구, 골프 특정 동작 분석 모드 (1장짜리 리포트 출력)
        elif sport in ["Golf", "Baseball"] and analysis not in ["Free", "None", ""] and "Measured_Value" in df.columns and not df["Measured_Value"].isna().all():
            with PdfPages(pdf_path) as pdf:
                fig = plt.figure(figsize=(8.27, 11.69))
                gs = GridSpec(2, 2, height_ratios=[1, 1.2], figure=fig)
                
                display_analysis = analysis

                fig.suptitle(pdf_title, fontsize=18, fontweight='bold', y=0.97)
                fig.text(0.5, 0.93, f"{header_prefix}Analysis: {display_analysis}", fontsize=13, fontweight='bold', ha='center')
                fig.text(0.95, 0.89, f"{lbl_name} {pt_name_str}", fontsize=12, fontweight='bold', ha='right', va='top')
                if pt_date not in ["00.00.00", "0.0.0", "..", ""]:
                    fig.text(0.95, 0.87, f"{lbl_date} {pt_date}", fontsize=10, ha='right', va='top')

                ax_img = fig.add_subplot(gs[0, 0])
                ax_table = fig.add_subplot(gs[0, 1])
                ax_graph = fig.add_subplot(gs[1, :])

                valid_params = df['Analysis_Param_Name'].dropna() if 'Analysis_Param_Name' in df else pd.Series()
                param_name = valid_params.iloc[0] if not valid_params.empty else "Measured Parameter"

                valid_targets = df['Target_Value'].dropna() if 'Target_Value' in df else pd.Series()
                target_val = valid_targets.iloc[0] if not valid_targets.empty else 0
                
                valid_df = df.dropna(subset=['Measured_Value']).copy()
                if len(valid_df) > 0:
                    valid_df['Deviation'] = (valid_df['Measured_Value'] - target_val).abs()
                    peak_idx = valid_df['Deviation'].idxmax()
                    peak_val = valid_df.loc[peak_idx, 'Measured_Value']
                else:
                    peak_idx = 0
                    peak_val = 0

                has_error = df['Error_Flag'].sum() > 0
                status_text = o.get("status_fail", "FAIL (Limit Reached)") if has_error else o.get("status_pass", "PASS (In Range)")
                status_color = "red" if has_error else "green"

                cap = cv2.VideoCapture(self.temp_out)
                cap.set(cv2.CAP_PROP_POS_FRAMES, peak_idx)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = np.zeros((400, 400, 3), dtype=np.uint8)
                    cv2.putText(frame_rgb, "Image Load Error", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                ax_img.imshow(frame_rgb)
                ax_img.axis('off')
                ax_img.set_title(f"Peak Deviation Snapshot ({peak_val:.1f})", fontsize=12, fontweight='bold', pad=10)

                ax_table.axis('off')
                table_data = [
                    ["Metric", "Value"],
                    ["Parameter", param_name],
                    ["Target (Norm)", f"{target_val:.1f}"],
                    ["Peak Measured", f"{peak_val:.1f}"]
                ]
                
                if "Rep_Count" in df.columns and analysis in ["Squat", "Sit to Stand"]:
                    total_reps = int(df['Rep_Count'].max())
                    table_data.insert(4, ["Total Reps", f"{total_reps} reps"])
                
                if "Body_Speed_px_s" in df.columns:
                    max_spd = df["Body_Speed_px_s"].max()
                    avg_spd = df["Body_Speed_px_s"].mean()
                    table_data.append(["Max Speed (Body)", f"{max_spd:.1f} px/s"])
                    table_data.append(["Avg Speed (Body)", f"{avg_spd:.1f} px/s"])
                    
                for tool_name in ["Ball", "Baseball_Bat", "Tennis_Racket", "Golf_Club"]:
                    spd_col = f"{tool_name}_Speed_px_s"
                    if spd_col in df.columns and not df[spd_col].isna().all():
                        table_data.append([f"Max Speed ({tool_name.replace('_', ' ')})", f"{df[spd_col].max():.1f} px/s"])

                table_data.append(["Status", status_text])

                table = ax_table.table(cellText=table_data, loc='center', cellLoc='center', bbox=[0, 0.05, 1, 0.82])
                table.auto_set_font_size(False)
                table.set_fontsize(11)
                
                for (row, col), cell in table.get_celld().items():
                    if col == 0:
                        cell.set_text_props(weight='bold', color='white')
                        cell.set_facecolor('#4472C4')
                    if row == (len(table_data)-1) and col == 1:
                        cell.set_text_props(weight='bold', color=status_color)

                ax_table.set_title("Analysis Summary", fontsize=12, fontweight='bold', y=0.9)

                ax_graph.plot(valid_df['Time(sec)'], valid_df['Measured_Value'], label=f'{param_name} Trend', color='blue', linewidth=2)
                ax_graph.axhline(y=target_val, color='orange', linestyle='--', label='Target / Threshold', linewidth=2)
                
                if "Is_Active_Phase" in valid_df.columns:
                    ax_graph.fill_between(valid_df['Time(sec)'], ax_graph.get_ylim()[0], ax_graph.get_ylim()[1], 
                                          where=valid_df['Is_Active_Phase']==1, color='yellow', alpha=0.2, label='Active Phase (Moving)')

                error_df = df[df['Error_Flag'] == 1]
                if not error_df.empty:
                    ax_graph.scatter(error_df['Time(sec)'], error_df['Measured_Value'], color='red', zorder=5, label='Error Detected', s=15)

                ax_graph.set_xlabel("Time (sec)", fontsize=12)
                ax_graph.set_ylabel("Value", fontsize=12)
                ax_graph.set_title(f"{param_name} over Time", fontsize=14, fontweight='bold')
                ax_graph.legend(loc='upper right')
                ax_graph.grid(True, linestyle=':', alpha=0.7)

                fig.text(0.10, 0.02, examiner_str, fontsize=13, fontweight='bold')
                fig.text(0.60, 0.02, f"{lbl_signature} ________________________", fontsize=13, fontweight='bold')
                plt.tight_layout(rect=[0, 0.06, 1, 0.84])
                
                pdf.savefig(fig)
                plt.close(fig)

# 4-1. Individual (Custom JSON) 모드 (JSON 내 여러 분석 존재 시 다중 페이지 생성)
        elif sport == "Individual" and analysis in self.custom_analyses_data:
            with PdfPages(pdf_path) as pdf:
                run_analyses = self.custom_analyses_data[analysis]

                # 루프에서 인덱스(i)를 가져와 페이지 번호를 구분합니다.
                for i, custom_cfg in enumerate(run_analyses):
                    a_name = custom_cfg.get("analysis_name", "Custom")
                    val_col = f"{a_name}_Measured_Value"
                    tgt_col = f"{a_name}_Target_Value"
                    param_col = f"{a_name}_Param_Name"

                    if val_col not in df.columns or df[val_col].isna().all():
                        continue

                    fig = plt.figure(figsize=(8.27, 11.69))
                    gs = GridSpec(2, 2, height_ratios=[1, 1.2], figure=fig)
                    
                    # [1] 첫 번째 페이지와 마지막 페이지에만 제목, 부제목, 환자 이름 출력
                    if i == 0 or i == len(run_analyses) - 1:
                        fig.suptitle(pdf_title, fontsize=18, fontweight='bold', y=0.97)
                        # 부제목: JSON 파일 이름(analysis)만 깔끔하고 돋보이게 표시
                        fig.text(0.5, 0.93, f"{analysis}", fontsize=14, fontweight='bold', color='midnightblue', ha='center')
                        fig.text(0.95, 0.89, f"{lbl_name} {pt_name_str}", fontsize=12, fontweight='bold', ha='right', va='top')
                        if pt_date not in ["00.00.00", "0.0.0", "..", ""]:
                            fig.text(0.95, 0.87, f"{lbl_date} {pt_date}", fontsize=10, ha='right', va='top')
                        top_margin = 0.84 # 상단 여백 확보
                    else:
                        top_margin = 0.94 # 중간 페이지는 제목이 없으므로 차트 영역을 위로 올림

                    ax_img = fig.add_subplot(gs[0, 0])
                    ax_table = fig.add_subplot(gs[0, 1])
                    ax_graph = fig.add_subplot(gs[1, :])

                    param_name = df[param_col].dropna().iloc[0] if param_col in df else "Custom Value"
                    target_val = df[tgt_col].dropna().iloc[0] if tgt_col in df else 0
                    
                    valid_df = df.dropna(subset=[val_col]).copy()
                    if len(valid_df) > 0:
                        valid_df['Deviation'] = (valid_df[val_col] - target_val).abs()
                        peak_idx = valid_df['Deviation'].idxmax()
                        peak_val = valid_df.loc[peak_idx, val_col]
                    else:
                        peak_idx, peak_val = 0, 0

                    condition = custom_cfg.get("condition", "over")
                    is_error_triggered = (condition == "over" and (valid_df[val_col] > target_val).any()) or \
                                         (condition == "under" and (valid_df[val_col] < target_val).any())
                    
                    # --- 추가된 부분: 처음 에러가 발생하기까지 걸린 시간 계산 ---
                    if is_error_triggered:
                        if condition == "over":
                            first_err_idx = (valid_df[val_col] > target_val).idxmax()
                        else:
                            first_err_idx = (valid_df[val_col] < target_val).idxmax()
                            
                        start_time = valid_df['Time(sec)'].iloc[0]
                        first_err_time = valid_df.loc[first_err_idx, 'Time(sec)']
                        time_to_error = first_err_time - start_time
                        time_to_error_str = f"{time_to_error:.2f} s"
                    else:
                        time_to_error_str = "No Error"
                    # -------------------------------------------------------------

                    status_text = o.get("status_fail", "FAIL (Limit Reached)") if is_error_triggered else o.get("status_pass", "PASS (In Range)")
                    status_color = "red" if is_error_triggered else "green"

                    cap = cv2.VideoCapture(self.temp_out)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, peak_idx)
                    ret, frame = cap.read()
                    cap.release()
                    
                    if ret:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = np.zeros((400, 400, 3), dtype=np.uint8)
                        cv2.putText(frame_rgb, "Image Load Error", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

                    ax_img.imshow(frame_rgb)
                    ax_img.axis('off')
                    ax_img.set_title(f"{a_name} - Peak Snapshot", fontsize=12, fontweight='bold', pad=10)

                    ax_table.axis('off')
                    
                    # --- 수정된 부분: Metric / Value 대신 Time to First Error 삽입 ---
                    table_data = [
                        ["Time to First Error", time_to_error_str],
                        ["Parameter", param_name],
                        ["Target (Norm)", f"{target_val:.1f}"],
                        ["Peak Measured", f"{peak_val:.1f}"]
                    ]
                    # -----------------------------------------------------------------
                    
                    if "Body_Speed_px_s" in df.columns:
                        table_data.append(["Avg Speed (Body)", f"{df['Body_Speed_px_s'].mean():.1f} px/s"])

                    table_data.append(["Status", status_text])

                    table = ax_table.table(cellText=table_data, loc='center', cellLoc='center', bbox=[0, 0.1, 1, 0.8])
                    table.auto_set_font_size(False)
                    table.set_fontsize(11)
                    
                    for (row, col), cell in table.get_celld().items():
                        if col == 0:
                            cell.set_text_props(weight='bold', color='white')
                            cell.set_facecolor('#4472C4')
                        if row == (len(table_data)-1) and col == 1:
                            cell.set_text_props(weight='bold', color=status_color)

                    ax_table.set_title(f"{a_name} Summary", fontsize=12, fontweight='bold', y=0.9)

                    ax_graph.plot(valid_df['Time(sec)'], valid_df[val_col], label=f'{param_name} Trend', color='blue', linewidth=2)
                    ax_graph.axhline(y=target_val, color='orange', linestyle='--', label='Target / Threshold', linewidth=2)
                    
                    if "Is_Active_Phase" in valid_df.columns:
                        ax_graph.fill_between(valid_df['Time(sec)'], ax_graph.get_ylim()[0], ax_graph.get_ylim()[1], 
                                              where=valid_df['Is_Active_Phase']==1, color='yellow', alpha=0.2, label='Active Phase (Moving)')

                    ax_graph.set_xlabel("Time (sec)", fontsize=12)
                    ax_graph.set_ylabel("Value", fontsize=12)
                    ax_graph.set_title(f"{param_name} Tracker", fontsize=14, fontweight='bold')
                    ax_graph.legend(loc='upper right')
                    ax_graph.grid(True, linestyle=':', alpha=0.7)

                    # [2] 마지막 페이지만 검사자 정보 및 서명란 출력
                    if i == len(run_analyses) - 1:
                        fig.text(0.10, 0.02, examiner_str, fontsize=13, fontweight='bold')
                        fig.text(0.60, 0.02, f"{lbl_signature} ________________________", fontsize=13, fontweight='bold')
                        bottom_margin = 0.06 # 하단 여백 확보
                    else:
                        bottom_margin = 0.02 # 서명란이 없으므로 하단 공간 활용

                    plt.tight_layout(rect=[0, bottom_margin, 1, top_margin])
                    
                    pdf.savefig(fig)
                    plt.close(fig)

        # 5. Free 모드 선택 시
        else:
            with PdfPages(pdf_path) as pdf:
                fig1 = plt.figure(figsize=(8.27, 11.69))
                gs1 = GridSpec(2, 2, height_ratios=[1, 1.2], figure=fig1)
                
                display_analysis = "________________________"
                
                fig1.suptitle(pdf_title, fontsize=18, fontweight='bold', y=0.97)
                fig1.text(0.5, 0.93, f"{header_prefix}Analysis: {display_analysis}", fontsize=13, fontweight='bold', ha='center')
                fig1.text(0.95, 0.89, f"{lbl_name} {pt_name_str}", fontsize=12, fontweight='bold', ha='right', va='top')
                if pt_date not in ["00.00.00", "0.0.0", "..", ""]:
                    fig1.text(0.95, 0.87, f"{lbl_date} {pt_date}", fontsize=10, ha='right', va='top')

                ax_img = fig1.add_subplot(gs1[0, 0])
                ax_table = fig1.add_subplot(gs1[0, 1])
                ax_graph = fig1.add_subplot(gs1[1, :])

                cap = cv2.VideoCapture(self.temp_out)
                mid_frame = int(self.total_frames / 2)
                cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
                ret, frame = cap.read()
                cap.release()
                
                if ret:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = np.zeros((400, 400, 3), dtype=np.uint8)

                ax_img.imshow(frame_rgb)
                ax_img.axis('off')
                ax_img.set_title("Reference Snapshot", fontsize=12, fontweight='bold', pad=10)

                plot_data = {}
                max_all_speed = 0
                all_speeds = []
                
                for key in self.selected_keys:
                    name = self.landmark_names.get(key, "").split(". ")[-1].replace(" ", "_")
                    col_x = f"{name}_X"
                    col_y = f"{name}_Y"
                    if col_x in df.columns and col_y in df.columns and not df[col_x].isna().all():
                        dx = df[col_x].diff().fillna(0)
                        dy = df[col_y].diff().fillna(0)
                        dt = df["Time(sec)"].diff().fillna(1/self.fps)
                        dt[dt == 0] = 1/self.fps 
                        speed = np.sqrt(dx**2 + dy**2) / dt
                        
                        win_len = min(11, len(speed) if len(speed) % 2 != 0 else len(speed) - 1)
                        if win_len > 2:
                            speed = pd.Series(savgol_filter(speed.values, window_length=win_len, polyorder=2))
                        
                        plot_data[name] = speed
                        max_all_speed = max(max_all_speed, speed.max())
                        all_speeds.extend(speed.dropna().values)
                
                for tool_name in ["Ball", "Baseball_Bat", "Tennis_Racket", "Golf_Club"]:
                    spd_col = f"{tool_name}_Speed_px_s"
                    if spd_col in df.columns and not df[spd_col].isna().all():
                        speed = df[spd_col]
                        plot_data[tool_name.replace("_", " ")] = speed
                        max_all_speed = max(max_all_speed, speed.max())
                        all_speeds.extend(speed.dropna().values)

                avg_all_speed = np.mean(all_speeds) if len(all_speeds) > 0 else 0

                ax_table.axis('off')
                table_data = [
                    ["Metric", "Value"],
                    ["Mode", "Custom Tracking"],
                    ["Tracked Points", f"{len(self.selected_keys)} joints"],
                    ["Max Speed (Peak)", f"{max_all_speed:.1f} px/s"],
                    ["Avg Speed (Overall)", f"{avg_all_speed:.1f} px/s"]
                ]
                
                for tool_name in ["Ball", "Baseball_Bat", "Tennis_Racket", "Golf_Club"]:
                    spd_col = f"{tool_name}_Speed_px_s"
                    if spd_col in df.columns and not df[spd_col].isna().all():
                        table_data.append([f"Max Speed ({tool_name.replace('_', ' ')})", f"{df[spd_col].max():.1f} px/s"])

                table_data.append(["Status", "Manual Review"])
                
                table = ax_table.table(cellText=table_data, loc='center', cellLoc='center', bbox=[0, 0.05, 1, 0.82])
                table.auto_set_font_size(False)
                table.set_fontsize(11)
                
                for (row, col), cell in table.get_celld().items():
                    if col == 0:
                        cell.set_text_props(weight='bold', color='white')
                        cell.set_facecolor('#4472C4')

                ax_table.set_title("Tracking Summary", fontsize=12, fontweight='bold', y=0.9)

                plotted_something = False
                for name, speed in plot_data.items():
                    ax_graph.plot(df['Time(sec)'], speed, label=name, linewidth=1.5)
                    plotted_something = True
                        
                if plotted_something:
                    ax_graph.set_ylabel("Movement Velocity (px/sec)", fontsize=12)
                    ax_graph.set_title("Selected Landmarks & Objects Speed Trend", fontsize=14, fontweight='bold')
                    
                    if len(plot_data) > 10:
                        ax_graph.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), fontsize=8, ncol=4)
                    else:
                        ax_graph.legend(loc='upper right', fontsize=8)
                else:
                    ax_graph.text(0.5, 0.5, "No landmarks/objects selected for tracking", ha='center', va='center')
                    ax_graph.set_title("Movement Speed Tracker", fontsize=14, fontweight='bold')
                    
                ax_graph.set_xlabel("Time (sec)", fontsize=12)
                ax_graph.grid(True, linestyle=':', alpha=0.7)

                if "Is_Active_Phase" in df.columns:
                    ax_graph.fill_between(df['Time(sec)'], ax_graph.get_ylim()[0], ax_graph.get_ylim()[1], 
                                          where=df['Is_Active_Phase']==1, color='yellow', alpha=0.2, label='Active Phase')

                if len(plot_data) > 10:
                    plt.tight_layout(rect=[0, 0.08, 1, 0.84]) 
                else:
                    plt.tight_layout(rect=[0, 0.06, 1, 0.84])
                
                pdf.savefig(fig1)
                plt.close(fig1)

                fig2 = plt.figure(figsize=(8.27, 11.69))
                
                # Split the page into 2 rows, place the chart in the top row to move it up
                gs2 = GridSpec(2, 1, height_ratios=[1, 1], figure=fig2)
                
                # Create the chart at the top row (Title will be centered automatically)
                ax_angle = fig2.add_subplot(gs2[0, 0])
                
                angle_defs = {
                    "Left Elbow (11-13-15)": (11, 13, 15),
                    "Right Elbow (12-14-16)": (12, 14, 16),
                    "Left Shoulder (23-11-13)": (23, 11, 13),
                    "Right Shoulder (24-12-14)": (24, 12, 14),
                    "Left Hip (11-23-25)": (11, 23, 25),
                    "Right Hip (12-24-26)": (12, 24, 26),
                    "Left Knee (23-25-27)": (23, 25, 27),
                    "Right Knee (24-26-28)": (24, 26, 28)
                }
                
                plotted_angles = False
                for ang_name, (p1, p2, p3) in angle_defs.items():
                    if p1 in self.selected_keys and p2 in self.selected_keys and p3 in self.selected_keys:
                        raw_n1 = self.landmark_names.get(p1, "")
                        raw_n2 = self.landmark_names.get(p2, "")
                        raw_n3 = self.landmark_names.get(p3, "")

                        n1 = raw_n1.split(".", 1)[-1].strip().replace(" ", "_") if raw_n1 else "Unknown1"
                        n2 = raw_n2.split(".", 1)[-1].strip().replace(" ", "_") if raw_n2 else "Unknown2"
                        n3 = raw_n3.split(".", 1)[-1].strip().replace(" ", "_") if raw_n3 else "Unknown3"
                        
                        c1x, c1y = f"{n1}_X", f"{n1}_Y"
                        c2x, c2y = f"{n2}_X", f"{n2}_Y"
                        c3x, c3y = f"{n3}_X", f"{n3}_Y"
                    
                        if all(c in df.columns for c in [c1x, c1y, c2x, c2y, c3x, c3y]):
                            v1_x = df[c1x] - df[c2x]
                            v1_y = df[c1y] - df[c2y]
                            v2_x = df[c3x] - df[c2x]
                            v2_y = df[c3y] - df[c2y]
                            
                            dot = v1_x * v2_x + v1_y * v2_y
                            mag1 = np.sqrt(v1_x**2 + v1_y**2)
                            mag2 = np.sqrt(v2_x**2 + v2_y**2)
                            
                            denom = mag1 * mag2
                            denom = denom.replace(0, np.nan)
                            
                            cos_ang = np.clip(dot / denom, -1.0, 1.0)
                            ang_deg = np.degrees(np.arccos(cos_ang))
                            
                            valid_mask = ang_deg.notna()
                            if valid_mask.sum() > 3:
                                win_len = min(11, valid_mask.sum() if valid_mask.sum() % 2 != 0 else valid_mask.sum() - 1)
                                if win_len > 2:
                                    ang_deg.loc[valid_mask] = savgol_filter(ang_deg.loc[valid_mask], window_length=win_len, polyorder=2)
                            
                            label_name = " ".join(ang_name.split(" ")[:2])
                            ax_angle.plot(df['Time(sec)'], ang_deg, label=label_name, linewidth=1.5)
                            plotted_angles = True
                
                if plotted_angles:
                    ax_angle.set_ylabel("Joint Angle (deg)", fontsize=12)
                    ax_angle.set_title("Selected Major Joint Angles Trend", fontsize=14, fontweight='bold')
                    ax_angle.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=10, ncol=4)
                else:
                    ax_angle.text(0.5, 0.5, "To see angle graphs, please select at least 3 connected joints in 'Body Landmarks' tab.\n(e.g., Shoulder + Elbow + Wrist)", ha='center', va='center', fontsize=12)
                    ax_angle.set_title("Movement Angle Tracker", fontsize=14, fontweight='bold')
                
                ax_angle.set_xlabel("Time (sec)", fontsize=12)
                ax_angle.grid(True, linestyle=':', alpha=0.7)

                if "Is_Active_Phase" in df.columns:
                    ax_angle.fill_between(df['Time(sec)'], ax_angle.get_ylim()[0], ax_angle.get_ylim()[1], 
                                          where=df['Is_Active_Phase']==1, color='yellow', alpha=0.2, label='Active Phase')
                
                fig2.text(0.10, 0.02, examiner_str, fontsize=13, fontweight='bold')
                fig2.text(0.60, 0.02, f"{lbl_signature} ________________________", fontsize=13, fontweight='bold')

                if len(self.selected_keys) > 10:
                    plt.tight_layout(rect=[0, 0.08, 1, 0.98]) 
                else:
                    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
                
                pdf.savefig(fig2)
                plt.close(fig2)

    def finish_analysis(self):
        self.status_var.set("Processing calculations and smoothing data...")
        self.root.update_idletasks()

        # 웹캠 컨트롤 팝업창 닫기 및 무한 로딩 바 정지
        if hasattr(self, 'webcam_ctrl_win') and self.webcam_ctrl_win.winfo_exists():
            self.webcam_ctrl_win.destroy()
            
        if getattr(self, 'is_webcam', False):
            self.progress.stop()
            self.progress.configure(mode='determinate', value=100)

        if not os.path.exists(self.temp_csv_path):
            self.status_var.set("No data processed.")
            self.btn_run.config(state=tk.NORMAL)
            if hasattr(self, 'btn_webcam_main'):
                self.btn_webcam_main.config(state=tk.NORMAL)
            self.is_running = False
            return

        # RAM에 있던 리스트 대신, 디스크에 안전하게 분할 저장된 CSV를 읽어옴
        df = pd.read_csv(self.temp_csv_path)
        dt = df["Time(sec)"].diff().replace(0, np.nan).fillna(1/self.fps) 
        df = df.interpolate(method='linear', limit_direction='both')
        win_len = max(3, len(df) if len(df) % 2 != 0 else len(df) - 1)
        if win_len > 11: win_len = 11

        for col in df.columns:
            if col not in ["Time(sec)", "Error_Flag", "Analysis_Param_Name", "Measured_Value", "Target_Value", "Rep_Count", "Is_Active_Phase", "Start_Time"] and df[col].notna().any():
                df[col] = savgol_filter(df[col].bfill().ffill().values, window_length=win_len, polyorder=3)

        if "L_Hip_X" in df.columns and "R_Hip_X" in df.columns and "L_Hip_Y" in df.columns and "R_Hip_Y" in df.columns:
            mid_hip_x = (df["L_Hip_X"] + df["R_Hip_X"]) / 2
            mid_hip_y = (df["L_Hip_Y"] + df["R_Hip_Y"]) / 2
            dx = mid_hip_x.diff().fillna(0)
            dy = mid_hip_y.diff().fillna(0)
            dt_s = df["Time(sec)"].diff().fillna(1/self.fps)
            dt_s[dt_s == 0] = 1/self.fps
            b_speed = np.sqrt(dx**2 + dy**2) / dt_s
            
            valid_mask = b_speed.notna()
            if valid_mask.sum() > 3:
                w_len = min(11, valid_mask.sum() if valid_mask.sum() % 2 != 0 else valid_mask.sum() - 1)
                if w_len > 2:
                    b_speed.loc[valid_mask] = savgol_filter(b_speed.loc[valid_mask], window_length=w_len, polyorder=2)
            df["Body_Speed_px_s"] = b_speed

        if "L_Ankle_X" in df.columns and "R_Ankle_X" in df.columns and "L_Ankle_Y" in df.columns and "R_Ankle_Y" in df.columns:
            df["Step_Width_px"] = np.sqrt((df["L_Ankle_X"] - df["R_Ankle_X"])**2 + (df["L_Ankle_Y"] - df["R_Ankle_Y"])**2)

        for tool_name in ["Ball", "Baseball_Bat", "Tennis_Racket", "Golf_Club"]:
            col_x, col_y = f"{tool_name}_X", f"{tool_name}_Y"
            if col_x in df.columns and col_y in df.columns and not df[col_x].isna().all():
                df[col_x] = df[col_x].interpolate(method='linear', limit_direction='both')
                df[col_y] = df[col_y].interpolate(method='linear', limit_direction='both')

                dx = df[col_x].diff().fillna(0)
                dy = df[col_y].diff().fillna(0)
                dt_s = df["Time(sec)"].diff().fillna(1/self.fps)
                dt_s[dt_s == 0] = 1/self.fps
                t_speed = np.sqrt(dx**2 + dy**2) / dt_s

                valid_mask = t_speed.notna()
                if valid_mask.sum() > 3:
                    w_len = min(11, valid_mask.sum() if valid_mask.sum() % 2 != 0 else valid_mask.sum() - 1)
                    if w_len > 2:
                        t_speed.loc[valid_mask] = savgol_filter(t_speed.loc[valid_mask], window_length=w_len, polyorder=2)
                df[f"{tool_name}_Speed_px_s"] = t_speed

        # --- [수정] Step_Count, TUG_Time_sec 추가 ---
        base_cols = ["Time(sec)", "Error_Flag", "Analysis_Param_Name", "Measured_Value", "Target_Value", "Rep_Count", "Is_Active_Phase", "Start_Time", "Step_Count", "TUG_Time_sec"]
        coord_cols = [c for c in df.columns if "_X" in c or "_Y" in c]
        final_col_order = base_cols + ["Body_Speed_px_s", "Step_Width_px"] + coord_cols + [c for c in df.columns if "Angle" in c or "Velocity" in c or "Acceleration" in c or "Speed" in c]
        final_col_order = list(dict.fromkeys([c for c in final_col_order if c in df.columns]))

        # ----------------------------------------------------
        # 수정 2: 파일명에 이름_날짜_종목_분석명_시간 통합 반영
        c_name = self.client_name_var.get().strip()
        safe_c_name = c_name if c_name and c_name != "None" else "Unknown"
        
        # 날짜 포맷 (미입력시 오늘 날짜)
        date_str = f"{self.date_yy_var.get().strip()}{self.date_mm_var.get().strip()}{self.date_dd_var.get().strip()}"
        safe_date = date_str if date_str != "000000" else datetime.now().strftime('%y%m%d')
        
        sport_str = self.sport_var.get()
        analysis_str = self.analysis_var.get()
        timestamp = datetime.now().strftime('%H%M%S')

        # 최종 파일명 조합: 예) 홍길동_260421_Workout_Squat_142500
        name_part = f"{safe_c_name}_{safe_date}_{sport_str}_{analysis_str}_{timestamp}"
        # ----------------------------------------------------

        df[final_col_order].to_excel(os.path.join(self.current_excel_dir, f"{name_part}_tracked_data.xlsx"), index=False)

        # PDF 리포트 생성 (sport, analysis 변수도 새로 만든 변수 적용)
        try:
            self.generate_pdf_report(df, name_part, sport_str, analysis_str)
        except Exception as e:
            print(f"PDF generation error: {e}")

        try:
            dst = os.path.join(self.current_result_dir, f"analyzed_{name_part}.mp4")
            if os.path.exists(self.temp_out):
                shutil.move(self.temp_out, dst)
            # 웹캠 모드에서는 temp_in이 정수(카메라 인덱스)이므로 삭제 불가
            if isinstance(self.temp_in, str) and os.path.exists(self.temp_in):
                os.remove(self.temp_in)
        except Exception as e:
            print(f"File cleanup/move error: {e}")

        self.btn_run.config(state=tk.NORMAL)
        if hasattr(self, 'btn_webcam_main'):
            self.btn_webcam_main.config(state=tk.NORMAL)
            
        self.status_var.set("Analysis complete!")
        messagebox.showinfo("Success", "Analysis, kinematics, and PDF Report completed successfully!")
        
        # [수정] 제어 플래그들을 확실하게 False로 변경
        self.is_running = False
        self.is_webcam = False
        self.is_recording = False

if __name__ == "__main__":
    # 암호 확인 없이 바로 메인 윈도우를 생성하고 실행합니다.
    root = tk.Tk()
    app = BiomechanicsAnalyzer(root)
    root.mainloop()