import os
import json
import uuid
import time
import sys
import shutil
import hashlib
import traceback
from datetime import datetime
import cv2
import numpy as np

# ==========================================================
#
# DATASET TYPE
#
# 1 = Videos IA
# 2 = Videos Reales
# 3 = Replay Attack
# 4 = Fotografía impresa
# 5 = Fotografía mostrada en teléfono
# 6 = Monitor / Pantalla
# 7 = Casos desconocidos
# 8 = Casos rechazados manualmente
#
# ==========================================================
DATASET_TYPE = 1

class DatasetLogger:
    def __init__(self, dataset_type=DATASET_TYPE):
        self.dataset_type = dataset_type
        self.type_folders = {
            1: "01_videos_ia",
            2: "02_videos_reales",
            3: "03_replay_attack",
            4: "04_fotografia_impresa",
            5: "05_foto_telefono",
            6: "06_monitor",
            7: "07_desconocidos",
            8: "08_rechazados"
        }
        
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        micro = now.strftime("%f")
        
        base_name = f"{timestamp}_{micro}"
        
        # Permite configurar la ruta raíz mediante variable de entorno (útil para GCP, AWS, etc.)
        self.dataset_root = os.getenv("DATASET_ROOT_PATH", os.path.join(os.getcwd(), "dataset"))
        type_folder = self.type_folders.get(self.dataset_type, "07_desconocidos")
        
        proposed_path = os.path.join(self.dataset_root, type_folder, base_name)
        
        if os.path.exists(proposed_path):
            unique_id = uuid.uuid4().hex[:6]
            base_name = f"{timestamp}_{micro}_{unique_id}"
            proposed_path = os.path.join(self.dataset_root, type_folder, base_name)
            
        self.base_name = base_name
        self.run_dir = proposed_path
        self.subdirs = {
            "video": os.path.join(self.run_dir, "video"),
            "capturas": os.path.join(self.run_dir, "capturas"),
            "frames_originales": os.path.join(self.run_dir, "frames_originales"),
            "frames_analizados": os.path.join(self.run_dir, "frames_analizados"),
            "rostros": os.path.join(self.run_dir, "rostros"),
            "landmarks": os.path.join(self.run_dir, "landmarks"),
            "roi": os.path.join(self.run_dir, "roi"),
            "debug": os.path.join(self.run_dir, "debug"),
            "analysis": os.path.join(self.run_dir, "analysis"),
            "metadata": os.path.join(self.run_dir, "metadata"),
            "logs": os.path.join(self.run_dir, "logs"),
            "checksums": os.path.join(self.run_dir, "checksums")
        }
        
        self._create_structure()
        self.hashes = {}
        
    def _create_structure(self):
        try:
            for path in self.subdirs.values():
                os.makedirs(path, exist_ok=True)
        except Exception as e:
            print(f"[DatasetLogger Error] No se pudieron crear los directorios: {e}", file=sys.stderr)

    def _safe_execute(self, func, *args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            self.log_error(e)

    def log_error(self, e):
        try:
            error_file = os.path.join(self.run_dir, "error.txt")
            with open(error_file, "a") as f:
                f.write(f"[{datetime.now().isoformat()}] {str(e)}\n")
            
            trace_file = os.path.join(self.run_dir, "stacktrace.txt")
            with open(trace_file, "a") as f:
                f.write(f"[{datetime.now().isoformat()}]\n")
                f.write(traceback.format_exc())
                f.write("\n")
        except Exception:
            pass

    def _calc_hash(self, file_path):
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()

    def _save_file_and_hash(self, src, dest, key):
        try:
            shutil.copy2(src, dest)
            h = self._calc_hash(dest)
            self.hashes[key] = h
            return h
        except Exception as e:
            self.log_error(e)
            return None

    def _save_image_and_hash(self, img, dest, key):
        try:
            cv2.imwrite(dest, img)
            h = self._calc_hash(dest)
            self.hashes[key] = h
            return h
        except Exception as e:
            self.log_error(e)
            return None
            
    def _save_json(self, data, dest):
        try:
            with open(dest, "w") as f:
                json.dump(data, f, indent=4, default=str)
        except Exception as e:
            self.log_error(e)

    def save_video(self, video_path):
        def _do():
            ext = os.path.splitext(video_path)[1]
            dest = os.path.join(self.subdirs["video"], f"{self.base_name}{ext}")
            self._save_file_and_hash(video_path, dest, "video")
        self._safe_execute(_do)
        
    def save_video_metadata(self, metadata):
        def _do():
            dest = os.path.join(self.subdirs["video"], f"{self.base_name}.json")
            self._save_json(metadata, dest)
        self._safe_execute(_do)

    def save_normal_image(self, img_or_path, metadata=None):
        def _do():
            if isinstance(img_or_path, str):
                ext = os.path.splitext(img_or_path)[1]
                if not ext: ext = ".jpg"
                dest = os.path.join(self.subdirs["capturas"], f"{self.base_name}_normal{ext}")
                self._save_file_and_hash(img_or_path, dest, "normal")
            else:
                dest = os.path.join(self.subdirs["capturas"], f"{self.base_name}_normal.jpg")
                self._save_image_and_hash(img_or_path, dest, "normal")
                
            if metadata:
                dest_json = os.path.join(self.subdirs["capturas"], f"{self.base_name}_normal.json")
                self._save_json(metadata, dest_json)
        self._safe_execute(_do)
        
    def save_challenge_image(self, img_or_path, metadata=None):
        def _do():
            if isinstance(img_or_path, str):
                ext = os.path.splitext(img_or_path)[1]
                if not ext: ext = ".jpg"
                dest = os.path.join(self.subdirs["capturas"], f"{self.base_name}_challenge{ext}")
                self._save_file_and_hash(img_or_path, dest, "challenge")
            else:
                dest = os.path.join(self.subdirs["capturas"], f"{self.base_name}_challenge.jpg")
                self._save_image_and_hash(img_or_path, dest, "challenge")
                
            if metadata:
                dest_json = os.path.join(self.subdirs["capturas"], f"{self.base_name}_challenge.json")
                self._save_json(metadata, dest_json)
        self._safe_execute(_do)

    def save_original_frame(self, index, img):
        def _do():
            name = f"frame_{index:06d}.jpg"
            dest = os.path.join(self.subdirs["frames_originales"], name)
            self._save_image_and_hash(img, dest, f"original_frame_{index}")
        self._safe_execute(_do)
        
    def save_analyzed_frame(self, index, img, metadata):
        def _do():
            name = f"frame_{index:06d}.jpg"
            dest = os.path.join(self.subdirs["frames_analizados"], name)
            self._save_image_and_hash(img, dest, f"analyzed_frame_{index}")
            
            dest_json = os.path.join(self.subdirs["frames_analizados"], f"frame_{index:06d}.json")
            self._save_json(metadata, dest_json)
        self._safe_execute(_do)

    def save_face(self, index, img):
        def _do():
            name = f"face_{index:06d}.jpg"
            dest = os.path.join(self.subdirs["rostros"], name)
            self._save_image_and_hash(img, dest, f"face_{index}")
        self._safe_execute(_do)
        
    def save_landmarks(self, index, img):
        def _do():
            name = f"landmarks_{index:06d}.jpg"
            dest = os.path.join(self.subdirs["landmarks"], name)
            self._save_image_and_hash(img, dest, f"landmarks_{index}")
        self._safe_execute(_do)

    def save_roi(self, name, img):
        def _do():
            dest = os.path.join(self.subdirs["roi"], f"{name}.jpg")
            self._save_image_and_hash(img, dest, f"roi_{name}")
        self._safe_execute(_do)

    def save_debug_image(self, name, img):
        def _do():
            dest = os.path.join(self.subdirs["debug"], f"{name}.jpg")
            self._save_image_and_hash(img, dest, f"debug_{name}")
        self._safe_execute(_do)
        
    def save_decision(self, decision_data):
        def _do():
            dest = os.path.join(self.subdirs["analysis"], "decision.json")
            self._save_json(decision_data, dest)
        self._safe_execute(_do)

    def save_response(self, response_data):
        def _do():
            dest = os.path.join(self.subdirs["analysis"], "response.json")
            self._save_json(response_data, dest)
        self._safe_execute(_do)

    def save_timeline(self, timeline_data):
        def _do():
            dest = os.path.join(self.subdirs["analysis"], "timeline.json")
            self._save_json(timeline_data, dest)
        self._safe_execute(_do)

    def save_scores_csv(self, rows):
        def _do():
            dest = os.path.join(self.subdirs["analysis"], "scores.csv")
            with open(dest, "w") as f:
                f.write("frame,timestamp,deepfake_probability,classification\n")
                for r in rows:
                    f.write(f"{r['frame']},{r['timestamp']},{r['deepfake_probability']},{r['classification']}\n")
        self._safe_execute(_do)

    def save_metadata(self, metadata):
        def _do():
            metadata["dataset_type"] = self.dataset_type
            metadata["timestamp"] = datetime.now().isoformat()
            dest = os.path.join(self.subdirs["metadata"], "metadata.json")
            self._save_json(metadata, dest)
        self._safe_execute(_do)
        
    def save_checksums(self):
        def _do():
            dest = os.path.join(self.subdirs["checksums"], "checksums.json")
            self._save_json(self.hashes, dest)
        self._safe_execute(_do)
