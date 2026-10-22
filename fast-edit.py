import customtkinter as ctk
from tkinter import filedialog
from PIL import Image, ImageTk
import rawpy
import cv2
import numpy as np
from scipy.interpolate import CubicSpline
import exifread
from datetime import datetime
import os

# --- SETĂRI INTERFAȚĂ ---
ctk.set_appearance_mode("Dark")  
ctk.set_default_color_theme("blue")

# --- FUNCȚII DE PROCESARE OPTICĂ (Motorul din spate) ---
class ImageEngine:
    @staticmethod
    def create_tone_curve(x, y):
        cs = CubicSpline(x, y, bc_type='clamped')
        x_new = np.arange(256)
        return np.clip(cs(x_new), 0, 255).astype(np.uint8)

    @staticmethod
    def add_bloom(img, amount=0.15, radius=15):
        blur = cv2.GaussianBlur(img, (0, 0), radius)
        bloom = img.astype(np.float32) + (blur.astype(np.float32) * amount)
        return np.clip(bloom, 0, 255).astype(np.uint8)

    @staticmethod
    def add_film_grain(img, intensity=15):
        noise = np.random.normal(0, intensity, img.shape[:2]).astype(np.float32)
        noise_rgb = cv2.cvtColor(noise, cv2.COLOR_GRAY2RGB)
        noisy_img = cv2.add(img.astype(np.float32), noise_rgb)
        return np.clip(noisy_img, 0, 255).astype(np.uint8)

    @staticmethod
    def add_vignette(img, intensity=0.5):
        rows, cols = img.shape[:2]
        kernel_x = cv2.getGaussianKernel(cols, cols / 2)
        kernel_y = cv2.getGaussianKernel(rows, rows / 2)
        kernel = kernel_y * kernel_x.T
        mask = kernel / kernel.max()
        mask = np.dstack([mask] * 3)
        vignette_img = img.astype(np.float32) * (1.0 - intensity + mask * intensity)
        return np.clip(vignette_img, 0, 255).astype(np.uint8)

    @staticmethod
    def add_chromatic_aberration(img, shift=5):
        r, g, b = cv2.split(img)
        r_shifted = np.roll(r, -shift, axis=1)
        b_shifted = np.roll(b, shift, axis=1)
        return cv2.merge((r_shifted, g, b_shifted))

    @staticmethod
    def add_scanlines(img, intensity=0.15):
        mask = np.ones_like(img, dtype=np.float32)
        mask[::3, :, :] = 1.0 - intensity
        return np.clip(img.astype(np.float32) * mask, 0, 255).astype(np.uint8)

    @staticmethod
    def get_exif_date(file_path):
        try:
            with open(file_path, 'rb') as f:
                tags = exifread.process_file(f, details=False, stop_tag='EXIF DateTimeOriginal')
                if 'EXIF DateTimeOriginal' in tags:
                    dt = datetime.strptime(str(tags['EXIF DateTimeOriginal']), "%Y:%m:%d %H:%M:%S")
                    return dt.strftime("%b. %d %Y").upper() + "  " + dt.strftime("%p %I:%M")
        except:
            pass
        return "OCT. 20 2026  PM 11:35"

    @staticmethod
    def apply_vcr_overlay(img, date_str):
        h, w = img.shape[:2]
        scale = max(0.5, h / 900.0)
        thickness = max(1, int(scale * 1.5))
        font = cv2.FONT_HERSHEY_SIMPLEX

        x_play, y_play = int(w * 0.05), int(h * 0.10)
        x_date, y_date = int(w * 0.65), int(h * 0.90)

        def draw_text(img, text, x, y):
            cv2.putText(img, text, (x+3, y+3), font, scale, (0, 0, 0), thickness * 3, cv2.LINE_AA)
            cv2.putText(img, text, (x, y), font, scale, (240, 250, 255), thickness, cv2.LINE_AA)

        draw_text(img, "PLAY  > ", x_play, y_play)
        draw_text(img, date_str, x_date, y_date)
        return img

    # --- PRESETURI ---
    @classmethod
    def apply_sage(cls, rgb_img, mode='light'):
        img = cls.add_bloom(rgb_img, amount=(0.10 if mode == 'light' else 0.20))
        
        if mode == 'light':
            lut = cls.create_tone_curve([0, 50, 128, 210, 255], [25, 60, 140, 230, 245])
        else:
            lut = cls.create_tone_curve([0, 60, 128, 190, 255], [45, 65, 120, 200, 225])
        
        img = cv2.LUT(img, lut)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        h, s, v = cv2.split(hsv)
        green_mask = (h > 20) & (h < 90)

        if mode == 'light':
            h[green_mask] = np.clip(h[green_mask] + 12, 0, 179) 
            s[green_mask] *= 0.25
            s = np.clip(s * 0.85, 0, 255)
        else:
            h[green_mask] = np.clip(h[green_mask] - 5, 0, 179) 
            s[green_mask] *= 0.10 
            s = np.clip(s * 0.55, 0, 255) 

        hsv_adjusted = cv2.merge((h, s, v)).astype(np.uint8)
        img = cv2.cvtColor(hsv_adjusted, cv2.COLOR_HSV2RGB).astype(np.float32)

        luminance = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        shadows, highlights = 1.0 - luminance, luminance

        if mode == 'light':
            img[:,:,0] += highlights * 15; img[:,:,1] += highlights * 10; img[:,:,2] -= highlights * 15
            img[:,:,1] += shadows * 10; img[:,:,2] += shadows * 15
        else:
            img[:,:,0] += highlights * 20; img[:,:,1] += highlights * 5; img[:,:,2] -= highlights * 20
            img[:,:,1] += shadows * 25; img[:,:,0] -= shadows * 10

        img = np.clip(img, 0, 255).astype(np.uint8)
        img = cls.add_vignette(img, 0.3 if mode == 'light' else 0.7)
        return cls.add_film_grain(img, 8 if mode == 'light' else 18)

    @classmethod
    def apply_jdm(cls, rgb_img, date_str, mode='wangan'):
        if mode == 'wangan':
            lut = cls.create_tone_curve([0, 40, 128, 200, 255], [15, 30, 110, 215, 240])
            img = cv2.LUT(rgb_img, lut).astype(np.float32)
            luminance = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
            img[:,:,0] += luminance * 25; img[:,:,2] += luminance * 10
            img[:,:,2] += (1.0 - luminance) * 30; img[:,:,1] += (1.0 - luminance) * 15
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = cls.add_bloom(img, 0.25, 35)
            img = cls.add_chromatic_aberration(img, 3)
            img = cls.add_film_grain(img, 10)
        else: # vhs
            lut = cls.create_tone_curve([0, 60, 128, 200, 255], [45, 75, 130, 190, 220])
            img = cv2.LUT(rgb_img, lut)
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[:, :, 1] *= 0.60
            img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB).astype(np.float32)
            img[:,:,0] += 15; img[:,:,1] += 10; img[:,:,2] -= 15
            img = np.clip(img, 0, 255).astype(np.uint8)
            img = cls.add_bloom(img, 0.15)
            img = cls.add_chromatic_aberration(img, 8)
            img = cls.add_scanlines(img, 0.20)
            img = cls.add_film_grain(img, 25)

        return cls.apply_vcr_overlay(img, date_str)


# --- INTERFAȚA GRAFICĂ (GUI) ---
class RawEditorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("JDM & Sage RAW Studio")
        self.geometry("1300x800")
        self.minsize(900, 600)

        # Variabile de stare
        self.original_image_rgb = None
        self.current_image_rgb = None
        self.camera_date = ""
        self.file_path = ""

        self.setup_ui()

    def setup_ui(self):
        # Grid principal: Coloana 0 (Panou comenzi), Coloana 1 (Canvas imagine)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- PANOU STÂNGA (Comenzi) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=250, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)
        self.sidebar_frame.grid_rowconfigure(7, weight=1) # Spatiu gol inainte de save

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="RAW STUDIO", font=ctk.CTkFont(size=24, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.load_btn = ctk.CTkButton(self.sidebar_frame, text="Încărcare Imagine", command=self.load_image)
        self.load_btn.grid(row=1, column=0, padx=20, pady=10)

        self.preset_label = ctk.CTkLabel(self.sidebar_frame, text="Selectează Preset:", font=ctk.CTkFont(size=14))
        self.preset_label.grid(row=2, column=0, padx=20, pady=(20, 5))

        # Butoane Preset-uri
        self.btn_orig = ctk.CTkButton(self.sidebar_frame, text="Original", fg_color="gray", command=lambda: self.apply_preset('original'))
        self.btn_orig.grid(row=3, column=0, padx=20, pady=5)

        self.btn_light = ctk.CTkButton(self.sidebar_frame, text="Light Sage", fg_color="#638C75", hover_color="#4F705E", command=lambda: self.apply_preset('light_sage'))
        self.btn_light.grid(row=4, column=0, padx=20, pady=5)

        self.btn_deep = ctk.CTkButton(self.sidebar_frame, text="Deep Sage", fg_color="#2C4033", hover_color="#1D2A22", command=lambda: self.apply_preset('deep_sage'))
        self.btn_deep.grid(row=5, column=0, padx=20, pady=5)

        self.btn_wangan = ctk.CTkButton(self.sidebar_frame, text="Wangan Midnight", fg_color="#5B2C6F", hover_color="#4A235A", command=lambda: self.apply_preset('wangan'))
        self.btn_wangan.grid(row=6, column=0, padx=20, pady=5)

        self.btn_vhs = ctk.CTkButton(self.sidebar_frame, text="VHS Drift", fg_color="#A98307", hover_color="#8A6A05", command=lambda: self.apply_preset('vhs'))
        self.btn_vhs.grid(row=7, column=0, padx=20, pady=5)

        self.status_label = ctk.CTkLabel(self.sidebar_frame, text="Aștept fișier...", text_color="gray")
        self.status_label.grid(row=8, column=0, padx=20, pady=10)

        self.save_btn = ctk.CTkButton(self.sidebar_frame, text="Salvează JPEG", fg_color="#196F3D", hover_color="#145A32", command=self.save_image)
        self.save_btn.grid(row=9, column=0, padx=20, pady=20)

        # --- PANOU DREAPTA (Imagine) ---
        self.image_frame = ctk.CTkFrame(self)
        self.image_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.image_frame.grid_rowconfigure(0, weight=1)
        self.image_frame.grid_columnconfigure(0, weight=1)

        self.image_label = ctk.CTkLabel(self.image_frame, text="Nicio imagine încărcată")
        self.image_label.grid(row=0, column=0, sticky="nsew")

    def set_status(self, text):
        self.status_label.configure(text=text)
        self.update()

    def load_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Imagini", "*.ARW *.jpg *.jpeg *.png")])
        if not file_path:
            return

        self.file_path = file_path
        self.set_status("Decodare RAW...")
        
        try:
            if file_path.lower().endswith('.arw'):
                with rawpy.imread(file_path) as raw:
                    # Folosim half_size pentru viteză în aplicație. La export "serios" se poate face toggle.
                    self.original_image_rgb = raw.postprocess(use_camera_wb=True, half_size=True)
            else:
                img_bgr = cv2.imread(file_path)
                self.original_image_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            self.camera_date = ImageEngine.get_exif_date(file_path)
            self.current_image_rgb = self.original_image_rgb.copy()
            
            self.display_image(self.current_image_rgb)
            self.set_status("Imagine încărcată!")
        except Exception as e:
            self.set_status(f"Eroare: {str(e)}")

    def apply_preset(self, preset_name):
        if self.original_image_rgb is None:
            self.set_status("Încarcă o poză mai întâi!")
            return

        self.set_status("Procesez (poate dura 1-2 sec)...")
        base_img = self.original_image_rgb.copy()

        if preset_name == 'original':
            self.current_image_rgb = base_img
        elif preset_name == 'light_sage':
            self.current_image_rgb = ImageEngine.apply_sage(base_img, 'light')
        elif preset_name == 'deep_sage':
            self.current_image_rgb = ImageEngine.apply_sage(base_img, 'deep')
        elif preset_name == 'wangan':
            self.current_image_rgb = ImageEngine.apply_jdm(base_img, self.camera_date, 'wangan')
        elif preset_name == 'vhs':
            self.current_image_rgb = ImageEngine.apply_jdm(base_img, self.camera_date, 'vhs')

        self.display_image(self.current_image_rgb)
        self.set_status(f"Preset aplicat: {preset_name}")

    def display_image(self, rgb_array):
        # Redimensionare logică pentru a încăpea frumos pe ecran fără a strivi proporțiile
        frame_width = self.image_frame.winfo_width()
        frame_height = self.image_frame.winfo_height()
        
        # Fallback dacă fereastra nu e încă desenată complet
        if frame_width <= 10: frame_width = 900
        if frame_height <= 10: frame_height = 600

        img = Image.fromarray(rgb_array)
        img.thumbnail((frame_width - 20, frame_height - 20), Image.Resampling.LANCZOS)
        
        ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=img.size)
        self.image_label.configure(image=ctk_img, text="")
        self.image_label.image = ctk_img

    def save_image(self):
        if self.current_image_rgb is None:
            self.set_status("Nimic de salvat.")
            return

        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG", "*.jpg")])
        if save_path:
            # OpenCV salvează în BGR, deci convertim înapoi din spațiul nostru de lucru RGB
            bgr_img = cv2.cvtColor(self.current_image_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(save_path, bgr_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            self.set_status("Poza a fost salvată!")

if __name__ == "__main__":
    app = RawEditorApp()
    app.mainloop()
