# quant_dashboard_modern.py
"""
📊 QUANT DASHBOARD - Modern Edition
====================================
• Dark / Light tema (üstte toggle)
• Catppuccin paleti + font fallback
• Renkli terminal (satır bazlı)
• Notebook sekmeleri: Terminal + Grafik + L1...L10 (script açıklamaları)
• L9 için ⏱️ 60dk ipucu
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinter import font as tkfont
import threading
import sys
import os
import re
import importlib.util
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

try:
    from PIL import Image, ImageTk
    PIL_KURULU = True
except ImportError:
    PIL_KURULU = False


# ═══════════════════════════════════════════════════════════════
#  FONT YARDIMCILARI
# ═══════════════════════════════════════════════════════════════
def _pick_ui_font():
    available = set(tkfont.families())
    for f in ('Segoe UI', 'Helvetica Neue', 'Ubuntu', 'Cantarell',
              'DejaVu Sans', 'Arial'):
        if f in available:
            return f
    return 'TkDefaultFont'


def _pick_mono_font():
    available = set(tkfont.families())
    for f in ('Consolas', 'Cascadia Code', 'JetBrains Mono',
              'Fira Code', 'Menlo', 'Monaco', 'DejaVu Sans Mono',
              'Courier New'):
        if f in available:
            return f
    return 'TkFixedFont'


UI_FONT = 'TkDefaultFont'
MONO_FONT = 'TkFixedFont'


def ui(size=10, weight='normal'):
    return (UI_FONT, size, weight)


def mono(size=10, weight='normal'):
    return (MONO_FONT, size, weight)


# ═══════════════════════════════════════════════════════════════
#  TEMA YÖNETİCİSİ
# ═══════════════════════════════════════════════════════════════
class ThemeManager:
    THEMES = {
        'dark': {
            'bg': '#1e1e2e', 'bg_alt': '#181825',
            'panel': '#313244', 'panel_alt': '#45475a',
            'text': '#cdd6f4', 'text_dim': '#a6adc8',
            'accent': '#89b4fa', 'accent_hover': '#b4befe',
            'success': '#a6e3a1', 'warning': '#f9e2af',
            'danger': '#f38ba8', 'border': '#45475a',
            'term_bg': '#11111b', 'term_fg': '#cdd6f4',
        },
        'light': {
            'bg': '#eff1f5', 'bg_alt': '#e6e9ef',
            'panel': '#ffffff', 'panel_alt': '#dce0e8',
            'text': '#4c4f69', 'text_dim': '#6c6f85',
            'accent': '#1e66f5', 'accent_hover': '#7287fd',
            'success': '#40a02b', 'warning': '#df8e1d',
            'danger': '#d20f39', 'border': '#ccd0da',
            'term_bg': '#ffffff', 'term_fg': '#4c4f69',
        }
    }

    def __init__(self):
        self.current = 'dark'
        self.colors = self.THEMES[self.current]

    def toggle(self):
        self.current = 'light' if self.current == 'dark' else 'dark'
        self.colors = self.THEMES[self.current]
        return self.current


# ═══════════════════════════════════════════════════════════════
#  THREAD-SAFE STDOUT (Renkli)
# ═══════════════════════════════════════════════════════════════
class ThreadSafeStdout:
    ANSI_STRIP = re.compile(r'\033\[[0-9;]*[A-Za-z]')

    LINE_RULES = [
        (re.compile(r'^[═=]{10,}'),             'style_separator'),
        (re.compile(r'^[─-]{10,}'),             'style_separator_dim'),
        (re.compile(r'^\s*🚀'),                  'style_header'),
        (re.compile(r'^\s*▶'),                   'style_header'),
        (re.compile(r'^\s*\d+\.\s'),             'style_section'),
        (re.compile(r'^\s*[✅✔]'),               'style_success'),
        (re.compile(r'^\s*[❌✗]'),               'style_danger'),
        (re.compile(r'^\s*[⚠❗]'),               'style_warning'),
        (re.compile(r'^\s*💡'),                  'style_info_cyan'),
        (re.compile(r'^\s*➡️'),                  'style_info'),
        (re.compile(r'^\s*[🎯🎲⚖🔗]'),           'style_accent'),
        (re.compile(r'^\s*[🟡🟢🔴]'),            'style_accent'),
        (re.compile(r'^\s*[📥📋📊📈📉📁]'),       'style_dim'),
        (re.compile(r'[:=]\s*\+\s*\d'),          'style_positive'),
        (re.compile(r'[:=]\s*-\s*\d'),           'style_negative'),
        (re.compile(r'\+\d+[.,]\d+\s*%'),        'style_positive'),
        (re.compile(r'-\d+[.,]\d+\s*%'),         'style_negative'),
        (re.compile(r'^\s*•'),                   'style_dim'),
        (re.compile(r'^\s*[-•]\s'),              'style_dim'),
    ]

    def __init__(self, text_widget, root):
        self.text_widget = text_widget
        self.root = root
        self._buffer = ''

    def write(self, text):
        if not text:
            return
        self._buffer += text
        if '\n' not in self._buffer:
            return
        lines = self._buffer.split('\n')
        complete = lines[:-1]
        self._buffer = lines[-1]
        for line in complete:
            self._schedule_line(line)

    def _schedule_line(self, line):
        clean = self.ANSI_STRIP.sub('', line)
        tag = self._detect_tag(clean)
        self.root.after(0, lambda l=clean, t=tag: self._render(l, t))

    def _detect_tag(self, line):
        for pattern, tag in self.LINE_RULES:
            if pattern.match(line):
                return tag
        return None

    def _render(self, text, tag):
        try:
            self.text_widget.config(state='normal')
            if tag:
                self.text_widget.insert(tk.END, text + '\n', tag)
            else:
                self.text_widget.insert(tk.END, text + '\n')
            self.text_widget.see(tk.END)
            self.text_widget.config(state='disabled')
        except Exception:
            pass

    def flush(self):
        if self._buffer:
            remaining = self._buffer
            self._buffer = ''
            self._schedule_line(remaining)


# ═══════════════════════════════════════════════════════════════
#  ANA UYGULAMA
# ═══════════════════════════════════════════════════════════════
class QuantDashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("📊 Quant Dashboard")
        self.root.geometry("1500x950")
        self.root.minsize(1300, 750)

        self.theme = ThemeManager()
        self.csv_dizini = None
        self.calisiyor = False
        self.current_image = None
        self.grafik_image_path = None
        self.grafik_zoom = 1.0
        self._term_bold_font = None
        self.level_text_widgets = {}   # L1..L10 sekme Text widget'ları
        self.level_bars = {}            # L1..L10 üst bilgi çubukları

        self.level_tanimlari = {
            1: {'mod': 'adim_1_sadece_olasilik', 'sinif': 'Level1OlasilikAnalizi',
                'tek': True, 'kisa': 'Beklenen Değer & Volatilite',
                'aciklama': 'EV, Win Rate, Volatilite'},
            2: {'mod': 'adim_2_istatistik', 'sinif': 'Level2Istatistik',
                'tek': True, 'kisa': 't-stat & Regresyon',
                'aciklama': 't-stat, Sharpe, OLS, R²'},
            3: {'mod': 'adim_3_lineer_cebir', 'sinif': 'Level3LineerCebir',
                'tek': False, 'kisa': 'PCA & Portföy Riski',
                'aciklama': 'Kovaryans, PCA, Shrinkage'},
            4: {'mod': 'adim_4_kalkulus_optimizasyon', 'sinif': 'Level4KalkulusOptimizasyon',
                'tek': False, 'kisa': 'Portföy Optimizasyonu',
                'aciklama': 'Taylor, Gradient, Markowitz'},
            5: {'mod': 'adim_5_zaman_serileri', 'sinif': 'Level5ZamanSerileri',
                'tek': False, 'kisa': 'Zaman Serisi Analizi',
                'aciklama': 'ADF, Eşbütünleşme, GARCH(1,1)'},
            6: {'mod': 'adim_6_stokastik_kalkulus', 'sinif': 'Level6StokastikKalkulus',
                'tek': True, 'kisa': 'Monte Carlo Simülasyonu',
                'aciklama': 'GBM, Itô, Martingale'},
            7: {'mod': 'adim_7_turevler', 'sinif': 'Level7Turevler',
                'tek': True, 'kisa': 'Opsiyon Fiyatlama',
                'aciklama': 'Black-Scholes, Yunanlar, IV'},
            8: {'mod': 'adim_8_portfoy_ve_risk', 'sinif': 'Level8PortfoyVeRisk',
                'tek': False, 'kisa': 'Faktör Modelleri & Risk',
                'aciklama': 'CAPM, Fama-French, VaR/ES'},
            9: {'mod': 'adim_9_mikroyapi_ve_icra', 'sinif': 'Level9MikroyapiOHLCV',
                'tek': True, 'kisa': 'Mikroyapı & Emir İcra',
                'aciklama': 'Corwin-Schultz, CLV, VWAP/TWAP',
                'ipucu': '60dk bar ile bakmak lehinize'},
            10: {'mod': 'adim_10_arastirma_ve_ml', 'sinif': 'Level10ResearchML',
                 'tek': True, 'kisa': 'ML & Araştırma',
                 'aciklama': 'Walk-Forward, Deflated Sharpe, HMM'},
        }

        # Script docstring'lerinin ilk 40 satırı
        # Script docstring'leri — Level başına satır sayısı
        satir_sayilari = {1:36, 3:31, 4:36,5:32, 6:10, 8:160, 9: 50, 9:48, 10: 48}   # geri kalanlar 40
        self.script_aciklamalari = {}
        for n in range(1, 11):
            adet = satir_sayilari.get(n, 40)
            self.script_aciklamalari[n] = self._script_ilk_satirlar(n, adet)

        self._setup_styles()
        self.gui_olustur()
        self._update_tk_widgets()
        self.varsayilan_dizin_yukle()

    # ─────────────────────────────────────────────────────────
    #  SCRIPT OKUMA
    # ─────────────────────────────────────────────────────────
    def _script_ilk_satirlar(self, level_num, satir_sayisi=40):
        """Docstring + gerekiyorsa devamındaki kod satırlarını döndür"""
        tanim = self.level_tanimlari.get(level_num)
        if not tanim:
            return ""
        mod_yolu = Path(__file__).parent / f"{tanim['mod']}.py"
        if not mod_yolu.exists():
            return f"⚠️ Dosya bulunamadı: {tanim['mod']}.py"
        try:
            with open(mod_yolu, 'r', encoding='utf-8') as f:
                tum = f.read()
        except Exception as e:
            return f"⚠️ Okuma hatası: {e}"

        # Docstring var mı?
        m = re.search(r'^\s*"""(.*?)"""', tum, re.DOTALL | re.MULTILINE)

        if m:
            # Docstring içeriği
            docstring_metni = m.group(1).strip('\n')
            satirlar = docstring_metni.split('\n')

            # İstenen sayıya yetmiyorsa, docstring'den sonraki kodu ekle
            if len(satirlar) < satir_sayisi:
                sonrasi = tum[m.end():].split('\n')
                # Baştaki boş satırları atla
                while sonrasi and not sonrasi[0].strip():
                    sonrasi.pop(0)

                ihtiyac = satir_sayisi - len(satirlar)
                if sonrasi:
                    satirlar.append("")   # docstring ile kod arasında boş satır
                    satirlar.extend(sonrasi[:ihtiyac - 1])

            return '\n'.join(satirlar[:satir_sayisi])
        else:
            # Docstring yok → dosyanın ilk N satırı
            return '\n'.join(tum.split('\n')[:satir_sayisi])

    # ─────────────────────────────────────────────────────────
    #  STİL KURULUMU
    # ─────────────────────────────────────────────────────────
    def _setup_styles(self):
        c = self.theme.colors
        style = ttk.Style()
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass

        style.configure('.', background=c['bg'], foreground=c['text'], font=ui(10))
        style.configure('TFrame', background=c['bg'])
        style.configure('TLabel', background=c['bg'], foreground=c['text'], font=ui(10))

        style.configure('TButton', background=c['panel'], foreground=c['text'],
                        borderwidth=0, padding=(12, 7), font=ui(10),
                        relief='flat', focusthickness=0)
        style.map('TButton',
                  background=[('active', c['panel_alt']), ('pressed', c['accent'])],
                  foreground=[('active', c['accent']), ('pressed', c['bg'])])

        style.configure('Accent.TButton', background=c['accent'], foreground=c['bg'],
                        padding=(14, 8), font=ui(10, 'bold'),
                        borderwidth=0, relief='flat')
        style.map('Accent.TButton',
                  background=[('active', c['accent_hover']), ('pressed', c['accent'])],
                  foreground=[('active', c['bg'])])

        style.configure('Run.TButton', background=c['success'], foreground=c['bg'],
                        padding=(14, 8), font=ui(10, 'bold'),
                        borderwidth=0, relief='flat')
        style.map('Run.TButton', background=[('active', c['accent_hover'])])

        style.configure('TLabelframe', background=c['bg'], foreground=c['accent'],
                        borderwidth=1, relief='solid',
                        bordercolor=c['border'], lightcolor=c['border'], darkcolor=c['border'])
        style.configure('TLabelframe.Label', background=c['bg'], foreground=c['accent'],
                        font=ui(10, 'bold'))

        style.configure('TNotebook', background=c['bg'], borderwidth=0)
        style.configure('TNotebook.Tab', background=c['panel'], foreground=c['text_dim'],
                        padding=(14, 8), borderwidth=0, font=ui(9, 'bold'))
        style.map('TNotebook.Tab',
                  background=[('selected', c['accent']), ('active', c['panel_alt'])],
                  foreground=[('selected', c['bg']), ('active', c['accent'])])

        style.configure('Title.TLabel', font=ui(20, 'bold'),
                        foreground=c['accent'], background=c['bg'])
        style.configure('Status.TLabel', background=c['panel'], foreground=c['text'],
                        padding=10, font=ui(10))
        style.configure('Dim.TLabel', foreground=c['text_dim'],
                        background=c['bg'], font=ui(9))
        style.configure('DimBig.TLabel', foreground=c['text_dim'],
                        background=c['bg'], font=ui(14))
        style.configure('Hint.TLabel', foreground=c['success'],
                        background=c['bg'], font=ui(9, 'italic'))

        style.configure('Vertical.TScrollbar', background=c['panel'],
                        troughcolor=c['bg'], borderwidth=0, arrowcolor=c['text_dim'])
        style.map('Vertical.TScrollbar', background=[('active', c['accent'])])
        style.configure('Horizontal.TScrollbar', background=c['panel'],
                        troughcolor=c['bg'], borderwidth=0, arrowcolor=c['text_dim'])
        style.map('Horizontal.TScrollbar', background=[('active', c['accent'])])

    # ─────────────────────────────────────────────────────────
    #  GUI
    # ─────────────────────────────────────────────────────────
    def gui_olustur(self):
        c = self.theme.colors
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=12, pady=12)

        # ÜST BAR
        ust_bar = ttk.Frame(main_frame)
        ust_bar.pack(fill='x', pady=(0, 10))

        baslik_frame = ttk.Frame(ust_bar)
        baslik_frame.pack(side='left')
        ttk.Label(baslik_frame, text="📊 QUANT DASHBOARD",
                  style='Title.TLabel').pack(anchor='w')
        ttk.Label(baslik_frame,
                  text="Level 1-10 · Olasılıktan ML'e Quant Analiz",
                  style='Dim.TLabel').pack(anchor='w')

        self.theme_btn = ttk.Button(
            ust_bar,
            text="🌙 Karanlık" if self.theme.current == 'dark' else "☀️ Aydınlık",
            style='Accent.TButton', command=self.toggle_theme, cursor='hand2')
        self.theme_btn.pack(side='right', padx=4)

        self._separator = tk.Frame(main_frame, height=1, bg=c['border'])
        self._separator.pack(fill='x', pady=(0, 10))

        # PANED
        self.paned = ttk.PanedWindow(main_frame, orient=tk.HORIZONTAL)
        self.paned.pack(fill=tk.BOTH, expand=True)

        # ─── SOL PANEL ───
        sol_frame = ttk.Frame(self.paned, width=400)
        self.paned.add(sol_frame, weight=0)

        self.sol_canvas = tk.Canvas(sol_frame, highlightthickness=0)
        sol_scroll = ttk.Scrollbar(sol_frame, orient='vertical', command=self.sol_canvas.yview)
        sol_inner = ttk.Frame(self.sol_canvas)

        sol_inner.bind("<Configure>",
                       lambda e: self.sol_canvas.configure(scrollregion=self.sol_canvas.bbox("all")))
        self._sol_window = self.sol_canvas.create_window((0, 0), window=sol_inner, anchor='nw')
        self.sol_canvas.configure(yscrollcommand=sol_scroll.set)

        self.sol_canvas.bind("<Configure>",
                             lambda e: self.sol_canvas.itemconfig(self._sol_window, width=e.width))
        self.sol_canvas.pack(side='left', fill='both', expand=True)
        sol_scroll.pack(side='right', fill='y')

        def _wheel_sol(event):
            self.sol_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        self.sol_canvas.bind("<Enter>", lambda e: self.sol_canvas.bind_all("<MouseWheel>", _wheel_sol))
        self.sol_canvas.bind("<Leave>", lambda e: self.sol_canvas.unbind_all("<MouseWheel>"))

        # Dizin
        dizin_frame = ttk.LabelFrame(sol_inner, text="📁  Veri Dizini", padding=12)
        dizin_frame.pack(fill='x', padx=8, pady=6)
        self.dizin_label = ttk.Label(dizin_frame, text="Dizin seçilmedi",
                                      wraplength=340, style='Dim.TLabel')
        self.dizin_label.pack(fill='x', pady=(0, 8))
        ttk.Button(dizin_frame, text="📂  Dizin Seç...",
                   command=self.dizin_sec, cursor='hand2').pack(fill='x')

        # Hisse
        hisse_frame = ttk.LabelFrame(sol_inner, text="📈  Hisse Seçimi", padding=12)
        hisse_frame.pack(fill='both', expand=True, padx=8, pady=6)
        ttk.Label(hisse_frame, text="Ctrl+Click ile çoklu seçim",
                  style='Dim.TLabel').pack(anchor='w', pady=(0, 6))

        listbox_frame = ttk.Frame(hisse_frame)
        listbox_frame.pack(fill='both', expand=True)
        self.hisse_listbox = tk.Listbox(
            listbox_frame, selectmode=tk.MULTIPLE, exportselection=False,
            height=8, font=ui(10), relief='flat', borderwidth=0,
            highlightthickness=1, activestyle='none')
        hisse_scroll = ttk.Scrollbar(listbox_frame, orient='vertical',
                                      command=self.hisse_listbox.yview)
        self.hisse_listbox.configure(yscrollcommand=hisse_scroll.set)
        self.hisse_listbox.pack(side='left', fill='both', expand=True)
        hisse_scroll.pack(side='right', fill='y')

        btn_frame = ttk.Frame(hisse_frame)
        btn_frame.pack(fill='x', pady=(8, 0))
        ttk.Button(btn_frame, text="✓ Tümü", cursor='hand2',
                   command=lambda: self.hisse_listbox.select_set(0, tk.END)
                   ).pack(side='left', expand=True, padx=2)
        ttk.Button(btn_frame, text="✗ Temizle", cursor='hand2',
                   command=lambda: self.hisse_listbox.selection_clear(0, tk.END)
                   ).pack(side='right', expand=True, padx=2)

        # Level
        level_frame = ttk.LabelFrame(sol_inner, text="🎯  Level Seçimi", padding=12)
        level_frame.pack(fill='both', expand=True, padx=8, pady=6)
        ttk.Label(level_frame, text="🎯 tek hisse    👥 2+ hisse",
                  style='Dim.TLabel').pack(anchor='w', pady=(0, 8))

        self.level_buttons = {}
        for level_num in range(1, 11):
            tanim = self.level_tanimlari[level_num]
            ikon = '🎯' if tanim['tek'] else '👥'
            gerekli = 'tek hisse' if tanim['tek'] else '2+ hisse'

            satir = ttk.Frame(level_frame)
            satir.pack(fill='x', pady=2)

            btn = ttk.Button(satir,
                             text=f"{ikon}  L{level_num} · {tanim['kisa']}  ·  {gerekli}",
                             command=lambda n=level_num: self.level_calistir(n),
                             cursor='hand2')
            btn.pack(fill='x')
            self.level_buttons[level_num] = btn

            if tanim.get('ipucu'):
                ttk.Label(satir, text=f"      💡 {tanim['ipucu']}",
                          style='Hint.TLabel').pack(anchor='w')

        ttk.Button(level_frame, text="▶  TÜMÜNÜ ÇALIŞTIR (1-10)",
                   style='Run.TButton', cursor='hand2',
                   command=self.tumunu_calistir).pack(fill='x', pady=(10, 0))

        self.durum_label = ttk.Label(sol_inner, text="⚡ Hazır",
                                      style='Status.TLabel', anchor='w')
        self.durum_label.pack(fill='x', padx=8, pady=8)

        # ─── SAĞ PANEL: NOTEBOOK ───
        sag_frame = ttk.Frame(self.paned)
        self.paned.add(sag_frame, weight=1)

        self.notebook = ttk.Notebook(sag_frame)
        self.notebook.pack(fill='both', expand=True)

        # ── Sekme 1: Terminal
        metin_frame = ttk.Frame(self.notebook)
        self.notebook.add(metin_frame, text="📝  Terminal")
        terminal_wrap = ttk.Frame(metin_frame)
        terminal_wrap.pack(fill='both', expand=True, padx=4, pady=4)

        self.metin_area = tk.Text(
            terminal_wrap, wrap=tk.WORD, font=mono(10), state='disabled',
            padx=14, pady=14, relief='flat', borderwidth=0,
            highlightthickness=1, insertbackground='white')
        term_scroll = ttk.Scrollbar(terminal_wrap, orient='vertical',
                                     command=self.metin_area.yview)
        self.metin_area.configure(yscrollcommand=term_scroll.set)
        self.metin_area.pack(side='left', fill='both', expand=True)
        term_scroll.pack(side='right', fill='y')

        metin_btn_frame = ttk.Frame(metin_frame)
        metin_btn_frame.pack(fill='x', padx=4, pady=4)
        ttk.Button(metin_btn_frame, text="🗑️  Temizle", cursor='hand2',
                   command=self.metin_temizle).pack(side='left', padx=2)
        ttk.Button(metin_btn_frame, text="📋  Kopyala", cursor='hand2',
                   command=self.metin_kopyala).pack(side='left', padx=2)

        # ── Sekme 2: Grafik
        grafik_frame = ttk.Frame(self.notebook)
        self.notebook.add(grafik_frame, text="🖼️  Grafik")
        self.grafik_container = ttk.Frame(grafik_frame)
        self.grafik_container.pack(fill='both', expand=True, padx=4, pady=4)

        self.grafik_label = ttk.Label(
            self.grafik_container,
            text="📊 Grafik burada görünecek\n\nBir Level çalıştırın...",
            style='DimBig.TLabel', justify='center')
        self.grafik_label.pack(expand=True)
        self.grafik_image_label = None

        grafik_btn_frame = ttk.Frame(grafik_frame)
        grafik_btn_frame.pack(fill='x', padx=4, pady=4)
        ttk.Button(grafik_btn_frame, text="🔍  Yakınlaştır", cursor='hand2',
                   command=self.grafik_zoom_in).pack(side='left', padx=2)
        ttk.Button(grafik_btn_frame, text="🔎  Uzaklaştır", cursor='hand2',
                   command=self.grafik_zoom_out).pack(side='left', padx=2)
        ttk.Button(grafik_btn_frame, text="💾  Kaydet", cursor='hand2',
                   command=self.grafik_kaydet).pack(side='right', padx=2)

        # ── Sekme 3..12: L1 ... L10 (script açıklamaları)
        # ── Sağa yaslamak için geniş spacer sekmesi (görünmez, tıklanamaz)
        spacer = ttk.Frame(self.notebook)
        self.notebook.add(spacer, text="\u00A0" * 200)
        self.notebook.tab(spacer, state='disabled')

        # ── Sekme 3..12: L1 ... L10 (sağa yaslı)
        for level_num in range(1, 11):
            self._level_sekmesi_olustur(self.notebook, level_num)

    # ─────────────────────────────────────────────────────────
    #  LEVEL SEKMESİ (script metni)
    # ─────────────────────────────────────────────────────────
    def _level_sekmesi_olustur(self, notebook, level_num):
        tanim = self.level_tanimlari[level_num]
        c = self.theme.colors

        tab = ttk.Frame(notebook)
        notebook.add(tab, text=f"L{level_num}")

        # ── Üst bilgi çubuğu
        bar = tk.Frame(tab, bg=c['panel_alt'], height=36)
        bar.pack(fill='x')
        bar.pack_propagate(False)

        # Renkli rozet
        tk.Label(bar, text=f" L{level_num} ",
                 bg=c['accent'], fg=c['bg'],
                 font=ui(10, 'bold'),
                 padx=4).pack(side='left', padx=(10, 8), pady=6)

        tk.Label(bar, text=tanim['kisa'],
                 bg=c['panel_alt'], fg=c['text'],
                 font=ui(10, 'bold'),
                 anchor='w').pack(side='left', fill='x', expand=True, pady=6)

        gerekli = '1 hisse' if tanim['tek'] else '2+ hisse'
        tk.Label(bar, text=gerekli,
                 bg=c['accent'], fg=c['bg'],
                 font=ui(9, 'bold'),
                 padx=8, pady=2).pack(side='right', padx=10, pady=6)

        # ── Script içeriği
        icerik = self.script_aciklamalari.get(level_num, "(okunamadı)")

        text_frame = ttk.Frame(tab)
        text_frame.pack(fill='both', expand=True, padx=4, pady=4)

        text = tk.Text(text_frame, wrap='none', font=mono(10),
                       bg=c['term_bg'], fg=c['term_fg'],
                       relief='flat', borderwidth=0,
                       padx=14, pady=14,
                       highlightthickness=1,
                       highlightbackground=c['border'],
                       insertbackground=c['accent'])
        text.insert('1.0', icerik)
        text.config(state='disabled')

        ys = ttk.Scrollbar(text_frame, orient='vertical', command=text.yview)
        xs = ttk.Scrollbar(tab, orient='horizontal', command=text.xview)
        text.configure(yscrollcommand=ys.set, xscrollcommand=xs.set)

        text.pack(side='left', fill='both', expand=True)
        ys.pack(side='right', fill='y')
        xs.pack(fill='x')

        # Referansları sakla (tema değişimi için)
        self.level_text_widgets[level_num] = text
        self.level_bars[level_num] = bar

    # ─────────────────────────────────────────────────────────
    #  TEMA
    # ─────────────────────────────────────────────────────────
    def toggle_theme(self):
        self.theme.toggle()
        self._setup_styles()
        self._update_tk_widgets()
        self.theme_btn.config(
            text="🌙 Karanlık" if self.theme.current == 'dark' else "☀️ Aydınlık")
        self.durum_guncelle(f"🎨 Tema: {self.theme.current.capitalize()}")

    def _update_tk_widgets(self):
        c = self.theme.colors

        self.sol_canvas.configure(bg=c['bg'])
        self._separator.configure(bg=c['border'])

        self.hisse_listbox.configure(
            bg=c['term_bg'], fg=c['term_fg'],
            selectbackground=c['accent'], selectforeground=c['bg'],
            highlightbackground=c['border'], highlightcolor=c['accent'])

        self.metin_area.configure(
            bg=c['term_bg'], fg=c['term_fg'],
            highlightbackground=c['border'], highlightcolor=c['accent'],
            insertbackground=c['accent'])

        self._configure_text_tags()

        # L1..L10 sekme text'leri ve barları
        for n, text in self.level_text_widgets.items():
            text.configure(bg=c['term_bg'], fg=c['term_fg'],
                           highlightbackground=c['border'],
                           insertbackground=c['accent'])

        for n, bar in self.level_bars.items():
            bar.configure(bg=c['panel_alt'])
            for child in bar.winfo_children():
                # rozet (accent zemin) / metin (panel_alt) / gerekli (accent zemin)
                try:
                    if child.cget('bg') == c['accent'] or \
                       child.cget('bg') == self.theme.THEMES['dark' if self.theme.current == 'light' else 'light']['accent']:
                        child.configure(bg=c['accent'], fg=c['bg'])
                    else:
                        child.configure(bg=c['panel_alt'], fg=c['text'])
                except Exception:
                    pass

    def _configure_text_tags(self):
        dark = self.theme.current == 'dark'
        if dark:
            colors = {
                'style_separator': '#89b4fa', 'style_separator_dim': '#585b70',
                'style_header': '#89b4fa', 'style_section': '#fab387',
                'style_success': '#a6e3a1', 'style_danger': '#f38ba8',
                'style_warning': '#f9e2af', 'style_info': '#89b4fa',
                'style_info_cyan': '#94e2d5', 'style_accent': '#f5c2e7',
                'style_dim': '#6c7086',
                'style_positive': '#a6e3a1', 'style_negative': '#f38ba8',
            }
        else:
            colors = {
                'style_separator': '#1e66f5', 'style_separator_dim': '#acb0be',
                'style_header': '#1e66f5', 'style_section': '#fe640b',
                'style_success': '#40a02b', 'style_danger': '#d20f39',
                'style_warning': '#df8e1d', 'style_info': '#1e66f5',
                'style_info_cyan': '#179299', 'style_accent': '#ea76cb',
                'style_dim': '#8c8fa1',
                'style_positive': '#40a02b', 'style_negative': '#d20f39',
            }
        for name, color in colors.items():
            self.metin_area.tag_configure(name, foreground=color)

        try:
            bold = tkfont.Font(family=MONO_FONT, size=10, weight='bold')
            self.metin_area.tag_configure('style_header', font=bold)
            self.metin_area.tag_configure('style_separator', font=bold)
            self.metin_area.tag_configure('style_section', font=bold)
            self._term_bold_font = bold
        except Exception:
            pass

    # ─────────────────────────────────────────────────────────
    #  DİZİN
    # ─────────────────────────────────────────────────────────
    def varsayilan_dizin_yukle(self):
        self.dizin_guncelle(str(Path(__file__).parent))

    def dizin_sec(self):
        dizin = filedialog.askdirectory(title="CSV Dosyalarının Olduğu Dizini Seçin")
        if dizin:
            self.dizin_guncelle(dizin)

    def dizin_guncelle(self, dizin_yolu):
        self.csv_dizini = dizin_yolu
        self.dizin_label.config(text=f"📁 {dizin_yolu}")
        self.hisse_listbox.delete(0, tk.END)
        try:
            csv_dosyalari = sorted([
                f.stem.upper() for f in Path(dizin_yolu).glob('*.csv')
                if f.stem.upper() not in ['EMERGING_5_FACTORS']])
        except Exception:
            csv_dosyalari = []
        for hisse in csv_dosyalari:
            self.hisse_listbox.insert(tk.END, hisse)
        self.metin_yaz(f"✅ Dizin güncellendi: {dizin_yolu}\n")
        self.metin_yaz(f"📋 {len(csv_dosyalari)} hisse bulundu\n\n")

    def secili_hisseleri_al(self):
        return [self.hisse_listbox.get(i) for i in self.hisse_listbox.curselection()]

    # ─────────────────────────────────────────────────────────
    #  TERMİNAL
    # ─────────────────────────────────────────────────────────
    def metin_yaz(self, metin):
        def _yaz():
            try:
                self.metin_area.config(state='normal')
                for line in metin.splitlines(keepends=True):
                    stripped = line.rstrip('\n')
                    tag = self._detect_text_tag(stripped) if stripped else None
                    if tag:
                        self.metin_area.insert(tk.END, line, tag)
                    else:
                        self.metin_area.insert(tk.END, line)
                self.metin_area.see(tk.END)
                self.metin_area.config(state='disabled')
            except Exception:
                pass
        self.root.after(0, _yaz)

    def _detect_text_tag(self, line):
        for pattern, tag in ThreadSafeStdout.LINE_RULES:
            if pattern.match(line):
                return tag
        return None

    def metin_temizle(self):
        self.metin_area.config(state='normal')
        self.metin_area.delete('1.0', tk.END)
        self.metin_area.config(state='disabled')

    def metin_kopyala(self):
        icerik = self.metin_area.get('1.0', tk.END)
        self.root.clipboard_clear()
        self.root.clipboard_append(icerik)
        self.durum_guncelle("📋 Panoya kopyalandı")

    def durum_guncelle(self, mesaj):
        self.root.after(0, lambda: self.durum_label.config(text=mesaj))

    # ─────────────────────────────────────────────────────────
    #  GRAFİK
    # ─────────────────────────────────────────────────────────
    def grafik_goster(self, dosya_yolu):
        def _goster():
            try:
                self.grafik_label.pack_forget()
                if self.grafik_image_label:
                    self.grafik_image_label.destroy()
                    self.grafik_image_label = None
                if not PIL_KURULU:
                    self.metin_yaz(f"⚠️  Pillow kurulu değil. Grafik: {dosya_yolu}\n")
                    self.grafik_label.pack(expand=True)
                    return
                img = Image.open(dosya_yolu)
                self.grafik_image_path = dosya_yolu
                self.grafik_container.update_idletasks()
                cw = self.grafik_container.winfo_width()
                ch = self.grafik_container.winfo_height()
                if cw < 100 or ch < 100:
                    cw, ch = 1200, 700
                nw = int(img.width * self.grafik_zoom)
                nh = int(img.height * self.grafik_zoom)
                mw, mh = int(cw * 0.95), int(ch * 0.95)
                if nw > mw or nh > mh:
                    r = min(mw / nw, mh / nh)
                    nw, nh = int(nw * r), int(nh * r)
                img_resized = img.resize((nw, nh), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img_resized)
                self.current_image = photo
                self.grafik_image_label = ttk.Label(self.grafik_container, image=photo)
                self.grafik_image_label.pack(expand=True)
                self.durum_guncelle(f"🖼️ {Path(dosya_yolu).name}")
            except Exception as e:
                self.metin_yaz(f"⚠️  Grafik hatası: {e}\n")
                self.grafik_label.pack(expand=True)
        self.root.after(0, _goster)

    def grafik_zoom_in(self):
        self.grafik_zoom = min(3.0, self.grafik_zoom * 1.2)
        if self.grafik_image_path:
            self.grafik_goster(self.grafik_image_path)

    def grafik_zoom_out(self):
        self.grafik_zoom = max(0.3, self.grafik_zoom / 1.2)
        if self.grafik_image_path:
            self.grafik_goster(self.grafik_image_path)

    def grafik_kaydet(self):
        if self.grafik_image_path:
            hedef = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("All files", "*.*")],
                initialfile=Path(self.grafik_image_path).name)
            if hedef:
                import shutil
                shutil.copy2(self.grafik_image_path, hedef)
                self.durum_guncelle("💾 Kaydedildi")

    def grafik_bul_ve_goster(self, level_num, tek_hisse):
        dizin = Path(__file__).parent
        if tek_hisse:
            patterns = [f"{tek_hisse}_level{level_num}_*.png",
                        f"{tek_hisse}_level*.png",
                        f"level{level_num}_*.png"]
        else:
            patterns = [f"level{level_num}_*.png"]
        for pattern in patterns:
            dosyalar = list(dizin.glob(pattern))
            if dosyalar:
                en_yeni = max(dosyalar, key=os.path.getmtime)
                self.grafik_goster(str(en_yeni))
                return
        self.metin_yaz(f"⚠️  Level {level_num} için grafik bulunamadı\n")

    # ─────────────────────────────────────────────────────────
    #  LEVEL ÇALIŞTIRMA
    # ─────────────────────────────────────────────────────────
    def level_calistir(self, level_num):
        if self.calisiyor:
            messagebox.showwarning("Uyarı", "Bir işlem zaten çalışıyor!")
            return
        hisseler = self.secili_hisseleri_al()
        if not hisseler:
            messagebox.showwarning("Uyarı", "1️⃣ Lütfen en az bir hisse seçin!")
            return
        if level_num not in self.level_tanimlari:
            messagebox.showerror("Hata", f"Level {level_num} tanımlı değil!")
            return
        tanim = self.level_tanimlari[level_num]
        if tanim['tek'] and len(hisseler) > 1:
            if not messagebox.askyesno("Bilgi",
                                        f"Level {level_num} tek hisse kullanır.\n"
                                        f"Kullanılacak: {hisseler[0]}\n\nDevam?"):
                return
        self.calisiyor = True
        self.durum_guncelle(f"⏳ Level {level_num} çalışıyor...")
        self.metin_yaz(f"\n{'═' * 70}\n")
        self.metin_yaz(f"🚀 LEVEL {level_num}: {tanim['kisa']}\n")
        self.metin_yaz(f"{'═' * 70}\n\n")
        threading.Thread(target=self._level_calistir_thread,
                         args=(level_num, hisseler), daemon=True).start()

    def _level_calistir_thread(self, level_num, hisseler):
        try:
            tanim = self.level_tanimlari[level_num]
            mod_ad = tanim['mod']
            sinif_ad = tanim['sinif']
            tek_hisse = tanim['tek']
            mod_yolu = Path(__file__).parent / f"{mod_ad}.py"
            if not mod_yolu.exists():
                self.metin_yaz(f"❌ Modül bulunamadı: {mod_ad}.py\n")
                return
            spec = importlib.util.spec_from_file_location(mod_ad, mod_yolu)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[mod_ad] = mod
            spec.loader.exec_module(mod)
            sinif = getattr(mod, sinif_ad)
            redirector = ThreadSafeStdout(self.metin_area, self.root)
            old_stdout = sys.stdout
            sys.stdout = redirector
            try:
                if tek_hisse:
                    obj = self._level_objesi_olustur(level_num, sinif, hisseler[0])
                else:
                    obj = self._level_objesi_olustur_coklu(level_num, sinif, hisseler)
                obj.raporla()
            finally:
                sys.stdout = old_stdout
                redirector.flush()
            self.grafik_bul_ve_goster(level_num, hisseler[0] if tek_hisse else None)
            self.durum_guncelle(f"✅ Level {level_num} tamamlandı")
            self.metin_yaz(f"\n✅ Level {level_num} başarıyla tamamlandı!\n")
        except Exception as e:
            self.metin_yaz(f"\n❌ HATA: {e}\n")
            import traceback
            self.metin_yaz(traceback.format_exc())
            self.durum_guncelle(f"❌ Level {level_num} başarısız")
        finally:
            self.calisiyor = False

    def _level_objesi_olustur(self, level_num, sinif, hisse):
        import numpy as np
        if level_num == 6:
            from veri_periyot import VeriPeriyot
            vp = VeriPeriyot()
            df = vp.get(hisse, 'gunluk')
            if df is None or len(df) < 30:
                raise ValueError(f"{hisse} için yeterli veri yok!")
            S0 = float(df['Close'].iloc[-1])
            log_getiriler = np.log(df['Close'] / df['Close'].shift(1)).dropna()
            m = log_getiriler.mean()
            s = log_getiriler.std()
            mu_GBM = 252 * m + 0.5 * 252 * (s ** 2)
            sigma_annual = np.sqrt(252) * s
            try:
                return sinif(hisse=hisse, S0=S0, mu_GBM=mu_GBM,
                             sigma_annual=sigma_annual, r=0.45, n_simulasyon=5000)
            except TypeError:
                return sinif(hisse=hisse, S0=S0, mu_GBM=mu_GBM,
                             sigma_annual=sigma_annual, n_simulasyon=5000)
        elif level_num == 7:
            try:
                return sinif(hisse=hisse, K_carpan=1.0, T_gun=30, r=0.45, q=0.02)
            except TypeError:
                return sinif(hisse=hisse, K_carpan=1.0, T_gun=30)
        elif level_num == 9:
            try:
                return sinif(hisse=hisse, gunluk_bar_sayisi=60)
            except TypeError:
                try:
                    return sinif(hisse=hisse, gun_sayisi=60)
                except TypeError:
                    return sinif(hisse=hisse)
        elif level_num == 10:
            try:
                return sinif(hisse=hisse, periyot='gunluk')
            except TypeError:
                return sinif(hisse=hisse)
        else:
            try:
                return sinif(hisse=hisse, periyot='gunluk')
            except TypeError:
                return sinif(hisse=hisse)

    def _level_objesi_olustur_coklu(self, level_num, sinif, hisseler):
        if level_num == 8:
            try:
                return sinif(hisse_listesi=hisseler,
                             csv_dizini=str(self.csv_dizini), max_weight=0.40)
            except TypeError:
                try:
                    return sinif(hisse_listesi=hisseler, r_f_yillik=0.45, max_weight=0.40)
                except TypeError:
                    return sinif(hisse_listesi=hisseler, max_weight=0.40)
        else:
            try:
                return sinif(hisse_listesi=hisseler, periyot='gunluk')
            except TypeError:
                return sinif(hisse_listesi=hisseler)

    def tumunu_calistir(self):
        hisseler = self.secili_hisseleri_al()
        if not hisseler:
            messagebox.showwarning("Uyarı", "1️⃣ Lütfen en az bir hisse seçin!")
            return
        if not messagebox.askyesno("Onay",
                                    f"Tüm Level'ları (1-10) çalıştırmak?\n\n"
                                    f"Hisseler: {', '.join(hisseler)}"):
            return
        self.metin_temizle()

        def _run_all():
            for level_num in range(1, 11):
                if not self.calisiyor:
                    self._level_calistir_thread(level_num, hisseler)
        threading.Thread(target=_run_all, daemon=True).start()


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════
def main():
    global UI_FONT, MONO_FONT
    root = tk.Tk()
    root.withdraw()
    UI_FONT = _pick_ui_font()
    MONO_FONT = _pick_mono_font()
    try:
        tkfont.nametofont('TkDefaultFont').configure(family=UI_FONT, size=10)
        tkfont.nametofont('TkTextFont').configure(family=UI_FONT, size=10)
    except Exception:
        pass
    root.deiconify()
    app = QuantDashboard(root)
    root.mainloop()


if __name__ == "__main__":
    main()