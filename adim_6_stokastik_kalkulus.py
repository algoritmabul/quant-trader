# adim_6_stokastik_kalkulus_nihai.py
"""
============================================================================
🔴 LEVEL 6: STOCHASTIC CALCULUS (Nihai ve Eksiksiz Versiyon)
============================================================================
DÜZELTMELER:
1. 6.1 (Random Walk) ve 6.2 (Brownian Motion) bölümleri geri eklendi.
2. Euler negatif fiyat kontrolü artık tüm yolu (np.any) kapsıyor.
3. np.log hesaplamasından önce <= 0 olan yollar filtrelenerek NaN önleniyor.
4. r=0.45, "sürekli bileşik yıllık risksiz faiz" olarak belirtildi. 
   Martingale hatasının Monte Carlo örnekleme hatası nedeniyle tam 0 olmayacağı not edildi.
5. np.random.seed yerine yerel ve temiz np.random.default_rng(42) kullanıldı.
============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from veri_periyot import VeriPeriyot
import warnings
warnings.filterwarnings('ignore')


class Level6StokastikKalkulus:
    def __init__(self, hisse, S0, mu_GBM, sigma_annual, r=0.45, n_simulasyon=10000):
        self.hisse = hisse
        self.S0 = S0
        self.mu_GBM = mu_GBM
        self.sigma_annual = sigma_annual
        self.r = r  # Sürekli bileşik yıllık risksiz faiz oranı
        self.n_simulasyon = n_simulasyon
        self.n_adim = 252
        # Düzeltme 5: Yerel ve temiz rastgele sayı üreteci
        self.rng = np.random.default_rng(42)
        
    def hesapla(self):
        dt = 1.0 / self.n_adim
        t = np.linspace(0, 1, self.n_adim + 1)
        
        # ==================== 6.1 RANDOM WALK ====================
        baslangic_rw = 100
        rw_steps = self.rng.standard_normal((self.n_simulasyon, self.n_adim))
        rw = baslangic_rw + np.cumsum(rw_steps, axis=1)
        
        rw_sonuc = {
            'std_son': np.std(rw[:, -1]),
            'teorik_std': np.sqrt(self.n_adim),
            'tipik_mesafe_10': np.std(rw[:, 9]),
            'teorik_mesafe_10': np.sqrt(10)
        }
        
        # ==================== 6.2 BROWNIAN MOTION ====================
        dW = np.sqrt(dt) * self.rng.standard_normal((self.n_simulasyon, self.n_adim))
        W = np.hstack((np.zeros((self.n_simulasyon, 1)), np.cumsum(dW, axis=1)))
        
        zamanlar = [21, 63, 126, 252]
        bm_istatistikler = {}
        for t_idx in zamanlar:
            bm_istatistikler[t_idx] = {
                'std': np.std(W[:, t_idx]),
                'teorik_std': np.sqrt(t_idx * dt)
            }
            
        bm_sonuc = {
            'istatistikler': bm_istatistikler,
            'ortalama_dw_kare': np.mean(dW**2),
            'teorik_dt': dt,
            'W': W
        }
        
        # ==================== 6.3 GEOMETRIC BROWNIAN MOTION ====================
        S_analytical = self.S0 * np.exp((self.mu_GBM - 0.5 * self.sigma_annual**2) * t + self.sigma_annual * W)
        
        S_euler = np.zeros((self.n_simulasyon, self.n_adim + 1))
        S_euler[:, 0] = self.S0
        for i in range(1, self.n_adim + 1):
            S_euler[:, i] = S_euler[:, i-1] * (1 + self.mu_GBM * dt + self.sigma_annual * dW[:, i-1])
            
        # Düzeltme 2: Herhangi bir adımda negatif veya sıfır olan yolları tespit et
        negatif_yol_sayisi = np.sum(np.any(S_euler <= 0, axis=1))
        
        gbm_sonuc = {
            'S_analytical': S_analytical,
            'S_euler': S_euler,
            'negatif_yol_sayisi': negatif_yol_sayisi,
            'teorik_beklenen': self.S0 * np.exp(self.mu_GBM * 1.0),
            'gercek_beklenen_analitik': np.mean(S_analytical[:, -1])
        }
        
        # ==================== 6.4 ITÔ'S LEMMA ====================
        # Düzeltme 3: NaN oluşmasını önlemek için sadece pozitif kalan yolları filtrele
        valid_paths = np.all(S_euler > 0, axis=1)
        S_euler_valid = S_euler[valid_paths]
        W_valid = W[valid_paths]
        
        ln_S_num = np.log(S_euler_valid)
        ln_S_ito = np.log(self.S0) + (self.mu_GBM - 0.5 * self.sigma_annual**2) * t + self.sigma_annual * W_valid
        
        # Son adımdaki ortalama mutlak hata (Ayrık zamanlama hatası)
        hata_ito = np.mean(np.abs(ln_S_num[:, -1] - ln_S_ito[:, -1]))
        
        ito_sonuc = {'hata_ito': hata_ito}
        
        # ==================== 6.5 MARTINGALES & GAMBLER'S RUIN ====================
        # Risk-nötr ölçüde simülasyon (Drift = r)
        S_Q = self.S0 * np.exp((self.r - 0.5 * self.sigma_annual**2) * t + self.sigma_annual * W)
        M_t = S_Q * np.exp(-self.r * t)
        
        # E[M_t] = S_0 olmalı. Fark, Monte Carlo örnekleme hatasıdır (~sigma/sqrt(N))
        martingale_hata = abs(np.mean(M_t[:, -1]) - self.S0)
        
        # Gambler's Ruin (30 -> 100)
        baslangic_sermaye, hedef_sermaye = 30, 100
        teorik_kazanma = baslangic_sermaye / hedef_sermaye
        
        n_oyun = 10000
        kazanma_sayisi = 0
        for _ in range(n_oyun):
            sermaye = baslangic_sermaye
            while 0 < sermaye < hedef_sermaye:
                sermaye += 1 if self.rng.random() < 0.5 else -1
            if sermaye >= hedef_sermaye:
                kazanma_sayisi += 1
                
        gambler_sonuc = {
            'martingale_hata': martingale_hata,
            'teorik_kazanma_olasiligi': teorik_kazanma,
            'gercek_kazanma_olasiligi': kazanma_sayisi / n_oyun
        }
        
        return {
            'rw': rw_sonuc,
            'bm': bm_sonuc,
            'gbm': gbm_sonuc,
            'ito': ito_sonuc,
            'gambler': gambler_sonuc
        }

    def grafik_ciz(self, sonuclar):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. GBM Gerçek Fiyat Senaryoları
        ax1 = axes[0, 0]
        S_analitik = sonuclar['gbm']['S_analytical']
        for i in range(200):
            ax1.plot(S_analitik[i], alpha=0.2, linewidth=0.5, color='blue')
        ax1.set_title(f'{self.hisse} - 200 Gelecek Senaryosu (Gerçek Fiyat TL)', fontweight='bold')
        ax1.set_xlabel('İşlem Günü (1 Yıl)')
        ax1.set_ylabel('Fiyat (TL)')
        ax1.grid(True, alpha=0.3)
        
        # 2. Son Fiyat Dağılımı
        ax2 = axes[0, 1]
        son_fiyatlar = S_analitik[:, -1]
        ax2.hist(son_fiyatlar, bins=50, color='green', alpha=0.7, edgecolor='black')
        ax2.axvline(x=np.median(son_fiyatlar), color='red', linestyle='--', label=f'Medyan: {np.median(son_fiyatlar):.2f} TL')
        ax2.set_title('1 Yıl Sonraki Olası Fiyat Dağılımı', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Brownian Motion Yolları
        ax3 = axes[1, 0]
        W = sonuclar['bm']['W']
        for i in range(100):
            ax3.plot(W[i], alpha=0.3, linewidth=0.5, color='purple')
        ax3.set_title('Standart Brownian Hareketi (W_t)', fontweight='bold')
        ax3.set_xlabel('Zaman Adımı')
        ax3.set_ylabel('W_t')
        ax3.grid(True, alpha=0.3)
        
        # 4. Özet Parametreler
        ax4 = axes[1, 1]
        ax4.axis('off')
        metrikler = [
            ['Hisse', self.hisse],
            ['Başlangıç Fiyatı (S_0)', f"{self.S0:.2f} TL"],
            ['GBM Drift (μ)', f"%{self.mu_GBM*100:.2f}"],
            ['Yıllık Volatilite (σ)', f"%{self.sigma_annual*100:.2f}"],
            ['Risksiz Faiz (r)', f"%{self.r*100:.2f} (Sürekli Bileşik)"],
            ['Martingale Hatası', f"{sonuclar['gambler']['martingale_hata']:.4f} TL (MC Hatası)"],
            ['Itô Ayrık Hata', f"{sonuclar['ito']['hata_ito']:.6f}"],
            ['Negatif Euler Yolu', f"{sonuclar['gbm']['negatif_yol_sayisi']} / {self.n_simulasyon}"]
        ]
        table = ax4.table(cellText=metrikler, colLabels=['Parametre', 'Değer'], 
                          loc='center', bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        for j in range(2):
            table[0, j].set_facecolor('#34495e')
            table[0, j].set_text_props(color='white', fontweight='bold')
        ax4.set_title('Simülasyon Doğrulama Metrikleri', fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(f'{self.hisse}_level6_nihai.png', dpi=150)
        print(f"\n📊 Grafik kaydedildi: {self.hisse}_level6_nihai.png")
        plt.show()

    def raporla(self):
        sonuclar = self.hesapla()
        
        print("="*70)
        print(f"🔴 LEVEL 6: STOCHASTIC CALCULUS ({self.hisse} GERÇEK VERİSİ)")
        print("="*70)
        print(f"📈 Başlangıç Fiyatı (S_0)   : {self.S0:.2f} TL")
        print(f"📈 GBM Drift (μ)            : %{self.mu_GBM*100:.2f} (252m + 0.5*252s²)")
        print(f"⚡ Yıllık Volatilite (σ)     : %{self.sigma_annual*100:.2f} (√252 * s)")
        print(f"🏦 Risksiz Faiz (r)         : %{self.r*100:.2f} (Sürekli Bileşik)")
        print("="*70)
        
        # 6.1
        rw = sonuclar['rw']
        print(f"\n🎲 6.1 RANDOM WALK")
        print(f"   • 1 Yıl Sonra Std Sapma: {rw['std_son']:.4f} (Teorik: {rw['teorik_std']:.4f})")
        print(f"   • 10 Adım Tipik Mesafe : {rw['tipik_mesafe_10']:.4f} (Teorik: {rw['teorik_mesafe_10']:.4f})")
        
        # 6.2
        bm = sonuclar['bm']
        print(f"\n🌊 6.2 BROWNIAN MOTION")
        print(f"   • (dW)² Ortalaması: {bm['ortalama_dw_kare']:.6f} (Teorik dt: {bm['teorik_dt']:.6f})")
        
        # 6.3
        gbm = sonuclar['gbm']
        print(f"\n📈 6.3 GEOMETRIC BROWNIAN MOTION")
        print(f"   • Negatif Fiyat Gören Yol Sayısı: {gbm['negatif_yol_sayisi']} / {self.n_simulasyon}")
        print(f"   • Teorik Beklenen Değer (E[S_t]): {gbm['teorik_beklenen']:.2f} TL")
        print(f"   • Simülasyon Ortalaması         : {gbm['gercek_beklenen_analitik']:.2f} TL")
        
        # 6.4
        ito = sonuclar['ito']
        print(f"\n🔬 6.4 ITÔ'S LEMMA (Ayrık Zaman Doğrulaması)")
        print(f"   • Euler Çözümü ile Itô Açılımı Arası Hata: {ito['hata_ito']:.6f}")
        print(f"   • Not: Sadece pozitif kalan yollar hesaba katılarak NaN önlenmiştir.")
        
        # 6.5
        gambler = sonuclar['gambler']
        print(f"\n🎰 6.5 MARTINGALES & GAMBLER'S RUIN")
        print(f"   • Risk-Nötr Martingale İndirgeme Hatası : {gambler['martingale_hata']:.4f} TL")
        print(f"     (Not: Monte Carlo örnekleme hatası nedeniyle tam 0 beklenmez)")
        print(f"   • Gambler's Ruin (30 -> 100):")
        print(f"      Teorik Kazanma Olasılığı             : %{gambler['teorik_kazanma_olasiligi']*100:.1f}")
        print(f"      Simülasyon Kazanma Olasılığı         : %{gambler['gercek_kazanma_olasiligi']*100:.1f}")

        self.grafik_ciz(sonuclar)


# ========== ÇALIŞTIRMA ==========
if __name__ == "__main__":
    SECILEN_HISSE = 'ahgaz'
    PERIYOT = 'gunluk'
    
    print(f"📥 {SECILEN_HISSE} verisi çekiliyor...")
    vp = VeriPeriyot()
    df = vp.get(SECILEN_HISSE, PERIYOT)
    
    if df is None or len(df) < 30:
        raise ValueError(f"❌ {SECILEN_HISSE} için yeterli veri bulunamadı!")
        
    S0_gercek = float(df['Close'].iloc[-1])
    
    log_getiriler = np.log(df['Close'] / df['Close'].shift(1)).dropna()
    m = log_getiriler.mean()
    s = log_getiriler.std()
    
    mu_GBM = 252 * m + 0.5 * 252 * (s**2)
    sigma_annual = np.sqrt(252) * s
    
    r_risksiz = 0.45 # Sürekli bileşik yıllık risksiz faiz
    
    print(f"✅ Parametreler: S0={S0_gercek:.2f}, μ_GBM={mu_GBM:.4f}, σ={sigma_annual:.4f}\n")
    
    analiz = Level6StokastikKalkulus(
        hisse=SECILEN_HISSE,
        S0=S0_gercek,
        mu_GBM=mu_GBM,
        sigma_annual=sigma_annual,
        r=r_risksiz,
        n_simulasyon=10000
    )
    
    analiz.raporla()