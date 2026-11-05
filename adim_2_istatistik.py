# adim_2_sadece_istatistik.py
"""
============================================================================
🟢 LEVEL 2: STATISTICS (İSTATİSTİK) - "Gerçek mi, Gürültü mü?"
============================================================================
AMAÇ: Level 1'de ölçtüğümüz olasılık değerlerinin istatistiksel olarak 
      anlamlı olup olmadığını test etmek. "Edge" gerçek mi yoksa şans mı?

KAYNAK: Quant Roadmap - Level 2 (Görseldeki 5 kutuya tam sadık)
-----------------------------------------------------------------------------

2.1 STANDART HATA (Standard Error)
    FORMÜL: SE = σ / √n
    ANLAMI: Ölçtüğümüz ortalamanın (EV) ne kadar güvenilir olduğu.
    NE İŞE YARAR: EV pozitif çıksa bile SE büyükse, sonuç şans eseri olabilir.
    MÜLAKAT: "Hatayı yarıya indirmek için veri sayısını kaç katına çıkarmalısın?"
             Cevap: 4 katına (√n kuralı).

2.2 t-İSTATİSTİĞİ (t-stat of a Sharpe)
    FORMÜL: t ≈ SR × √T
    ANLAMI: Ölçülen değerin istatistiksel anlamlılığı. |t| > 2 genelde anlamlı.
    NE İŞE YARAR: Backtest sonucu "gerçek edge" mi yoksa "şans" mı ayrıştırır.
    MÜLAKAT: "4 yıllık aylık SR 0.2 olan strateji anlamlı mı?" 
             Cevap: t = 0.2 × √48 ≈ 1.39 → Anlamsız (borderline).

2.3 OLS REGRESSION (Ordinary Least Squares)
    FORMÜL: β = (X'X)^{-1} X'y
    ANLAMI: En iyi doğrusal ilişkiyi bulan regresyon katsayısı.
    NE İŞE YARAR: Hedge oranları, faktör maruziyetleri, sinyal ağırlıkları.
    MÜLAKAT: "Tek değişkenli β'yı sıfırdan türet."
             Cevap: β = Cov(X,Y) / Var(X) (basitleştirilmiş formül).

2.4 CORRELATION AND R²
    FORMÜL: ρ = Cov(X,Y) / (σₓσᵧ)  |  R² = 1 - SS_res / SS_tot
    ANLAMI: İki seri ne kadar birlikte hareket ediyor? Model ne kadar açıklıyor?
    NE İŞE YARAR: Piyasada küçük bir R² bile gerçek, trade edilebilir edge olabilir.
    MÜLAKAT: "X simetrik etrafında 0: Corr(X, X²) = 0, yet they are not independent."

2.5 MULTIPLE TESTING
    FORMÜL: P(1 false win) = 1 - (1 - α)^n
    ANLAMI: Yeterli sayıda test yap ve bazıları şansa kazanır.
    NE İŞE YARAR: 20 testte %5 anlamlılıkta %64 ihtimalle en az 1 sahte kazanan var.
    MÜLAKAT: "En iyi backtest'in neden abartılmış olabilir?"
             Cevap: Multiple testing bias - en iyisi seçildiğinde şans abartılır.

============================================================================
KULLANIM:
    python adim_2_sadece_istatistik.py
============================================================================
"""

import pandas as pd
import numpy as np
from scipy import stats
from veri_periyot import VeriPeriyot
import warnings
warnings.filterwarnings('ignore')

class Level2Istatistik:
    def __init__(self, hisse, periyot='gunluk'):
        self.hisse = hisse.upper()
        self.periyot = periyot.lower()
        self.vp = VeriPeriyot()
        self.df = None
        
    def veriyi_hazirla(self):
        """Veriyi çeker"""
        print(f"📥 {self.hisse} ({self.periyot}) verisi yükleniyor...")
        self.df = self.vp.get(self.hisse, self.periyot)
        
        if self.df is None or len(self.df) < 30:
            raise ValueError(f"❌ {self.hisse} için yeterli veri yok!")
            
        self.df['Getiri'] = self.df['Close'].pct_change()
        self.df['Hacim_Degisim'] = self.df['Volume'].pct_change()
        self.df = self.df.dropna()
        print(f"✅ {len(self.df)} bar veri hazır.\n")

    def hesapla(self):
        """Görseldeki Level 2'nin 5 kutusunu hesapla"""
        getiriler = self.df['Getiri'].values
        hacim_degisim = self.df['Hacim_Degisim'].values
        n = len(getiriler)
        
        # Level 1'den gelen değerler
        ev = np.mean(getiriler)
        std_sapma = np.std(getiriler, ddof=1)
        
        # ==================== 2.1 STANDART HATA ====================
        standart_hata = std_sapma / np.sqrt(n)
        
        # ==================== 2.2 t-İSTATİSTİĞİ ====================
        # t ≈ SR × √T formülü
        # Günlük SR = EV / σ
        sharpe_gunluk = ev / std_sapma if std_sapma > 0 else 0
        # Yıllık SR (252 işlem günü)
        sharpe_yillik = sharpe_gunluk * np.sqrt(252)
        # t-stat of Sharpe: t ≈ SR_yıllık × √(T_yıl)
        # T_yıl = n / 252
        t_yil = n / 252
        t_stat_sharpe = sharpe_yillik * np.sqrt(t_yil)
        
        # ==================== 2.3 OLS REGRESSION ====================
        # Basit OLS: y = α + βx
        # x = bir önceki günün getirisi, y = bugünün getirisi
        # β = (X'X)^{-1} X'y formülünü sıfırdan implement et
        
        # X vektörü (intercept için 1'ler + bir önceki gün getirisi)
        x_ols = getiriler[:-1]  # t-1 günün getirisi
        y_ols = getiriler[1:]   # t günün getirisi
        
        # X matrisi: [1, x] şeklinde (intercept + slope)
        X_matris = np.column_stack([np.ones(len(x_ols)), x_ols])
        
        # β = (X'X)^{-1} X'y (sıfırdan OLS)
        XtX = X_matris.T @ X_matris
        XtX_inv = np.linalg.inv(XtX)
        Xty = X_matris.T @ y_ols
        beta_ols = XtX_inv @ Xty
        
        alpha_ols = beta_ols[0]  # Intercept
        beta_ols_slope = beta_ols[1]  # Slope (eğim)
        
        # ==================== 2.4 CORRELATION AND R² ====================
        # Korelasyon: ρ = Cov(X,Y) / (σₓσᵧ)
        # x = getiri, y = hacim değişimi
        cov_xy = np.cov(getiriler, hacim_degisim, ddof=1)[0, 1]
        std_x = np.std(getiriler, ddof=1)
        std_y = np.std(hacim_degisim, ddof=1)
        korelasyon = cov_xy / (std_x * std_y) if (std_x > 0 and std_y > 0) else 0
        
        # R² = 1 - SS_res / SS_tot (OLS regresyonunun açıklayıcılığı)
        # Tahmin edilen değerler
        y_pred = alpha_ols + beta_ols_slope * x_ols
        # SS_res (residual sum of squares)
        ss_res = np.sum((y_ols - y_pred) ** 2)
        # SS_tot (total sum of squares)
        ss_tot = np.sum((y_ols - np.mean(y_ols)) ** 2)
        r_kare = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        # ==================== 2.5 MULTIPLE TESTING ====================
        n_test = 20
        alpha = 0.05
        sahte_kazanan_olasiligi = 1 - (1 - alpha) ** n_test
        
        return {
            'n': n,
            'ev': ev,
            'std_sapma': std_sapma,
            'standart_hata': standart_hata,
            'sharpe_gunluk': sharpe_gunluk,
            'sharpe_yillik': sharpe_yillik,
            't_stat_sharpe': t_stat_sharpe,
            't_yil': t_yil,
            'alpha_ols': alpha_ols,
            'beta_ols_slope': beta_ols_slope,
            'korelasyon': korelasyon,
            'r_kare': r_kare,
            'ss_res': ss_res,
            'ss_tot': ss_tot,
            'sahte_kazanan_olasiligi': sahte_kazanan_olasiligi,
            'n_test': n_test
        }

    def raporla(self):
        """Görseldeki 5 kutuyu raporla"""
        self.veriyi_hazirla()
        m = self.hesapla()
        
        print("="*70)
        print(f"🟢 {self.hisse} - LEVEL 2: İSTATİSTİK (Görseldeki 5 Kutu)")
        print("="*70)
        
        # ==================== 2.1 STANDART HATA ====================
        print(f"\n📏 2.1 STANDART HATA (Standard Error)")
        print(f"   FORMÜL: SE = σ / √n")
        print(f"   • σ (Günlük Volatilite) = {m['std_sapma']:.6f}")
        print(f"   • n (Bar Sayısı) = {m['n']}")
        print(f"   • SE = {m['standart_hata']:.6f}")
        print(f"   • EV %95 Güven Aralığı: [{m['ev'] - 1.96*m['standart_hata']:+.4f}, {m['ev'] + 1.96*m['standart_hata']:+.4f}]")
        if abs(m['ev']) < 2 * m['standart_hata']:
            print("   ⚠️  YORUM: EV, standart hatadan küçük! Sonuç istatistiksel olarak anlamsız olabilir.")
        else:
            print("   ✅ YORUM: EV, standart hatadan büyük. Sonuç nispeten güvenilir.")
        print(f"   💡 MÜLAKAT: 'Hatayı yarıya indirmek için veri sayısını kaç katına çıkarmalısın?'")
        print(f"              Cevap: 4 katına (√n kuralı).")

        # ==================== 2.2 t-İSTATİSTİĞİ ====================
        print(f"\n📊 2.2 t-İSTATİSTİĞİ (t-stat of a Sharpe)")
        print(f"   FORMÜL: t ≈ SR × √T")
        print(f"   • Günlük Sharpe (SR) = {m['sharpe_gunluk']:.4f}")
        print(f"   • Yıllık Sharpe = {m['sharpe_yillik']:.3f}")
        print(f"   • T (Yıl) = {m['t_yil']:.2f}")
        print(f"   • t-stat = {m['t_stat_sharpe']:.3f}")
        
        if abs(m['t_stat_sharpe']) > 2:
            print("   ✅ YORUM: |t| > 2 → Sharpe istatistiksel olarak ANLAMLI (gerçek edge).")
        elif abs(m['t_stat_sharpe']) > 1.5:
            print("   ⚠️  YORUM: |t| 1.5-2 arası → Borderline. Daha fazla veri gerekli.")
        else:
            print("   ❌ YORUM: |t| < 1.5 → Sharpe istatistiksel olarak ANLAMSIZ (şans olabilir).")
        print(f"   💡 MÜLAKAT: '4 yıllık aylık SR 0.2 olan strateji anlamlı mı?'")
        print(f"              Cevap: t = 0.2 × √48 ≈ 1.39 → Anlamsız (borderline).")

        # ==================== 2.3 OLS REGRESSION ====================
        print(f"\n📈 2.3 OLS REGRESSION (Ordinary Least Squares)")
        print(f"   FORMÜL: β = (X'X)^{{-1}} X'y")
        print(f"   • Model: Getiri_t = α + β × Getiri_{{t-1}}")
        print(f"   • α (Intercept) = {m['alpha_ols']:.6f}")
        print(f"   • β (Slope) = {m['beta_ols_slope']:.6f}")
        
        if m['beta_ols_slope'] > 0.05:
            print("   📈 YORUM: Pozitif β → MOMENTUM (dünkü yükseliş bugünkü yükselişi getirir).")
        elif m['beta_ols_slope'] < -0.05:
            print("   🔄 YORUM: Negatif β → MEAN-REVERSION (dünkü düşüş bugünkü toparlamayı getirir).")
        else:
            print("   ➡️  YORUM: β ≈ 0 → Rastgele yürüyüş (Random Walk).")
        print(f"   💡 MÜLAKAT: 'Tek değişkenli β'yı sıfırdan türet.'")
        print(f"              Cevap: β = Cov(X,Y) / Var(X) (basitleştirilmiş formül).")

        # ==================== 2.4 CORRELATION AND R² ====================
        print(f"\n🔗 2.4 CORRELATION AND R²")
        print(f"   FORMÜL: ρ = Cov(X,Y) / (σₓσᵧ)  |  R² = 1 - SS_res / SS_tot")
        print(f"   • Korelasyon (Getiri vs Hacim Değişimi) = {m['korelasyon']:.4f}")
        print(f"   • R² (OLS Regresyonunun Açıklayıcılığı) = {m['r_kare']:.4f} (%{m['r_kare']*100:.2f})")
        print(f"   • SS_res = {m['ss_res']:.6f}")
        print(f"   • SS_tot = {m['ss_tot']:.6f}")
        
        if m['r_kare'] > 0.1:
            print("   ✅ YORUM: R² > 0.1 → Model getirilerin %10'undan fazlasını açıklıyor (güçlü).")
        elif m['r_kare'] > 0.02:
            print("   ⚠️  YORUM: R² 0.02-0.1 arası → Zayıf ama trade edilebilir edge olabilir.")
        else:
            print("   ❌ YORUM: R² < 0.02 → Model neredeyse hiçbir şeyi açıklamıyor.")
        print(f"   💡 MÜLAKAT: 'X simetrik etrafında 0: Corr(X, X²) = 0, yet they are not independent.'")
        print(f"              Cevap: Korelasyon lineer ilişkiyi ölçer, non-lineer ilişkileri kaçırır.")

        # ==================== 2.5 MULTIPLE TESTING ====================
        print(f"\n⚠️ 2.5 MULTIPLE TESTING")
        print(f"   FORMÜL: P(1 false win) = 1 - (1 - α)^n")
        print(f"   • α (Anlamlılık Düzeyi) = {0.05}")
        print(f"   • n (Test Sayısı) = {m['n_test']}")
        print(f"   • P(Sahte Kazanan) = {m['sahte_kazanan_olasiligi']:.4f} (%{m['sahte_kazanan_olasiligi']*100:.1f})")
        print("   💡 YORUM: 20 farklı strateji denersen, %64 ihtimalle en az biri şans eseri 'iyi' çıkar.")
        print("             Backtest'te 'en iyi' stratejiyi seçerken dikkatli ol!")
        print(f"   💡 MÜLAKAT: 'En iyi backtest'in neden abartılmış olabilir?'")
        print(f"              Cevap: Multiple testing bias - en iyisi seçildiğinde şans abartılır.")

        print("\n" + "="*70)
        print("💡 SONUÇ: Level 2 tamamlandı (Görseldeki 5 kutu).")
        print("Verinin istatistiksel güvenilirliği ölçüldü.")
        print("Bir sonraki adımda (Level 3: Lineer Cebir) korelasyon matrisi ve PCA eklenebilir.")
        print("="*70)


# ========== ÇALIŞTIRMA ==========
if __name__ == "__main__":
    HISSE = 'THYAO'
    PERIYOT = 'gunluk'
    
    try:
        analiz = Level2Istatistik(HISSE, PERIYOT)
        analiz.raporla()
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")