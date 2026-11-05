# adim_7_turevler.py
"""
============================================================================
🟤 LEVEL 7: DERIVATIVES (Türev Araçlar) - "Geleceği Fiyatla"
============================================================================
AMAÇ: Black-Scholes-Merton modeli ile Avrupa tipi opsiyonları fiyatlamak, 
      risk hassasiyetlerini (Yunanlar) hesaplamak ve piyasa beklentisini 
      (Örtülü Volatilite) analiz etmek.
KAYNAK: Quant Roadmap - Level 7 (Görseldeki 5 kutuya tam sadık)
7.1 BLACK-SCHOLES MODELİ
    FORMÜL: C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
            P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)
            d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
            d2 = d1 - σ√T
    ANLAMI: Risksiz portföy oluşturarak (delta-hedging) türevin adil değeri.

7.2 PUT-CALL PARITY (Put-Call Paritesi)
    FORMÜL: C - P = S·e^(-qT) - K·e^(-rT)
    ANLAMI: Aynı strike ve vadeye sahip Call ve Put arasındaki arbitrajsız ilişki.
    NE İŞE YARAR: Modelin tutarlılığını test etmek ve sentetik pozisyon oluşturmak.

7.3 THE GREEKS (Yunanlar)
    FORMÜL: Δ (Delta), Γ (Gamma), ν (Vega), Θ (Theta), ρ (Rho)
    ANLAMI: Opsiyon fiyatının parametrelere (S, σ, T, r) olan birinci dereceden hassasiyeti.
    NE İŞE YARAR: Delta-hedging ve risk yönetimi.

7.4 IMPLIED VOLATILITY (Örtülü Volatilite)
    FORMÜL: σ_imp öyle ki BS(S, K, T, r, q, σ_imp) = Piyasa Fiyatı
    ANLAMI: Piyasanın gelecekteki volatilite beklentisi.
    NE İŞE YARAR: Opsiyonların pahalı mı ucuz mu olduğunu anlamak.
============================================================================
DÜZELTMELER:
1. 7.5 kutusu artık Gamma P&L (görseldeki formül). Volatility Smile ek içerik.
2. fiyat_hesapla() ve ortulu_volatilite() fonksiyonlarına K parametresi eklendi.
   Gülüş simülasyonu artık her strike için doğru fiyat/IV hesaplıyor.
3. Zaman değeri: max(0, S0-K) kullanılarak OTM call hatası düzeltildi.
4. IV alt sınırı: Temettülü Avrupa opsiyonu için doğru formül uygulandı.
   C_min = max(0, S·e^(-qT) - K·e^(-rT))
   P_min = max(0, K·e^(-rT) - S·e^(-qT))
5. 7.4'te sentetik örnek olduğu raporda açıkça belirtildi.
6. linspace(0.80, 1.20, 21) ile ATM tam ortada (indeks 10).
============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import brentq
from veri_periyot import VeriPeriyot
import warnings
warnings.filterwarnings('ignore')


class Level7Turevler:
    def __init__(self, hisse, K_carpan=1.0, T_gun=30, r=0.45, q=0.02):
        self.hisse = hisse.upper()
        self.K_carpan = K_carpan
        self.T = T_gun / 365.0
        self.r = r
        self.q = q
        self.vp = VeriPeriyot()
        self.S0 = 0.0
        self.sigma = 0.0
        self.K = 0.0
        
    def veriyi_hazirla(self):
        print(f"📥 {self.hisse} verisi çekiliyor...")
        df = self.vp.get(self.hisse, 'gunluk')
        if df is None or len(df) < 30:
            raise ValueError(f"❌ {self.hisse} için yeterli veri yok!")
            
        self.S0 = float(df['Close'].iloc[-1])
        self.K = self.S0 * self.K_carpan
        
        log_getiriler = np.log(df['Close'] / df['Close'].shift(1)).dropna()
        self.sigma = np.std(log_getiriler) * np.sqrt(252)
        
        print(f"✅ S0 = {self.S0:.2f} TL, K = {self.K:.2f} TL, T = {self.T:.4f} yıl")
        print(f"✅ Tarihsel σ = %{self.sigma*100:.2f}, r = %{self.r*100:.2f}, q = %{self.q*100:.2f}\n")

    def _d1_d2(self, S, K, sigma):
        """d1 ve d2'yi hesapla (S ve K parametreli)"""
        d1 = (np.log(S / K) + (self.r - self.q + 0.5 * sigma**2) * self.T) / (sigma * np.sqrt(self.T))
        d2 = d1 - sigma * np.sqrt(self.T)
        return d1, d2

    def fiyat_hesapla(self, option_type='call', sigma=None, S=None, K=None):
        """
        Black-Scholes fiyatı
        DÜZELTME: S ve K parametreleri eklendi (varsayılan: self.S0, self.K)
        """
        if sigma is None:
            sigma = self.sigma
        if S is None:
            S = self.S0
        if K is None:
            K = self.K
            
        d1, d2 = self._d1_d2(S, K, sigma)
        
        if option_type.lower() == 'call':
            fiyat = S * np.exp(-self.q * self.T) * norm.cdf(d1) - K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            fiyat = K * np.exp(-self.r * self.T) * norm.cdf(-d2) - S * np.exp(-self.q * self.T) * norm.cdf(-d1)
            
        return fiyat

    def yunanlar_hesapla(self, option_type='call', sigma=None, S=None, K=None):
        """Delta, Gamma, Vega, Theta, Rho hesapla"""
        if sigma is None:
            sigma = self.sigma
        if S is None:
            S = self.S0
        if K is None:
            K = self.K
            
        d1, d2 = self._d1_d2(S, K, sigma)
        S_disc = S * np.exp(-self.q * self.T)
        K_disc = K * np.exp(-self.r * self.T)
        nd1 = norm.pdf(d1)
        
        is_call = (option_type.lower() == 'call')
        
        delta = np.exp(-self.q * self.T) * norm.cdf(d1) if is_call else np.exp(-self.q * self.T) * (norm.cdf(d1) - 1)
        gamma = (np.exp(-self.q * self.T) * nd1) / (S * sigma * np.sqrt(self.T))
        vega = (S_disc * nd1 * np.sqrt(self.T)) / 100.0
        
        term1 = -(S_disc * nd1 * sigma) / (2 * np.sqrt(self.T))
        if is_call:
            theta = (term1 - self.r * K_disc * norm.cdf(d2) + self.q * S_disc * norm.cdf(d1)) / 365.0
        else:
            theta = (term1 + self.r * K_disc * norm.cdf(-d2) - self.q * S_disc * norm.cdf(-d1)) / 365.0
            
        rho = (K_disc * self.T * norm.cdf(d2) if is_call else -K_disc * self.T * norm.cdf(-d2)) / 100.0
        
        return {'Delta': delta, 'Gamma': gamma, 'Vega': vega, 'Theta': theta, 'Rho': rho}

    def ortulu_volatilite(self, piyasa_fiyati, option_type='call', S=None, K=None):
        """
        Brentq ile örtülü volatilite hesapla
        DÜZELTME: Temettülü Avrupa opsiyonu için doğru alt sınır
        """
        if S is None:
            S = self.S0
        if K is None:
            K = self.K
        
        # DÜZELTME 4: Temettülü Avrupa opsiyonu için doğru alt sınır
        if option_type.lower() == 'call':
            min_fiyat = max(0, S * np.exp(-self.q * self.T) - K * np.exp(-self.r * self.T))
        else:
            min_fiyat = max(0, K * np.exp(-self.r * self.T) - S * np.exp(-self.q * self.T))
            
        if piyasa_fiyati < min_fiyat - 1e-4:
            return np.nan
            
        def objective(vol):
            return self.fiyat_hesapla(option_type, vol, S, K) - piyasa_fiyati
            
        try:
            iv = brentq(objective, 0.01, 3.00, xtol=1e-6)
            return iv
        except ValueError:
            return np.nan

    def volatilite_gulusu_simulasyonu(self):
        """
        Farklı strike fiyatları için örtülü volatilite eğrisi
        DÜZELTME 2: Her strike için doğru K kullanılıyor
        DÜZELTME 6: 21 nokta ile ATM tam ortada (indeks 10)
        """
        strike_oranlari = np.linspace(0.80, 1.20, 21)  # 21 nokta, ATM = indeks 10
        iv_listesi = []
        piyasa_fiyatlari = []
        
        for k_oran in strike_oranlari:
            K_strike = self.S0 * k_oran
            
            # Simüle edilmiş piyasa fiyatı: skew ile
            skew_adjusted_sigma = self.sigma * (1.0 + 0.15 * (1.0 - k_oran))
            simule_piyasa_fiyati = self.fiyat_hesapla('call', sigma=skew_adjusted_sigma, K=K_strike)
            piyasa_fiyatlari.append(simule_piyasa_fiyati)
            
            # Geri hesapla (standart BSM ile)
            iv = self.ortulu_volatilite(simule_piyasa_fiyati, 'call', K=K_strike)
            iv_listesi.append(iv)
            
        return strike_oranlari * self.S0, iv_listesi, piyasa_fiyatlari

    def gamma_pnl_hesapla(self):
        """
        7.5 Gamma P&L (Görseldeki formül)
        P&L ≈ ½·Γ·S²·(σ²_realized - σ²_implied)·dt
        """
        # ATM call için Yunanlar
        greeks = self.yunanlar_hesapla('call')
        Gamma = greeks['Gamma']
        
        # Senaryolar: Gerçekleşen volatilite vs Örtülü volatilite
        sigma_implied = self.sigma  # Örtülü = tarihsel (varsayım)
        
        senaryolar = {
            'Düşük Volatilite (σ_real = 0.8·σ_imp)': 0.8 * sigma_implied,
            'Eşit Volatilite (σ_real = σ_imp)': sigma_implied,
            'Yüksek Volatilite (σ_real = 1.2·σ_imp)': 1.2 * sigma_implied,
            'Çok Yüksek Volatilite (σ_real = 1.5·σ_imp)': 1.5 * sigma_implied,
        }
        
        dt = 1.0 / 252  # 1 günlük P&L
        
        pnl_sonuclari = {}
        for senaryo_adi, sigma_real in senaryolar.items():
            # P&L ≈ ½·Γ·S²·(σ²_real - σ²_imp)·dt
            pnl = 0.5 * Gamma * (self.S0 ** 2) * (sigma_real**2 - sigma_implied**2) * dt
            pnl_sonuclari[senaryo_adi] = {
                'sigma_real': sigma_real,
                'pnl_gunluk': pnl,
                'pnl_yillik': pnl * 252
            }
        
        return {
            'Gamma': Gamma,
            'sigma_implied': sigma_implied,
            'senaryolar': pnl_sonuclari,
            'dt': dt
        }

    def put_call_parity_kontrol(self):
        """C - P = S·e^(-qT) - K·e^(-rT)"""
        C = self.fiyat_hesapla('call')
        P = self.fiyat_hesapla('put')
        
        sol_taraf = C - P
        sag_taraf = self.S0 * np.exp(-self.q * self.T) - self.K * np.exp(-self.r * self.T)
        fark = abs(sol_taraf - sag_taraf)
        
        return {'Call': C, 'Put': P, 'Sol_Taraf': sol_taraf, 'Sag_Taraf': sag_taraf, 'Fark': fark}

    def raporla(self):
        self.veriyi_hazirla()
        
        print("="*70)
        print(f"🟤 LEVEL 7: DERIVATIVES ({self.hisse} Opsiyon Analizi)")
        print("="*70)
        
        # 7.1 Black-Scholes
        C = self.fiyat_hesapla('call')
        P = self.fiyat_hesapla('put')
        icsel_deger_call = max(0, self.S0 - self.K)
        zaman_degeri_call = C - icsel_deger_call  # DÜZELTME 3
        
        print(f"\n📜 7.1 BLACK-SCHOLES MODELİ")
        print(f"   • Call Opsiyon Değeri: {C:.4f} TL")
        print(f"   • Put Opsiyon Değeri : {P:.4f} TL")
        print(f"   • İçsel Değer (Call) : {icsel_deger_call:.4f} TL")
        print(f"   • Zaman Değeri (Call): {zaman_degeri_call:.4f} TL")

        # 7.2 Put-Call Parity
        pcp = self.put_call_parity_kontrol()
        print(f"\n⚖️ 7.2 PUT-CALL PARITY")
        print(f"   FORMÜL: C - P = S·e^(-qT) - K·e^(-rT)")
        print(f"   • Sol Taraf (C - P)                : {pcp['Sol_Taraf']:.4f}")
        print(f"   • Sağ Taraf (S·e^(-qT) - K·e^(-rT)): {pcp['Sag_Taraf']:.4f}")
        print(f"   • Arbitraj Farkı                   : {pcp['Fark']:.8f} TL")
        print(f"   💡 MÜLAKAT: 'Put-call parity neyi kanıtlar?'")
        print(f"              Cevap: Arbitraj olmayan piyasada sentetik long/short pozisyon eşitliğini.")

        # 7.3 The Greeks
        greeks = self.yunanlar_hesapla('call')
        print(f"\n🏛️ 7.3 THE GREEKS (Call Opsiyonu)")
        print(f"   • Delta (Δ) : {greeks['Delta']:.4f} (Hedge için {greeks['Delta']*100:.0f} hisse short)")
        print(f"   • Gamma (Γ) : {greeks['Gamma']:.6f} (Delta'nın S'ye göre değişim hızı)")
        print(f"   • Vega (ν)  : {greeks['Vega']:.4f} (Volatilite %1 artarsa fiyat değişimi)")
        print(f"   • Theta (Θ) : {greeks['Theta']:.4f} (Günlük zaman erimesi)")
        print(f"   • Rho (ρ)   : {greeks['Rho']:.4f} (Faiz %1 artarsa fiyat değişimi)")

        # 7.4 Implied Volatility (SENTETİK ÖRNEK)
        simule_piyasa_fiyati = self.fiyat_hesapla('call', sigma=self.sigma * 1.10)
        iv = self.ortulu_volatilite(simule_piyasa_fiyati, 'call')
        print(f"\n🔍 7.4 IMPLIED VOLATILITY (Örtülü Volatilite) - SENTETİK ÖRNEK")
        print(f"   ⚠️  NOT: Bu örnek sentetik piyasa fiyatı kullanmaktadır.")
        print(f"            Gerçek piyasa opsiyon zinciri verisi gerekli değildir.")
        print(f"   • Simüle Piyasa Fiyatı (σ·1.10): {simule_piyasa_fiyati:.4f} TL")
        print(f"   • Hesaplanan Örtülü σ          : %{iv*100:.2f}")
        print(f"   • Tarihsel σ                   : %{self.sigma*100:.2f}")
        print(f"   💡 MÜLAKAT: 'IV, tarihsel volatiliteden yüksekse ne anlama gelir?'")
        print(f"              Cevap: Piyasa, gelecekte tarihsel ortalamadan daha büyük hareketler bekliyor.")

        # 7.5 Gamma P&L (GÖRSELDEKİ FORMÜL)
        gamma_pnl = self.gamma_pnl_hesapla()
        print(f"\n📊 7.5 GAMMA P&L (Görseldeki Formül)")
        print(f"   FORMÜL: P&L ≈ ½·Γ·S²·(σ²_realized - σ²_implied)·dt")
        print(f"   • Gamma (Γ)              : {gamma_pnl['Gamma']:.6f}")
        print(f"   • Örtülü Volatilite (σ_imp): %{gamma_pnl['sigma_implied']*100:.2f}")
        print(f"   • Zaman Adımı (dt)       : {gamma_pnl['dt']:.6f} (1 gün)")
        print(f"\n   📋 Senaryo Analizi (Günlük P&L):")
        for senaryo_adi, sonuc in gamma_pnl['senaryolar'].items():
            pnl_gunluk = sonuc['pnl_gunluk']
            pnl_yillik = sonuc['pnl_yillik']
            emoji = "📈" if pnl_gunluk > 0 else "📉" if pnl_gunluk < 0 else "➡️"
            print(f"      {emoji} {senaryo_adi}:")
            print(f"         σ_real = %{sonuc['sigma_real']*100:.2f}")
            print(f"         Günlük P&L: {pnl_gunluk:+.4f} TL")
            print(f"         Yıllık P&L: {pnl_yillik:+.2f} TL")
        print(f"   💡 MÜLAKAT: 'Delta-hedged opsiyonun kazancı nereden gelir?'")
        print(f"              Cevap: Gerçekleşen volatilite ile örtülü volatilite farkından (Gamma scalping).")

        # EK: Volatility Smile (Görselde yok ama faydalı)
        strikes, ivs, _ = self.volatilite_gulusu_simulasyonu()
        print(f"\n😊 EK: VOLATILITY SMILE (Görselde yok, ek analiz)")
        print(f"   • ATM (K={self.S0:.0f}) Volatilite : %{ivs[10]*100:.2f}")  # İndeks 10 = ATM
        print(f"   • OTM Put (K={strikes[0]:.0f}) Volatilite : %{ivs[0]*100:.2f}")
        print(f"   • OTM Call (K={strikes[-1]:.0f}) Volatilite : %{ivs[-1]*100:.2f}")

        self.grafik_ciz(greeks, strikes, ivs, gamma_pnl)

    def grafik_ciz(self, greeks, strikes, ivs, gamma_pnl):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Opsiyon Değeri vs Hisse Fiyatı
        ax1 = axes[0, 0]
        S_range = np.linspace(self.S0 * 0.7, self.S0 * 1.3, 100)
        C_range, P_range = [], []
        for S in S_range:
            d1 = (np.log(S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
            d2 = d1 - self.sigma * np.sqrt(self.T)
            C_range.append(S * np.exp(-self.q * self.T) * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2))
            P_range.append(self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - S * np.exp(-self.q * self.T) * norm.cdf(-d1))
            
        ax1.plot(S_range, C_range, label='Call Değeri', color='green', linewidth=2)
        ax1.plot(S_range, P_range, label='Put Değeri', color='red', linewidth=2)
        ax1.plot(S_range, np.maximum(0, S_range - self.K), 'g--', alpha=0.5, label='Call İçsel Değer')
        ax1.axvline(self.S0, color='blue', linestyle=':', label=f'Mevcut S0 ({self.S0:.0f})')
        ax1.set_title('Opsiyon Değeri vs. Dayanak Varlık Fiyatı', fontweight='bold')
        ax1.set_xlabel('Hisse Fiyatı (S)')
        ax1.set_ylabel('Opsiyon Değeri')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Delta ve Gamma
        ax2 = axes[0, 1]
        delta_range, gamma_range = [], []
        for S in S_range:
            d1 = (np.log(S / self.K) + (self.r - self.q + 0.5 * self.sigma**2) * self.T) / (self.sigma * np.sqrt(self.T))
            nd1 = norm.pdf(d1)
            delta_range.append(np.exp(-self.q * self.T) * norm.cdf(d1))
            gamma_range.append((np.exp(-self.q * self.T) * nd1) / (S * self.sigma * np.sqrt(self.T)))
            
        ax2.plot(S_range, delta_range, label='Delta (Sol Eksen)', color='blue', linewidth=2)
        ax2.set_ylabel('Delta', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        
        ax2_twin = ax2.twinx()
        ax2_twin.plot(S_range, gamma_range, label='Gamma (Sağ Eksen)', color='purple', linewidth=2)
        ax2_twin.set_ylabel('Gamma', color='purple')
        ax2_twin.tick_params(axis='y', labelcolor='purple')
        
        ax2.axvline(self.S0, color='gray', linestyle=':')
        ax2.set_title('Delta ve Gamma (Call)', fontweight='bold')
        ax2.set_xlabel('Hisse Fiyatı (S)')
        ax2.grid(True, alpha=0.3)

        # 3. Gamma P&L Senaryoları (7.5 - Görseldeki ana konu)
        ax3 = axes[1, 0]
        senaryo_isimleri = list(gamma_pnl['senaryolar'].keys())
        pnl_degerleri = [gamma_pnl['senaryolar'][s]['pnl_yillik'] for s in senaryo_isimleri]
        renkler = ['red' if p < 0 else 'green' for p in pnl_degerleri]
        
        ax3.barh(senaryo_isimleri, pnl_degerleri, color=renkler, alpha=0.7, edgecolor='black')
        ax3.axvline(x=0, color='black', linewidth=1)
        ax3.set_xlabel('Yıllık P&L (TL)', fontweight='bold')
        ax3.set_title('7.5 Gamma P&L Senaryoları\n(Gerçekleşen vs Örtülü Volatilite)', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='x')
        
        # Değerleri etiketle
        for i, pnl in enumerate(pnl_degerleri):
            ax3.text(pnl + (100 if pnl >= 0 else -100), i, f'{pnl:+.0f} TL', 
                    va='center', ha='left' if pnl >= 0 else 'right', fontweight='bold')

        # 4. Volatility Smile (Ek içerik)
        ax4 = axes[1, 1]
        ax4.plot(strikes, np.array(ivs)*100, marker='o', color='darkorange', linewidth=2)
        ax4.axvline(self.S0, color='gray', linestyle=':', label=f'ATM (S0={self.S0:.0f})')
        ax4.set_title('Volatility Smile/Skew (Ek Analiz)', fontweight='bold')
        ax4.set_xlabel('Strike Fiyatı (K)')
        ax4.set_ylabel('Örtülü Volatilite (%)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f'{self.hisse}_level7_turevler.png', dpi=150)
        print(f"\n📊 Grafik kaydedildi: {self.hisse}_level7_turevler.png")
        plt.show()


# ========== ÇALIŞTIRMA ==========
if __name__ == "__main__":
    analiz = Level7Turevler(
        hisse='THYAO', 
        K_carpan=1.0,   # ATM
        T_gun=30,       # 1 aylık vade
        r=0.45,         # %45 yıllık risksiz faiz
        q=0.02          # %2 yıllık temettü verimi
    )
    analiz.raporla()