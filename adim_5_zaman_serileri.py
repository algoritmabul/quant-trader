# adim_5_zaman_serileri.py
"""
============================================================================
🟠 LEVEL 5: TIME SERIES (Zaman Serileri) - "Piyasaların Hafızası Vardır, Bazen"
============================================================================
AMAÇ: Zaman serilerinin istatistiksel özelliklerini analiz etmek, durağanlık 
      testleri yapmak, eşbütünleşme ilişkilerini bulmak ve volatiliteyi modellemek.

KAYNAK: Quant Roadmap - Level 5 (Görseldeki 5 kutuya tam sadık)
-----------------------------------------------------------------------------

5.1 LOG RETURNS (Logaritmik Getiriler)
    FORMÜL: r_t = ln(P_t / P_{t-1})
    YILLIK VOLATİLİTE: σ_yıllık = σ_günlük × √252

5.2 AUTOCORRELATION (Otokorelasyon)
    FORMÜL: ρ_k = Corr(r_t, r_{t-k})

5.3 STATIONARITY (ADF Testi)
    FORMÜL: Δy_t = α + γy_{t-1} + Σ δ_i Δy_{t-i} + ε_t
    H0: γ = 0 (seride birim kök var, durağan değil)
    H1: γ < 0 (seri durağan)

5.4 COINTEGRATION AND HALF-LIFE (Eşbütünleşme ve Yarı Ömür)
    FORMÜL: s_t = ln(A_t) - β·ln(B_t) - c
    YARI ÖMÜR (Tam Ayrık Zaman): half-life = ln(0.5) / ln(1 + λ)
    NOT: 0 < 1+λ < 1 koşulu sağlanmalıdır.

5.5 GARCH(1,1)
    FORMÜL: σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}
    UZUN DÖNEM: σ²_LR = ω / (1 - α - β)
    TAHMİN: 1-adım ileri volatilite tahmini (forecast)

============================================================================
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller, coint
from arch import arch_model
from veri_periyot import VeriPeriyot
import warnings
warnings.filterwarnings('ignore')


class Level5ZamanSerileri:
    def __init__(self, hisse_listesi, periyot='gunluk'):
        self.hisse_listesi = [h.upper() for h in hisse_listesi]
        self.periyot = periyot.lower()
        self.vp = VeriPeriyot()
        self.getiri_df = None
        self.fiyat_df = None
        self.log_fiyat_df = None
        self.yuklenen_hisseler = []
        
    def veriyi_hazirla(self):
        print(f"📥 {len(self.hisse_listesi)} hisse verisi yükleniyor...")
        
        getiriler, fiyatlar, log_fiyatlar = {}, {}, {}
        
        for hisse in self.hisse_listesi:
            try:
                df = self.vp.get(hisse, self.periyot)
                if df is not None and len(df) >= 30:
                    fiyatlar[hisse] = df['Close']
                    log_fiyatlar[hisse] = np.log(df['Close'])
                    getiriler[hisse] = np.log(df['Close'] / df['Close'].shift(1))
                    self.yuklenen_hisseler.append(hisse)
                    print(f"   ✅ {hisse}: {len(df)} bar")
            except Exception as e:
                print(f"   ❌ {hisse}: Hata - {e}")
        
        if len(getiriler) < 1:
            raise ValueError("❌ En az 1 hisse için yeterli veri gerekli!")
        
        self.getiri_df = pd.DataFrame(getiriler).dropna()
        self.fiyat_df = pd.DataFrame(fiyatlar).dropna()
        self.log_fiyat_df = pd.DataFrame(log_fiyatlar).dropna()
        
        print(f"\n✅ {len(self.getiri_df)} ortak bar, {len(self.yuklenen_hisseler)} hisse hazır.\n")
        return self.getiri_df

    def hesapla(self):
        if self.getiri_df is None:
            self.veriyi_hazirla()
        
        n_hisse = len(self.yuklenen_hisseler)
        
        # ==================== 5.1 LOG RETURNS ====================
        log_vs_basit_farklar = {}
        yillik_volatilite = {}
        
        for hisse in self.yuklenen_hisseler:
            fiyatlar = self.fiyat_df[hisse].values
            log_get = np.log(fiyatlar[1:] / fiyatlar[:-1])
            basit_get = (fiyatlar[1:] - fiyatlar[:-1]) / fiyatlar[:-1]
            
            log_vs_basit_farklar[hisse] = np.mean(log_get - basit_get)
            gunluk_std = self.getiri_df[hisse].std()
            yillik_volatilite[hisse] = gunluk_std * np.sqrt(252)
        
        # ==================== 5.2 AUTOCORRELATION ====================
        lags = [1, 5, 10, 20]
        otokorelasyonlar = {lag: [] for lag in lags}
        
        for hisse in self.yuklenen_hisseler:
            seri = self.getiri_df[hisse].values
            for lag in lags:
                if len(seri) > lag:
                    rho = np.corrcoef(seri[lag:], seri[:-lag])[0, 1]
                    otokorelasyonlar[lag].append(rho)
        
        otokorelasyon_ort = {lag: np.mean(rhos) for lag, rhos in otokorelasyonlar.items()}
        lag1_rho = otokorelasyon_ort[1]
        
        # ==================== 5.3 STATIONARITY (ADF Testi) ====================
        adf_sonuclari = {}
        for hisse in self.yuklenen_hisseler:
            getiri_seri = self.getiri_df[hisse].values
            log_fiyat_seri = self.log_fiyat_df[hisse].values
            
            adf_sonuclari[hisse] = {
                'getiri': {
                    'adf_stat': adfuller(getiri_seri, maxlag=None, autolag='AIC')[0],
                    'p_value': adfuller(getiri_seri, maxlag=None, autolag='AIC')[1],
                    'durağan': adfuller(getiri_seri, maxlag=None, autolag='AIC')[1] < 0.05
                },
                'log_fiyat': {
                    'adf_stat': adfuller(log_fiyat_seri, maxlag=None, autolag='AIC')[0],
                    'p_value': adfuller(log_fiyat_seri, maxlag=None, autolag='AIC')[1],
                    'durağan': adfuller(log_fiyat_seri, maxlag=None, autolag='AIC')[1] < 0.05
                }
            }
        
        # ==================== 5.4 COINTEGRATION AND HALF-LIFE ====================
        esbütünlesme_sonuclari = {}
        
        if n_hisse >= 2:
            hisse1, hisse2 = self.yuklenen_hisseler[0], self.yuklenen_hisseler[1]
            log_fiyat1 = self.log_fiyat_df[hisse1].values
            log_fiyat2 = self.log_fiyat_df[hisse2].values
            
            # Eşbütünleşme testi (log fiyatlarla)
            esbütünlesme_stat, p_value, _ = coint(log_fiyat1, log_fiyat2)
            esbütünlesik = p_value < 0.05
            
            # Spread regresyonu: ln(A) = β·ln(B) + c + ε  =>  s = ln(A) - β·ln(B) - c
            log_fiyat2_with_const = np.column_stack([log_fiyat2, np.ones(len(log_fiyat2))])
            beta_with_const, c_sabit = np.linalg.lstsq(log_fiyat2_with_const, log_fiyat1, rcond=None)[0]
            spread = log_fiyat1 - beta_with_const * log_fiyat2 - c_sabit
            
            # Half-life için ayrık zaman AR(1) regresyonu: Δs_t = λ·s_{t-1} + ε_t
            spread_lag = spread[:-1]
            spread_diff = np.diff(spread)
            spread_lag_with_const = np.column_stack([spread_lag, np.ones(len(spread_lag))])
            lambda_ols, _ = np.linalg.lstsq(spread_lag_with_const, spread_diff, rcond=None)[0]
            
            # ✅ DÜZELTME: Tam ayrık zaman formülü: ln(0.5) / ln(1 + λ)
            phi = 1 + lambda_ols
            if 0 < phi < 1:
                half_life = np.log(0.5) / np.log(phi)
            else:
                half_life = np.inf  # Ortalamaya dönüş yok
            
            esbütünlesme_sonuclari = {
                'hisse1': hisse1, 'hisse2': hisse2,
                'esbütünlesme_stat': esbütünlesme_stat,
                'p_value': p_value,
                'esbütünlesik': esbütünlesik,
                'beta': beta_with_const,
                'c_sabit': c_sabit,
                'lambda_ols': lambda_ols,
                'half_life': half_life
            }
        
        # ==================== 5.5 GARCH(1,1) ====================
        garch_sonuclari = {}
        
        for hisse in self.yuklenen_hisseler:
            seri = self.getiri_df[hisse].values * 100
            
            try:
                model = arch_model(seri, vol='GARCH', p=1, q=1, mean='Zero')
                sonuc = model.fit(disp='off')
                
                omega = sonuc.params['omega']
                alpha = sonuc.params['alpha[1]']
                beta = sonuc.params['beta[1]']
                
                # Uzun dönem varyans
                if (alpha + beta) < 1:
                    uzun_donem_varyans = omega / (1 - alpha - beta)
                    uzun_donem_volatilite = np.sqrt(uzun_donem_varyans)
                else:
                    uzun_donem_varyans = np.inf
                    uzun_donem_volatilite = np.inf
                
                # ✅ DÜZELTME: 1-adım ileri gerçek volatilite tahmini (forecast)
                forecast = sonuc.forecast(horizon=1, reindex=False)
                next_step_variance = forecast.variance.iloc[-1, 0]
                tahmini_volatilite = np.sqrt(next_step_variance)
                
                garch_sonuclari[hisse] = {
                    'omega': omega,
                    'alpha': alpha,
                    'beta': beta,
                    'volatiliti_kalicilik': alpha + beta,
                    'uzun_donem_volatilite': uzun_donem_volatilite,
                    'tahmini_volatilite': tahmini_volatilite, # Artık bu bir forecast
                    'mean_zero': True
                }
            except Exception as e:
                garch_sonuclari[hisse] = {'hata': str(e)}
        
        return {
            'n_hisse': n_hisse,
            'yuklenen_hisseler': self.yuklenen_hisseler,
            'ortalama_fark': np.mean(list(log_vs_basit_farklar.values())),
            'yillik_volatilite': yillik_volatilite,
            'otokorelasyon_ort': otokorelasyon_ort,
            'lag1_rho': lag1_rho,
            'adf_sonuclari': adf_sonuclari,
            'esbütünlesme_sonuclari': esbütünlesme_sonuclari,
            'garch_sonuclari': garch_sonuclari
        }

    def raporla(self):
        self.veriyi_hazirla()
        m = self.hesapla()
        
        print("="*70)
        print(f"🟠 LEVEL 5: TIME SERIES (Görseldeki 5 Kutu)")
        print(f"📊 {m['n_hisse']} Hisse Zaman Serisi Analizi")
        print("="*70)
        
        # 5.1
        print(f"\n📈 5.1 LOG RETURNS")
        print(f"   FORMÜL: r_t = ln(P_t / P_{{t-1}})  |  σ_yıllık = σ_günlük × √252")
        print(f"   • Log/Basit Getiri Farkı (Ort.): {m['ortalama_fark']:.6f}")
        for hisse in m['yuklenen_hisseler']:
            print(f"      {hisse:<8}: Yıllık Vol = %{m['yillik_volatilite'][hisse]*100:.2f}")

        # 5.2
        print(f"\n🔁 5.2 AUTOCORRELATION")
        print(f"   FORMÜL: ρ_k = Corr(r_t, r_{{t-k}})")
        print(f"   • Lag-1 Otokorelasyon (Ort.): {m['lag1_rho']:.4f}")
        if m['lag1_rho'] > 0.05: print("   📈 YORUM: Pozitif → MOMENTUM")
        elif m['lag1_rho'] < -0.05: print("   🔄 YORUM: Negatif → MEAN-REVERSION")
        else: print("   ➡️  YORUM: Sıfıra yakın → Rastgele yürüyüş")

        # 5.3
        print(f"\n📊 5.3 STATIONARITY (ADF Testi)")
        print(f"   H0: γ = 0 (birim kök var) | H1: γ < 0 (durağan)")
        for hisse, sonuc in m['adf_sonuclari'].items():
            g_durum = "✅ Durağan" if sonuc['getiri']['durağan'] else "❌ Değil"
            f_durum = "✅ Durağan" if sonuc['log_fiyat']['durağan'] else "❌ Değil"
            print(f"   {hisse:<8}: Getiri p={sonuc['getiri']['p_value']:.4f} ({g_durum}) | Fiyat p={sonuc['log_fiyat']['p_value']:.4f} ({f_durum})")

        # 5.4
        if m['esbütünlesme_sonuclari']:
            es = m['esbütünlesme_sonuclari']
            print(f"\n🔗 5.4 COINTEGRATION AND HALF-LIFE")
            print(f"   FORMÜL: s_t = ln(A) - β·ln(B) - c  |  half-life = ln(0.5) / ln(1 + λ)")
            print(f"   • Çift: {es['hisse1']} & {es['hisse2']} | Eşbütünleşik: {'✅ Evet' if es['esbütünlesik'] else '❌ Hayır'} (p={es['p_value']:.4f})")
            print(f"   • β: {es['beta']:.4f} | Sabit (c): {es['c_sabit']:.4f} | λ: {es['lambda_ols']:.4f}")
            if es['half_life'] != np.inf:
                print(f"   • Half-life (Tam Ayrık Zaman): {es['half_life']:.1f} bar")
            else:
                print(f"   • Half-life: ∞ (Ortalamaya dönüş yok│monotonik yarı ömür hesaplanamıyor, φ = 1+λ koşulu sağlanmıyor)")

        # 5.5
        print(f"\n📉 5.5 GARCH(1,1)")
        print(f"   FORMÜL: σ²_t = ω + α·ε²_{{t-1}} + β·σ²_{{t-1}}  |  σ²_LR = ω / (1 - α - β)")
        print(f"   NOT: mean='Zero' (bilinçli seçim) | Tahmin: 1-adım ileri forecast")
        for hisse, sonuc in m['garch_sonuclari'].items():
            if 'hata' not in sonuc:
                print(f"\n   {hisse}:")
                print(f"      α (ARCH): {sonuc['alpha']:.4f} | β (GARCH): {sonuc['beta']:.4f} | α+β: {sonuc['volatiliti_kalicilik']:.4f}")
                if sonuc['uzun_donem_volatilite'] != np.inf:
                    print(f"      Uzun Dönem Vol: %{sonuc['uzun_donem_volatilite']:.2f}")
                print(f"      1-Adım İleri Tahmin Vol: %{sonuc['tahmini_volatilite']:.2f}")
            else:
                print(f"\n   {hisse}: ❌ Hata - {sonuc['hata']}")

        print("\n" + "="*70)


if __name__ == "__main__":
    HISSE_LISTESI = ['THYAO', 'GARAN', 'ASELS', 'BIMAS']
    analiz = Level5ZamanSerileri(HISSE_LISTESI, 'gunluk')
    analiz.raporla()