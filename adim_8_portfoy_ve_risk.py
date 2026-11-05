# adim_8_portfoy_ve_risk.py
"""
============================================================================
⚫ LEVEL 8: PORTFOLIO AND RISK (Portföy ve Risk) - "Ne Kadar Bahis, Ne Kadar Kayıp"
============================================================================
AMAÇ: Sistemik riski ölçmek, getiriyi riske göre ayarlamak, optimal bahis 
      boyutunu bulmak ve kuyruk riskini (VaR/ES) hesaplamak.

KAYNAK: Quant Roadmap - Level 8 (Görseldeki 5 kutuya tam sadık)
-----------------------------------------------------------------------------

8.1 CAPM (Capital Asset Pricing Model)
    FORMÜL: E[R] = r_f + β(E[R_m] - r_f)  |  β = Cov(R, R_m) / Var(R_m)
    ANLAMI: Sadece piyasa riskini alarak beklenen getiri.
    NE İŞE YARAR: Gerçek alpha'yı (fazla getiri) piyasa maruziyetinden ayırır.

8.2 FAMA-FRENCH 3-FAKTÖR
    FORMÜL: R - r_f = α + b·MKT + s·SMB + h·HML + ε
    ANLAMI: Piyasa (MKT), Boyut (SMB) ve Değer (HML) faktörleri getiriyi açıklar.
    NE İŞE YARAR: Alpha'nızın sadece bir "boyut" veya "değer" eğilimi olup olmadığını test eder.

8.3 SHARPE RATIO (Sharpe Oranı)
    FORMÜL: SR = (R - r_f) / σ  |  SR_yıllık = SR_günlük × √252
    ANLAMI: Birim risk başına düşen getiri.
    NE İŞE YARAR: Günlük SR 0.1 ≈ Yıllık 1.59. Strateji karşılaştırma standardı.

8.4 KELLY CRITERION (Kelly Kriteri)
    FORMÜL: f* = (bp - q) / b  (Ayrık)  |  f* = (μ - r) / σ² (Sürekli)
    ANLAMI: Uzun vadeli büyümeyi maksimize eden bahis boyutu.
    NE İŞE YARAR: p=0.55, b=1 ise %10 bahis. Profesyoneller genelde "Half-Kelly" kullanır.

8.5 VaR, ES AND DRAWDOWN (VaR, Beklenen Kısa Vade ve Maksimum Düşüş)
    FORMÜL: VaR_95 = 1.645 × σ  |  ES_95 = σ × φ(1.645) / 0.05
    ANLAMI: Kötü bir gündeki kayıp (VaR) ve o eşiği aşan ortalama kayıp (ES).
    NE İŞE YARAR: $1M, %2 günlük vol -> VaR $32.9k, ES $41.3k. Max DD en büyük tepe-dip düşüşü.

============================================================================
"""

# adim_8_portfoy_ve_risk_nihai.py
"""
============================================================================
⚫ LEVEL 8: PORTFOLIO AND RISK (Portföy ve Risk) - "Ne Kadar Bahis, Ne Kadar Kayıp"
============================================================================
DÜZELTMELER:
1. r_f matematiği: 0.45 / 252 = 0.001785 olarak düzeltildi.
2. Piyasa Getirisi (R_m): Önce 'XU100' (BIST 100) çekilir. Yoksa uyarı verip 
   bağımsız sentetik bir piyasa serisi kullanılır (hisse içi korelasyon önlenir).
3. Fama-French: MKT faktörü artık (R_m - r_f) olarak (excess return) kullanılıyor.
4. Portföy Riski: Metrikler artık "Eşit Ağırlıklı Portföy" üzerinden hesaplanıyor.
5. Max Drawdown: Büyüme serisinin başına 0.0 getiri (değer = 1.0) eklenerek 
   ilk günkü olası düşüşün tepe değeri kaçırması engellendi.
============================================================================
"""
# adim_8_portfoy_ve_risk_cvxpy.py
"""
============================================================================
⚫ LEVEL 8: PORTFOLIO AND RISK (Portföy ve Risk) - "Ne Kadar Bahis, Ne Kadar Kayıp"
============================================================================
DÜZELTME: Görseldeki TOOLS kısmına sadık kalarak cvxpy eklendi.
Artık eşit ağırlıklı portföy yerine, cvxpy ile optimize edilmiş (Minimum Varyans 
veya Maksimum Sharpe) portföy ağırlıkları bulunuyor ve risk metrikleri bu 
optimal portföy üzerinden hesaplanıyor.
============================================================================
"""

# adim_8_portfoy_ve_risk_cvxpy_duzeltilmis.py
"""
============================================================================
⚫ LEVEL 8: PORTFOLIO AND RISK (Portföy ve Risk) - cvxpy ile Optimize
============================================================================
DÜZELTMELER:
1. İki ayrı optimizasyon modu: Minimum Varyans VE Maksimum Sharpe
2. Maksimum Sharpe için Efficient Frontier taraması
3. Diversifikasyon kısıtı: Tek hisseye yığılmayı önlemek için max_weight
4. Yorumlar düzeltildi
============================================================================
"""
# adim_8_portfoy_ve_risk_cvxpy_hata_raporlu.py
"""
============================================================================
⚫ LEVEL 8: PORTFOLIO AND RISK (cvxpy) - Hata Raporlaması Eklenmiş
============================================================================
DÜZELTMELER:
1. cp.installed_solvers() ile kurulu çözücüleri listele
2. except bloklarında hatayı print et (pass/continue kaldırıldı)
3. problem.solve(verbose=True) ile çözücü detaylarını göster
4. ECOS yerine daha stabil OSQP/SCS çözücülerini dene
============================================================================
"""

# adim_8_portfoy_ve_risk_gercek_faktorler.py
"""
============================================================================
⚫ LEVEL 8: PORTFOLIO AND RISK (Gerçek Fama-French 5-Faktör Verisi)
============================================================================
VERİ KAYNAĞI: Emerging_5_Factors.csv (Bloomberg, 202609)
- Mkt-RF: Piyasa fazlası getirisi (Market minus Risk-Free)
- SMB: Küçük-Büyük şirket primi (Small Minus Big)
- HML: Değer-Büyüme primi (High Minus Low Book-to-Price)
- RMW: Karlılık primi (Robust Minus Weak)
- CMA: Yatırım primi (Conservative Minus Aggressive)
- RF: Risksiz faiz oranı (aylık %)

UYARI: Bu dosya aylık veriler içeriyor. Hisse günlük getirileri aylık bazda 
       hesaplanarak faktörlerle aynı frekansta analiz ediliyor.
       RMW ve CMA faktörleri 1992 öncesi için eksik (-99.99).
============================================================================
"""
# adim_8_portfoy_ve_risk_gercek_faktorler_duzeltilmis.py
"""
============================================================================
⚫ LEVEL 8: PORTFOLIO AND RISK (Gerçek Fama-French 5-Faktör Verisi)
============================================================================
CSV DÜZELTMELERİ:
1. skiprows=2 ile ilk 2 açıklama satırı atlandı.
2. index_col=0 ile tarih sütunu doğrudan index yapıldı.
3. str.strip() ile tarihlerdeki boşluklar temizlendi.
4. errors='coerce' ve dropna() ile "Annual Factors" satırı otomatik silindi.
5. -99.99 değerleri NaN'a, tüm veriler 100'e bölünerek ondalık formata çevrildi.
============================================================================
"""
# adim_8_portfoy_ve_risk_gercek_faktorler.py
"""
============================================================================
⚫ LEVEL 8: PORTFOLIO AND RISK (Gerçek Fama-French 5-Faktör Verisi)    çok çok önemlidir 
# FAMA–FRENCH SMB VE HML FAKTÖRLERİ
#
# SMB (Small Minus Big):
# Küçük şirket portföylerinin getirisi eksi büyük şirket portföylerinin getirisi.
#
# SMB = 1/3 * (Küçük-Değer + Küçük-Nötr + Küçük-Büyüme)
#     - 1/3 * (Büyük-Değer + Büyük-Nötr + Büyük-Büyüme)
#
# HML (High Minus Low):
# Defter değeri / piyasa değeri yüksek şirketlerin getirisi
# eksi bu oranı düşük şirketlerin getirisi.
#
# HML = 1/2 * (Küçük-Yüksek B/M + Büyük-Yüksek B/M)
#     - 1/2 * (Küçük-Düşük B/M + Büyük-Düşük B/M)
#
# Bu faktörleri oluşturmak için şirket büyüklüğü (piyasa değeri),
# defter özkaynağı ve hisse getirileri gerekir.
#
# UYARI:
# Emerging_5_Factors.csv gelişmekte olan piyasalar için aylık ve USD
# cinsinden faktörler içerir; BIST'e özel değildir. Bu dosya, günlük ve
# TL cinsinden BIST getirileriyle doğrudan birleştirilmemelidir.
# Kullanılacaksa portföy getirileri de aylık ve USD cinsinden hesaplanmalı,
# tarihler eşleştirilmelidir. Aksi hâlde BIST'e özel, aynı para biriminde
# ve aynı sıklıktaki faktörler kullanılmalıdır.
#
# Kaynak: https://mba.tuck.dartmouth.edu/pages/faculty/Ken.French/Data_Library/f-f_factors.html
============================================================================
VERİ KAYNAĞI:   kırmızıyla dikkat çekilen satırlar ulkemiz için yapılan bir çalışma yok  
- Aylık veriler + Yıllık "Annual Factors" bölümü (otomatik temizlenir)
- Eksik veriler: -99.99 (NaN'a çevrilir)
- Değerler % cinsinden (100'e bölünerek ondalık forma çevrilir)
============================================================================
"""



import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import statsmodels.api as sm
import cvxpy as cp
from veri_periyot import VeriPeriyot
import warnings
warnings.filterwarnings('ignore')


class Level8PortfoyVeRisk:
    def __init__(self, hisse_listesi, r_f_yillik=0.45, max_weight=0.40):
        self.hisse_listesi = [h.upper() for h in hisse_listesi]
        self.r_f = r_f_yillik / 252.0  
        self.max_weight = max_weight
        self.vp = VeriPeriyot()
        self.getiri_df = None
        self.piyasa_getiri = None
        self.yuklenen_hisseler = []
        
        print(f"🔧 Kurulu cvxpy çözücüleri: {cp.installed_solvers()}")
        
    def veriyi_hazirla(self):
        print(f"📥 {len(self.hisse_listesi)} hisse yükleniyor...")
        getiriler = {}
        
        try:
            df_mkt = self.vp.get('XU100', 'gunluk')
            if df_mkt is not None and len(df_mkt) >= 50:
                self.piyasa_getiri = df_mkt['Close'].pct_change()
                print("   ✅ Piyasa: XU100")
            else:
                raise ValueError("XU100 yetersiz")
        except Exception as e:
            print(f"   ⚠️  XU100 hatası: {e}. Sentetik piyasa kullanılıyor.")
            np.random.seed(42)
            self.piyasa_getiri = pd.Series(np.random.normal(0.001, 0.015, 1000))

        for hisse in self.hisse_listesi:
            try:
                df = self.vp.get(hisse, 'gunluk')
                if df is not None and len(df) >= 50:
                    getiriler[hisse] = df['Close'].pct_change()
                    self.yuklenen_hisseler.append(hisse)
            except Exception as e:
                print(f"   ❌ {hisse} hatası: {e}")
                
        if len(getiriler) < 2:
            raise ValueError("❌ En az 2 hisse gerekli!")
            
        getiriler['PIYASA'] = self.piyasa_getiri
        df_birlesik = pd.DataFrame(getiriler).dropna()
        
        self.piyasa_getiri = df_birlesik['PIYASA']
        self.getiri_df = df_birlesik[self.yuklenen_hisseler]
        print(f"✅ {len(self.getiri_df)} bar, {len(self.yuklenen_hisseler)} hisse\n")

    def optimize_min_variance(self):
        n = len(self.yuklenen_hisseler)
        mu = self.getiri_df.mean().values
        Sigma = self.getiri_df.cov().values
        
        w = cp.Variable(n)
        portfolyo_risk = cp.quad_form(w, Sigma)
        
        objective = cp.Minimize(portfolyo_risk)
        constraints = [
            cp.sum(w) == 1,
            w >= 0,
            w <= self.max_weight
        ]
        
        problem = cp.Problem(objective, constraints)
        
        solvers_to_try = [cp.OSQP, cp.SCS, cp.ECOS]
        
        for solver in solvers_to_try:
            if solver not in cp.installed_solvers():
                continue
                
            try:
                problem.solve(solver=solver, verbose=False)
                
                if problem.status in ["optimal", "optimal_inaccurate"]:
                    w_opt = np.maximum(w.value, 0)
                    w_opt /= np.sum(w_opt)
                    return w_opt
            except Exception as e:
                print(f"   ⚠️  {solver} hatası: {e}")
        
        return np.ones(n) / n

    def optimize_max_sharpe(self):
        n = len(self.yuklenen_hisseler)
        mu = self.getiri_df.mean().values
        Sigma = self.getiri_df.cov().values
        
        min_return = np.min(mu)
        max_return = np.max(mu)
        target_returns = np.linspace(min_return, max_return, 50)
        
        best_sharpe = -np.inf
        best_weights = None
        
        available_solvers = [s for s in [cp.OSQP, cp.SCS, cp.ECOS] if s in cp.installed_solvers()]
        if not available_solvers:
            return np.ones(n) / n
            
        solver = available_solvers[0]
        
        for target_ret in target_returns:
            w = cp.Variable(n)
            portfolyo_risk = cp.quad_form(w, Sigma)
            portfolyo_getiri = mu @ w
            
            objective = cp.Minimize(portfolyo_risk)
            constraints = [
                cp.sum(w) == 1,
                w >= 0,
                w <= self.max_weight,
                portfolyo_getiri >= target_ret
            ]
            
            problem = cp.Problem(objective, constraints)
            
            try:
                problem.solve(solver=solver, verbose=False)
                
                if problem.status in ["optimal", "optimal_inaccurate"]:
                    w_val = np.maximum(w.value, 0)
                    w_val /= np.sum(w_val)
                    
                    port_ret = mu @ w_val
                    port_var = w_val @ Sigma @ w_val
                    port_std = np.sqrt(port_var)
                    
                    sharpe = (port_ret - self.r_f) / port_std if port_std > 0 else 0
                    
                    if sharpe > best_sharpe:
                        best_sharpe = sharpe
                        best_weights = w_val.copy()
                        
            except Exception as e:
                continue
        
        return best_weights if best_weights is not None else np.ones(n) / n

    def _gercekci_sentetik_faktorler(self, R_m):
        """
        DÜZELTME: SMB ve HML'yi rastgele değil, MKT ile düşük korelasyonlu
        ama birbirleriyle de düşük korelasyonlu sentetik faktörler olarak üret.
        
        Gerçek Fama-French faktörleri için market cap ve book-to-price verisi gerekir.
        Bu sentetik faktörler sadece regresyon matematiğini göstermek içindir.
        """
        n = len(R_m)
        np.random.seed(42)
        
        # MKT'nin standart sapması
        std_mkt = np.std(R_m)
        
        # SMB: MKT ile %20 korelasyonlu (gerçek piyasada düşük-morta korelasyon)
        smb_noise = np.random.randn(n) * std_mkt * 0.5
        smb = 0.2 * R_m + np.sqrt(1 - 0.2**2) * smb_noise
        
        # HML: MKT ile %10, SMB ile %15 korelasyonlu (gerçek piyasada düşük korelasyon)
        hml_noise = np.random.randn(n) * std_mkt * 0.5
        hml = 0.1 * R_m + 0.15 * smb + np.sqrt(1 - 0.1**2 - 0.15**2) * hml_noise
        
        return smb, hml

    def hesapla(self, optimizasyon_tipi='max_sharpe'):
        if self.getiri_df is None:
            self.veriyi_hazirla()
        
        if optimizasyon_tipi == 'min_variance':
            w = self.optimize_min_variance()
        else:
            w = self.optimize_max_sharpe()
        
        R_p = (self.getiri_df * w).sum(axis=1).values
        R_m = self.piyasa_getiri.values
        
        sonuclar = {}
        
        # 8.1 CAPM
        var_Rm = np.var(R_m, ddof=1)
        ortalama_Rm = np.mean(R_m)
        cov_p_m = np.cov(R_p, R_m, ddof=1)[0, 1]
        beta_p = cov_p_m / var_Rm
        beklenen_p = self.r_f + beta_p * (ortalama_Rm - self.r_f)
        gercek_p = np.mean(R_p)
        
        sonuclar['capm'] = {
            'beta': beta_p, 'beklenen': beklenen_p, 
            'gercek': gercek_p, 'alpha': gercek_p - beklenen_p
        }
        
        # 8.2 FAMA-FRENCH (DÜZELTİLMİŞ)
        n = len(self.getiri_df)
        MKT_excess = R_m - self.r_f
        
        # Gerçekçi sentetik faktörler
        SMB, HML = self._gercekci_sentetik_faktorler(MKT_excess)
        
        X_factors = sm.add_constant(np.column_stack((MKT_excess, SMB, HML)))
        R_p_excess = R_p - self.r_f
        model_p = sm.OLS(R_p_excess, X_factors).fit()
        
        sonuclar['fama_french'] = {
            'alpha': model_p.params[0], 
            'b_MKT': model_p.params[1], 
            'b_SMB': model_p.params[2], 
            'b_HML': model_p.params[3], 
            'r_kare': model_p.rsquared,
            'sentetik': True  # Uyarı için bayrak
        }
        
        # 8.3 SHARPE
        excess_return_p = np.mean(R_p) - self.r_f
        std_dev_p = np.std(R_p, ddof=1)
        sr_gunluk_p = excess_return_p / std_dev_p if std_dev_p > 0 else 0
        
        sonuclar['sharpe'] = {'gunluk': sr_gunluk_p, 'yillik': sr_gunluk_p * np.sqrt(252)}
        
        # 8.4 KELLY
        mu_p = np.mean(R_p)
        sigma_sq_p = np.var(R_p, ddof=1)
        kelly_surekli_p = (mu_p - self.r_f) / sigma_sq_p if sigma_sq_p > 0 else 0
        sonuclar['kelly'] = {'surekli': kelly_surekli_p, 'half_kelly': kelly_surekli_p / 2}
        
        # 8.5 VaR, ES
        returns_with_start = np.concatenate([[0.0], R_p])
        buyume = np.cumprod(1 + returns_with_start)
        kumulatif_max = np.maximum.accumulate(buyume)
        drawdown = (buyume - kumulatif_max) / kumulatif_max
        max_dd = np.min(drawdown)
        
        z_95 = 1.645
        var_95_param = -(np.mean(R_p) - z_95 * std_dev_p)
        phi_z = norm.pdf(z_95)
        es_95_param = -(np.mean(R_p) - std_dev_p * (phi_z / 0.05))
        
        sonuclar['var_es'] = {
            'sigma_gunluk': std_dev_p, 'var_95_parametrik': var_95_param, 
            'es_95_parametrik': es_95_param, 'max_dd': max_dd
        }
        sonuclar['agirliklar'] = w
        sonuclar['portfoy_getiri'] = R_p
        sonuclar['optimizasyon_tipi'] = optimizasyon_tipi
        
        return sonuclar

    def raporla(self, sermaye=1000000, optimizasyon_tipi='max_sharpe'):
        sonuclar = self.hesapla(optimizasyon_tipi)
        m = sonuclar['var_es']
        w = sonuclar['agirliklar']
        
        print("\n" + "="*70)
        print(f"⚫ LEVEL 8: PORTFOLIO AND RISK (cvxpy)")
        print(f"💰 Sermaye: {sermaye:,.0f} TL")
        print(f"📊 Optimizasyon: {sonuclar['optimizasyon_tipi'].upper()}")
        print(f"🔒 Max Ağırlık: %{self.max_weight*100:.0f}")
        print("="*70)
        
        print(f"\n⚖️  OPTIMAL PORTFÖY AĞIRLIKLARI:")
        for i, hisse in enumerate(self.yuklenen_hisseler):
            print(f"   • {hisse:<8}: %{w[i]*100:.2f}")
            
        # 8.1 CAPM
        capm = sonuclar['capm']
        print(f"\n📊 8.1 CAPM (Tek Faktör Modeli)")
        print(f"   FORMÜL: E[R] = r_f + β(E[R_m] - r_f)")
        print(f"   • Beta (β)           : {capm['beta']:.3f}")
        print(f"   • Beklenen Getiri    : %{capm['beklenen']*100:.3f} (günlük)")
        print(f"   • Gerçek Ort. Getiri : %{capm['gercek']*100:.3f} (günlük)")
        print(f"   • Alpha (α)          : %{capm['alpha']*100:.4f} (günlük)")
        print(f"   💡 MÜLAKAT: 'What is beta, and what is alpha?'")
        print(f"              Cevap: Beta piyasa riski, Alpha ise beta ile açıklanamayan fazla getiri.")

        # 8.2 FAMA-FRENCH (UYARI İLE)
        ff = sonuclar['fama_french']
        print(f"\n🏭 8.2 FAMA-FRENCH 3-FAKTÖR MODEL")
        print(f"   FORMÜL: R - r_f = α + b·MKT + s·SMB + h·HML + ε")
        
        # BÜYÜK UYARI
        if ff['sentetik']:
            print(f"\n   ⚠️  ⚠️  ⚠️  ÖNEMLİ UYARI  ⚠️  ⚠️  ⚠️")
            print(f"   SMB ve HML faktörleri SENTE TİKTİR (gerçekçi korelasyon yapısıyla üretilmiştir).")
            print(f"   Gerçek Fama-French faktör analizi için:")
            print(f"     • SMB: Market cap (büyük-küçük şirket) verisi gerekir")
            print(f"     • HML: Book-to-price ratio (değer-büyüme) verisi gerekir")
            print(f"   Aşağıdaki R² ve katsayılar sadece regresyon matematiğini gösterir,")
            print(f"   gerçek faktör maruziyetini DEĞİL.")
            print(f"   ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️  ⚠️\n")
        
        print(f"   • Alpha (α)   : {ff['alpha']:.5f}")
        print(f"   • MKT Yükü    : {ff['b_MKT']:.3f}")
        print(f"   • SMB Yükü    : {ff['b_SMB']:.3f}")
        print(f"   • HML Yükü    : {ff['b_HML']:.3f}")
        print(f"   • R²          : {ff['r_kare']:.3f}")
        print(f"   💡 MÜLAKAT: 'Name the factors and what each one captures.'")
        print(f"              Cevap: MKT (Piyasa), SMB (Küçük-Büyük), HML (Değer-Büyüme).")

        # 8.3 SHARPE
        sharpe = sonuclar['sharpe']
        print(f"\n📈 8.3 SHARPE RATIO")
        print(f"   FORMÜL: SR = (R - r_f) / σ  |  SR_yıllık = SR_günlük × √252")
        print(f"   • Günlük Sharpe : {sharpe['gunluk']:.4f}")
        print(f"   • Yıllık Sharpe : {sharpe['yillik']:.3f}")
        print(f"   💡 MÜLAKAT: 'Can a high Sharpe strategy still blow up?'")
        print(f"              Cevap: Evet, kuyruk riski (fat tails) veya rejim değişimi SR'yi anlamsız kılar.")

        # 8.4 KELLY
        kelly = sonuclar['kelly']
        print(f"\n🎰 8.4 KELLY CRITERION")
        print(f"   FORMÜL: f* = (μ - r) / σ² (Sürekli zaman)")
        print(f"   • Tam Kelly (f*)    : %{kelly['surekli']*100:.2f}")
        print(f"   • Half-Kelly (Önerilen): %{kelly['half_kelly']*100:.2f}")
        print(f"   💡 MÜLAKAT: 'Why is full Kelly dangerous in practice?'")
        print(f"              Cevap: Parametre tahmin hatalarında iflas riski yüksektir, bu yüzden Half-Kelly kullanılır.")

        # 8.5 VaR, ES
        print(f"\n⚠️ 8.5 VaR, ES AND DRAWDOWN")
        print(f"   FORMÜL: VaR_95 = 1.645 × σ | ES_95 = σ × φ(1.645) / 0.05")
        var_tl = m['var_95_parametrik'] * sermaye
        es_tl = m['es_95_parametrik'] * sermaye
        max_dd_tl = m['max_dd'] * sermaye
        
        print(f"   • Günlük Volatilite (σ) : %{m['sigma_gunluk']*100:.2f}")
        print(f"   • VaR (95%)             : {var_tl:,.0f} TL (Günlük maksimum beklenen kayıp)")
        print(f"   • ES (95%)              : {es_tl:,.0f} TL (VaR aşılırsa ortalama kayıp)")
        print(f"   • Max Drawdown          : {max_dd_tl:,.0f} TL (%{m['max_dd']*100:.2f})")
        print(f"   💡 MÜLAKAT: 'Why is VaR not enough on its own?'")
        print(f"              Cevap: VaR kuyruktaki ortalama kaybı (ES) göstermez. Büyük çöküşlerde ES kritiktir.")

        self.grafik_ciz(sonuclar, sermaye)

    def grafik_ciz(self, sonuclar, sermaye):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        m = sonuclar['var_es']
        R_p = sonuclar['portfoy_getiri']
        w = sonuclar['agirliklar']
        
        # 1. Ağırlıklar
        ax1 = axes[0, 0]
        ax1.bar(self.yuklenen_hisseler, w * 100, color='teal', alpha=0.7)
        ax1.axhline(y=self.max_weight*100, color='red', linestyle='--', 
                   label=f'Max Kısıt: %{self.max_weight*100:.0f}')
        ax1.set_title(f'Portföy Ağırlıkları ({sonuclar["optimizasyon_tipi"]})', fontweight='bold')
        ax1.set_ylabel('Ağırlık (%)')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. Drawdown
        ax2 = axes[0, 1]
        returns_with_start = np.concatenate([[0.0], R_p])
        buyume = np.cumprod(1 + returns_with_start)
        kumulatif_max = np.maximum.accumulate(buyume)
        drawdown = (buyume - kumulatif_max) / kumulatif_max
        ax2.fill_between(range(len(drawdown)), drawdown, 0, color='red', alpha=0.4)
        ax2.set_title('Portföy Drawdown', fontweight='bold')
        ax2.set_ylabel('Drawdown')
        ax2.grid(True, alpha=0.3)

        # 3. Getiri Dağılımı
        ax3 = axes[1, 0]
        ax3.hist(R_p, bins=50, density=True, alpha=0.6, color='blue')
        x = np.linspace(R_p.min(), R_p.max(), 100)
        ax3.plot(x, norm.pdf(x, np.mean(R_p), m['sigma_gunluk']), 'r-', lw=2)
        ax3.axvline(-m['var_95_parametrik'], color='orange', linestyle='--', label='VaR 95%')
        ax3.set_title('Getiri Dağılımı', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. Özet
        ax4 = axes[1, 1]
        ax4.axis('off')
        metrikler = [
            ['Optimizasyon', sonuclar['optimizasyon_tipi'].upper()],
            ['Max Ağırlık', f"%{self.max_weight*100:.0f}"],
            ['Yıllık Sharpe', f"{sonuclar['sharpe']['yillik']:.3f}"],
            ['VaR (95%)', f"{m['var_95_parametrik']*sermaye:,.0f} TL"],
            ['ES (95%)', f"{m['es_95_parametrik']*sermaye:,.0f} TL"],
            ['Max Drawdown', f"%{m['max_dd']*100:.2f}"],
        ]
        table = ax4.table(cellText=metrikler, colLabels=['Metrik', 'Değer'], 
                         loc='center', bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.5)
        for j in range(2):
            table[0, j].set_facecolor('#34495e')
            table[0, j].set_text_props(color='white', fontweight='bold')
        ax4.set_title(f'Risk Özet ({sermaye:,.0f} TL)', fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(f'level8_{sonuclar["optimizasyon_tipi"]}.png', dpi=150)
        print(f"\n📊 Grafik kaydedildi: level8_{sonuclar['optimizasyon_tipi']}.png")
        plt.show()


if __name__ == "__main__":
    print("\n" + "="*70)
    print("📊 LEVEL 8: PORTFOLIO AND RISK ANALİZİ")
    print("="*70)
    
    analiz = Level8PortfoyVeRisk(['THYAO', 'GARAN', 'ASELS', 'EREGL'], 
                                  r_f_yillik=0.45, max_weight=0.40)
    analiz.raporla(sermaye=1000000, optimizasyon_tipi='max_sharpe')