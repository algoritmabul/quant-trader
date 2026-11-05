# adim_4_kalkulus_optimizasyon.py
"""
============================================================================
🟣 LEVEL 4: CALCULUS & OPTIMIZATION (Kalkülüs ve Optimizasyon) - "En İyi Kararı Bul"
============================================================================
AMAÇ: Matematiksel optimizasyon ile en iyi portföy ağırlıklarını bulmak, 
      risk-getiri dengesini optimize etmek ve karar verme süreçlerini modellemek.

KAYNAK: Quant Roadmap - Level 4 (Görseldeki 5 kutuya tam sadık)
-----------------------------------------------------------------------------

4.1 TAYLOR EXPANSION (Taylor Açılımı)
    FORMÜL: f(x+h) ≈ f(x) + f'(x)h + ½f''(x)h²
    ANLAMI: Karmaşık fonksiyonları basit polinomlarla yaklaşım.
    NE İŞE YARAR: Opsiyon P&L ≈ Δ·dS + ½Γ·dS² + Θ·dt

4.2 GRADIENT DESCENT (Gradyan İnişi)
    FORMÜL: θ ← θ - η∇L(θ)
    ANLAMI: Hata fonksiyonunu minimize etmek için tekrarlı optimizasyon.
    NE İŞE YARAR: Tüm makine öğrenmesi modellerinin eğitim algoritması.

4.3 LAGRANGE MULTIPLIERS (Lagrange Çarpanları)
    FORMÜL: ∇f = λ∇g
    ANLAMI: Kısıtlar altında optimizasyon.
    PROBLEM: min (1/2)w'Σw - w'μ  s.t.  1'w = 1
    LAGRANGIAN: L(w,λ) = (1/2)w'Σw - w'μ + λ(1'w - 1)
    ÇÖZÜM: ∇_w L = Σw - μ + λ1 = 0  =>  w = Σ⁻¹(μ - λ1)

4.4 MEAN-VARIANCE WEIGHTS (Ortalama-Varyans Ağırlıkları)
    FORMÜL: max w'μ - (γ/2)w'Σw  <=>  min (γ/2)w'Σw - w'μ
    ANLAMI: Markowitz portföy optimizasyonu (γ: riskten kaçınma katsayısı).
    NE İŞE YARAR: γ büyüdükçe varyans cezası artar, daha konservatif portföy oluşur.

4.5 CONVEXITY AND JENSEN (Konvekslik ve Jensen Eşitsizliği)
    FORMÜL: E[f(X)] ≤ f(E[X]) (içbükey f için, örn: log)
    ANLAMI: Volatilite drag: geometrik getiri ≤ aritmetik getiri - σ²/2.

============================================================================
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
from veri_periyot import VeriPeriyot
import warnings
warnings.filterwarnings('ignore')

def project_simplex(v):
    """
    Bir vektörü olasılık simpleksine projekte eder.
    Kısıtlar: sum(w) = 1 ve w >= 0 (Long-only garantisi)
    (Duchi et al., 2008 algoritması)
    """
    n = len(v)
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    return np.maximum(v - theta, 0)


class Level4KalkulusOptimizasyon:
    def __init__(self, hisse_listesi, periyot='gunluk', risk_aversion=1.0):
        if risk_aversion <= 0:
            raise ValueError("Risk aversion (γ) pozitif olmalı!")
        
        self.hisse_listesi = [h.upper() for h in hisse_listesi]
        self.periyot = periyot.lower()
        self.vp = VeriPeriyot()
        self.getiri_df = None
        self.yuklenen_hisseler = []
        self.risk_aversion = risk_aversion  # γ (riskten kaçınma)
        
    def veriyi_hazirla(self):
        print(f"📥 {len(self.hisse_listesi)} hisse verisi yükleniyor...")
        getiriler = {}
        for hisse in self.hisse_listesi:
            try:
                df = self.vp.get(hisse, self.periyot)
                if df is not None and len(df) >= 30:
                    getiriler[hisse] = df['Close'].pct_change()
                    self.yuklenen_hisseler.append(hisse)
                    print(f"   ✅ {hisse}: {len(df)} bar")
            except Exception as e:
                print(f"   ❌ {hisse}: Hata - {e}")
        
        if len(getiriler) < 2:
            raise ValueError("❌ En az 2 hisse için yeterli veri gerekli!")
        
        self.getiri_df = pd.DataFrame(getiriler).dropna()
        print(f"\n✅ {len(self.getiri_df)} ortak bar, {len(self.yuklenen_hisseler)} hisse hazır.\n")
        return self.getiri_df

    def hesapla(self):
        if self.getiri_df is None:
            self.veriyi_hazirla()
        
        n_hisse = len(self.yuklenen_hisseler)
        getiriler = self.getiri_df.values
        mu = np.mean(getiriler, axis=0)
        sigma = np.cov(getiriler, rowvar=False)
        
        # ==================== 4.1 TAYLOR EXPANSION ====================
        # √101 yaklaşımı
        x, h = 100.0, 1.0
        f_x = np.sqrt(x)
        f_prime = 1 / (2 * np.sqrt(x))
        f_double_prime = -1 / (4 * x**(3/2))
        taylor_approx = f_x + f_prime * h + 0.5 * f_double_prime * h**2
        
        # Opsiyon P&L: Δ·dS + ½Γ·dS² + Θ·dt
        S, dS = 100, 2
        delta, gamma, theta = 0.6, 0.02, -0.05  # theta günlük zaman erimesi
        dt = 1  # 1 gün
        option_pnl_taylor = delta * dS + 0.5 * gamma * dS**2 + theta * dt
        
        # ==================== 4.2 GRADIENT DESCENT ====================
        w_gd = np.ones(n_hisse) / n_hisse
        learning_rate = 0.01
        tolerance = 1e-8
        prev_loss = float('inf')
        
        for _ in range(1000):
            gradient = 2 * sigma @ w_gd  # ∇(w'Σw) = 2Σw
            w_gd = w_gd - learning_rate * gradient
            
            # Long-only ve sum=1 projeksiyonu (Simplex)
            w_gd = project_simplex(w_gd)
            
            loss = w_gd @ sigma @ w_gd
            if abs(prev_loss - loss) < tolerance:
                break
            prev_loss = loss
            
        portfolyo_var_gd = w_gd @ sigma @ w_gd
        portfolyo_getiri_gd = w_gd @ mu
        
        # ==================== 4.3 LAGRANGE MULTIPLIERS ====================
        # Problem: min (1/2)w'Σw - w'μ  s.t.  1'w = 1
        try:
            sigma_inv = np.linalg.inv(sigma)
            ones = np.ones(n_hisse)
            
            # ∇_w L = Σw - μ + λ1 = 0  =>  w = Σ⁻¹(μ - λ1)
            # 1'w = 1 kısıtından: λ = (1'Σ⁻¹μ - 1) / (1'Σ⁻¹1)
            lambda_lagrange = (ones @ sigma_inv @ mu - 1) / (ones @ sigma_inv @ ones)
            w_lagrange = sigma_inv @ (mu - lambda_lagrange * ones)
            
            # Not: Bu çözüm long-only garanti etmez, sadece sum=1 garanti eder.
            w_lagrange = w_lagrange / np.sum(w_lagrange) # Normalize
            lagrange_stabil = True
        except np.linalg.LinAlgError:
            w_lagrange = np.ones(n_hisse) / n_hisse
            lambda_lagrange = 0
            lagrange_stabil = False
            
        portfolyo_var_lagrange = w_lagrange @ sigma @ w_lagrange
        portfolyo_getiri_lagrange = w_lagrange @ mu
        
        # ==================== 4.4 MEAN-VARIANCE WEIGHTS ====================
        # Problem: min (γ/2)w'Σw - w'μ  s.t. sum(w)=1, w>=0
        def objective(w):
            return 0.5 * self.risk_aversion * (w @ sigma @ w) - w @ mu
        
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(n_hisse)]
        w0 = np.ones(n_hisse) / n_hisse
        
        result = minimize(objective, w0, method='SLSQP', 
                         bounds=bounds, constraints=constraints,
                         options={'ftol': 1e-9, 'maxiter': 1000})
        
        optimization_success = result.success
        w_mv = result.x if optimization_success else np.ones(n_hisse) / n_hisse
        
        # Matris koşulluluğunu ayrı değerlendir
        try:
            eigenvalues = np.linalg.eigvalsh(sigma)
            condition_number = eigenvalues[-1] / eigenvalues[0] if eigenvalues[0] > 1e-10 else np.inf
            matrix_well_conditioned = condition_number < 100
        except:
            condition_number = np.inf
            matrix_well_conditioned = False
            
        portfolyo_var_mv = w_mv @ sigma @ w_mv
        portfolyo_getiri_mv = w_mv @ mu
        sharpe_mv = portfolyo_getiri_mv / np.sqrt(portfolyo_var_mv) if portfolyo_var_mv > 0 else 0
        
        # ==================== 4.5 CONVEXITY AND JENSEN ====================
        volatilite_drag = 0.5 * portfolyo_var_mv
        geometrik_getiri_approx = portfolyo_getiri_mv - volatilite_drag
        
        portfolyo_degerler = 1 + self.getiri_df @ w_mv
        if np.all(portfolyo_degerler > 0):
            geometrik_getiri_gercek = np.exp(np.mean(np.log(portfolyo_degerler))) - 1
            jensen_sol = np.mean(np.log(portfolyo_degerler))
            jensen_sag = np.log(np.mean(portfolyo_degerler))
            jensen_dogru = jensen_sol <= jensen_sag
        else:
            geometrik_getiri_gercek = np.nan
            jensen_dogru = False
        
        return {
            'n_hisse': n_hisse, 'yuklenen_hisseler': self.yuklenen_hisseler,
            'taylor_approx': taylor_approx, 'option_pnl_taylor': option_pnl_taylor,
            'w_gd': w_gd, 'portfolyo_var_gd': portfolyo_var_gd, 'portfolyo_getiri_gd': portfolyo_getiri_gd,
            'w_lagrange': w_lagrange, 'lambda_lagrange': lambda_lagrange, 
            'portfolyo_var_lagrange': portfolyo_var_lagrange, 'portfolyo_getiri_lagrange': portfolyo_getiri_lagrange,
            'lagrange_stabil': lagrange_stabil,
            'w_mv': w_mv, 'portfolyo_var_mv': portfolyo_var_mv, 'portfolyo_getiri_mv': portfolyo_getiri_mv,
            'sharpe_mv': sharpe_mv, 'optimization_success': optimization_success,
            'matrix_well_conditioned': matrix_well_conditioned, 'condition_number': condition_number,
            'portfolyo_ortalama_getiri': portfolyo_getiri_mv, 'portfolyo_volatilite': np.sqrt(portfolyo_var_mv),
            'volatilite_drag': volatilite_drag, 'geometrik_getiri_approx': geometrik_getiri_approx,
            'geometrik_getiri_gercek': geometrik_getiri_gercek, 'jensen_dogru': jensen_dogru
        }

    def raporla(self):
        self.veriyi_hazirla()
        m = self.hesapla()
        
        print("="*70)
        print(f"🟣 LEVEL 4: CALCULUS & OPTIMIZATION (Görseldeki 5 Kutu)")
        print(f"📊 {m['n_hisse']} Hisse Portföy Optimizasyonu (γ={self.risk_aversion})")
        print("="*70)
        
        print(f"\n📐 4.1 TAYLOR EXPANSION")
        print(f"   FORMÜL: f(x+h) ≈ f(x) + f'(x)h + ½f''(x)h²")
        print(f"   • √101 Yaklaşımı: {m['taylor_approx']:.6f} (Gerçek: {np.sqrt(101):.6f})")
        print(f"   • Opsiyon P&L (Δ·dS + ½Γ·dS² + Θ·dt): {m['option_pnl_taylor']:.4f}")
        
        print(f"\n⬇️ 4.2 GRADIENT DESCENT")
        print(f"   FORMÜL: θ ← θ - η∇L(θ)")
        print(f"   • Yöntem: Simpleks projeksiyonu ile Long-Only + Sum=1 garantisi")
        print(f"   • Portföy Volatilite: %{np.sqrt(m['portfolyo_var_gd'])*100:.2f}")
        
        print(f"\n⚖️ 4.3 LAGRANGE MULTIPLIERS")
        print(f"   FORMÜL: ∇f = λ∇g")
        print(f"   • Problem: min (1/2)w'Σw - w'μ  s.t.  1'w = 1")
        print(f"   • Lagrange Çarpanı (λ): {m['lambda_lagrange']:.6f}")
        print(f"   • Matris Stabil: {'✅ Evet' if m['lagrange_stabil'] else '❌ Hayır'}")
        
        print(f"\n🎯 4.4 MEAN-VARIANCE WEIGHTS")
        print(f"   FORMÜL: min (γ/2)w'Σw - w'μ  s.t.  1'w=1, w≥0")
        print(f"   • Optimizasyon Başarılı: {'✅ Evet' if m['optimization_success'] else '❌ Hayır'}")
        print(f"   • Matris İyi Koşullu (κ<100): {'✅ Evet' if m['matrix_well_conditioned'] else '❌ Hayır'} (κ={m['condition_number']:.2f})")
        print(f"   • Sharpe Oranı: {m['sharpe_mv']:.4f}")
        print(f"   • Optimal Ağırlıklar (SLSQP):")
        for i, hisse in enumerate(m['yuklenen_hisseler']):
            print(f"      {hisse:<8}: %{m['w_mv'][i]*100:.1f}")
            
        print(f"\n📊 4.5 CONVEXITY AND JENSEN")
        print(f"   FORMÜL: E[log(X)] ≤ log(E[X])")
        print(f"   • Aritmetik Getiri: %{m['portfolyo_ortalama_getiri']*100:.2f}")
        print(f"   • Volatilite Drag (σ²/2): %{m['volatilite_drag']*100:.2f}")
        print(f"   • Geometrik Getiri (Gerçek): %{m['geometrik_getiri_gercek']*100:.2f}")
        print(f"   • Jensen Eşitsizliği Sağlanıyor: {'✅ Evet' if m['jensen_dogru'] else '❌ Hayır'}")

        print("\n" + "="*70)


if __name__ == "__main__":
    HISSE_LISTESI = ['THYAO', 'GARAN', 'ASELS', 'BIMAS', 'TUPRS', 'EREGL']
    analiz = Level4KalkulusOptimizasyon(HISSE_LISTESI, risk_aversion=2.0) # γ=2: Daha konservatif
    analiz.raporla()