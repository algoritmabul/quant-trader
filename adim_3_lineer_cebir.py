# adim_3_lineer_cebir.py
"""
============================================================================
🔵 LEVEL 3: LINEAR ALGEBRA (Lineer Cebir) - "Her Şey Nasıl Birlikte Hareket Eder"
============================================================================
AMAÇ: Birden fazla varlığın birlikte hareketini anlamak, portföy riskini 
      hesaplamak ve gizli sürücüleri (PCA) ortaya çıkarmak.

KAYNAK: Quant Roadmap - Level 3 (Görseldeki 5 kutuya tam sadık)
-----------------------------------------------------------------------------

3.1 COVARIANCE MATRIX (Kovaryans Matrisi)
    FORMÜL: Σᵢⱼ = Cov(rᵢ, rⱼ)
    ANLAMI: Her çift varlığın birlikte hareket etme tablosu.
    NE İŞE YARAR: Her portföy ve risk modelinin girdisi.

3.2 PORTFOLIO VARIANCE (Portföy Varyansı)
    FORMÜL: σ²ₚ = w'Σw
    ANLAMI: Ağırlıklar ve Σ'den toplam portföy riski.

3.3 EIGENVALUES AND PCA (Özdeğerler ve Temel Bileşen Analizi)
    FORMÜL: Σ = QΛQ', share = λₖ/Σλ
    ANLAMI: Çoklu serinin arkasındaki birkaç gizli sürücüyü bul.

3.4 INVERSE AND CONDITIONING (Ters Matris ve Koşulluluk)
    FORMÜL: κ = λₘₐₓ/λₘᵢₙ
    ANLAMI: Büyük κ, Σ⁻¹'nin küçük hataları büyüttüğü anlamına gelir.

3.5 SHRINKAGE (Büzüşme)
    FORMÜL: Σ' = δF + (1-δ)S
    ANLAMI: Gürültülü örnek matris S'yi basit hedef F ile karıştır (Ledoit-Wolf).

============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.covariance import LedoitWolf  # ✅ Endüstri standardı Ledoit-Wolf
from veri_periyot import VeriPeriyot
import warnings
warnings.filterwarnings('ignore')

class Level3LineerCebir:
    def __init__(self, hisse_listesi, periyot='gunluk'):
        self.hisse_listesi = [h.upper() for h in hisse_listesi]
        self.periyot = periyot.lower()
        self.vp = VeriPeriyot()
        self.getiri_df = None
        self.kovaryans_matrisi = None
        self.korelasyon_matrisi = None
        self.yuklenen_hisseler = []
        
    def veriyi_hazirla(self):
        """Birden fazla hisse verisi çek ve getirileri hesapla"""
        print(f"📥 {len(self.hisse_listesi)} hisse verisi yükleniyor...")
        
        getiriler = {}
        for hisse in self.hisse_listesi:
            try:
                df = self.vp.get(hisse, self.periyot)
                if df is not None and len(df) >= 30:
                    getiriler[hisse] = df['Close'].pct_change()
                    self.yuklenen_hisseler.append(hisse)
                    print(f"   ✅ {hisse}: {len(df)} bar")
                else:
                    print(f"   ⚠️  {hisse}: Yetersiz veri, atlanıyor")
            except Exception as e:
                print(f"   ❌ {hisse}: Hata - {e}")
        
        if len(getiriler) < 2:
            raise ValueError("❌ En az 2 hisse için yeterli veri gerekli!")
        
        self.getiri_df = pd.DataFrame(getiriler).dropna()
        print(f"\n✅ {len(self.getiri_df)} ortak bar, {len(self.yuklenen_hisseler)} hisse hazır.")
        print(f"   📋 Yüklenen hisseler: {self.yuklenen_hisseler}\n")
        
        self.kovaryans_matrisi = self.getiri_df.cov()
        self.korelasyon_matrisi = self.getiri_df.corr()
        
        return self.getiri_df

    def ledoit_wolf_shrinkage(self, returns):
        """
        Ledoit–Wolf kovaryans tahmini (scikit-learn implementasyonu).
        returns: satırlar tarihler, sütunlar hisseler.
        """
        X = np.asarray(returns, dtype=float)

        if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 2:
            raise ValueError("Ledoit-Wolf için en az 2 tarih ve 2 hisse gerekli.")
        if not np.isfinite(X).all():
            raise ValueError("Getiri verisinde NaN veya sonsuz değer var.")

        estimator = LedoitWolf(assume_centered=False).fit(X)

        # scikit-learn'in kullandığı örnek kovaryans ve hedef F = μI
        S = np.cov(X, rowvar=False, ddof=0)
        p = X.shape[1]
        mu = float(np.trace(S) / p)
        F = mu * np.eye(p)

        delta = float(estimator.shrinkage_)
        sigma_shrunk = estimator.covariance_

        return delta, F, sigma_shrunk

    def hesapla(self):
        """Görseldeki Level 3'ün 5 kutusunu hesapla"""
        if self.getiri_df is None:
            self.veriyi_hazirla()
        
        # ✅ DÜZELTME: Hisse sayısını başarıyla yüklenen sütunlardan al
        n_hisse = len(self.yuklenen_hisseler)
        n_bar = len(self.getiri_df)
        
        # ==================== 3.1 COVARIANCE MATRIX ====================
        sigma = self.kovaryans_matrisi.values
        
        # ==================== 3.2 PORTFOLIO VARIANCE ====================
        # ✅ DÜZELTME: Ağırlıkları yüklenen hisse sayısına göre kur
        w = np.ones(n_hisse) / n_hisse
        
        # σ²ₚ = w'Σw
        portfolyo_varyans = w @ sigma @ w
        portfolyo_volatilite = np.sqrt(portfolyo_varyans)
        
        bireysel_voller = np.sqrt(np.diag(sigma))
        ortalama_bireysel_vol = np.mean(bireysel_voller)
        diversifikasyon_indirimi = ortalama_bireysel_vol - portfolyo_volatilite
        
        # ==================== 3.3 EIGENVALUES AND PCA ====================
        eigenvalues, eigenvectors = np.linalg.eigh(sigma)
        
        # Büyükten küçüğe sırala (PCA için)
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        toplam_varyans = np.sum(eigenvalues)
        varyans_oranlari = eigenvalues / toplam_varyans
        kumulatif_varyans = np.cumsum(varyans_oranlari)
        ilk_bilesen = eigenvectors[:, 0]
        
        # ==================== 3.4 INVERSE AND CONDITIONING ====================
        # ✅ DÜZELTME: κ = λₘₐₓ/λₘᵢₙ (en büyük / en küçük)
        lambda_max = eigenvalues[0]  # Artık büyükten küçüğe sıralı
        lambda_min = eigenvalues[-1]
        
        condition_number = lambda_max / lambda_min if lambda_min > 1e-10 else np.inf
        
        try:
            sigma_inv = np.linalg.inv(sigma)
            ters_matris_stabil = True
        except np.linalg.LinAlgError:
            sigma_inv = None
            ters_matris_stabil = False
        
        # ==================== 3.5 SHRINKAGE (Ledoit-Wolf) ====================
        # ✅ YENİ: Scikit-learn Ledoit-Wolf implementasyonu
        returns = self.getiri_df.to_numpy(dtype=float)
        delta, F, sigma_shrunk = self.ledoit_wolf_shrinkage(returns)
        
        # Shrinkage sonrası condition number
        eigenvalues_shrunk = np.linalg.eigvalsh(sigma_shrunk)  # küçükten büyüğe
        lambda_min_shrunk = eigenvalues_shrunk[0]
        lambda_max_shrunk = eigenvalues_shrunk[-1]
        
        condition_number_shrunk = (
            lambda_max_shrunk / lambda_min_shrunk
            if lambda_min_shrunk > 0
            else np.inf
        )
        
        return {
            'n_hisse': n_hisse,
            'n_bar': n_bar,
            'yuklenen_hisseler': self.yuklenen_hisseler,
            'kovaryans_matrisi': sigma,
            'korelasyon_matrisi': self.korelasyon_matrisi.values,
            'portfolyo_varyans': portfolyo_varyans,
            'portfolyo_volatilite': portfolyo_volatilite,
            'ortalama_bireysel_vol': ortalama_bireysel_vol,
            'diversifikasyon_indirimi': diversifikasyon_indirimi,
            'eigenvalues': eigenvalues,
            'eigenvectors': eigenvectors,
            'varyans_oranlari': varyans_oranlari,
            'kumulatif_varyans': kumulatif_varyans,
            'ilk_bilesen': ilk_bilesen,
            'lambda_max': lambda_max,
            'lambda_min': lambda_min,
            'condition_number': condition_number,
            'ters_matris_stabil': ters_matris_stabil,
            'delta': delta,
            'sigma_shrunk': sigma_shrunk,
            'condition_number_shrunk': condition_number_shrunk
        }

    def raporla(self):
        """Görseldeki 5 kutuyu raporla"""
        self.veriyi_hazirla()
        m = self.hesapla()
        
        print("="*70)
        print(f"🔵 LEVEL 3: LINEAR ALGEBRA (Görseldeki 5 Kutu)")
        print(f"📊 {m['n_hisse']} Hisse Portföy Analizi")
        print("="*70)
        
        # 3.1
        print(f"\n📊 3.1 COVARIANCE MATRIX (Kovaryans Matrisi)")
        print(f"   FORMÜL: Σᵢⱼ = Cov(rᵢ, rⱼ)")
        print(f"   • Boyut: {m['n_hisse']} × {m['n_hisse']}")
        print(f"   • Ortalama Kovaryans: {np.mean(m['kovaryans_matrisi']):.6f}")
        print(f"   • Ortalama Korelasyon: {np.mean(m['korelasyon_matrisi'][np.triu_indices(m['n_hisse'], k=1)]):.4f}")
        
        if m['n_hisse'] <= 6:
            print(f"\n   📋 Korelasyon Matrisi:")
            print(self.korelasyon_matrisi.round(3).to_string())

        # 3.2
        print(f"\n⚖️ 3.2 PORTFOLIO VARIANCE (Portföy Varyansı)")
        print(f"   FORMÜL: σ²ₚ = w'Σw")
        print(f"   • Portföy: Eşit Ağırlıklı ({m['n_hisse']} hisse)")
        print(f"   • Portföy Volatilite (Günlük): %{m['portfolyo_volatilite']*100:.2f}")
        print(f"   • Portföy Volatilite (Yıllık): %{m['portfolyo_volatilite']*np.sqrt(252)*100:.2f}")
        print(f"   • Ortalama Bireysel Volatilite: %{m['ortalama_bireysel_vol']*100:.2f}")
        print(f"   • Diversifikasyon İndirimi: %{m['diversifikasyon_indirimi']*100:.2f}")
        
        if m['diversifikasyon_indirimi'] > 0:
            print(f"   ✅ YORUM: Diversifikasyon çalışıyor! Portföy riski bireysel riskten düşük.")

        # 3.3
        print(f"\n🔍 3.3 EIGENVALUES AND PCA (Özdeğerler ve Temel Bileşen Analizi)")
        print(f"   FORMÜL: Σ = QΛQ', share = λₖ/Σλ")
        print(f"   • Toplam Bileşen: {m['n_hisse']}")
        print(f"\n   📊 Bileşen Analizi:")
        print(f"   {'Bileşen':<10} {'Eigenvalue':<12} {'Varyans Oranı':<15} {'Kumulatif':<10}")
        print(f"   {'-'*50}")
        for i in range(min(m['n_hisse'], 5)):
            print(f"   PC{i+1:<9} {m['eigenvalues'][i]:<12.6f} "
                  f"%{m['varyans_oranlari'][i]*100:<14.2f} "
                  f"%{m['kumulatif_varyans'][i]*100:<9.2f}")
        
        print(f"\n   🎯 İlk Bileşen (Piyasa Faktörü) Ağırlıkları:")
        for i, hisse in enumerate(m['yuklenen_hisseler']):
            print(f"      {hisse:<8}: {m['ilk_bilesen'][i]:+.4f}")

        # 3.4
        print(f"\n🔢 3.4 INVERSE AND CONDITIONING (Ters Matris ve Koşulluluk)")
        print(f"   FORMÜL: κ = λₘₐₓ/λₘᵢₙ")
        print(f"   • λₘₐₓ = {m['lambda_max']:.6f}")
        print(f"   • λₘᵢₙ = {m['lambda_min']:.6f}")
        print(f"   • Condition Number (κ) = {m['condition_number']:.2f}")
        print(f"   • Ters Matris Stabil: {'✅ Evet' if m['ters_matris_stabil'] else '❌ Hayır'}")
        
        if m['condition_number'] > 100:
            print(f"   ⚠️  YORUM: Yüksek condition number! Optimizör kararsız olabilir.")
        elif m['condition_number'] > 10:
            print(f"   ⚠️  YORUM: Orta condition number. Dikkatli olunmalı.")
        else:
            print(f"   ✅ YORUM: Düşük condition number. Matris iyi koşullu.")

        # 3.5
        print(f"\n🎯 3.5 SHRINKAGE (Büzüşme - Ledoit-Wolf)")
        print(f"   FORMÜL: Σ' = δF + (1-δ)S")
        print(f"   • Shrinkage Intensity (δ) = {m['delta']:.4f} (Ledoit-Wolf optimal)")
        print(f"   • Hedef Matris (F): μI (köşegende ortalama varyans)")  # ✅ GÜNCELLENDİ
        print(f"   • Örnek Matris (S): Ham kovaryans")
        print(f"\n   📊 Shrinkage Öncesi vs Sonrası:")
        print(f"   • Condition Number (Öncesi): {m['condition_number']:.2f}")
        print(f"   • Condition Number (Sonrası): {m['condition_number_shrunk']:.2f}")
        
        if m['condition_number'] > 0 and m['condition_number'] != np.inf:
            iyilesme = ((m['condition_number'] - m['condition_number_shrunk']) / m['condition_number']) * 100
            print(f"   • İyileşme: %{iyilesme:.1f}")
        
        if m['condition_number_shrunk'] < m['condition_number']:
            print(f"   ✅ YORUM: Shrinkage matrisi iyileştirdi! Daha stabil optimizasyon.")
        else:
            print(f"   ⚠️  YORUM: Shrinkage etkisi minimal veya matris zaten stabil.")

        print("\n" + "="*70)
        print("💡 SONUÇ: Level 3 tamamlandı (Görseldeki 5 kutu).")
        print("Portföy riski, PCA ve matris stabilitesi analiz edildi.")
        print("="*70)


# ========== ÇALIŞTIRMA ==========
if __name__ == "__main__":
    HISSE_LISTESI = ['THYAO', 'GARAN', 'ASELS', 'BIMAS', 'TUPRS', 'EREGL']
    PERIYOT = 'gunluk'
    
    try:
        analiz = Level3LineerCebir(HISSE_LISTESI, PERIYOT)
        analiz.raporla()
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")

# adim_3_lineer_cebir.py (son kısmı)

