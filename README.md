# 📊 Quant Roadmap - Kantitatif Finans Eğitim Platformu

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Status: Active](https://img.shields.io/badge/Status-Active-green.svg)]()

**Sıfırdan ileri seviyeye kantitatif finans yolculuğu — 10 aşamalı, uygulamalı, akademik standartlarda**

---

## 🎯 Proje Hakkında

Bu proje, kantitatif finansın temel kavramlarını **teorik formüller** ve **pratik Python implementasyonları** ile öğretmeyi amaçlayan kapsamlı bir eğitim platformudur. Her seviye, akademik makalelerdeki formülleri gerçek piyasa verileriyle uygulayarak, teorik bilgiyi pratik beceriye dönüştürür.

### ✨ Öne Çıkan Özellikler

- 🎓 **10 Seviyeli Yapı**: Olasılıktan makine öğrenmesine, kademeli ilerleme
- 📈 **Gerçek Veri Entegrasyonu**: MetaStock CSV formatı ile BIST hisseleri
- 🧮 **Akademik Formüller**: Bailey-López de Prado DSR, Corwin-Schultz spread, Avellaneda-Stoikov
- 🖥️ **Modern Dashboard**: Tkinter tabanlı görsel arayüz, dark/light tema
- 📊 **Profesyonel Görselleştirme**: Matplotlib ile interaktif grafikler
- 🇹🇷 **Türkçe Dokümantasyon**: Tüm açıklamalar ve mülakat soruları Türkçe
- 🔬 **Hata Düzeltmeli**: Her seviye akademik review'dan geçirilmiş

---

## 📚 İçerik Haritası

| Seviye | Konu | Ana Kavramlar | Dosya |
|--------|------|---------------|-------|
| **Level 1** | Olasılık | Beklenen Değer, Win Rate, Volatilite, Kelly Kriteri | `adim_1_sadece_olasilik.py` |
| **Level 2** | İstatistik | t-stat, Sharpe, OLS Regresyon, R², Multiple Testing | `adim_2_sadece_istatistik.py` |
| **Level 3** | Lineer Cebir | Kovaryans Matrisi, PCA, Ledoit-Wolf Shrinkage | `adim_3_lineer_cebir.py` |
| **Level 4** | Kalkülüs & Optimizasyon | Taylor, Gradient Descent, Lagrange, Markowitz | `adim_4_kalkulus_optimizasyon.py` |
| **Level 5** | Zaman Serileri | ADF Testi, Eşbütünleşme, GARCH(1,1), Half-Life | `adim_5_zaman_serileri.py` |
| **Level 6** | Stokastik Kalkülüs | Brownian Hareket, Itô Lemması, Monte Carlo | `adim_6_stokastik_kalkulus_duzeltilmis.py` |
| **Level 7** | Türev Araçlar | Black-Scholes, Yunanlar, Örtülü Volatilite | `adim_7_turevler.py` |
| **Level 8** | Portföy & Risk | CAPM, Fama-French 5-Faktör, VaR/ES, cvxpy | `adim_8_portfoy_ve_risk_gercek_faktorler.py` |
| **Level 9** | Mikroyapı | Corwin-Schultz, CLV, VWAP/TWAP, Impact | `adim_9_mikroyapi_ohlcv.py` |
| **Level 10** | Araştırma & ML | Walk-Forward, Deflated Sharpe, HMM Rejimleri | `adim_10_arastirma_ve_ml_dsr_gercek.py` |

## 💻 Kullanım

### Yöntem 1: Dashboard (Önerilen)

Görsel arayüz ile tüm seviyeleri tek ekranda çalıştırın:

```bash
python quant_dashboard.py
```

**Dashboard Özellikleri:**
- 📁 Otomatik CSV dizin taraması
- 🎯 Seviye seçimi (tek tıkla)
- 📊 Gerçek zamanlı çıktı görüntüleme
- 🖼️ Grafik önizleme (PIL tabanlı)
- 🌙/☀️ Dark/Light tema
- 📋 Çoklu hisse seçimi (Ctrl+Click)

### Yöntem 2: Bireysel Seviyeler

Her seviyeyi ayrı çalıştırın:

```bash
# Level 1: Olasılık Analizi
python adim_1_sadece_olasilik.py

# Level 3: Lineer Cebir (çoklu hisse)
python adim_3_lineer_cebir.py

# Level 8: Portföy & Risk (Fama-French ile)
python adim_8_portfoy_ve_risk_gercek_faktorler.py
```
## 📁 Proje Yapısı

```

│
├── 📄 README.md                              # Bu dosya
├── 📄 requirements.txt                       # Python bağımlılıkları
├── 📄                         
│
├── 🖥️ Dashboard
│   └── quant_dashboard.py                   # Ana dashboard
│
├── 🎓 Seviye Dosyaları
│   ├── adim_1_sadece_olasilik.py
│   ├── adim_2_sadece_istatistik.py
│   ├── adim_3_lineer_cebir.py
│   ├── adim_4_kalkulus_optimizasyon.py
│   ├── adim_5_zaman_serileri.py
│   ├── adim_6_stokastik_kalkulus_duzeltilmis.py
│   ├── adim_7_turevler.py
│   ├── adim_8_portfoy_ve_risk_gercek_faktorler.py
│   ├── adim_9_mikroyapi_ohlcv.py
│   └── adim_10_arastirma_ve_ml_dsr_gercek.py
│
├── 🛠️ Yardımcı Modüller
│   ├── veri_periyot.py                      # CSV veri yükleme
│   ├──                          
│   └──                     
│
└── 📊 Veri Dosyaları
    └──             
```

---

## 📖 Seviye Detayları

### Level 1: Olasılık ve Beklenen Değer

**Kavramlar:**
- Beklenen Değer (EV) = Σ(P_i × X_i)
- Win Rate = Kazanan İşlemler / Toplam İşlemler
- Volatilite (σ) = √Var(R)
- Kelly Kriteri: f* = (bp - q) / b

### Level 2: İstatistik

**Kavramlar:**
- Standart Hata: SE = σ / √n
- t-İstatistiği: t ≈ SR × √T
- OLS Regresyon: β = (X'X)^{-1} X'y
- Korelasyon ve R²
- Multiple Testing Bias

### Level 3: Lineer Cebir

**Kavramlar:**
- Kovaryans Matrisi: Σᵢⱼ = Cov(rᵢ, rⱼ)
- Portföy Varyansı: σ²ₚ = w'Σw
- Eigenvalues ve PCA
- Condition Number ve Shrinkage (Ledoit-Wolf)

### Level 4: Kalkülüs & Optimizasyon

**Kavramlar:**
- Taylor Açılımı: f(x+h) ≈ f(x) + f'(x)h + ½f''(x)h²
- Gradient Descent: θ ← θ - η∇L(θ)
- Lagrange Çarpanları: ∇f = λ∇g
- Mean-Variance Weights: w* = (1/γ)Σ⁻¹μ
- Jensen Eşitsizliği

### Level 5: Zaman Serileri

**Kavramlar:**
- Log Returns: r_t = ln(P_t / P_{t-1})
- Autocorrelation: ρ_k = Corr(r_t, r_{t-k})
- ADF Testi (Durağanlık)
- Eşbütünleşme ve Half-Life: half-life = ln(0.5) / ln(1 + λ)
- GARCH(1,1): σ²_t = ω + α·ε²_{t-1} + β·σ²_{t-1}

### Level 6: Stokastik Kalkülüs

**Kavramlar:**
- Random Walk: S_t = S_{t-1} + X_t
- Brownian Motion: W_t ~ N(0,t)
- Geometric Brownian Motion: dS = μS dt + σS dW
- Itô's Lemma: df = (f_t + μf_s + ½σ²f_ss)dt + σf_s dW
- Martingales ve Gambler's Ruin

**Özel:** Gerçek hisse verisinden μ ve σ hesaplanarak Monte Carlo simülasyonu yapılır.

### Level 7: Türev Araçlar

**Kavramlar:**
- Black-Scholes Modeli (temettü verimi dahil)
- Put-Call Parity: C - P = S·e^(-qT) - K·e^(-rT)
- Yunanlar: Delta, Gamma, Vega, Theta, Rho
- Implied Volatility (Brentq ile kök bulma)
- Gamma P&L: P&L ≈ ½·Γ·S²·(σ²_real - σ²_imp)·dt

### Level 8: Portföy & Risk

**Kavramlar:**
- CAPM: E[R] = r_f + β(E[R_m] - r_f)
- Fama-French 5-Faktör Modeli (gerçek veri)
- Sharpe Ratio: SR = (R - r_f) / σ
- Kelly Criterion: f* = (μ - r) / σ²
- VaR, ES ve Max Drawdown
- cvxpy ile portföy optimizasyonu (Minimum Varyans & Maximum Sharpe)

**Gerçek Faktör Verisi:**
`Emerging_5_Factors.csv` dosyasından 1989-2026 aylık faktör verileri otomatik yüklenir.

### Level 9: Mikroyapı

**Kavramlar:**
- Corwin-Schultz Spread Tahmini (High/Low tabanlı)
- Close Location Value (CLV): CLV = (2C - H - L) / (H - L)
- Square-Root Impact: ΔP/P ≈ Y · σ · √(Q / V)
- VWAP ve TWAP (Typical Price yaklaşımı)
- Avellaneda-Stoikov (kavramsal gösterim)

**Not:** OHLCV verisinden türetilen proxy'ler kullanılır. Gerçek L2 emir defteri verisi gerektirmez.

### Level 10: Araştırma & Makine Öğrenmesi

**Kavramlar:**
- Backtest Biases (Look-ahead, Survivorship, Data Snooping)
- Walk-Forward Validation
- **Deflated Sharpe Ratio** (Bailey & López de Prado, 2014)
  - Çarpıklık ve basıklık düzeltmeli
  - Çoklu strateji denemesi düzeltmesi
- Precision, Recall, F1 Score
- HMM Rejim Modelleri (GaussianHMM)

---

## 📦 requirements.txt

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
scikit-learn>=1.0.0
scipy>=1.7.0
statsmodels>=0.13.0
arch>=5.0.0
cvxpy>=1.1.0
hmmlearn>=0.2.6
Pillow>=8.0.0
```

## 📄 LICENSE (MIT)

```
MIT License

Copyright (c) 2026 [Adınız]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 📚 Akademik Kaynaklar

### Level Bazlı Kaynakça

| Seviye | Kaynak |
|--------|--------|
| **Level 2** | Student (1908) - t-dağılımı |
| **Level 3** | Ledoit & Wolf (2004) - Shrinkage estimation |
| **Level 4** | Markowitz (1952) - Portfolio Selection |
| **Level 5** | Engle (1982) - ARCH; Bollerslev (1986) - GARCH |
| **Level 6** | Hull (2018) - Options, Futures, and Other Derivatives |
| **Level 7** | Black & Scholes (1973); Merton (1973) |
| **Level 8** | Fama & French (1993, 2015) - 3 & 5 Factor Models |
| **Level 9** | Corwin & Schultz (2012) - Spread estimation |
| **Level 10** | Bailey & López de Prado (2014) - Deflated Sharpe Ratio |

### Temel Kitaplar

- 📘 **Hull, J.C.** (2018). *Options, Futures, and Other Derivatives* (10th ed.)
- 📗 **López de Prado, M.** (2018). *Advances in Financial Machine Learning*
- 📙 **Tsay, R.S.** (2010). *Analysis of Financial Time Series* (3rd ed.)
- 📕 **Chan, E.P.** (2013). *Algorithmic Trading*

---

## 🤝 Katkıda Bulunma

Katkılarınız memnuniyetle karşılanır! Lütfen:

1. Bu depoyu fork'layın
2. Feature branch oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -m 'Yeni özellik eklendi'`)
4. Branch'inizi push edin (`git push origin feature/yeni-ozellik`)
5. Pull Request açın

### Geliştirme Kuralları

- ✅ Her seviye görseldeki 5 kutuya tam sadık kalmalı
- ✅ Matematiksel formüller akademik kaynaklarla doğrulanmalı
- ✅ Kodda sınırlamalar açıkça belirtilmeli (özellikle sentetik veri kullanımlarında)
- ✅ Türkçe açıklamalar ve mülakat soruları eklenmeli
- ✅ Hata yönetimi ve edge case'ler dikkate alınmalı

---

## 🐛 Bilinen Sınırlamalar

| Seviye | Sınırlama | Açıklama |
|--------|-----------|----------|
| **Level 8** | SMB/HML faktörleri | `Emerging_5_Factors.csv` kullanılmalıdır |
| **Level 9** | Gerçek emir defteri | OHLCV proxy'leri kullanılır |
| **Level 9** | Avellaneda-Stoikov | Kavramsal gösterim, birim uyumsuzluğu var |
| **Level 10** | HMM | Geriye dönük etiketleme, canlı sinyal değil |

---


## 🙏 Teşekkürler

- **Quant Roadmap** - Eğitim yapısı için
- **Python topluluğu** - Harika kütüphaneler için

---

## 📈 Yol Haritası

- [x] Level 1-10 temel implementasyonlar
- [x] Dashboard arayüzü
- [x] Gerçek veri entegrasyonu
- [x] Akademik kaynaklar

---

## ⭐ Bu Projeyi Beğendiyseniz

Eğer bu proje işinize yaradıysa, lütfen bir yıldız vermeyi unutmayın! ⭐

---

<div align="center">

**🎓 Kantitatif finans öğrenmek hiç bu kadar kolay olmamıştı!**

</div>
