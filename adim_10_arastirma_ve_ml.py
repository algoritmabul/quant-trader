# adim_10_arastirma_ve_ml_dsr_gercek.py
"""
============================================================================
🟩 LEVEL 10: RESEARCH & MACHINE LEARNING (Kendini Kandırma Sanatı)
============================================================================

10.3 DEFLATED SHARPE RATIO (DSR) - GERÇEK FORMÜL
-----------------------------------------------------------------------------
KAYNAK: Bailey, D.H. & López de Prado, M. (2014). "The Deflated Sharpe 
        Ratio: Correcting for Selection Bias, Backtest Overfitting, 
        and Non-Normality." Journal of Portfolio Management, 40(5), 94-107.

FORMÜLLER:
1. Standart Hata (Formül 3):
   σ_SR = √[(1 - κ·SR + (χ-3)/4 · SR²) / T]
   
   Burada:
   - κ = çarpıklık (skewness)
   - χ = basıklık (kurtosis, Fisher değil; normal dağılım için χ=3)
   - T = gözlem sayısı
   - SR = gözlenen Sharpe oranı

2. Eşik Değer SR₀ (Formül 5):
   SR₀ = μ_SR + σ_SR_dist · E[max(Z)]
   
   Burada:
   - μ_SR = denenmiş stratejilerin Sharpe ortalaması
   - σ_SR_dist = denenmiş stratejilerin Sharpe standart sapması
   - E[max(Z)] = (1-γ)·Φ⁻¹(1-1/N) + γ·Φ⁻¹(1-1/(N·e))
   - γ = 0.5772156649 (Euler-Mascheroni sabiti)
   - N = gerçekten denenen farklı strateji sayısı

3. Deflated Sharpe Ratio (Formül 6):
   DSR = Φ[(SR_obs - SR₀) / σ_SR]
   
   Burada:
   - Φ = standart normal CDF
   - SR_obs = gözlenen Sharpe (en iyi stratejinin Sharpe'ı)
   - SR₀ = eşik değer (şansla elde edilebilecek maksimum Sharpe)
   - σ_SR = Sharpe standart hatası (çarpıklık/basıklık düzeltmeli)

YORUM:
- DSR > 0.95: Alpha istatistiksel olarak anlamlı (%95 güven)
- DSR > 0.90: Sınırda anlamlı
- DSR < 0.90: Muhtemelen seçim yanlılığı (selection bias) ürünü

NOT: N, walk-forward splits sayısı DEĞİLDİR. N, gerçekten denenmiş farklı 
     strateji/parametre kombinasyonlarının sayısıdır.
============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm, skew, kurtosis
from veri_periyot import VeriPeriyot
import warnings
warnings.filterwarnings('ignore')

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_KURULU = True
except ImportError:
    HMM_KURULU = False
    print("⚠️  Uyarı: 'hmmlearn' kurulu değil. HMM analizi atlanacak.")


class Level10ResearchML:
    def __init__(self, hisse='THYAO', periyot='gunluk'):
        self.hisse = hisse.upper()
        self.periyot = periyot
        self.vp = VeriPeriyot()
        self.df = None
        
    def veriyi_hazirla(self):
        print(f"📥 {self.hisse} verisi yükleniyor...")
        self.df = self.vp.get(self.hisse, self.periyot)
        
        if self.df is None or len(self.df) < 560:
            raise ValueError(f"❌ Yeterli veri yok (Min 560 bar)!")
            
        self.df['Ret'] = self.df['Close'].pct_change()
        self.df['NextRet'] = self.df['Ret'].shift(-1)
        self.df['Vol_20'] = self.df['Ret'].rolling(20).std()
        self.df['Lag1'] = self.df['Ret'].shift(1)
        self.df['Lag5'] = self.df['Ret'].shift(5)
        self.df['RSI_14'] = self._hesapla_rsi(self.df['Close'], 14)
        self.df['Target'] = (self.df['NextRet'] > 0).astype(int)
        
        self.df = self.df.dropna()
        
        self.features = ['Lag1', 'Lag5', 'Vol_20', 'RSI_14']
        self.X = self.df[self.features].values
        self.y = self.df['Target'].values
        self.NextRet = self.df['NextRet'].values
        
        print(f"✅ {len(self.df)} bar hazır.\n")

    def _hesapla_rsi(self, fiyatlar, periyot=14):
        delta = fiyatlar.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=periyot).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=periyot).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _egit_ve_tahmin_et(self, X_train, X_test, y_train, model_tipi='logistic', **kwargs):
        """Model eğit ve tahmin yap"""
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        if model_tipi == 'logistic':
            C = kwargs.get('C', 1.0)
            model = LogisticRegression(C=C, random_state=42, max_iter=1000)
        elif model_tipi == 'randomforest':
            n_estimators = kwargs.get('n_estimators', 100)
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
        
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        return y_pred

    def walk_forward_coklu_strateji(self):
        """
        10.1 & 10.2: Walk-Forward + 10.3: Çoklu Strateji DSR
        
        Farklı model tipleri ve parametrelerle birden fazla strateji dener.
        Her stratejinin OOS getirilerini ayrı ayrı toplar.
        """
        print("🔄 10.1 & 10.2: Walk-Forward Validasyon Başlatılıyor...")
        print("🎲 10.3: Çoklu Strateji DSR için farklı modeller deneniyor...\n")
        
        train_size = 500
        test_size = 60
        step = 60
        
        n_splits = (len(self.X) - train_size) // step
        
        if n_splits == 0:
            return None, None, None
        
        # ✅ DÜZELTME: Farklı stratejiler tanımla
        # Her strateji farklı model tipi veya parametre kombinasyonu
        stratejiler = [
            {'ad': 'Logistic C=0.1', 'tip': 'logistic', 'C': 0.1},
            {'ad': 'Logistic C=1.0', 'tip': 'logistic', 'C': 1.0},
            {'ad': 'Logistic C=10.0', 'tip': 'logistic', 'C': 10.0},
            {'ad': 'RandomForest n=50', 'tip': 'randomforest', 'n_estimators': 50},
            {'ad': 'RandomForest n=200', 'tip': 'randomforest', 'n_estimators': 200},
        ]
        
        N_strateji = len(stratejiler)
        
        # Her strateji için OOS getirilerini topla
        # oos_returns_by_strategy[strateji_idx] = birleştirilmiş OOS getiri serisi
        oos_returns_by_strategy = {i: [] for i in range(N_strateji)}
        
        # Her strateji için ortalama OOS Sharpe
        oos_sharpes_by_strategy = {i: [] for i in range(N_strateji)}
        
        for i in range(n_splits):
            start_idx = i * step
            end_train = start_idx + train_size
            end_test = end_train + test_size
            
            if end_test > len(self.X):
                break
                
            X_train = self.X[start_idx:end_train]
            X_test = self.X[end_train:end_test]
            y_train = self.y[start_idx:end_train]
            NextRet_test = self.NextRet[end_train:end_test]
            
            # Her strateji için
            for strat_idx, strateji in enumerate(stratejiler):
                y_pred = self._egit_ve_tahmin_et(
                    X_train, X_test, y_train,
                    model_tipi=strateji['tip'],
                    **{k: v for k, v in strateji.items() if k not in ['ad', 'tip']}
                )
                
                # Strateji getirisi = Sinyal × NextRet
                test_returns = y_pred * NextRet_test
                oos_returns_by_strategy[strat_idx].extend(test_returns)
                
                # Bu split için Sharpe
                if np.std(test_returns) > 0:
                    sharpe = (np.mean(test_returns) / np.std(test_returns)) * np.sqrt(252)
                else:
                    sharpe = 0
                oos_sharpes_by_strategy[strat_idx].append(sharpe)
        
        # Listeleri numpy array'e çevir
        for i in range(N_strateji):
            oos_returns_by_strategy[i] = np.array(oos_returns_by_strategy[i])
            oos_sharpes_by_strategy[i] = np.array(oos_sharpes_by_strategy[i])
        
        return N_strateji, oos_returns_by_strategy, oos_sharpes_by_strategy

    def calculate_deflated_sharpe_gercek(self, N_strateji, oos_returns_by_strategy, oos_sharpes_by_strategy):
        """
        ✅ DÜZELTME: Gerçek Bailey & López de Prado (2014) DSR Formülü
        
        Adımlar:
        1. Her strateji için Sharpe hesapla (birleştirilmiş OOS serisinden)
        2. En iyi stratejiyi seç (SR_obs)
        3. Çarpıklık ve basıklık hesapla (en iyi stratejinin serisinden)
        4. σ_SR hesapla (Formül 3)
        5. SR₀ eşik değerini hesapla (Formül 5)
        6. DSR hesapla (Formül 6)
        """
        if N_strateji is None or N_strateji < 2:
            return None
        
        # 1. Her strateji için Sharpe hesapla
        sharpes = []
        for i in range(N_strateji):
            ret_series = oos_returns_by_strategy[i]
            if len(ret_series) > 30 and np.std(ret_series) > 0:
                sr = (np.mean(ret_series) / np.std(ret_series)) * np.sqrt(252)
            else:
                sr = 0
            sharpes.append(sr)
        
        sharpes = np.array(sharpes)
        
        # 2. En iyi stratejiyi seç
        best_idx = np.argmax(sharpes)
        SR_obs = sharpes[best_idx]
        best_returns = oos_returns_by_strategy[best_idx]
        T = len(best_returns)
        
        print(f"   📊 Strateji Performansları:")
        for i, sr in enumerate(sharpes):
            marker = " ← EN İYİ" if i == best_idx else ""
            print(f"      {i+1}. Sharpe: {sr:.4f}{marker}")
        print()
        
        # 3. Çarpıklık (κ) ve Basıklık (χ) - en iyi stratejinin serisinden
        kappa = skew(best_returns)
        chi = kurtosis(best_returns, fisher=False)  # Fisher=False -> normal=3
        
        # 4. ✅ DÜZELTME: DSR Standart Hatası (Formül 3)
        # σ_SR = √[(1 - κ·SR + (χ-3)/4 · SR²) / T]
        var_SR_periodic = (1 - kappa * SR_obs/np.sqrt(252) + ((chi - 3) / 4) * (SR_obs/np.sqrt(252))**2) / T
        sigma_SR_periodic = np.sqrt(max(var_SR_periodic, 1e-8))
        sigma_SR_annual = sigma_SR_periodic * np.sqrt(252)
        
        # 5. ✅ DÜZELTME: SR₀ Eşik Değeri (Formül 5)
        # SR₀ = μ_SR + σ_SR_dist · E[max(Z)]
        mu_SR = np.mean(sharpes)
        sigma_SR_dist = np.std(sharpes, ddof=1) if N_strateji > 1 else 0
        
        # E[max(Z)] hesabı (Extreme Value Theory)
        gamma_euler = 0.5772156649
        z1 = norm.ppf(1 - 1/N_strateji)
        z2 = norm.ppf(1 - 1/(N_strateji * np.e))
        E_max_Z = (1 - gamma_euler) * z1 + gamma_euler * z2
        
        SR_0 = mu_SR + sigma_SR_dist * E_max_Z
        
        # 6. ✅ DÜZELTME: DSR (Formül 6)
        # DSR = Φ[(SR_obs - SR₀) / σ_SR]
        DSR = norm.cdf((SR_obs - SR_0) / sigma_SR_annual)
        
        # Geleneksel p-değeri (karşılaştırma)
        p_geleneksel = 1 - norm.cdf(SR_obs / sigma_SR_annual)
        
        return {
            'DSR': DSR,
            'SR_obs': SR_obs,
            'SR_0': SR_0,
            'sigma_SR_annual': sigma_SR_annual,
            'kappa': kappa,
            'chi': chi,
            'T': T,
            'N_strateji': N_strateji,
            'sharpes': sharpes,
            'p_geleneksel': p_geleneksel,
            'best_idx': best_idx
        }

    def ml_siniflandirma_metrikleri(self):
        print("🤖 10.4: ML Sınıflandırma Metrikleri Hesaplanıyor...")
        split_idx = int(len(self.X) * 0.8)
        X_train, X_test = self.X[:split_idx], self.X[split_idx:]
        y_train, y_test = self.y[:split_idx], self.y[split_idx:]
        NextRet_test = self.NextRet[split_idx:]
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        strateji_getiri = y_pred * NextRet_test
        sharpe_strateji = (np.mean(strateji_getiri) / np.std(strateji_getiri)) * np.sqrt(252) if np.std(strateji_getiri) > 0 else 0
        
        return precision, recall, f1, conf_matrix, y_test, y_pred, sharpe_strateji

    def hmm_rejim_tespiti(self, n_states=3):
        if not HMM_KURULU:
            return None
        print(f"🌊 10.5: HMM ile {n_states} Rejim Tespiti...")
        X_hmm = self.df[['Ret']].values
        model = GaussianHMM(n_components=n_states, covariance_type="full", n_iter=1000, random_state=42)
        model.fit(X_hmm)
        hidden_states = model.predict(X_hmm)
        
        rejim_istatistikleri = []
        for i in range(n_states):
            state_returns = X_hmm[hidden_states == i]
            mean_ret = np.mean(state_returns) * 252
            std_ret = np.std(state_returns) * np.sqrt(252)
            rejim_istatistikleri.append({
                'state': i, 'mean_annual': mean_ret, 'std_annual': std_ret,
                'frequency': np.sum(hidden_states == i) / len(hidden_states)
            })
            
        rejim_istatistikleri.sort(key=lambda x: x['mean_annual'])
        names = ['Ayı (Bear)', 'Yatay (Neutral)', 'Boğa (Bull)'][:n_states]
        for i, name in enumerate(names):
            if i < len(rejim_istatistikleri):
                rejim_istatistikleri[i]['name'] = name
                
        return hidden_states, rejim_istatistikleri, model

    def raporla(self):
        self.veriyi_hazirla()
        print("="*70)
        print(f"🟩 LEVEL 10: RESEARCH & ML ({self.hisse})")
        print("="*70)
        
        # 10.1, 10.2, 10.3 - Birleştirilmiş
        N_strateji, oos_returns_by_strategy, oos_sharpes_by_strategy = self.walk_forward_coklu_strateji()
        
        print(f"\n🔍 10.1 & 10.2: WALK-FORWARD VALIDATION")
        if N_strateji is not None:
            # İlk stratejinin ortalama OOS Sharpe'ını göster (temsilci)
            mean_oos = np.mean(oos_sharpes_by_strategy[0])
            print(f"   • Strateji Sayısı (N)             : {N_strateji}")
            print(f"   • İlk Strateji Ort. OOS Sharpe    : {mean_oos:.3f}")
            print(f"   • Not: Walk-forward splits ≠ farklı stratejiler")
            print(f"   • N = gerçekten denenmiş model/parametre kombinasyonları\n")
        
        # 10.3 - Gerçek DSR
        print(f"📉 10.3: DEFLATED SHARPE RATIO (Gerçek Formül)")
        print(f"   Kaynak: Bailey & López de Prado (2014)\n")
        
        if N_strateji is not None:
            dsr_sonuc = self.calculate_deflated_sharpe_gercek(
                N_strateji, oos_returns_by_strategy, oos_sharpes_by_strategy
            )
            
            if dsr_sonuc:
                print(f"   📊 DSR Bileşenleri:")
                print(f"      • Gözlenen Sharpe (SR_obs)      : {dsr_sonuc['SR_obs']:.4f}")
                print(f"      • Eşik Değer (SR₀)              : {dsr_sonuc['SR_0']:.4f}")
                print(f"      • σ_SR (Standart Hata)          : {dsr_sonuc['sigma_SR_annual']:.4f}")
                print(f"      • Çarpıklık (κ)                 : {dsr_sonuc['kappa']:.4f}")
                print(f"      • Basıklık (χ)                  : {dsr_sonuc['chi']:.4f}")
                print(f"      • Gözlem Sayısı (T)             : {dsr_sonuc['T']}")
                print(f"      • Denenen Strateji (N)          : {dsr_sonuc['N_strateji']}")
                print(f"      • Geleneksel p-değeri           : {dsr_sonuc['p_geleneksel']:.4f}")
                print(f"\n   🎯 SONUÇ:")
                print(f"      • Deflated Sharpe Ratio (DSR)   : {dsr_sonuc['DSR']:.4f}")
                
                if dsr_sonuc['DSR'] > 0.95:
                    print(f"      ✅ DSR > 0.95 → Alpha istatistiksel olarak ANLAMLI")
                elif dsr_sonuc['DSR'] > 0.90:
                    print(f"      ⚠️  0.90 < DSR < 0.95 → Sınırda anlamlı")
                else:
                    print(f"      ❌ DSR < 0.90 → Muhtemelen SEÇİM YANLILIĞI ürünü")
        else:
            print("   ⚠️  DSR hesaplanamadı (yetersiz veri).\n")

        # 10.4
        prec, rec, f1, conf_mat, y_test, y_pred, sharpe_strateji = self.ml_siniflandirma_metrikleri()
        print(f"\n🎯 10.4: PRECISION, RECALL, F1")
        print(f"   • Precision : {prec:.3f}")
        print(f"   • Recall    : {rec:.3f}")
        print(f"   • F1-Score  : {f1:.3f}")
        print(f"   • Sharpe    : {sharpe_strateji:.3f}")
        print(f"   • Confusion Matrix:\n{conf_mat}")

        # 10.5
        hmm_sonuc = self.hmm_rejim_tespiti(n_states=3)
        if hmm_sonuc:
            hidden_states, rejimler, model = hmm_sonuc
            print(f"\n🌊 10.5: REGIME MODELS (HMM)")
            print(f"   ⚠️  UYARI: Geriye dönük etiketleme, canlı sinyal DEĞİL.")
            print(f"   {'Rejim':<15} {'Yıllık Getiri':<15} {'Yıllık Vol':<15} {'Sıklık':<10}")
            print(f"   {'-'*55}")
            for r in rejimler:
                print(f"   {r['name']:<15} %{r['mean_annual']*100:<14.1f} %{r['std_annual']*100:<14.1f} %{r['frequency']*100:<9.1f}")
        else:
            print("\n🌊 10.5: HMM atlandı.")

        self.grafik_ciz(dsr_sonuc if 'dsr_sonuc' in locals() else None, y_test, y_pred, hmm_sonuc)

    def grafik_ciz(self, dsr_sonuc, y_test, y_pred, hmm_sonuc):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Strateji Performansları
        ax1 = axes[0, 0]
        if dsr_sonuc:
            sharpes = dsr_sonuc['sharpes']
            ax1.bar(range(len(sharpes)), sharpes, color='teal', alpha=0.7, edgecolor='black')
            ax1.axhline(y=dsr_sonuc['SR_0'], color='red', linestyle='--', linewidth=2, 
                       label=f'SR₀ eşiği: {dsr_sonuc["SR_0"]:.3f}')
            ax1.axhline(y=0, color='black', linewidth=0.5)
            ax1.set_xticks(range(len(sharpes)))
            ax1.set_xticklabels([f'Strat {i+1}' for i in range(len(sharpes))], rotation=45, ha='right')
        ax1.set_title('10.3: Strateji Sharpe Oranları', fontweight='bold')
        ax1.set_ylabel('Annualized Sharpe')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')

        # 2. DSR Görselleştirme
        ax2 = axes[0, 1]
        if dsr_sonuc:
            ax2.axvline(x=dsr_sonuc['SR_0'], color='red', linestyle='--', linewidth=2, 
                       label=f'SR₀: {dsr_sonuc["SR_0"]:.3f}')
            ax2.axvline(x=dsr_sonuc['SR_obs'], color='green', linestyle='-', linewidth=2, 
                       label=f'SR_obs: {dsr_sonuc["SR_obs"]:.3f}')
            if dsr_sonuc['SR_obs'] > dsr_sonuc['SR_0']:
                ax2.axvspan(dsr_sonuc['SR_0'], dsr_sonuc['SR_obs'], alpha=0.2, color='green', 
                           label=f'DSR={dsr_sonuc["DSR"]:.3f}')
            else:
                ax2.axvspan(dsr_sonuc['SR_obs'], dsr_sonuc['SR_0'], alpha=0.2, color='red', 
                           label=f'DSR={dsr_sonuc["DSR"]:.3f}')
        ax2.set_title('10.3: Deflated Sharpe Ratio\n(Bailey & López de Prado, 2014)', fontweight='bold')
        ax2.set_xlabel('Sharpe Ratio')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

        # 3. Confusion Matrix
        ax3 = axes[1, 0]
        cm = confusion_matrix(y_test, y_pred)
        ax3.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax3.set_title('10.4: Confusion Matrix', fontweight='bold')
        ax3.set_xlabel('Predicted')
        ax3.set_ylabel('Actual')
        ax3.set_xticks([0, 1])
        ax3.set_xticklabels(['Down', 'Up'])
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(['Down', 'Up'])
        for i in range(2):
            for j in range(2):
                ax3.text(j, i, str(cm[i, j]), ha="center", va="center", 
                        color="white" if cm[i, j] > cm.max()/2 else "black", fontsize=12)

        # 4. HMM
        ax4 = axes[1, 1]
        if hmm_sonuc:
            hidden_states, rejimler, _ = hmm_sonuc
            plot_len = min(500, len(hidden_states))
            states_plot = hidden_states[-plot_len:]
            prices_plot = self.df['Close'].values[-plot_len:]
            state_to_name = {r['state']: r['name'] for r in rejimler}
            
            ax4.plot(self.df.index[-plot_len:], prices_plot, color='black', linewidth=1, label='Price')
            colors = {'Ayı (Bear)': 'red', 'Yatay (Neutral)': 'gray', 'Boğa (Bull)': 'green'}
            for state_id, name in state_to_name.items():
                mask = states_plot == state_id
                if np.any(mask):
                    ax4.scatter(self.df.index[-plot_len:][mask], prices_plot[mask], 
                               color=colors.get(name, 'blue'), s=10, label=name, alpha=0.6)
            ax4.set_title('10.5: HMM Regimes\n(⚠️  Geriye dönük)', fontweight='bold')
            ax4.legend(loc='upper left', fontsize=8)
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'HMM Yok', ha='center', va='center')

        plt.tight_layout()
        plt.savefig(f'{self.hisse}_level10_dsr_gercek.png', dpi=150)
        print(f"\n📊 Grafik kaydedildi: {self.hisse}_level10_dsr_gercek.png")
        plt.show()


if __name__ == "__main__":
    analiz = Level10ResearchML('bryat', 'gunluk')
    analiz.raporla()