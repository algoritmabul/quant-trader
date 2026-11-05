# adim_9_mikroyapi_ohlcv.py
"""
============================================================================
🟦 LEVEL 9: MICROSTRUCTURE & EXECUTION (OHLCV Tabanlı Pratik Yaklaşım)
============================================================================

⚠️  ÖNCEKİ SENTE TİK YAKLAŞIM TERK EDİLDİ.
Artık rastgele emir defteri simülasyonu YOK. Bunun yerine, gerçek OHLCV 
verisinden literatürde kabul görmüş proxy'ler kullanıyoruz.

-----------------------------------------------------------------------------
📊 KULLANILAN OHLCV PROXY'LERİ (Gerçeğe En Yakın Makul Karşılıklar):
-----------------------------------------------------------------------------

1. SPREAD → Corwin-Schultz (2012) Tahmini
   • High-Low aralığı tek başına spread DEĞİLDİR (oynaklığı da içerir).
   • Corwin-Schultz, iki günlük High/Low'dan spread'i izole eder.
   • Kaynak: Corwin & Schultz (2012), "A Simple Way to Estimate Bid-Ask 
     Spreads from Daily High and Low Prices", Journal of Finance.

2. MID → (High + Low) / 2
   • Gerçek bid-ask mid DEĞİLDİR.
   • Günlük işlem aralığının orta noktasıdır (range midpoint).

3. ORDER-BOOK IMBALANCE → CLV (Close Location Value)
   • CLV = ((C-L) - (H-C)) / (H-L) = (2C - H - L) / (H-L)
   • Kapanışın günlük aralıktaki konumunu ölçer (-1 ile +1 arası).
   • +1 = kapanış High'da (alış baskısı), -1 = kapanış Low'da (satış baskısı).
   • Gerçek OBI diye sunulmamalıdır; sadece bir vekildir.

4. SQUARE-ROOT IMPACT → Y · σ · √(Q / V)
   • Q ve V aynı hacim biriminde olmalı.
   • Y kalibre edilmediği için sonuç sadece demo tahminidir.

5. VWAP → Σ(TypicalPriceᵢ · Volumeᵢ) / ΣVolumeᵢ
   • Typical Price = (H + L + C) / 3
   • Bar verisinden YAKLAŞIK VWAP. Gerçek işlem bazlı VWAP değildir.

6. TWAP → 60dk Bar Close Ortalaması
   • 60 dakikalık barların Close fiyatlarının basit ortalaması.
   • Seans içi TWAP vekilidir (intraday execution proxy).
   • Günlük kapanışların yıllar boyunca ortalaması intraday TWAP DEĞİLDİR.

7. AVELLANEDA-STOIKOV → Kavramsal Gösterim
   • OHLCV'den güvenilir kotasyon hesabı YAPILAMAZ.
   • Gerçek kotasyon için emir geliş yoğunluğu, ölçek uyumlu volatilite 
     ve kalibre edilmiş parametreler gerekir.
   • Sadece formül yapısını gösterir.

============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from veri_periyot import VeriPeriyot
import warnings
warnings.filterwarnings('ignore')


class Level9MikroyapiOHLCV:
    def __init__(self, hisse, gun_sayisi=60):
        """
        :param hisse: Hisse kodu
        :param gun_sayisi: Analiz için kullanılacak son gün sayısı
        """
        self.hisse = hisse.upper()
        self.vp = VeriPeriyot()
        self.gun_sayisi = gun_sayisi
        
        # 60dk verisi (intraday TWAP ve daha hassas VWAP için)
        self.df_60dk = self.vp.get(self.hisse, '60dk')
        if self.df_60dk is None or len(self.df_60dk) < 100:
            raise ValueError(f"❌ {self.hisse} için yeterli 60dk veri yok!")
        
        # Günlük veri (Corwin-Schultz spread için)
        self.df_gunluk = self.vp.get(self.hisse, 'gunluk')
        if self.df_gunluk is None or len(self.df_gunluk) < 30:
            raise ValueError(f"❌ {self.hisse} için yeterli günlük veri yok!")
        
        self.S0 = float(self.df_gunluk['Close'].iloc[-1])
        
        # Günlük volatilite (impact ve Avellaneda için)
        log_getiri = np.log(self.df_gunluk['Close'] / self.df_gunluk['Close'].shift(1)).dropna()
        self.sigma_gunluk = np.std(log_getiri)

    def corwin_schultz_spread(self, df, window=20):
        """
        Corwin-Schultz (2012) Bid-Ask Spread Tahmini
        
        FORMÜL:
        β_t = (ln(H_t/L_t))² + (ln(H_{t-1}/L_{t-1}))²
        γ_t = (ln(H_{t,t-1}/L_{t,t-1}))²  (iki günlük birleşik range)
        α = (√(2β) - √β) / (3 - 2√2) - √(γ / (3 - 2√2))
        Spread = 2(e^α - 1) / (1 + e^α)
        
        Kaynak: Corwin & Schultz, Journal of Finance, 2012
        """
        H = df['High'].values
        L = df['Low'].values
        
        # ln(H/L) için güvenli hesaplama (L=0 durumunu önle)
        ln_HL = np.log(np.maximum(H / L, 1.0001))
        
        # β: iki ardışık günün ln(H/L) kareleri toplamı
        beta = ln_HL[1:]**2 + ln_HL[:-1]**2
        
        # γ: iki günlük birleşik range (max(H_t, H_{t-1}) / min(L_t, L_{t-1}))
        H_max = np.maximum(H[1:], H[:-1])
        L_min = np.minimum(L[1:], L[:-1])
        gamma = np.log(np.maximum(H_max / L_min, 1.0001))**2
        
        # α hesapla
        k = 3 - 2 * np.sqrt(2)
        alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        
        # α negatif olamaz (spread negatif olamaz)
        alpha = np.maximum(alpha, 0)
        
        # Spread = 2(e^α - 1) / (1 + e^α)
        spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
        
        # Son 'window' günün ortalaması
        return np.mean(spread[-window:])

    def clv_hesapla(self, df):
        """
        Close Location Value (CLV) - OBI vekili
        
        FORMÜL: CLV = ((C-L) - (H-C)) / (H-L) = (2C - H - L) / (H-L)
        
        • +1 = Kapanış High'da (güçlü alış baskısı)
        •  0 = Kapanış tam ortada
        • -1 = Kapanış Low'da (güçlü satış baskısı)
        
        NOT: High == Low durumunda (nadiren) CLV tanımsızdır; 0 atanır.
        """
        H = df['High'].values
        L = df['Low'].values
        C = df['Close'].values
        
        range_HL = H - L
        clv = np.where(range_HL > 0, (2 * C - H - L) / range_HL, 0.0)
        
        return clv

    def vwap_hesapla(self, df):
        """
        Yaklaşık VWAP: Σ(TypicalPrice · Volume) / ΣVolume
        Typical Price = (H + L + C) / 3
        """
        TP = (df['High'] + df['Low'] + df['Close']) / 3.0
        vwap = np.sum(TP * df['Volume']) / np.sum(df['Volume'])
        return vwap

    def twap_60dk(self, df_60dk, son_gun_tarih=None):
        """
        Intraday TWAP vekili: Son günün 60dk bar close ortalaması
        
        NOT: Gerçek TWAP her işlem için eşit ağırlık gerektirir.
        60dk bar close ortalaması bunun makul bir vekilidir.
        """
        if son_gun_tarih is None:
            son_gun = df_60dk.index[-1].date()
        else:
            son_gun = son_gun_tarih
        
        gun_bari = df_60dk[df_60dk.index.date == son_gun]
        if len(gun_bari) == 0:
            return np.nan
        
        return np.mean(gun_bari['Close'].values)

    def hesapla(self):
        sonuclar = {}
        
        # Son N günlük veri (Corwin-Schultz ve CLV için)
        df_gun = self.df_gunluk.tail(self.gun_sayisi).copy()
        
        # ============================================================
        # 9.1 SPREAD (Corwin-Schultz)
        # ============================================================
        spread_cs = self.corwin_schultz_spread(df_gun, window=20)
        spread_bps = spread_cs * 10000  # basis point
        
        sonuclar['spread'] = {
            'corwin_schultz': spread_cs,
            'bps': spread_bps,
            'yontem': 'Corwin-Schultz (2012) - High/Low tabanlı'
        }
        
        # ============================================================
        # 9.2 MID (Range Midpoint)
        # ============================================================
        son_gun = df_gun.iloc[-1]
        mid_range = (son_gun['High'] + son_gun['Low']) / 2.0
        
        sonuclar['mid'] = {
            'range_midpoint': mid_range,
            'high': son_gun['High'],
            'low': son_gun['Low'],
            'close': son_gun['Close'],
            'uyari': 'Gerçek bid-ask mid DEĞİLDİR. Günlük range ortasıdır.'
        }
        
        # ============================================================
        # 9.3 CLV (OBI vekili)
        # ============================================================
        clv_seri = self.clv_hesapla(df_gun)
        df_gun['CLV'] = clv_seri
        
        # CLV ile sonraki gün getirisi korelasyonu (sinyal gücü testi)
        df_gun['Sonraki_Getiri'] = df_gun['Close'].pct_change().shift(-1)
        clv_getiri_korr = df_gun['CLV'].corr(df_gun['Sonraki_Getiri'])
        
        sonuclar['clv'] = {
            'son_clv': clv_seri[-1],
            'ortalama_clv': np.mean(clv_seri),
            'korelasyon_sinyal': clv_getiri_korr,
            'uyari': 'Gerçek OBI DEĞİLDİR. CLV sadece kapanış konumunu ölçer.'
        }
        
        # ============================================================
        # 9.4 SQUARE-ROOT IMPACT (Demo)
        # ============================================================
        # Son günün hacmi
        V_gun = son_gun['Volume']
        Y_sabiti = 0.1  # ⚠️  KEYFİ, kalibre edilmemiş
        
        # Günlük hacmin %1'i kadar büyük emir
        Q_buyuk = V_gun * 0.01
        etki_orani = Y_sabiti * self.sigma_gunluk * np.sqrt(Q_buyuk / V_gun)
        etki_tl = self.S0 * etki_orani
        
        sonuclar['impact'] = {
            'V_gun': V_gun,
            'Q_buyuk': Q_buyuk,
            'Y_sabiti': Y_sabiti,
            'etki_bps': etki_orani * 10000,
            'etki_tl': etki_tl,
            'uyari': 'Y kalibre edilmedi. Sadece demo tahmini.'
        }
        
        # ============================================================
        # 9.5 VWAP (Typical Price ile)
        # ============================================================
        vwap_gun = self.vwap_hesapla(df_gun.tail(1))  # Son gün
        vwap_60dk = self.vwap_hesapla(
            self.df_60dk[self.df_60dk.index.date == df_gun.index[-1].date()]
        )
        
        sonuclar['vwap'] = {
            'gunluk_typical': vwap_gun,
            'intraday_60dk': vwap_60dk,
            'uyari': 'Gerçek işlem bazlı VWAP DEĞİLDİR. Typical Price yaklaşımıdır.'
        }
        
        # ============================================================
        # 9.6 TWAP (60dk close ortalaması)
        # ============================================================
        son_gun_tarih = df_gun.index[-1].date()
        twap_60dk = self.twap_60dk(self.df_60dk, son_gun_tarih)
        
        # Implementation Shortfall (varış fiyatı = gün açılışı)
        P_varis = son_gun['Open']
        is_hisse = twap_60dk - P_varis if not np.isnan(twap_60dk) else np.nan
        is_bps = (is_hisse / P_varis) * 10000 if not np.isnan(is_hisse) else np.nan
        
        sonuclar['twap'] = {
            'twap_60dk': twap_60dk,
            'P_varis': P_varis,
            'IS_hisse': is_hisse,
            'IS_bps': is_bps,
            'uyari': '60dk close ortalaması intraday TWAP VEKİLİDİR. Gerçek TWAP DEĞİLDİR.'
        }
        
        # ============================================================
        # 9.7 AVELLANEDA-STOIKOV (Kavramsal)
        # ============================================================
        # ⚠️  Birim uyumsuzluğu uyarısı ile kavramsal gösterim
        gamma = 0.1
        kappa = 1.0
        T, t, q = 1.0, 0.5, 500
        s = mid_range  # Mid olarak range midpoint kullan
        
        r = s - q * gamma * (self.sigma_gunluk**2) * (T - t)
        delta = gamma * (self.sigma_gunluk**2) * (T - t) + (2/gamma) * np.log(1 + gamma/kappa)
        
        sonuclar['avellaneda'] = {
            's': s, 'q': q, 'gamma': gamma, 'kappa': kappa,
            'r': r, 'delta': delta,
            'bid': r - delta/2, 'ask': r + delta/2,
            'uyari': 'OHLCV\'den güvenilir kotasyon YAPILAMAZ. Kavramsal gösterim.'
        }
        
        # Ek: Son 120 günlük OHLCV tablosu (görselleştirme için)
        sonuclar['df_gun'] = df_gun.tail(120)

        # 9.5-9.6 için son 5 günün 60dk barlarını birleştir
        son_gunler = df_gun.index[-5:]
        gun_tarihleri = [d.date() for d in son_gunler]
        tarih_index = pd.Index(self.df_60dk.index.date)
        mask = tarih_index.isin(gun_tarihleri)
        sonuclar['df_60dk_son_gun'] = self.df_60dk[mask]

        return sonuclar

    def raporla(self):
        sonuclar = self.hesapla()
        
        print("="*70)
        print(f"🟦 LEVEL 9: MICROSTRUCTURE (OHLCV Tabanlı Pratik Yaklaşım)")
        print(f"📊 {self.hisse} | Son Fiyat: {self.S0:.2f} TL | σ: %{self.sigma_gunluk*100:.2f}")
        print("="*70)
        
        # 9.1
        sp = sonuclar['spread']
        print(f"\n📏 9.1 SPREAD (Corwin-Schultz Tahmini)")
        print(f"   FORMÜL: β=(ln(H/L))², γ=(ln(H_max/L_min))², α=(√2β-√β)/k - √(γ/k)")
        print(f"           Spread = 2(e^α-1)/(1+e^α)")
        print(f"   • Tahmini Spread  : {sp['corwin_schultz']:.6f} ({sp['bps']:.2f} bps)")
        print(f"   • Yöntem          : {sp['yontem']}")
        print(f"   💡 MÜLAKAT: 'High-Low neden tek başına spread değildir?'")
        print(f"              Cevap: High-Low aralığı hem spread hem de gerçek oynaklığı içerir.")
        print(f"              Corwin-Schultz iki günlük range ile bunları ayırır.")

        # 9.2
        mid = sonuclar['mid']
        print(f"\n🎯 9.2 MID (Range Midpoint)")
        print(f"   FORMÜL: Mid = (High + Low) / 2")
        print(f"   • High              : {mid['high']:.2f} TL")
        print(f"   • Low               : {mid['low']:.2f} TL")
        print(f"   • Range Midpoint    : {mid['range_midpoint']:.2f} TL")
        print(f"   • Close             : {mid['close']:.2f} TL")
        print(f"   ⚠️  UYARI: {mid['uyari']}")

        # 9.3
        clv = sonuclar['clv']
        print(f"\n⚖️ 9.3 CLOSE LOCATION VALUE (CLV) - OBI Vekili")
        print(f"   FORMÜL: CLV = (2C - H - L) / (H - L)")
        print(f"   • Son Gün CLV       : {clv['son_clv']:+.4f}")
        print(f"   • Ortalama CLV      : {clv['ortalama_clv']:+.4f}")
        print(f"   • CLV-Getiri Korel. : {clv['korelasyon_sinyal']:+.4f}")
        
        if clv['son_clv'] > 0.3:
            print(f"   📈 YORUM: Kapanış High'a yakın → Gün içi alış baskısı vardı.")
        elif clv['son_clv'] < -0.3:
            print(f"   📉 YORUM: Kapanış Low'a yakın → Gün içi satış baskısı vardı.")
        else:
            print(f"   ➡️  YORUM: Kapanış aralığın ortasında → Dengeli seyir.")
        
        print(f"   ⚠️  UYARI: {clv['uyari']}")
        print(f"   💡 MÜLAKAT: 'CLV neyi ölçer?'")
        print(f"              Cevap: Kapanışın günlük High-Low aralığındaki göreceli konumunu.")

        # 9.4
        imp = sonuclar['impact']
        print(f"\n🌊 9.4 SQUARE-ROOT IMPACT (Demo)")
        print(f"   FORMÜL: ΔP/P ≈ Y · σ · √(Q / V)")
        print(f"   • Günlük Hacim (V)  : {imp['V_gun']:,.0f}")
        print(f"   • Büyük Emir (Q=%1) : {imp['Q_buyuk']:,.0f}")
        print(f"   • Y Sabiti          : {imp['Y_sabiti']} (⚠️  KEYFİ)")
        print(f"   • Fiyat Etkisi      : {imp['etki_tl']:.4f} TL ({imp['etki_bps']:.2f} bps)")
        print(f"   ⚠️  UYARI: {imp['uyari']}")

        # 9.5
        vwap = sonuclar['vwap']
        print(f"\n📊 9.5 VWAP (Typical Price Yaklaşımı)")
        print(f"   FORMÜL: VWAP = Σ((H+L+C)/3 · Volume) / ΣVolume")
        print(f"   • Günlük VWAP       : {vwap['gunluk_typical']:.2f} TL")
        print(f"   • Intraday 60dk VWAP: {vwap['intraday_60dk']:.2f} TL")
        print(f"   • Son Fiyat (S0)    : {self.S0:.2f} TL")
        if not np.isnan(vwap['intraday_60dk']):
            if self.S0 > vwap['intraday_60dk']:
                print(f"   📈 Fiyat VWAP üstünde → Gün içi ortalamanın üzerinde işlem")
            else:
                print(f"   📉 Fiyat VWAP altında → Gün içi ortalamanın altında işlem")
        print(f"   ⚠️  UYARI: {vwap['uyari']}")

        # 9.6
        twap = sonuclar['twap']
        print(f"\n⏱️  9.6 TWAP (60dk Close Ortalaması)")
        print(f"   FORMÜL: TWAP = (1/N) · Σ Close_60dk")
        print(f"   • Varış Fiyatı (Open): {twap['P_varis']:.2f} TL")
        print(f"   • TWAP (60dk)       : {twap['twap_60dk']:.2f} TL")
        if not np.isnan(twap['IS_hisse']):
            print(f"   • IS (hisse başı)   : {twap['IS_hisse']:+.4f} TL ({twap['IS_bps']:+.2f} bps)")
        print(f"   ⚠️  UYARI: {twap['uyari']}")
        print(f"   💡 MÜLAKAT: 'Neden 60dk close ortalaması intraday TWAP vekili olabilir?'")
        print(f"              Cevap: Seans içi eşit zamanlı örneklem, gerçek TWAP'a en yakın OHLCV vekilidir.")

        # 9.7
        av = sonuclar['avellaneda']
        print(f"\n🤖 9.7 AVELLANEDA-STOIKOV (Kavramsal Gösterim)")
        print(f"   FORMÜL: r = s - q·γ·σ²·(T-t)  |  δ = γ·σ²·(T-t) + (2/γ)·ln(1+γ/κ)")
        print(f"   • s (range mid)     : {av['s']:.2f} TL")
        print(f"   • q (stok)          : {av['q']} lot (KEYFİ)")
        print(f"   • γ, κ, T, t        : KEYFİ parametreler")
        print(f"   • Rezervasyon (r)   : {av['r']:.2f} TL")
        print(f"   • Optimal Spread (δ): {av['delta']:.4f} TL")
        print(f"   • Bid               : {av['bid']:.2f} TL")
        print(f"   • Ask               : {av['ask']:.2f} TL")
        print(f"   ⚠️  ⚠️  UYARI: {av['uyari']}")

        print("\n" + "="*70)
        print("📚 KAYNAKÇA:")
        print("   • Corwin & Schultz (2012) - J. Finance (spread tahmini)")
        print("   • Kyle (1985) - Continuous auctions (impact)")
        print("   • Avellaneda & Stoikov (2008) - Quant. Finance")
        print("   • Perold (1988) - Implementation Shortfall")
        print("="*70)

        self.grafik_ciz(sonuclar)

    def grafik_ciz(self, sonuclar):
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        df_gun = sonuclar['df_gun']
        df_60dk = sonuclar['df_60dk_son_gun']
        
        # 1. Son 20 gün: Close + VWAP + Range
        ax1 = axes[0, 0]
        ax1.plot(df_gun.index, df_gun['Close'], 'b-', label='Close', linewidth=1.5)
        ax1.plot(df_gun.index, df_gun['High'], 'g--', alpha=0.5, label='High')
        ax1.plot(df_gun.index, df_gun['Low'], 'r--', alpha=0.5, label='Low')
        ax1.fill_between(df_gun.index, df_gun['Low'], df_gun['High'], 
                        alpha=0.1, color='gray', label='Range')
        ax1.axhline(y=sonuclar['spread']['corwin_schultz'] * self.S0 + self.S0, 
                   color='orange', linestyle=':', alpha=0.5, 
                   label=f'CS Spread ({sonuclar["spread"]["bps"]:.1f} bps)')
        ax1.set_title('9.1-9.2: OHLCV + Corwin-Schultz Spread', fontweight='bold')
        ax1.set_ylabel('Fiyat (TL)')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)

        # 2. CLV zaman serisi
        ax2 = axes[0, 1]
        colors = ['green' if x > 0 else 'red' for x in df_gun['CLV']]
        ax2.bar(df_gun.index, df_gun['CLV'], color=colors, alpha=0.6, edgecolor='black')
        ax2.axhline(y=0, color='black', linewidth=1)
        ax2.axhline(y=sonuclar['clv']['ortalama_clv'], color='blue', 
                   linestyle='--', label=f'Ort: {sonuclar["clv"]["ortalama_clv"]:+.3f}')
        ax2.set_title('9.3: CLV (Close Location Value)\n(OBI vekili)', fontweight='bold')
        ax2.set_ylabel('CLV (-1 ile +1)')
        ax2.set_ylim(-1.1, 1.1)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.tick_params(axis='x', rotation=45)

        # 3. Intraday 60dk: VWAP ve TWAP
        ax3 = axes[1, 0]
        if len(df_60dk) > 0:
            ax3.plot(df_60dk.index, df_60dk['Close'], 'b-', label='60dk Close', linewidth=1.5)
            
            # 60dk VWAP (kümülatif)
            TP_60 = (df_60dk['High'] + df_60dk['Low'] + df_60dk['Close']) / 3.0
            cum_vwap = np.cumsum(TP_60 * df_60dk['Volume']) / np.cumsum(df_60dk['Volume'])
            ax3.plot(df_60dk.index, cum_vwap, 'g-', label='Kümülatif VWAP', linewidth=2)
            
            # TWAP (sabit çizgi)
            twap_val = sonuclar['twap']['twap_60dk']
            if not np.isnan(twap_val):
                ax3.axhline(y=twap_val, color='purple', linestyle='--', 
                           label=f'TWAP ({twap_val:.2f})')
            
            ax3.set_title('9.5-9.6: Intraday VWAP & TWAP\n(60dk bar verisi)', fontweight='bold')
        else:
            ax3.text(0.5, 0.5, '60dk verisi yok', ha='center', va='center')
            ax3.set_title('9.5-9.6: Intraday VWAP & TWAP', fontweight='bold')
        
        ax3.set_ylabel('Fiyat (TL)')
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis='x', rotation=45)

        # 4. Özet Tablo
        ax4 = axes[1, 1]
        ax4.axis('off')
        metrikler = [
            ['Corwin-Schultz Spread', f"{sonuclar['spread']['bps']:.2f} bps"],
            ['Range Midpoint', f"{sonuclar['mid']['range_midpoint']:.2f} TL"],
            ['Son CLV', f"{sonuclar['clv']['son_clv']:+.4f}"],
            ['CLV-Getiri korr.', f"{sonuclar['clv']['korelasyon_sinyal']:+.4f}"],
            ['Impact (bps)', f"{sonuclar['impact']['etki_bps']:.2f}"],
            ['Intraday VWAP', f"{sonuclar['vwap']['intraday_60dk']:.2f} TL"],
            ['TWAP (60dk)', f"{sonuclar['twap']['twap_60dk']:.2f} TL"],
            ['IS (bps)', f"{sonuclar['twap']['IS_bps']:+.2f}" if not np.isnan(sonuclar['twap']['IS_bps']) else "N/A"],
        ]
        table = ax4.table(cellText=metrikler, colLabels=['Metrik', 'Değer'], 
                         loc='center', bbox=[0.1, 0.1, 0.8, 0.8])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.5)
        for j in range(2):
            table[0, j].set_facecolor('#34495e')
            table[0, j].set_text_props(color='white', fontweight='bold')
        ax4.set_title(f'OHLCV Mikroyapı Özet ({self.hisse})', fontweight='bold', pad=20)

        plt.tight_layout()
        plt.savefig(f'{self.hisse}_level9_ohlcv.png', dpi=150)
        print(f"\n📊 Grafik kaydedildi: {self.hisse}_level9_ohlcv.png")
        plt.show()


# ========== ÇALIŞTIRMA ==========
if __name__ == "__main__":
    analiz = Level9MikroyapiOHLCV('bryat', gun_sayisi=120)
    analiz.raporla()