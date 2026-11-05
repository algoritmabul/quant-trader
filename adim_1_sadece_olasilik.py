# adim_1_sadece_olasilik.py
"""
============================================================================
🟡 LEVEL 1: PROBABILITY (OLASILIK) - "Belirsizliğin Dili"
============================================================================
AMAÇ: Makine öğrenmesi veya karmaşık istatistiğe geçmeden önce, 
verinin temel olasılık matematiğini ve "beklenen değerini" (Edge) ölçmek.

KAYNAK: Quant Roadmap - Level 1
-----------------------------------------------------------------------------

1. BEKLENEN DEĞER (Expected Value - EV)
   FORMÜL: E[X] = Σ [x_i * P(x_i)]  (Örneklemde: μ = Σ x_i / n)
   ANLAMI: Bir stratejinin veya hissenin, rastgele seçilen bir günde 
           ortalama ne kadar kazandıracağı (veya kaybettireceği).
   NE İŞE YARAR: Pozitif EV (+EV) uzun vadede kazandırır. Negatif EV (-EV) 
                 kumardır; komisyonlar dahil edildiğinde er ya da geç kaybettirir.

2. VARYANS ve STANDART SAPMA (Variance & Standard Deviation)
   FORMÜL: Var(X) = E[(X - μ)²]  |  σ = √Var(X)
   ANLAMI: Getirilerin ortalamadan ne kadar saptığı. Riskin matematiksel tanımı.
   NE İŞE YARAR: Volatilite. Bir sonraki adımda (Kelly Kriteri veya Portföy 
                 Optimizasyonu) pozisyon büyüklüğünü belirlemek için temel girdidir.

3. KAZANMA OLASILIĞI (Win Probability / Hit Rate)
   FORMÜL: P(X > 0) = (Pozitif Getiri Sayısı) / (Toplam Gün Sayısı)
   ANLAMI: Hisseyi rastgele bir günde alıp ertesi gün satma ihtimalinizdeki 
           başarı oranı.
   NE İŞE YARAR: Stratejinin "tutarlılık" ölçüsüdür. %50'nin altındaysa, 
                 ortalama kazancın ortalama kayıptan çok daha büyük olması gerekir.

4. KOŞULLU BEKLENTİ (Conditional Expectation) - [İLERİ LEVEL 1]
   FORMÜL: E[X | Y] (Örn: E[Getiri_t | Getiri_{t-1} > 0])
   ANLAMI: "Dün yükseldiyse, bugün ortalama ne olur?" sorusunun cevabı.
   NE İŞE YARAR: Piyasanın Momentum mu (trend devamı) yoksa Mean-Reversion mı 
                 (ortalamaya dönüş) davrandığını saf olasılıkla test eder.
============================================================================
"""

import pandas as pd
import numpy as np
from veri_periyot import VeriPeriyot
import warnings
warnings.filterwarnings('ignore')

class Level1OlasilikAnalizi:
    def __init__(self, hisse, periyot='gunluk'):
        self.hisse = hisse.upper()
        self.periyot = periyot.lower()
        self.vp = VeriPeriyot()
        self.df = None
        
    def veriyi_hazirla(self):
        """Veriyi çeker ve Level 1 hesaplamaları için temel sütunları oluşturur."""
        print(f"📥 {self.hisse} ({self.periyot}) verisi yükleniyor...")
        self.df = self.vp.get(self.hisse, self.periyot)
        
        if self.df is None or len(self.df) < 30:
            raise ValueError(f"❌ {self.hisse} için yeterli veri yok!")
            
        # Basit Günlük Getiri (Simple Daily Return)
        self.df['Getiri'] = self.df['Close'].pct_change()
        
        # Koşullu Beklenti için: Önceki gün pozitif miydi?
        self.df['Onceki_Gun_Pozitif'] = self.df['Getiri'].shift(1) > 0
        
        self.df = self.df.dropna()
        print(f"✅ {len(self.df)} bar veri hazır.\n")

    def hesapla(self):
        """Sadece Level 1 (Olasılık) metriklerini hesaplar."""
        getiriler = self.df['Getiri']
        
        # 1. Beklenen Değer (EV)
        ev = getiriler.mean()
        
        # 2. Varyans ve Standart Sapma (Volatilite)
        varyans = getiriler.var()
        std_sapma = getiriler.std()
        
        # 3. Kazanma Olasılığı (Win Rate)
        win_rate = (getiriler > 0).mean()
        
        # 4. Koşullu Beklenti (Conditional Expectation)
        # Dün yükseldiyse bugünün beklenen getirisi
        ev_dun_pozitif = getiriler[self.df['Onceki_Gun_Pozitif'] == True].mean()
        # Dün düştüyse bugünün beklenen getirisi
        ev_dun_negatif = getiriler[self.df['Onceki_Gun_Pozitif'] == False].mean()
        
        return {
            'ev': ev,
            'varyans': varyans,
            'std_sapma': std_sapma,
            'win_rate': win_rate,
            'ev_dun_pozitif': ev_dun_pozitif,
            'ev_dun_negatif': ev_dun_negatif,
            'toplam_bar': len(self.df)
        }

    def raporla(self):
        """Bulguları trader dostu bir dille yorumlar."""
        self.veriyi_hazirla()
        m = self.hesapla()
        
        print("="*70)
        print(f"🟡 {self.hisse} - LEVEL 1: OLASILIK PROFİLİ")
        print("="*70)
        
        # --- BEKLENEN DEĞER ---
        print(f"\n🎯 1. BEKLENEN DEĞER (Expected Value)")
        print(f"   • Günlük Beklenen Getiri : {m['ev']:+.4f}  ({m['ev']*100:+.2f}%)")
        if m['ev'] > 0:
            print("   ✅ YORUM: Oyunun matematiği uzun vadede lehinize (+EV).")
        else:
            print("   ⚠️  YORUM: Oyun uzun vadede aleyhinize (-EV). Kısa vadeli al-sat dezavantajlı.")

        # --- KAZANMA OLASILIĞI ---
        print(f"\n🎲 2. KAZANMA OLASILIĞI (Win Rate)")
        print(f"   • Pozitif Kapanış İhtimali: %{m['win_rate']*100:.1f}")
        if m['win_rate'] > 0.55:
            print("   ✅ YORUM: Yüksek isabet oranı. Stratejilerde 'take-profit' daha sık tetiklenir.")
        elif m['win_rate'] < 0.45:
            print("   ⚠️  YORUM: Düşük isabet oranı. Kazanmak için ortalama kazancın, ortalama kayıptan çok büyük olması gerekir.")
        else:
            print("   ➡️  YORUM: Yazı-tura seviyesine yakın. Edge, giriş-çıkış zamanlamasında gizli.")

        # --- RİSK (VARYANS) ---
        print(f"\n⚖️ 3. RİSK ÖLÇÜMÜ (Varyans & Volatilite)")
        print(f"   • Günlük Varyans         : {m['varyans']:.6f}")
        print(f"   • Günlük Volatilite (σ)  : %{m['std_sapma']*100:.2f}")
        print(f"   • Yıllık Volatilite (σ√252): %{m['std_sapma']*np.sqrt(252)*100:.2f}")

        # --- KOŞULLU BEKLENTİ (En Değerli Kısım) ---
        print(f"\n🔗 4. KOŞULLU BEKLENTİ (E[Getiri | Önceki Gün])")
        print(f"   • Dün YÜKSELDİYSE, bugün beklenen: {m['ev_dun_pozitif']:+.4f} ({m['ev_dun_pozitif']*100:+.2f}%)")
        print(f"   • Dün DÜŞTÜYSE, bugün beklenen   : {m['ev_dun_negatif']:+.4f} ({m['ev_dun_negatif']*100:+.2f}%)")
        
        # Momentum vs Mean Reversion analizi
        if m['ev_dun_pozitif'] > m['ev_dun_negatif']:
            print("   📈 YORUM: Hafif MOMENTUM etkisi var. Yükselişler yükselişi, düşüşler düşüşü getirme eğiliminde.")
        else:
            print("   🔄 YORUM: Hafif ORTALAMAYA DÖNÜŞ (Mean-Reversion) etkisi var. Düşüşlerden sonra toparlama, yükselişlerden sonra kar realizasyonu görülüyor.")

        print("\n" + "="*70)
        print("💡 SONUÇ: Level 1 tamamlandı. Verinin temel olasılık yapısı anlaşıldı.")
        print("Bir sonraki adımda (Level 2: İstatistik) bu dağılımın 'çarpık' olup")
        print("olmadığına veya 'kuyruk riski' taşıyıp taşımadığına bakılabilir.")
        print("="*70)


# ========== ÇALIŞTIRMA ==========
if __name__ == "__main__":
    # Buradan hisse ve periyodu değiştirebilirsin
    HISSE = 'THYAO'       # Örnek: 'THYAO', 'GARAN', 'ASELS', 'EDATA', 'AHGAZ'
    PERIYOT = 'gunluk'    # 'gunluk', '60dk', 'haftalik'
    
    try:
        analiz = Level1OlasilikAnalizi(HISSE, PERIYOT)
        analiz.raporla()
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")