# 🚀 HIZLI BAŞLANGIÇ REHBERİ

## 📌 PROJEYİ BAŞLATMA

### **Tek Komutla Başlat**
```batch
HIZLI_BASLAT.bat
```

Bu komut:
- ✅ Portları temizler
- ✅ API'yi başlatır (Port 8003)
- ✅ Frontend'i başlatır (Port 3001)
- ✅ Tarayıcıyı açar

---

## 🌐 ERİŞİM ADRESLERİ

| Sayfa | URL | Açıklama |
|-------|-----|----------|
| **Ana Sayfa** | http://localhost:3001 | Dashboard |
| **Öğrenme Yolu** | http://localhost:3001/learning-path | Kişisel plan oluştur |
| **Quiz** | http://localhost:3001/quiz | Test çöz |
| **API Docs** | http://localhost:8003/docs | API test et |
| **Demo** | http://localhost:8003/demo | Demo sayfası |

---

## 📁 ÖNEMLİ DOSYALAR

### **Başlatma Scriptleri**
```
HIZLI_BASLAT.bat          # Tümünü başlat ⭐
baslat_frontend_api.bat   # Sadece API
frontend_baslat.bat       # Sadece Frontend
```

### **Test Scriptleri**
```
test_basit.py            # Basit API testi
test_frontend_api.py     # Detaylı API testi
frontend/test.html       # Tarayıcı testi
```

### **Kaynak Kodlar**
```
src/
├── api_server_frontend.py    # Ana API sunucusu
├── agents/                   # AI ajanları
│   ├── learning_path_agent_v2.py
│   └── study_buddy_agent.py
└── mcp_server/              # Claude entegrasyonu

frontend/
├── src/
│   ├── App.jsx             # Ana uygulama
│   ├── pages/              # Sayfalar
│   │   ├── Dashboard.jsx
│   │   ├── LearningPath.jsx
│   │   └── Quiz.jsx
│   └── components/         # Bileşenler
```

---

## 🎯 HIZLI TEST

### **1. API Testi**
```python
py test_basit.py
```

### **2. Frontend Testi**
1. http://localhost:3001 açın
2. Sol menüden sayfalar arası geçiş yapın
3. "Öğrenme Yolu" sayfasında plan oluşturun

### **3. Sorun Giderme Testi**
```
frontend/test.html
```
Dosyayı tarayıcıda açın, sistemin durumunu görün.

---

## 🔧 SORUN GİDERME

### **"Port kullanımda" hatası**
```batch
# Portları temizle
taskkill /F /IM node.exe
taskkill /F /IM py.exe
```

### **Frontend açılmıyor**
```batch
cd frontend
npm install
npm run dev
```

### **API çalışmıyor**
```batch
cd src
py api_server_frontend.py
```

### **Hiçbir şey çalışmıyor**
```batch
# Her şeyi kapat
taskkill /F /IM node.exe
taskkill /F /IM py.exe

# Yeniden başlat
HIZLI_BASLAT.bat
```

---

## 📚 KULLANIM ÖRNEKLERİ

### **Öğrenme Yolu Oluşturma**
1. http://localhost:3001/learning-path git
2. Adınızı girin: "Ahmet"
3. Sınıf seçin: "8"
4. Zayıf konular: "Matematik, İngilizce"
5. "Öğrenme Yolu Oluştur" tıkla

### **Quiz Çözme**
1. http://localhost:3001/quiz git
2. Soruları cevapla
3. "İleri" butonu ile devam et
4. Sonuçları gör

### **API Test**
1. http://localhost:8003/docs git
2. Endpoint seç
3. "Try it out" tıkla
4. "Execute" ile test et

---

## 💡 İPUÇLARI

- 🔄 **Sayfayı yenile:** F5
- 🧹 **Cache temizle:** Ctrl+Shift+R
- 🔍 **Konsol aç:** F12
- 🚪 **Çıkış:** Ctrl+C

---

## 📊 SİSTEM BİLGİLERİ

| Özellik | Değer |
|---------|-------|
| **Python** | 3.11+ |
| **Node.js** | 18+ |
| **React** | 18.2 |
| **FastAPI** | 0.109 |
| **Database** | PostgreSQL |

---

## 🆘 YARDIM

**Sorun mu var?**
1. `SISTEM_ANALIZ_RAPORU.md` dosyasını incele
2. GitHub Issues'a bak
3. Logları kontrol et

---

*Hızlı başlangıç için bu kadar yeterli! 🚀*