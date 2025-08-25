# 🚀 MCP (Model Context Protocol) Kurulumu - TEKNOFEST 2025

## ✅ KURULUM TAMAMLANDI!

MCP server başarıyla kuruldu ve yapılandırıldı. Claude Desktop'ta TEKNOFEST araçlarını kullanabilirsiniz.

## 📋 Kurulum Detayları

### 1. Kurulu Paketler
- ✅ `mcp` (v1.13.0) - Python MCP SDK
- ✅ FastMCP framework
- ✅ Tüm bağımlılıklar

### 2. MCP Server Dosyaları
- **Ana Server:** `teknofest_mcp_server_v2.py`
- **Config:** `C:\Users\husey\AppData\Roaming\Claude\claude_desktop_config.json`
- **Test:** `test_mcp_server.py`

### 3. Claude Desktop Konfigürasyonu
```json
{
  "mcpServers": {
    "teknofest": {
      "command": "python",
      "args": ["C:\\Users\\husey\\teknofest-2025-egitim-eylemci\\teknofest_mcp_server_v2.py"],
      "env": {
        "PYTHONPATH": "C:\\Users\\husey\\teknofest-2025-egitim-eylemci",
        "HUGGING_FACE_HUB_TOKEN": "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"
      }
    }
  }
}
```

## 🛠️ MCP Araçları (Tools)

### 1. **generate_learning_path**
Kişiselleştirilmiş öğrenme yolu oluşturur.
```
Parametreler:
- student_id: Öğrenci ID
- topic: Konu
- grade_level: Sınıf (9-12)
- learning_style: Öğrenme stili (visual/auditory/reading/kinesthetic)
```

### 2. **generate_quiz**
Adaptif quiz oluşturur (IRT destekli).
```
Parametreler:
- topic: Quiz konusu
- difficulty: Zorluk (kolay/orta/zor)
- num_questions: Soru sayısı
- grade_level: Sınıf (9-12)
```

### 3. **answer_question**
Eğitim sorularını cevaplar.
```
Parametreler:
- question: Soru
- subject: Konu
```

### 4. **detect_learning_style**
VARK öğrenme stilini tespit eder.
```
Parametreler:
- responses: Öğrenci cevapları listesi
```

### 5. **create_study_plan**
Kişiselleştirilmiş çalışma planı oluşturur.
```
Parametreler:
- weak_topics: Zayıf konular listesi
- available_hours: Haftalık çalışma saati
```

### 6. **get_irt_analysis**
IRT (Item Response Theory) analizi yapar.
```
Parametreler:
- student_responses: Öğrenci cevapları (zorluk ve doğruluk bilgisi ile)
```

## 📚 MCP Kaynakları (Resources)

### 1. **teknofest://dataset**
TEKNOFEST eğitim veri seti (5000+ Türkçe soru)

### 2. **teknofest://model-info**
Model bilgileri (Huseyin/teknofest-2025-turkish-edu-v2)

### 3. **teknofest://curriculum**
MEB müfredat bilgileri

## 🎯 Kullanım Örnekleri

Claude Desktop'ta şu komutları kullanabilirsiniz:

### Örnek 1: Quiz Oluşturma
```
"TEKNOFEST MCP araçlarını kullanarak matematik konusunda orta zorlukta 5 soruluk bir quiz oluştur"
```

### Örnek 2: Öğrenme Yolu
```
"9. sınıf öğrencisi için fizik konusunda görsel öğrenme stiline uygun bir öğrenme yolu oluştur"
```

### Örnek 3: Soru Cevaplama
```
"Pisagor teoremi nedir? TEKNOFEST araçlarını kullanarak cevapla"
```

### Örnek 4: Çalışma Planı
```
"Denklemler ve geometri konularında zayıf olan bir öğrenci için haftalık 10 saatlik çalışma planı oluştur"
```

## 🔧 Sorun Giderme

### MCP Server Görünmüyorsa:
1. Claude Desktop'ı yeniden başlatın
2. Config dosyasının doğru yerde olduğunu kontrol edin:
   - Windows: `%APPDATA%\Claude\claude_desktop_config.json`
3. Python yolunun PATH'de olduğundan emin olun

### Server Başlamazsa:
1. Python 3.11+ kurulu olduğundan emin olun
2. Tüm bağımlılıkların kurulu olduğunu kontrol edin:
   ```bash
   pip install mcp
   ```

### Test Etmek İçin:
```bash
# MCP server'ı manuel test et
python teknofest_mcp_server_v2.py
```

## 📊 Sistem Durumu

| Bileşen | Durum | Açıklama |
|---------|-------|----------|
| MCP SDK | ✅ Kurulu | v1.13.0 |
| FastMCP | ✅ Aktif | Server framework |
| Claude Config | ✅ Yapılandırıldı | AppData/Roaming/Claude |
| Python Server | ✅ Hazır | teknofest_mcp_server_v2.py |
| Model | ✅ Yapılandırıldı | Rule-based (indirme gerekmez) |
| API | ✅ Çalışıyor | http://localhost:8000 |

## 🚀 Sonraki Adımlar

1. **Claude Desktop'ı yeniden başlatın**
2. **MCP araçlarının görünüp görünmediğini kontrol edin**
3. **Örnek komutları test edin**

## 📝 Notlar

- Model indirme gerekmez (rule-based sistem + tokenizer)
- Tüm araçlar Türkçe destekli
- IRT (Item Response Theory) ile adaptif değerlendirme
- VARK öğrenme stili tespiti
- MEB müfredatı entegreli

---

**Versiyon:** 1.0.0  
**Tarih:** 2025  
**Proje:** TEKNOFEST 2025 Eğitim Teknolojileri