# MCP Araçları Dokümantasyonu
## TEKNOFEST 2025 - Eğitim Teknolojileri

### Yapılandırma Tamamlandı

MCP sunucusu başarıyla yapılandırıldı ve aşağıdaki araçlar kullanıma hazır:

## 📚 MCP Araçları (Tools)

### 1. generate_learning_path
**Açıklama:** Kişiselleştirilmiş öğrenme yolu oluşturur

**Parametreler:**
- `student_id` (string): Öğrenci kimliği
- `topic` (string): Öğrenilecek konu
- `grade_level` (int): Sınıf seviyesi (9-12)
- `learning_style` (string): VARK öğrenme stili (visual/auditory/reading/kinesthetic)

**Örnek Kullanım:**
```json
{
  "student_id": "STD001",
  "topic": "Matematik - Fonksiyonlar",
  "grade_level": 10,
  "learning_style": "visual"
}
```

### 2. generate_quiz
**Açıklama:** Adaptif quiz oluşturur (IRT tabanlı)

**Parametreler:**
- `topic` (string): Quiz konusu
- `difficulty` (string): Zorluk seviyesi (kolay/orta/zor)
- `num_questions` (int): Soru sayısı
- `grade_level` (int): Sınıf seviyesi (9-12)

**Örnek Kullanım:**
```json
{
  "topic": "Fizik - Hareket",
  "difficulty": "orta",
  "num_questions": 5,
  "grade_level": 9
}
```

### 3. answer_question
**Açıklama:** Türkçe eğitim sorularını yanıtlar

**Parametreler:**
- `question` (string): Sorulacak soru
- `subject` (string): Konu bağlamı

**Örnek Kullanım:**
```json
{
  "question": "Mitoz ve mayoz arasındaki farklar nelerdir?",
  "subject": "Biyoloji"
}
```

### 4. detect_learning_style
**Açıklama:** VARK öğrenme stilini tespit eder

**Parametreler:**
- `responses` (list): Öğrenci yanıtları listesi

**Örnek Kullanım:**
```json
{
  "responses": [
    "Görsel materyalleri tercih ederim",
    "Video izleyerek öğrenirim",
    "Grafikler ve şemalar kullanırım"
  ]
}
```

### 5. create_study_plan
**Açıklama:** Kişiselleştirilmiş çalışma planı oluşturur

**Parametreler:**
- `weak_topics` (list): Geliştirilmesi gereken konular
- `available_hours` (int): Haftalık uygun çalışma saati

**Örnek Kullanım:**
```json
{
  "weak_topics": ["Trigonometri", "İntegral", "Türev"],
  "available_hours": 10
}
```

### 6. get_irt_analysis
**Açıklama:** IRT (Item Response Theory) ile öğrenci performans analizi

**Parametreler:**
- `student_responses` (list): Öğrenci yanıtları ve zorluk seviyeleri

**Örnek Kullanım:**
```json
{
  "student_responses": [
    {"difficulty": 0.3, "is_correct": true},
    {"difficulty": 0.5, "is_correct": true},
    {"difficulty": 0.7, "is_correct": false}
  ]
}
```

## 📦 MCP Kaynakları (Resources)

### 1. teknofest://dataset
**Açıklama:** TEKNOFEST eğitim veri seti
- 4025 eğitim sorusu ve yanıtı
- Türkçe eğitim içeriği
- MEB müfredatına uyumlu

### 2. teknofest://model-info
**Açıklama:** Model bilgileri
- Model: Huseyin/teknofest-2025-turkish-edu-v2
- Tip: Türkçe Eğitim Modeli
- Durum: Tokenizer yüklü (model ağırlıkları yok)
- Mod: Kural tabanlı + tokenizer

### 3. teknofest://curriculum
**Açıklama:** MEB müfredat bilgileri
- Sınıflar: 9, 10, 11, 12
- Dersler: Matematik, Fizik, Kimya, Biyoloji, Türkçe, Tarih, Coğrafya, İngilizce
- Toplam 150 konu
- Haftalık ders saatleri

## 🚀 Claude Desktop Yapılandırması

Config dosyası konumu:
```
C:\Users\husey\AppData\Roaming\Claude\claude_desktop_config.json
```

Config içeriği:
```json
{
  "mcpServers": {
    "teknofest-edu": {
      "command": "C:\\Users\\husey\\teknofest-2025-egitim-eylemci\\venv\\Scripts\\python.exe",
      "args": [
        "C:\\Users\\husey\\teknofest-2025-egitim-eylemci\\teknofest_mcp_server_v2.py"
      ],
      "env": {
        "PYTHONPATH": "C:\\Users\\husey\\teknofest-2025-egitim-eylemci",
        "PYTHONIOENCODING": "utf-8",
        "HUGGING_FACE_HUB_TOKEN": "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"
      }
    }
  }
}
```

## ✅ Yapılandırma Durumu

- [x] MCP sunucusu kuruldu
- [x] Claude Desktop config oluşturuldu
- [x] Python venv yapılandırıldı
- [x] MCP modülü yüklendi
- [x] Araçlar test edildi
- [x] Kaynaklar doğrulandı

## 📝 Kullanım Notları

1. **Claude Desktop'ı yeniden başlatın** - Config değişikliklerinin aktif olması için
2. **MCP sunucusu otomatik başlar** - Claude Desktop açıldığında
3. **Araçlar Claude içinde kullanılabilir** - MCP protokolü üzerinden

## 🔧 Sorun Giderme

### Sunucu başlamıyorsa:
1. Python yolunu kontrol edin
2. Venv'in aktif olduğundan emin olun
3. MCP modülünün yüklü olduğunu doğrulayın: `pip list | grep mcp`

### Araçlar görünmüyorsa:
1. Claude Desktop'ı yeniden başlatın
2. Config dosyasının doğru konumda olduğunu kontrol edin
3. Sunucu loglarını kontrol edin

## 📊 Test Sonuçları

Test edilen araçlar:
- ✅ generate_learning_path
- ✅ generate_quiz (parametre düzeltmesi gerekli)
- ✅ answer_question
- ✅ detect_learning_style
- ⚠️ create_study_plan (implementasyon eksik)
- ✅ get_irt_analysis

Kaynaklar:
- ✅ Dataset (4025 öğe)
- ✅ Model bilgisi
- ✅ Müfredat bilgisi