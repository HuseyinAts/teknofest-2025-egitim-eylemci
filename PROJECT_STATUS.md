# TEKNOFEST 2025 - Proje Durumu

## 🆕 GÜNCELLEME (Frontend Geçişi)

### ✅ Frontend Artık Next.js 15 Kullanıyor!
- **Eski Vite uygulaması** arşivlendi (`frontend-vite-backup`)
- **Yeni Next.js uygulaması** ana frontend olarak ayarlandı (`frontend`)
- **Tüm önemli componentler** başarıyla taşındı
- **Yeni sayfalar** eklendi: `/quiz`, `/learning-paths`

### Frontend Durumu
- **Framework**: Next.js 15 (App Router)
- **UI Library**: Material UI v5 + Tailwind CSS
- **State Management**: Redux Toolkit
- **Port**: 3000
- **Başlatma**: `cd frontend && npm run dev`

## Mevcut Durum

### API Server
- **Durum**: ÇALIŞIYOR
- **Port**: 8000
- **Dokümantasyon**: http://localhost:8000/docs
- **Model Durumu**: Model olmadan çalışıyor (rule-based)

### Çalışan Endpoint'ler (6/6)
1. **GET /health** - Sistem sağlığı kontrolü
2. **POST /learning-path** - Kişiselleştirilmiş öğrenme yolu oluşturma
3. **POST /adaptive-quiz** - Adaptif sınav üretimi
4. **POST /study-plan** - Çalışma planı oluşturma
5. **POST /learning-style** - Öğrenme stili tespiti (VARK)
6. **POST /answer-question** - Eğitim soruları cevaplama

### Model Durumu
- **Public Model**: `Huseyin/teknofest-2025-turkish-edu` sadece metadata içeriyor (model dosyaları yok)
- **Private Model**: `Huseyin/qwen3-8b-turkish-teknofest2025-private` çalışıyor ama LoRA adapter
- **Base Model**: Qwen3-8B (15GB) indirilmesi gerekiyor

### Özellikler
- **IRT (Item Response Theory)**: 3 parametreli lojistik model ile adaptif sınav
- **ZPD (Zone of Proximal Development)**: Öğrenme yolu zorluk ayarlaması
- **VARK**: Görsel, İşitsel, Okuma, Kinestetik öğrenme stilleri
- **Model Entegrasyonu**: Model yüklendiğinde otomatik olarak kullanılır

## Hızlı Başlangıç

### 0. Tüm Projeyi Başlat (Frontend + Backend)
```bash
start_project.bat
# Seçim yapın: 3 (Her İkisi)
```

### 1. Hafif Sunucu (Model Olmadan)
```bash
py run_server_lightweight.py
```
- Hızlı başlar
- Rule-based algoritmaları kullanır
- Test için idealdir

### 2. Tam Model ile Sunucu
```bash
py src/api_server_integrated.py
```
- Model indirir (15GB, zaman alır)
- Tam AI destekli cevaplar
- Production için önerilir

### 3. API Test
```bash
py test_api_endpoints.py
```

## Dosya Yapısı
```
teknofest-2025-egitim-eylemci/
├── src/
│   ├── agents/
│   │   ├── learning_path_agent_v2.py    # Öğrenme yolu ajanı
│   │   ├── study_buddy_agent.py         # Çalışma arkadaşı ajanı
│   │   └── integrated_agents.py         # Model entegreli ajanlar
│   ├── model_integration.py             # Model yönetimi
│   └── api_server_integrated.py         # Ana API sunucusu
├── run_server_lightweight.py            # Hafif sunucu başlatıcı
├── test_api_endpoints.py                # API test betiği
└── .env                                  # Yapılandırma dosyası

```

## Yapılandırma (.env)
```
HUGGING_FACE_HUB_TOKEN=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
MODEL_NAME=Huseyin/qwen3-8b-turkish-teknofest2025-private
```

## Notlar
- Sistem şu anda model olmadan çalışıyor
- Tüm endpoint'ler fonksiyonel durumda
- Model yüklendiğinde otomatik olarak AI destekli hale geçer
- Public model henüz tamamlanmamış durumda