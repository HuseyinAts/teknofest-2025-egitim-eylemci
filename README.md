# Teknofest 2025 Eğitim Eylemci

## 🎯 Proje Hakkında

Kişiselleştirilmiş eğitim asistanı ve öğrenme yolu planlayıcısı. Bu proje, öğrencilere kişiselleştirilmiş öğrenme deneyimi sunan, yapay zeka destekli bir eğitim platformudur.

## 🚀 Hızlı Başlangıç

### Gereksinimler

- Python 3.9+
- Docker (opsiyonel)
- PostgreSQL veya SQLite

### Frontend Kurulum (Next.js)

```bash
cd frontend
npm install
npm run dev
```

### Backend Kurulum

1. **Bağımlılıkları yükleyin:**
```bash
pip install -r requirements.txt
```

2. **Ortam değişkenlerini ayarlayın:**
```bash
cp .env.example .env
# .env dosyasını düzenleyin
```

3. **Sunucuyu başlatın:**

**Windows:**
```bash
start_server.bat
```

**Linux/Mac:**
```bash
python -m src.mcp_server.production_server
```

### Docker ile Çalıştırma

```bash
docker-compose -f docker-compose.production.yml up
```

## 📁 Proje Yapısı

```
├── frontend/              # Next.js 15 frontend uygulaması
├── frontend-vite-backup/  # Eski Vite uygulaması (arşiv)
├── src/
│   ├── agents/              # AI ajanları
│   │   ├── learning_path_agent_v2.py
│   │   └── study_buddy_agent_clean.py
│   ├── mcp_server/          # MCP sunucu
│   │   ├── production_server.py
│   │   └── tools/           # MCP araçları
│   ├── config.py            # Yapılandırma
│   ├── data_processor.py    # Veri işleme
│   └── model_inference_fixed.py  # Model çıkarımı
├── tests/                   # Test dosyaları
├── data/                    # Veri dosyaları
├── notebooks/               # Jupyter notebooks
└── scripts/                 # Yardımcı scriptler
```

## 🔧 Yapılandırma

Yapılandırma ayarları `configs/production.yaml` dosyasında bulunur:

- **Model Ayarları**: Hugging Face model yapılandırması
- **Veritabanı**: PostgreSQL veya SQLite bağlantı ayarları
- **API Limitleri**: Rate limiting ve güvenlik ayarları
- **MCP Sunucu**: Model Context Protocol ayarları

## 🧪 Test

```bash
# Tüm testleri çalıştır
pytest tests/

# Belirli bir test dosyasını çalıştır
pytest tests/test_core.py

# Coverage ile test
pytest --cov=src tests/
```

## 📊 API Endpoints

- `POST /api/v1/learning-path` - Öğrenme yolu oluştur
- `POST /api/v1/study-buddy` - Çalışma arkadaşı yanıtı al
- `GET /api/v1/health` - Sistem sağlık durumu
- `GET /api/v1/metrics` - Performans metrikleri

## 🔐 Güvenlik

- Rate limiting aktif
- API key doğrulaması
- SQL injection koruması
- XSS koruması

## 📝 Lisans

MIT License - Detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add some AmazingFeature'`)
4. Branch'inizi push edin (`git push origin feature/AmazingFeature`)
5. Pull Request açın
