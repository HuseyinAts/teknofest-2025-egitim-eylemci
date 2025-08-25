# TEKNOFEST 2025 - Frontend (Next.js)

## 🎯 Proje Hakkında

TEKNOFEST 2025 Eğitim Teknolojileri Eylemcisi projesinin modern Next.js 15 tabanlı frontend uygulaması.

## 🚀 Özellikler

- **Next.js 15** - App Router ve Server Components
- **TypeScript** - Tip güvenli geliştirme
- **Material UI** - Modern UI komponetleri
- **Redux Toolkit** - State yönetimi
- **React Query** - Veri yönetimi
- **Tailwind CSS** - Utility-first CSS
- **PWA Desteği** - Offline çalışma
- **i18n** - Çoklu dil desteği

## 📦 Kurulum

```bash
# Bağımlılıkları yükle
npm install

# Geliştirme sunucusunu başlat
npm run dev

# Production build
npm run build

# Production sunucusu
npm start
```

## 📁 Proje Yapısı

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router sayfaları
│   │   ├── dashboard/          # Dashboard sayfası
│   │   ├── quiz/              # Quiz sayfası
│   │   ├── learning-paths/    # Öğrenme yolları
│   │   ├── login/             # Giriş sayfası
│   │   └── register/          # Kayıt sayfası
│   ├── components/            # React komponetleri
│   │   ├── quiz/             # Quiz komponetleri
│   │   ├── learning/         # Öğrenme komponetleri
│   │   ├── common/           # Ortak komponetler
│   │   └── Layout/           # Layout komponetleri
│   ├── store/                # Redux store
│   │   └── slices/          # Redux slices
│   ├── hooks/               # Custom React hooks
│   ├── lib/                 # Yardımcı kütüphaneler
│   ├── theme/              # MUI tema ayarları
│   └── i18n/               # Çoklu dil dosyaları
├── public/                 # Statik dosyalar
├── .env.local             # Yerel ortam değişkenleri
└── package.json           # Proje bağımlılıkları
```

## 🔧 Ortam Değişkenleri

`.env.local` dosyası oluşturun:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_APP_NAME=TEKNOFEST 2025
NEXT_PUBLIC_APP_VERSION=1.0.0
```

## 📱 Sayfalar

- `/` - Ana sayfa
- `/dashboard` - Öğrenci paneli
- `/quiz` - Quiz merkezi
- `/learning-paths` - Öğrenme yolları
- `/login` - Giriş sayfası
- `/register` - Kayıt sayfası

## 🧩 Ana Komponetler

### Quiz Komponetleri
- `QuizInterface` - Adaptif quiz arayüzü
- `QuizResults` - Quiz sonuçları

### Öğrenme Komponetleri
- `LearningPathVisualization` - Öğrenme yolu görselleştirme
- `ProgressTracker` - İlerleme takibi

### Ortak Komponetler
- `OfflineIndicator` - Çevrimdışı durum göstergesi
- `ErrorBoundary` - Hata yakalama
- `LoadingScreen` - Yükleme ekranı

## 🎨 Stil ve Tema

Material UI ve Tailwind CSS kullanılmaktadır:

```tsx
// MUI tema örneği
import { createTheme } from '@mui/material/styles';

const theme = createTheme({
  palette: {
    primary: {
      main: '#3b82f6',
    },
    secondary: {
      main: '#8b5cf6',
    },
  },
});
```

## 🧪 Test

```bash
# Unit testleri çalıştır
npm test

# Coverage raporu
npm run test:coverage
```

## 📈 Performans Optimizasyonları

- **Code Splitting** - Otomatik sayfa bazlı kod bölme
- **Image Optimization** - Next/Image ile otomatik optimizasyon
- **Lazy Loading** - Komponetlerin lazy yüklenmesi
- **PWA** - Service Worker ile offline destek
- **Bundle Analyzer** - Bundle boyutu analizi

## 🚢 Deployment

```bash
# Production build
npm run build

# Docker ile çalıştırma
docker build -t teknofest-frontend .
docker run -p 3000:3000 teknofest-frontend
```

## 📝 Notlar

- Next.js 15 App Router kullanılmaktadır
- Tüm sayfalar 'use client' direktifi ile client-side rendering kullanmaktadır
- Redux persist ile state kalıcılığı sağlanmaktadır
- Material UI v5 kullanılmaktadır

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/amazing`)
3. Commit yapın (`git commit -m 'Add amazing feature'`)
4. Push edin (`git push origin feature/amazing`)
5. Pull Request açın

## 📄 Lisans

MIT License - Detaylar için [LICENSE](../LICENSE) dosyasına bakın.
