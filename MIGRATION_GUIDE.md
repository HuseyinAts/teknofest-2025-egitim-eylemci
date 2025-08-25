# TEKNOFEST 2025 - Vite'tan Next.js'e Geçiş Rehberi

## 📋 Geçiş Özeti

Proje artık **Next.js 15** kullanmaktadır. Vite tabanlı eski frontend, `frontend-vite-backup` klasöründe arşivlenmiştir.

## ✅ Tamamlanan İşlemler

### 1. Next.js Kurulumu
- ✅ Next.js 15 App Router yapısı kuruldu
- ✅ TypeScript konfigürasyonu tamamlandı
- ✅ Tailwind CSS entegrasyonu yapıldı
- ✅ Material UI v5 entegrasyonu tamamlandı

### 2. Component Taşıma
Aşağıdaki componentler başarıyla taşındı:
- ✅ `QuizInterface` → `/src/components/quiz/QuizInterface.tsx`
- ✅ `LearningPathVisualization` → `/src/components/learning/LearningPathVisualization.tsx`
- ✅ `OfflineIndicator` → `/src/components/common/OfflineIndicator.tsx`

### 3. Sayfa Oluşturma
Yeni Next.js sayfaları oluşturuldu:
- ✅ `/dashboard` - Ana panel sayfası
- ✅ `/quiz` - Quiz merkezi sayfası
- ✅ `/learning-paths` - Öğrenme yolları sayfası
- ✅ `/login` - Giriş sayfası (mevcut)
- ✅ `/register` - Kayıt sayfası (mevcut)

### 4. State Management
- ✅ Redux Toolkit kurulumu
- ✅ Redux slices taşındı
- ✅ Redux persist entegrasyonu

### 5. Hook'lar
- ✅ `useOffline` hook'u taşındı ve güncellendi
- ✅ Diğer custom hook'lar hazır

## 🔄 Icon Dönüşümleri

| Lucide React | Material UI |
|-------------|-------------|
| BookOpen | MenuBook |
| CheckCircle | CheckCircle |
| Circle | RadioButtonUnchecked |
| Lock | Lock |
| Star | Star |
| TrendingUp | TrendingUp |
| Clock | AccessTime |
| Target | GpsFixed |
| Award | EmojiEvents |
| ChevronRight | ChevronRight |
| Zap | Bolt |
| Brain | Psychology |
| Sparkles | AutoAwesome |

## 📂 Klasör Yapısı Değişiklikleri

### Eski Yapı (Vite)
```
frontend/
├── src/
│   ├── components/
│   ├── pages/
│   ├── services/
│   └── App.tsx
└── vite.config.ts
```

### Yeni Yapı (Next.js)
```
frontend/
├── src/
│   ├── app/           # Next.js App Router
│   ├── components/    # React Components
│   ├── store/        # Redux Store
│   ├── hooks/        # Custom Hooks
│   └── lib/          # Utilities
└── next.config.ts
```

## 🚀 Çalıştırma Komutları

### Eski (Vite)
```bash
npm run dev    # Vite dev server
npm run build  # Vite build
```

### Yeni (Next.js)
```bash
npm run dev    # Next.js dev server (Turbopack)
npm run build  # Next.js production build
npm start      # Production server
```

## ⚠️ Dikkat Edilecek Noktalar

1. **Routing**: Next.js App Router kullanılıyor, `react-router-dom` kaldırıldı
2. **Image Optimization**: `next/image` kullanılmalı
3. **Environment Variables**: `.env` yerine `.env.local` kullanılıyor
4. **Client Components**: Tüm interaktif componentler `'use client'` direktifi ile başlamalı
5. **API Routes**: Backend API'ler `/api` klasöründe oluşturulabilir

## 🔧 Eksik Özellikler ve TODO

### Yüksek Öncelikli
- [ ] Öğretmen paneli sayfası
- [ ] Veli paneli sayfası
- [ ] Profil sayfası
- [ ] Ayarlar sayfası
- [ ] Değerlendirmeler (assessments) sayfası

### Orta Öncelikli
- [ ] Gamification sistemi tam entegrasyonu
- [ ] Socket.io gerçek zamanlı özellikler
- [ ] PWA konfigürasyonu
- [ ] Service Worker setup

### Düşük Öncelikli
- [ ] E2E testler (Cypress/Playwright)
- [ ] Storybook entegrasyonu
- [ ] Analytics entegrasyonu

## 🗑️ Kaldırılan Bağımlılıklar

- `vite`
- `@vitejs/plugin-react`
- `react-router-dom`
- `lucide-react` (Material UI Icons kullanılıyor)

## 📦 Eklenen Bağımlılıklar

- `next` (v15.5.0)
- `sharp` (image optimization)
- `@next/bundle-analyzer`
- `next-pwa`
- `swr`
- `zustand`

## 🔗 Faydalı Linkler

- [Next.js 15 Dokümantasyonu](https://nextjs.org/docs)
- [App Router Rehberi](https://nextjs.org/docs/app)
- [Material UI v5](https://mui.com/)
- [Redux Toolkit](https://redux-toolkit.js.org/)

## 📝 Notlar

- Eski Vite uygulaması `frontend-vite-backup` klasöründe saklanmaktadır
- Gerekirse eski koda referans için bakılabilir
- Tüm yeni geliştirmeler Next.js üzerinde yapılmalıdır

## ✨ Sonuç

Proje başarıyla Next.js 15'e geçirilmiştir. Modern, performanslı ve ölçeklenebilir bir yapıya sahiptir. Eksik özellikler TODO listesinde belirtilmiştir ve öncelik sırasına göre tamamlanmalıdır.
