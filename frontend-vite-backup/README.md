# TEKNOFEST 2025 - Eğitim Teknolojileri Frontend

Production-ready React frontend application for the TEKNOFEST 2025 Education Technologies platform.

## 🚀 Features

- **Modern React 18** with TypeScript
- **Vite** for fast development and optimized builds
- **Material-UI** for consistent, responsive design
- **Redux Toolkit** for state management
- **React Query** for server state management
- **React Router v6** for navigation
- **i18n** support (Turkish/English)
- **WebSocket** support for real-time features
- **PWA** ready
- **Docker** containerization
- **Nginx** for production deployment

## 📋 Prerequisites

- Node.js 20+
- npm or yarn
- Docker (optional)

## 🛠️ Installation

```bash
# Install dependencies
npm install

# Copy environment variables
cp .env.example .env
```

## 🔧 Development

```bash
# Start development server
npm run dev

# Run tests
npm test

# Run linter
npm run lint

# Type checking
npm run type-check

# Format code
npm run format
```

## 🏗️ Build

```bash
# Build for production
npm run build

# Preview production build
npm run preview

# Analyze bundle size
npm run analyze
```

## 🐳 Docker

```bash
# Build Docker image
docker build -t teknofest-frontend .

# Run with Docker
docker run -p 3000:80 teknofest-frontend

# Run with Docker Compose
docker-compose up frontend
```

## 📁 Project Structure

```
frontend/
├── src/
│   ├── components/     # Reusable UI components
│   ├── pages/          # Page components
│   ├── services/       # API services
│   ├── store/          # Redux store and slices
│   ├── hooks/          # Custom React hooks
│   ├── utils/          # Utility functions
│   ├── contexts/       # React contexts
│   ├── layouts/        # Layout components
│   ├── assets/         # Static assets
│   ├── i18n/           # Internationalization
│   └── types/          # TypeScript types
├── public/             # Public assets
├── config/             # Configuration files
└── tests/              # Test files
```

## 🔐 Environment Variables

```env
VITE_API_URL=http://localhost:8000
VITE_WS_URL=ws://localhost:8000
VITE_APP_NAME=TEKNOFEST 2025 Eğitim Teknolojileri
VITE_ENVIRONMENT=development
```

## 🚀 Production Deployment

1. Set production environment variables
2. Build the application: `npm run build`
3. Deploy `dist/` folder to your web server
4. Configure nginx with provided `nginx.conf`

## 📊 Performance Optimizations

- Code splitting with React.lazy()
- Route-based chunking
- Image lazy loading
- Bundle size optimization
- Gzip compression
- Browser caching
- Service Worker for offline support

## 🔒 Security

- Content Security Policy headers
- XSS protection
- HTTPS enforcement
- Secure authentication with JWT
- Input validation
- Rate limiting on API calls

## 🧪 Testing

- Unit tests with Vitest
- Component testing with React Testing Library
- E2E tests (coming soon)
- Coverage reports

## 📝 License

MIT