import type { Metadata, Viewport } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Providers } from "./providers";
import { Toaster } from 'react-hot-toast';
import ErrorBoundary from '@/components/ErrorBoundary';

const inter = Inter({ 
  subsets: ["latin"],
  display: 'swap',
  preload: true,
  fallback: ['system-ui', 'arial']
});

export const metadata: Metadata = {
  title: "Teknofest 2025 - Eğitim Platformu",
  description: "Yapay Zeka Destekli Kişiselleştirilmiş Eğitim Platformu",
  keywords: "teknofest, eğitim, yapay zeka, öğrenme, kişiselleştirilmiş eğitim",
  authors: [{ name: "Teknofest Team" }],
  manifest: '/manifest.json',
  robots: 'index, follow',
  openGraph: {
    title: "Teknofest 2025 - Eğitim Platformu",
    description: "Yapay Zeka Destekli Kişiselleştirilmiş Eğitim Platformu",
    type: "website",
    locale: 'tr_TR',
    siteName: 'Teknofest 2025',
  },
  twitter: {
    card: 'summary_large_image',
    title: "Teknofest 2025 - Eğitim Platformu",
    description: "Yapay Zeka Destekli Kişiselleştirilmiş Eğitim Platformu",
  },
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5,
  userScalable: true,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0a0a0a' }
  ],
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="tr" suppressHydrationWarning>
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="dns-prefetch" href="https://fonts.googleapis.com" />
      </head>
      <body className={inter.className}>
        <a href="#main-content" className="skip-to-content">
          Ana içeriğe geç
        </a>
        <ErrorBoundary>
          <Providers>
            <main id="main-content" role="main">
              {children}
            </main>
            <Toaster
              position="top-right"
              toastOptions={{
                duration: 4000,
                style: {
                  background: '#363636',
                  color: '#fff',
                  fontSize: '14px',
                  borderRadius: '8px',
                  padding: '12px 16px',
                },
                success: {
                  style: {
                    background: '#10b981',
                  },
                  iconTheme: {
                    primary: '#fff',
                    secondary: '#10b981',
                  },
                },
                error: {
                  style: {
                    background: '#ef4444',
                  },
                  iconTheme: {
                    primary: '#fff',
                    secondary: '#ef4444',
                  },
                },
                loading: {
                  style: {
                    background: '#6366f1',
                  },
                },
              }}
              containerStyle={{
                top: 20,
                right: 20,
              }}
            />
          </Providers>
        </ErrorBoundary>
      </body>
    </html>
  );
}
