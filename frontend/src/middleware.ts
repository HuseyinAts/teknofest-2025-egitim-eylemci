import { NextResponse } from 'next/server';
import type { NextRequest } from 'next/server';

export function middleware(request: NextRequest) {
  const { pathname } = request.nextUrl;
  const token = request.cookies.get('access_token')?.value;
  
  const isAuthPage = pathname.startsWith('/login') || pathname.startsWith('/register');
  const isProtectedRoute = pathname.startsWith('/dashboard') ||
                           pathname.startsWith('/learning-paths') ||
                           pathname.startsWith('/assessments') ||
                           pathname.startsWith('/study-session') ||
                           pathname.startsWith('/profile') ||
                           pathname.startsWith('/settings');
  
  // Create response
  const response = NextResponse.next();
  
  // Security headers
  response.headers.set('X-Frame-Options', 'DENY');
  response.headers.set('X-Content-Type-Options', 'nosniff');
  response.headers.set('Referrer-Policy', 'strict-origin-when-cross-origin');
  response.headers.set('X-XSS-Protection', '1; mode=block');
  response.headers.set(
    'Permissions-Policy',
    'camera=(), microphone=(), geolocation=()'
  );
  
  // Simple CSP for production
  if (process.env.NODE_ENV === 'production') {
    response.headers.set(
      'Content-Security-Policy',
      "default-src 'self'; script-src 'self' 'unsafe-eval' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' blob: data: https:; font-src 'self' data: https://fonts.gstatic.com; connect-src 'self' http://localhost:8000 ws://localhost:8000 https://api.teknofest.com;"
    );
  }

  // Authentication redirects
  if (isProtectedRoute && !token) {
    const loginUrl = new URL('/login', request.url);
    loginUrl.searchParams.set('redirect', pathname);
    return NextResponse.redirect(loginUrl);
  }

  if (isAuthPage && token) {
    return NextResponse.redirect(new URL('/dashboard', request.url));
  }
  
  // Language preference
  const acceptLanguage = request.headers.get('accept-language');
  const savedLanguage = request.cookies.get('language')?.value;
  
  if (!savedLanguage && acceptLanguage) {
    const preferredLanguage = acceptLanguage.includes('tr') ? 'tr' : 'en';
    response.cookies.set('language', preferredLanguage, {
      httpOnly: false,
      secure: process.env.NODE_ENV === 'production',
      sameSite: 'lax',
      maxAge: 60 * 60 * 24 * 365,
    });
  }

  return response;
}

export const config = {
  matcher: [
    '/dashboard/:path*',
    '/learning-paths/:path*',
    '/assessments/:path*',
    '/study-session/:path*',
    '/profile/:path*',
    '/login',
    '/register',
  ],
};