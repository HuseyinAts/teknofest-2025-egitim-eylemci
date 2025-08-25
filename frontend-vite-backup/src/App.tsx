import React, { useState } from 'react';
import './App.css';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import AuthPage from './components/auth/AuthPage';
import StudentDashboard from './components/StudentDashboard';
import QuizInterface from './components/QuizInterface';
import LearningPathVisualization from './components/LearningPathVisualization';
import { LogOut, User, Settings, Bell } from 'lucide-react';

// Main App Component
function AppContent() {
  const { user, isAuthenticated, isLoading, logout } = useAuth();
  const [currentView, setCurrentView] = useState<'dashboard' | 'quiz' | 'learning-path'>('dashboard');
  const [showUserMenu, setShowUserMenu] = useState(false);

  const handleQuizComplete = (score: number, answers: any[]) => {
    console.log('Quiz completed with score:', score);
    console.log('Answers:', answers);
    // Navigate back to dashboard after quiz
    setTimeout(() => {
      setCurrentView('dashboard');
    }, 3000);
  };

  const handleLogout = async () => {
    await logout();
    setShowUserMenu(false);
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gradient-to-br from-blue-50 to-purple-50">
        <div className="text-center">
          <div className="w-16 h-16 border-4 border-blue-200 border-t-blue-500 rounded-full animate-spin mx-auto mb-4"></div>
          <p className="text-gray-600">Yükleniyor...</p>
        </div>
      </div>
    );
  }

  // If not authenticated, show auth page
  if (!isAuthenticated || !user) {
    return <AuthPage />;
  }

  return (
    <div className="App">
      {/* Navigation Bar */}
      <nav className="bg-white shadow-lg sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-4">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center gap-8">
              <div className="flex items-center gap-2">
                <span className="text-2xl">🎓</span>
                <h1 className="text-xl font-bold text-gray-800">TEKNOFEST 2025</h1>
              </div>
              
              <div className="hidden md:flex gap-4">
                <button
                  onClick={() => setCurrentView('dashboard')}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    currentView === 'dashboard' 
                      ? 'bg-blue-500 text-white' 
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  Dashboard
                </button>
                
                <button
                  onClick={() => setCurrentView('quiz')}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    currentView === 'quiz' 
                      ? 'bg-blue-500 text-white' 
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  Quiz
                </button>
                
                <button
                  onClick={() => setCurrentView('learning-path')}
                  className={`px-4 py-2 rounded-lg font-medium transition-all ${
                    currentView === 'learning-path' 
                      ? 'bg-blue-500 text-white' 
                      : 'text-gray-600 hover:bg-gray-100'
                  }`}
                >
                  Öğrenme Yolu
                </button>
              </div>
            </div>
            
            <div className="flex items-center gap-4">
              {/* Notification Bell */}
              <button className="relative p-2 text-gray-600 hover:bg-gray-100 rounded-lg transition-colors">
                <Bell className="w-5 h-5" />
                <span className="absolute top-1 right-1 w-2 h-2 bg-red-500 rounded-full"></span>
              </button>

              {/* User Menu */}
              <div className="relative">
                <button
                  onClick={() => setShowUserMenu(!showUserMenu)}
                  className="flex items-center gap-3 p-2 hover:bg-gray-100 rounded-lg transition-colors"
                >
                  <div className="text-right">
                    <p className="text-sm font-medium text-gray-800">{user.name}</p>
                    <p className="text-xs text-gray-500">
                      {user.role === 'student' 
                        ? `${user.grade || '10'}. Sınıf` 
                        : 'Öğretmen'}
                    </p>
                  </div>
                  
                  <div className="w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-500 rounded-full flex items-center justify-center text-white font-bold">
                    {user.name.split(' ').map(n => n[0]).join('').toUpperCase().slice(0, 2)}
                  </div>
                </button>

                {/* Dropdown Menu */}
                {showUserMenu && (
                  <div className="absolute right-0 mt-2 w-56 bg-white rounded-lg shadow-lg border border-gray-200 py-2">
                    <div className="px-4 py-3 border-b border-gray-200">
                      <p className="text-sm font-medium text-gray-800">{user.name}</p>
                      <p className="text-xs text-gray-500">{user.email}</p>
                    </div>
                    
                    <button className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 flex items-center gap-2">
                      <User className="w-4 h-4" />
                      Profilim
                    </button>
                    
                    <button className="w-full px-4 py-2 text-left text-sm text-gray-700 hover:bg-gray-100 flex items-center gap-2">
                      <Settings className="w-4 h-4" />
                      Ayarlar
                    </button>
                    
                    <div className="border-t border-gray-200 mt-2 pt-2">
                      <button
                        onClick={handleLogout}
                        className="w-full px-4 py-2 text-left text-sm text-red-600 hover:bg-red-50 flex items-center gap-2"
                      >
                        <LogOut className="w-4 h-4" />
                        Çıkış Yap
                      </button>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="min-h-screen">
        {currentView === 'dashboard' && <StudentDashboard />}
        {currentView === 'quiz' && (
          <QuizInterface 
            topic="Matematik"
            grade={user.grade || 10}
            onComplete={handleQuizComplete}
          />
        )}
        {currentView === 'learning-path' && (
          <LearningPathVisualization 
            subject="Matematik"
            grade={user.grade || 10}
            studentLevel={0.6}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="bg-gray-800 text-white py-8 mt-12">
        <div className="max-w-7xl mx-auto px-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            <div>
              <h3 className="text-lg font-bold mb-4">TEKNOFEST 2025</h3>
              <p className="text-gray-400 text-sm">
                Eğitim Teknolojileri Yarışması için geliştirilmiştir.
              </p>
            </div>
            
            <div>
              <h4 className="font-medium mb-3">Hızlı Linkler</h4>
              <ul className="space-y-2 text-sm text-gray-400">
                <li><a href="#" className="hover:text-white">Hakkımızda</a></li>
                <li><a href="#" className="hover:text-white">Özellikler</a></li>
                <li><a href="#" className="hover:text-white">Destek</a></li>
              </ul>
            </div>
            
            <div>
              <h4 className="font-medium mb-3">İletişim</h4>
              <p className="text-sm text-gray-400">
                teknofest2025@egitim.com<br />
                +90 XXX XXX XX XX
              </p>
            </div>
          </div>
          
          <div className="border-t border-gray-700 mt-8 pt-8 text-center text-sm text-gray-400">
            © 2025 TEKNOFEST Eğitim Platformu. Tüm hakları saklıdır.
          </div>
        </div>
      </footer>

      {/* Click outside to close menu */}
      {showUserMenu && (
        <div 
          className="fixed inset-0 z-40" 
          onClick={() => setShowUserMenu(false)}
        />
      )}
    </div>
  );
}

// Root App Component with AuthProvider
function App() {
  return (
    <AuthProvider>
      <AppContent />
    </AuthProvider>
  );
}

export default App;
