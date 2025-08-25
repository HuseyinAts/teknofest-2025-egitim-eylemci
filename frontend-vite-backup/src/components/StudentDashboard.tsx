import React, { useState, useEffect } from 'react';
import { 
  BookOpen, 
  Target, 
  TrendingUp, 
  Award, 
  Clock, 
  Calendar,
  Brain,
  Users,
  ChevronRight,
  Trophy,
  Zap,
  BarChart3,
  Play,
  CheckCircle
} from 'lucide-react';

const StudentDashboard = () => {
  const [activeTab, setActiveTab] = useState('overview');
  const [userStats, setUserStats] = useState({
    name: 'Öğrenci',
    grade: 10,
    totalXP: 2450,
    level: 5,
    streak: 7,
    completedLessons: 24,
    totalLessons: 40,
    averageScore: 85
  });

  const [todaysTasks, setTodaysTasks] = useState([
    { id: 1, title: 'Matematik: Türev Konusu', type: 'lesson', completed: false, xp: 50 },
    { id: 2, title: 'Fizik Quiz: Newton Yasaları', type: 'quiz', completed: false, xp: 100 },
    { id: 3, title: 'Kimya: Periyodik Tablo Alıştırması', type: 'practice', completed: true, xp: 30 },
    { id: 4, title: 'Biyoloji: Hücre Yapısı', type: 'lesson', completed: false, xp: 50 }
  ]);

  const [achievements, setAchievements] = useState([
    { id: 1, name: 'İlk Adım', icon: '🎯', unlocked: true, description: 'İlk dersini tamamla' },
    { id: 2, name: 'Hafta Savaşçısı', icon: '🔥', unlocked: true, description: '7 gün üst üste çalış' },
    { id: 3, name: 'Quiz Ustası', icon: '🏆', unlocked: false, description: '10 quiz\'de 90+ puan al' },
    { id: 4, name: 'Hız Şampiyonu', icon: '⚡', unlocked: false, description: 'Bir dersi 5 dakikada bitir' }
  ]);

  const [learningPath, setLearningPath] = useState([
    { id: 1, subject: 'Matematik', progress: 75, currentTopic: 'Türev ve İntegral', nextTopic: 'Limit' },
    { id: 2, subject: 'Fizik', progress: 60, currentTopic: 'Newton Yasaları', nextTopic: 'Enerji' },
    { id: 3, subject: 'Kimya', progress: 45, currentTopic: 'Periyodik Tablo', nextTopic: 'Kimyasal Bağlar' },
    { id: 4, subject: 'Biyoloji', progress: 80, currentTopic: 'Hücre', nextTopic: 'Genetik' }
  ]);

  const [weeklyProgress, setWeeklyProgress] = useState([
    { day: 'Pzt', score: 75, tasks: 5 },
    { day: 'Sal', score: 85, tasks: 6 },
    { day: 'Çar', score: 90, tasks: 7 },
    { day: 'Per', score: 70, tasks: 4 },
    { day: 'Cum', score: 95, tasks: 8 },
    { day: 'Cmt', score: 80, tasks: 5 },
    { day: 'Paz', score: 88, tasks: 6 }
  ]);

  const progressPercentage = (userStats.completedLessons / userStats.totalLessons) * 100;
  const xpToNextLevel = (userStats.level + 1) * 500 - userStats.totalXP;

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 p-4">
      {/* Header */}
      <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold text-gray-800">
              Merhaba, {userStats.name}! 👋
            </h1>
            <p className="text-gray-600 mt-2">
              {userStats.grade}. Sınıf Öğrencisi • Bugün {new Date().toLocaleDateString('tr-TR', { weekday: 'long', day: 'numeric', month: 'long' })}
            </p>
          </div>
          
          <div className="flex gap-4">
            {/* Streak Badge */}
            <div className="bg-orange-100 rounded-xl p-4 text-center">
              <div className="text-2xl mb-1">🔥</div>
              <div className="text-2xl font-bold text-orange-600">{userStats.streak}</div>
              <div className="text-xs text-gray-600">Gün Serisi</div>
            </div>
            
            {/* Level Badge */}
            <div className="bg-purple-100 rounded-xl p-4 text-center">
              <div className="text-2xl mb-1">⭐</div>
              <div className="text-2xl font-bold text-purple-600">Seviye {userStats.level}</div>
              <div className="text-xs text-gray-600">{xpToNextLevel} XP sonraki seviye</div>
            </div>
          </div>
        </div>

        {/* XP Progress Bar */}
        <div className="mt-6">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>XP İlerlemesi</span>
            <span>{userStats.totalXP} / {(userStats.level + 1) * 500} XP</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div 
              className="bg-gradient-to-r from-purple-500 to-blue-500 h-3 rounded-full transition-all duration-500"
              style={{ width: `${(userStats.totalXP % 500) / 5}%` }}
            />
          </div>
        </div>
      </div>

      {/* Navigation Tabs */}
      <div className="bg-white rounded-xl shadow-md p-2 mb-6 flex gap-2">
        {['overview', 'learning', 'quizzes', 'achievements'].map((tab) => (
          <button
            key={tab}
            onClick={() => setActiveTab(tab)}
            className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${
              activeTab === tab 
                ? 'bg-blue-500 text-white' 
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            {tab === 'overview' && 'Genel Bakış'}
            {tab === 'learning' && 'Öğrenme Yolu'}
            {tab === 'quizzes' && 'Sınavlar'}
            {tab === 'achievements' && 'Başarılar'}
          </button>
        ))}
      </div>

      {/* Main Content Area */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Main Content */}
        <div className="lg:col-span-2 space-y-6">
          {activeTab === 'overview' && (
            <>
              {/* Today's Tasks */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <div className="flex items-center justify-between mb-4">
                  <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
                    <Target className="text-blue-500" />
                    Bugünkü Görevler
                  </h2>
                  <span className="text-sm text-gray-500">
                    {todaysTasks.filter(t => t.completed).length}/{todaysTasks.length} tamamlandı
                  </span>
                </div>
                
                <div className="space-y-3">
                  {todaysTasks.map((task) => (
                    <div 
                      key={task.id}
                      className={`border rounded-lg p-4 flex items-center justify-between transition-all hover:shadow-md ${
                        task.completed ? 'bg-green-50 border-green-200' : 'bg-white border-gray-200'
                      }`}
                    >
                      <div className="flex items-center gap-3">
                        <div className={`w-10 h-10 rounded-full flex items-center justify-center ${
                          task.completed ? 'bg-green-500' : 'bg-gray-200'
                        }`}>
                          {task.completed ? (
                            <CheckCircle className="w-6 h-6 text-white" />
                          ) : (
                            task.type === 'lesson' ? <BookOpen className="w-5 h-5 text-gray-600" /> :
                            task.type === 'quiz' ? <Brain className="w-5 h-5 text-gray-600" /> :
                            <Play className="w-5 h-5 text-gray-600" />
                          )}
                        </div>
                        <div>
                          <h3 className={`font-medium ${task.completed ? 'text-gray-500 line-through' : 'text-gray-800'}`}>
                            {task.title}
                          </h3>
                          <p className="text-sm text-gray-500">+{task.xp} XP</p>
                        </div>
                      </div>
                      {!task.completed && (
                        <button className="bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 transition-colors">
                          Başla
                        </button>
                      )}
                    </div>
                  ))}
                </div>
              </div>

              {/* Weekly Performance */}
              <div className="bg-white rounded-2xl shadow-lg p-6">
                <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2 mb-4">
                  <BarChart3 className="text-green-500" />
                  Haftalık Performans
                </h2>
                
                <div className="flex items-end justify-between h-40 mb-2">
                  {weeklyProgress.map((day) => (
                    <div key={day.day} className="flex flex-col items-center flex-1">
                      <div className="text-xs text-gray-500 mb-1">{day.score}%</div>
                      <div 
                        className="w-12 bg-gradient-to-t from-blue-500 to-purple-500 rounded-t-lg transition-all hover:opacity-80"
                        style={{ height: `${(day.score / 100) * 120}px` }}
                      />
                      <div className="text-sm font-medium text-gray-600 mt-2">{day.day}</div>
                      <div className="text-xs text-gray-400">{day.tasks} görev</div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {activeTab === 'learning' && (
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2 mb-4">
                <BookOpen className="text-indigo-500" />
                Öğrenme Yolun
              </h2>
              
              <div className="space-y-4">
                {learningPath.map((subject) => (
                  <div key={subject.id} className="border rounded-lg p-4 hover:shadow-md transition-all">
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <h3 className="font-bold text-lg text-gray-800">{subject.subject}</h3>
                        <p className="text-sm text-gray-600">
                          Mevcut: {subject.currentTopic} → Sonraki: {subject.nextTopic}
                        </p>
                      </div>
                      <span className="text-2xl font-bold text-blue-600">{subject.progress}%</span>
                    </div>
                    
                    <div className="w-full bg-gray-200 rounded-full h-3">
                      <div 
                        className="bg-gradient-to-r from-indigo-500 to-purple-500 h-3 rounded-full transition-all"
                        style={{ width: `${subject.progress}%` }}
                      />
                    </div>
                    
                    <button className="mt-3 text-blue-600 font-medium text-sm hover:text-blue-700 flex items-center gap-1">
                      Devam Et <ChevronRight className="w-4 h-4" />
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {activeTab === 'achievements' && (
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2 mb-4">
                <Trophy className="text-yellow-500" />
                Başarılarım
              </h2>
              
              <div className="grid grid-cols-2 gap-4">
                {achievements.map((achievement) => (
                  <div 
                    key={achievement.id}
                    className={`border-2 rounded-xl p-4 text-center transition-all ${
                      achievement.unlocked 
                        ? 'border-yellow-400 bg-yellow-50' 
                        : 'border-gray-200 bg-gray-50 opacity-50'
                    }`}
                  >
                    <div className="text-4xl mb-2">{achievement.icon}</div>
                    <h3 className="font-bold text-gray-800">{achievement.name}</h3>
                    <p className="text-xs text-gray-600 mt-1">{achievement.description}</p>
                    {achievement.unlocked && (
                      <div className="mt-2 text-xs text-yellow-600 font-medium">✓ Kazanıldı</div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Column - Stats & Quick Actions */}
        <div className="space-y-6">
          {/* Quick Stats */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-lg font-bold text-gray-800 mb-4">İstatistikler</h2>
            
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Tamamlanan Ders</span>
                <span className="font-bold text-gray-800">{userStats.completedLessons}/{userStats.totalLessons}</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Ortalama Puan</span>
                <span className="font-bold text-green-600">{userStats.averageScore}%</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Toplam XP</span>
                <span className="font-bold text-purple-600">{userStats.totalXP}</span>
              </div>
              
              <div className="flex items-center justify-between">
                <span className="text-gray-600">Aktif Gün</span>
                <span className="font-bold text-orange-600">{userStats.streak} 🔥</span>
              </div>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-lg font-bold text-gray-800 mb-4">Hızlı Eylemler</h2>
            
            <div className="space-y-3">
              <button className="w-full bg-gradient-to-r from-blue-500 to-purple-500 text-white rounded-lg py-3 font-medium hover:shadow-lg transition-all">
                🎯 Günlük Quiz'e Başla
              </button>
              
              <button className="w-full bg-gradient-to-r from-green-500 to-teal-500 text-white rounded-lg py-3 font-medium hover:shadow-lg transition-all">
                📚 Yeni Ders Seç
              </button>
              
              <button className="w-full bg-gradient-to-r from-orange-500 to-red-500 text-white rounded-lg py-3 font-medium hover:shadow-lg transition-all">
                🏆 Liderlik Tablosu
              </button>
            </div>
          </div>

          {/* Study Buddy */}
          <div className="bg-gradient-to-br from-purple-100 to-blue-100 rounded-2xl shadow-lg p-6">
            <div className="flex items-center gap-3 mb-3">
              <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center">
                <Zap className="w-6 h-6 text-purple-600" />
              </div>
              <div>
                <h3 className="font-bold text-gray-800">AI Çalışma Arkadaşı</h3>
                <p className="text-xs text-gray-600">Sorularını yanıtlamaya hazır!</p>
              </div>
            </div>
            
            <button className="w-full bg-white text-purple-600 rounded-lg py-2 font-medium hover:shadow-md transition-all">
              Sohbete Başla
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default StudentDashboard;