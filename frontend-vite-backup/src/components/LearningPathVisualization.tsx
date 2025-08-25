import React, { useState, useEffect } from 'react';
import { 
  BookOpen, 
  CheckCircle, 
  Circle, 
  Lock, 
  Star, 
  TrendingUp,
  Clock,
  Target,
  Award,
  ChevronRight,
  Zap,
  Brain,
  Sparkles
} from 'lucide-react';

interface Topic {
  id: number;
  title: string;
  description: string;
  status: 'completed' | 'current' | 'locked';
  progress: number;
  estimatedTime: number;
  difficulty: number;
  xp: number;
  prerequisites: number[];
  subtopics: string[];
}

interface LearningPathProps {
  subject?: string;
  grade?: number;
  studentLevel?: number;
}

const LearningPathVisualization: React.FC<LearningPathProps> = ({
  subject = "Matematik",
  grade = 10,
  studentLevel = 0.6
}) => {
  const [topics, setTopics] = useState<Topic[]>([
    {
      id: 1,
      title: "Sayılar ve İşlemler",
      description: "Temel sayı sistemleri, işlemler ve sayı kümeleri",
      status: 'completed',
      progress: 100,
      estimatedTime: 120,
      difficulty: 0.3,
      xp: 100,
      prerequisites: [],
      subtopics: ["Doğal Sayılar", "Tam Sayılar", "Rasyonel Sayılar", "Gerçek Sayılar"]
    },
    {
      id: 2,
      title: "Cebirsel İfadeler",
      description: "Polinomlar, çarpanlara ayırma ve cebirsel denklemler",
      status: 'completed',
      progress: 100,
      estimatedTime: 180,
      difficulty: 0.4,
      xp: 150,
      prerequisites: [1],
      subtopics: ["Polinomlar", "Çarpanlara Ayırma", "Özdeşlikler", "Denklemler"]
    },
    {
      id: 3,
      title: "Fonksiyonlar",
      description: "Fonksiyon kavramı, türleri ve grafikleri",
      status: 'completed',
      progress: 100,
      estimatedTime: 240,
      difficulty: 0.5,
      xp: 200,
      prerequisites: [2],
      subtopics: ["Fonksiyon Tanımı", "Doğrusal Fonksiyonlar", "Paraboller", "Trigonometrik Fonksiyonlar"]
    },
    {
      id: 4,
      title: "Limit ve Süreklilik",
      description: "Limit kavramı, limit alma kuralları ve süreklilik",
      status: 'current',
      progress: 65,
      estimatedTime: 200,
      difficulty: 0.6,
      xp: 250,
      prerequisites: [3],
      subtopics: ["Limit Tanımı", "Limit Kuralları", "Süreklilik", "Süreksizlik Türleri"]
    },
    {
      id: 5,
      title: "Türev",
      description: "Türev alma kuralları ve uygulamaları",
      status: 'locked',
      progress: 0,
      estimatedTime: 300,
      difficulty: 0.7,
      xp: 350,
      prerequisites: [4],
      subtopics: ["Türev Tanımı", "Türev Kuralları", "Zincir Kuralı", "Türev Uygulamaları"]
    },
    {
      id: 6,
      title: "İntegral",
      description: "Belirsiz ve belirli integral, integral uygulamaları",
      status: 'locked',
      progress: 0,
      estimatedTime: 350,
      difficulty: 0.8,
      xp: 400,
      prerequisites: [5],
      subtopics: ["Belirsiz İntegral", "Belirli İntegral", "Alan Hesabı", "Hacim Hesabı"]
    },
    {
      id: 7,
      title: "Diferansiyel Denklemler",
      description: "Birinci ve ikinci dereceden diferansiyel denklemler",
      status: 'locked',
      progress: 0,
      estimatedTime: 280,
      difficulty: 0.9,
      xp: 500,
      prerequisites: [6],
      subtopics: ["Birinci Derece", "İkinci Derece", "Uygulamalar"]
    }
  ]);

  const [selectedTopic, setSelectedTopic] = useState<Topic | null>(topics[3]);
  const [viewMode, setViewMode] = useState<'path' | 'grid' | 'tree'>('path');
  const [showRecommendation, setShowRecommendation] = useState(true);

  const totalXP = topics.reduce((sum, topic) => 
    topic.status === 'completed' ? sum + topic.xp : sum, 0
  );
  
  const completedCount = topics.filter(t => t.status === 'completed').length;
  const totalProgress = (completedCount / topics.length) * 100;

  const getDifficultyColor = (difficulty: number) => {
    if (difficulty < 0.33) return 'text-green-500';
    if (difficulty < 0.67) return 'text-yellow-500';
    return 'text-red-500';
  };

  const getDifficultyText = (difficulty: number) => {
    if (difficulty < 0.33) return 'Kolay';
    if (difficulty < 0.67) return 'Orta';
    return 'Zor';
  };

  const getStatusIcon = (status: string) => {
    switch(status) {
      case 'completed':
        return <CheckCircle className="w-6 h-6 text-green-500" />;
      case 'current':
        return <Circle className="w-6 h-6 text-blue-500 animate-pulse" />;
      case 'locked':
        return <Lock className="w-6 h-6 text-gray-400" />;
      default:
        return <Circle className="w-6 h-6 text-gray-400" />;
    }
  };

  const getRecommendedNext = () => {
    const current = topics.find(t => t.status === 'current');
    if (current && current.progress >= 80) {
      const next = topics.find(t => t.prerequisites.includes(current.id));
      return next;
    }
    return null;
  };

  const recommendedNext = getRecommendedNext();

  return (
    <div className="min-h-screen bg-gradient-to-br from-indigo-50 via-purple-50 to-pink-50 p-4">
      {/* Header */}
      <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
        <div className="flex justify-between items-start">
          <div>
            <h1 className="text-3xl font-bold text-gray-800 mb-2">
              {subject} Öğrenme Yolu
            </h1>
            <p className="text-gray-600">{grade}. Sınıf • Öğrenci Seviyesi: {(studentLevel * 100).toFixed(0)}%</p>
          </div>
          
          <div className="flex gap-4">
            <div className="bg-purple-100 rounded-xl p-4 text-center">
              <Award className="w-8 h-8 text-purple-600 mx-auto mb-1" />
              <div className="text-2xl font-bold text-purple-800">{totalXP}</div>
              <div className="text-xs text-gray-600">Toplam XP</div>
            </div>
            
            <div className="bg-blue-100 rounded-xl p-4 text-center">
              <Target className="w-8 h-8 text-blue-600 mx-auto mb-1" />
              <div className="text-2xl font-bold text-blue-800">{completedCount}/{topics.length}</div>
              <div className="text-xs text-gray-600">Tamamlanan</div>
            </div>
          </div>
        </div>

        {/* Overall Progress */}
        <div className="mt-6">
          <div className="flex justify-between text-sm text-gray-600 mb-2">
            <span>Genel İlerleme</span>
            <span>{totalProgress.toFixed(0)}%</span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-3">
            <div 
              className="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 h-3 rounded-full transition-all duration-500"
              style={{ width: `${totalProgress}%` }}
            />
          </div>
        </div>
      </div>

      {/* View Mode Selector */}
      <div className="bg-white rounded-xl shadow-md p-2 mb-6 flex gap-2">
        {['path', 'grid', 'tree'].map((mode) => (
          <button
            key={mode}
            onClick={() => setViewMode(mode as any)}
            className={`flex-1 py-3 px-4 rounded-lg font-medium transition-all ${
              viewMode === mode 
                ? 'bg-gradient-to-r from-indigo-500 to-purple-500 text-white' 
                : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            {mode === 'path' && '📍 Yol Görünümü'}
            {mode === 'grid' && '⊞ Izgara Görünümü'}
            {mode === 'tree' && '🌳 Ağaç Görünümü'}
          </button>
        ))}
      </div>

      {/* AI Recommendation */}
      {showRecommendation && recommendedNext && (
        <div className="bg-gradient-to-r from-purple-100 to-blue-100 rounded-2xl shadow-lg p-6 mb-6 relative">
          <button 
            onClick={() => setShowRecommendation(false)}
            className="absolute top-4 right-4 text-gray-500 hover:text-gray-700"
          >
            ✕
          </button>
          
          <div className="flex items-start gap-4">
            <div className="w-12 h-12 bg-white rounded-full flex items-center justify-center">
              <Sparkles className="w-6 h-6 text-purple-600" />
            </div>
            <div className="flex-1">
              <h3 className="font-bold text-gray-800 mb-1">AI Önerisi</h3>
              <p className="text-gray-700">
                Mevcut konudaki ilerlemeniz %{topics.find(t => t.status === 'current')?.progress} seviyesinde. 
                Yakında <span className="font-bold">{recommendedNext.title}</span> konusuna geçmeye hazır olacaksınız!
              </p>
              <button className="mt-3 text-purple-600 font-medium hover:text-purple-700 flex items-center gap-1">
                Hazırlık Önerileri <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left Column - Path Visualization */}
        <div className="lg:col-span-2">
          {viewMode === 'path' && (
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-6 flex items-center gap-2">
                <TrendingUp className="text-indigo-500" />
                Öğrenme Rotası
              </h2>
              
              <div className="relative">
                {/* Connection Lines */}
                <div className="absolute left-8 top-0 bottom-0 w-0.5 bg-gray-200"></div>
                
                {/* Topics */}
                <div className="space-y-6">
                  {topics.map((topic, index) => (
                    <div key={topic.id} className="relative">
                      {/* Connection Line Highlight */}
                      {topic.status === 'completed' && index < topics.length - 1 && (
                        <div className="absolute left-8 top-12 h-24 w-0.5 bg-green-500"></div>
                      )}
                      
                      <div 
                        onClick={() => setSelectedTopic(topic)}
                        className={`flex gap-4 p-4 rounded-xl cursor-pointer transition-all ${
                          selectedTopic?.id === topic.id 
                            ? 'bg-indigo-50 border-2 border-indigo-300' 
                            : 'hover:bg-gray-50'
                        } ${topic.status === 'locked' ? 'opacity-50' : ''}`}
                      >
                        {/* Status Icon */}
                        <div className="relative z-10 bg-white rounded-full p-2 shadow-md">
                          {getStatusIcon(topic.status)}
                        </div>
                        
                        {/* Content */}
                        <div className="flex-1">
                          <div className="flex items-start justify-between">
                            <div>
                              <h3 className="font-bold text-lg text-gray-800">{topic.title}</h3>
                              <p className="text-sm text-gray-600 mt-1">{topic.description}</p>
                              
                              {/* Subtopics */}
                              <div className="flex flex-wrap gap-2 mt-2">
                                {topic.subtopics.slice(0, 3).map((subtopic, idx) => (
                                  <span key={idx} className="text-xs bg-gray-100 text-gray-600 px-2 py-1 rounded-full">
                                    {subtopic}
                                  </span>
                                ))}
                                {topic.subtopics.length > 3 && (
                                  <span className="text-xs text-gray-500">+{topic.subtopics.length - 3} daha</span>
                                )}
                              </div>
                            </div>
                            
                            {/* Stats */}
                            <div className="text-right">
                              <div className="flex items-center gap-2 text-sm">
                                <Clock className="w-4 h-4 text-gray-400" />
                                <span className="text-gray-600">{topic.estimatedTime} dk</span>
                              </div>
                              <div className="flex items-center gap-2 text-sm mt-1">
                                <Zap className="w-4 h-4 text-yellow-500" />
                                <span className="font-bold text-purple-600">+{topic.xp} XP</span>
                              </div>
                              <div className={`text-sm mt-1 font-medium ${getDifficultyColor(topic.difficulty)}`}>
                                {getDifficultyText(topic.difficulty)}
                              </div>
                            </div>
                          </div>
                          
                          {/* Progress Bar */}
                          {topic.status !== 'locked' && (
                            <div className="mt-3">
                              <div className="w-full bg-gray-200 rounded-full h-2">
                                <div 
                                  className={`h-2 rounded-full transition-all duration-500 ${
                                    topic.status === 'completed' 
                                      ? 'bg-green-500' 
                                      : 'bg-gradient-to-r from-blue-500 to-purple-500'
                                  }`}
                                  style={{ width: `${topic.progress}%` }}
                                />
                              </div>
                              <span className="text-xs text-gray-500 mt-1">{topic.progress}% tamamlandı</span>
                            </div>
                          )}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          )}

          {viewMode === 'grid' && (
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h2 className="text-xl font-bold text-gray-800 mb-6">Konu Kartları</h2>
              
              <div className="grid grid-cols-2 gap-4">
                {topics.map((topic) => (
                  <div 
                    key={topic.id}
                    onClick={() => setSelectedTopic(topic)}
                    className={`border-2 rounded-xl p-4 cursor-pointer transition-all ${
                      topic.status === 'completed' ? 'border-green-400 bg-green-50' :
                      topic.status === 'current' ? 'border-blue-400 bg-blue-50' :
                      'border-gray-200 bg-gray-50 opacity-50'
                    } ${selectedTopic?.id === topic.id ? 'ring-2 ring-indigo-500' : ''}`}
                  >
                    <div className="flex items-start justify-between mb-3">
                      {getStatusIcon(topic.status)}
                      <span className="text-lg font-bold text-purple-600">+{topic.xp}</span>
                    </div>
                    
                    <h3 className="font-bold text-gray-800 mb-1">{topic.title}</h3>
                    <p className="text-xs text-gray-600 mb-3 line-clamp-2">{topic.description}</p>
                    
                    <div className="flex items-center justify-between text-xs">
                      <span className="text-gray-500">{topic.estimatedTime} dk</span>
                      <span className={`font-medium ${getDifficultyColor(topic.difficulty)}`}>
                        {getDifficultyText(topic.difficulty)}
                      </span>
                    </div>
                    
                    {topic.status !== 'locked' && (
                      <div className="mt-3">
                        <div className="w-full bg-gray-200 rounded-full h-1.5">
                          <div 
                            className={`h-1.5 rounded-full ${
                              topic.status === 'completed' ? 'bg-green-500' : 'bg-blue-500'
                            }`}
                            style={{ width: `${topic.progress}%` }}
                          />
                        </div>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Right Column - Topic Details */}
        <div className="space-y-6">
          {selectedTopic && (
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h2 className="text-lg font-bold text-gray-800 mb-4">Konu Detayları</h2>
              
              <div className="space-y-4">
                <div>
                  <h3 className="font-bold text-xl text-gray-800">{selectedTopic.title}</h3>
                  <p className="text-gray-600 mt-1">{selectedTopic.description}</p>
                </div>
                
                <div className="border-t pt-4">
                  <h4 className="font-medium text-gray-700 mb-2">Alt Konular</h4>
                  <div className="space-y-2">
                    {selectedTopic.subtopics.map((subtopic, idx) => (
                      <div key={idx} className="flex items-center gap-2">
                        <div className={`w-4 h-4 rounded-full ${
                          selectedTopic.status === 'completed' ? 'bg-green-500' :
                          selectedTopic.status === 'current' && idx === 0 ? 'bg-blue-500' :
                          'bg-gray-300'
                        }`} />
                        <span className="text-sm text-gray-700">{subtopic}</span>
                      </div>
                    ))}
                  </div>
                </div>
                
                <div className="border-t pt-4 space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-600">Zorluk</span>
                    <span className={`font-bold ${getDifficultyColor(selectedTopic.difficulty)}`}>
                      {getDifficultyText(selectedTopic.difficulty)}
                    </span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Tahmini Süre</span>
                    <span className="font-bold text-gray-800">{selectedTopic.estimatedTime} dakika</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">Kazanılacak XP</span>
                    <span className="font-bold text-purple-600">+{selectedTopic.xp} XP</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-600">İlerleme</span>
                    <span className="font-bold text-blue-600">{selectedTopic.progress}%</span>
                  </div>
                </div>
                
                {selectedTopic.status === 'current' && (
                  <button className="w-full bg-gradient-to-r from-blue-500 to-purple-500 text-white py-3 rounded-lg font-medium hover:shadow-lg transition-all">
                    Öğrenmeye Devam Et
                  </button>
                )}
                
                {selectedTopic.status === 'locked' && (
                  <div className="bg-gray-100 rounded-lg p-3">
                    <p className="text-sm text-gray-600 text-center">
                      🔒 Bu konuyu açmak için önce önceki konuları tamamlayın
                    </p>
                  </div>
                )}
                
                {selectedTopic.status === 'completed' && (
                  <button className="w-full bg-green-100 text-green-700 py-3 rounded-lg font-medium hover:bg-green-200 transition-all">
                    ✓ Tamamlandı - Tekrar Et
                  </button>
                )}
              </div>
            </div>
          )}
          
          {/* Learning Tips */}
          <div className="bg-gradient-to-br from-yellow-50 to-orange-50 rounded-2xl shadow-lg p-6">
            <div className="flex items-center gap-3 mb-3">
              <Brain className="w-8 h-8 text-orange-500" />
              <h3 className="font-bold text-gray-800">Öğrenme İpucu</h3>
            </div>
            
            <p className="text-sm text-gray-700">
              {selectedTopic?.status === 'current' 
                ? "Bu konuyu tamamlamak üzeresiniz! Düzenli tekrar yaparak kalıcı öğrenmeyi sağlayabilirsiniz."
                : "Her konu bir sonraki için temel oluşturur. Sıralı ilerleme başarınızı artırır."}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LearningPathVisualization;