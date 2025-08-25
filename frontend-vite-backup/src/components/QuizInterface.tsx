import React, { useState, useEffect, useRef } from 'react';
import './QuizInterface.css';

interface Question {
  id: number;
  text: string;
  options: string[];
  correctAnswer?: string;
  difficulty: number;
  topic: string;
}

interface QuizInterfaceProps {
  topic?: string;
  grade?: number;
  onComplete?: (score: number, answers: any[]) => void;
}

const QuizInterface: React.FC<QuizInterfaceProps> = ({ 
  topic = "Matematik", 
  grade = 10,
  onComplete 
}) => {
  const [questions, setQuestions] = useState<Question[]>([
    {
      id: 1,
      text: "Bir fonksiyonun türevi f'(x) = 3x² + 2x ise, f(x) fonksiyonu aşağıdakilerden hangisi olabilir?",
      options: ["x³ + x² + 5", "x³ + x² + C", "3x³ + 2x²", "x³ + 2x"],
      correctAnswer: "x³ + x² + C",
      difficulty: 0.6,
      topic: "Türev"
    },
    {
      id: 2,
      text: "lim(x→0) (sin x)/x limitinin değeri kaçtır?",
      options: ["0", "1", "∞", "Tanımsız"],
      correctAnswer: "1",
      difficulty: 0.5,
      topic: "Limit"
    },
    {
      id: 3,
      text: "∫(2x + 3)dx integralinin sonucu nedir?",
      options: ["x² + 3x + C", "2x² + 3x", "x² + 3", "2x + 3x + C"],
      correctAnswer: "x² + 3x + C",
      difficulty: 0.4,
      topic: "İntegral"
    },
    {
      id: 4,
      text: "f(x) = x³ - 3x² + 2 fonksiyonunun yerel maksimum noktası nerede bulunur?",
      options: ["x = 0", "x = 1", "x = 2", "x = -1"],
      correctAnswer: "x = 0",
      difficulty: 0.7,
      topic: "Ekstremum"
    },
    {
      id: 5,
      text: "Hangi fonksiyon sürekli ama türevlenemez?",
      options: ["|x|", "x²", "sin(x)", "e^x"],
      correctAnswer: "|x|",
      difficulty: 0.8,
      topic: "Süreklilik"
    }
  ]);

  const [currentQuestion, setCurrentQuestion] = useState(0);
  const [selectedAnswers, setSelectedAnswers] = useState<string[]>([]);
  const [showResults, setShowResults] = useState(false);
  const [timeLeft, setTimeLeft] = useState(300); // 5 minutes
  const [isAnswered, setIsAnswered] = useState(false);
  const [score, setScore] = useState(0);
  const [showFeedback, setShowFeedback] = useState(false);
  const timerRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    timerRef.current = setInterval(() => {
      setTimeLeft(prev => {
        if (prev <= 1) {
          handleQuizComplete();
          return 0;
        }
        return prev - 1;
      });
    }, 1000);

    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, []);

  const handleAnswerSelect = (answer: string) => {
    if (isAnswered) return;
    
    const newAnswers = [...selectedAnswers];
    newAnswers[currentQuestion] = answer;
    setSelectedAnswers(newAnswers);
    setIsAnswered(true);
    setShowFeedback(true);

    // Check if answer is correct
    if (answer === questions[currentQuestion].correctAnswer) {
      setScore(prev => prev + 1);
    }

    // Auto proceed after 2 seconds
    setTimeout(() => {
      if (currentQuestion < questions.length - 1) {
        handleNextQuestion();
      } else {
        handleQuizComplete();
      }
    }, 2000);
  };

  const handleNextQuestion = () => {
    setCurrentQuestion(prev => prev + 1);
    setIsAnswered(false);
    setShowFeedback(false);
  };

  const handlePreviousQuestion = () => {
    if (currentQuestion > 0) {
      setCurrentQuestion(prev => prev - 1);
      setIsAnswered(true);
    }
  };

  const handleQuizComplete = () => {
    if (timerRef.current) clearInterval(timerRef.current);
    setShowResults(true);
    const finalScore = (score / questions.length) * 100;
    if (onComplete) {
      onComplete(finalScore, selectedAnswers);
    }
  };

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60);
    const secs = seconds % 60;
    return `${mins}:${secs.toString().padStart(2, '0')}`;
  };

  const getDifficultyColor = (difficulty: number) => {
    if (difficulty < 0.33) return 'bg-green-500';
    if (difficulty < 0.67) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getDifficultyText = (difficulty: number) => {
    if (difficulty < 0.33) return 'Kolay';
    if (difficulty < 0.67) return 'Orta';
    return 'Zor';
  };

  if (showResults) {
    return (
      <div className="quiz-results min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 p-4">
        <div className="max-w-2xl mx-auto">
          <div className="bg-white rounded-2xl shadow-xl p-8">
            <div className="text-center mb-8">
              <div className="text-6xl mb-4">
                {score >= questions.length * 0.8 ? '🏆' : 
                 score >= questions.length * 0.6 ? '🎯' : 
                 score >= questions.length * 0.4 ? '👍' : '💪'}
              </div>
              <h2 className="text-3xl font-bold text-gray-800 mb-2">Quiz Tamamlandı!</h2>
              <p className="text-xl text-gray-600">
                Skorunuz: <span className="font-bold text-blue-600">{((score / questions.length) * 100).toFixed(0)}%</span>
              </p>
              <p className="text-gray-500 mt-2">
                {score} / {questions.length} doğru cevap
              </p>
            </div>

            <div className="space-y-4 mb-8">
              {questions.map((q, idx) => (
                <div key={q.id} className="border rounded-lg p-4">
                  <div className="flex items-start gap-3">
                    <div className={`w-8 h-8 rounded-full flex items-center justify-center text-white font-bold ${
                      selectedAnswers[idx] === q.correctAnswer ? 'bg-green-500' : 'bg-red-500'
                    }`}>
                      {idx + 1}
                    </div>
                    <div className="flex-1">
                      <p className="font-medium text-gray-800 mb-2">{q.text}</p>
                      <p className="text-sm text-gray-600">
                        Cevabınız: <span className={`font-medium ${
                          selectedAnswers[idx] === q.correctAnswer ? 'text-green-600' : 'text-red-600'
                        }`}>{selectedAnswers[idx] || 'Cevapsız'}</span>
                      </p>
                      {selectedAnswers[idx] !== q.correctAnswer && (
                        <p className="text-sm text-green-600">
                          Doğru cevap: <span className="font-medium">{q.correctAnswer}</span>
                        </p>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            <div className="flex gap-4">
              <button 
                onClick={() => window.location.reload()} 
                className="flex-1 bg-blue-500 text-white py-3 rounded-lg font-medium hover:bg-blue-600 transition-colors"
              >
                Yeni Quiz Başlat
              </button>
              <button 
                className="flex-1 bg-gray-200 text-gray-800 py-3 rounded-lg font-medium hover:bg-gray-300 transition-colors"
              >
                Dashboard'a Dön
              </button>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="quiz-interface min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 p-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-6">
          <div className="flex justify-between items-center">
            <div>
              <h1 className="text-2xl font-bold text-gray-800">{topic} Quiz</h1>
              <p className="text-gray-600">{grade}. Sınıf Seviyesi</p>
            </div>
            <div className="flex items-center gap-6">
              <div className="text-center">
                <p className="text-sm text-gray-500">Süre</p>
                <p className={`text-2xl font-bold ${timeLeft < 60 ? 'text-red-600' : 'text-gray-800'}`}>
                  {formatTime(timeLeft)}
                </p>
              </div>
              <div className="text-center">
                <p className="text-sm text-gray-500">İlerleme</p>
                <p className="text-2xl font-bold text-blue-600">
                  {currentQuestion + 1}/{questions.length}
                </p>
              </div>
            </div>
          </div>

          {/* Progress Bar */}
          <div className="mt-4">
            <div className="w-full bg-gray-200 rounded-full h-2">
              <div 
                className="bg-gradient-to-r from-blue-500 to-purple-500 h-2 rounded-full transition-all duration-500"
                style={{ width: `${((currentQuestion + 1) / questions.length) * 100}%` }}
              />
            </div>
          </div>
        </div>

        {/* Question Card */}
        <div className="bg-white rounded-2xl shadow-xl p-8">
          <div className="mb-6">
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center gap-3">
                <span className="text-lg font-medium text-gray-500">Soru {currentQuestion + 1}</span>
                <span className={`px-3 py-1 rounded-full text-white text-sm font-medium ${getDifficultyColor(questions[currentQuestion].difficulty)}`}>
                  {getDifficultyText(questions[currentQuestion].difficulty)}
                </span>
                <span className="px-3 py-1 bg-blue-100 text-blue-700 rounded-full text-sm font-medium">
                  {questions[currentQuestion].topic}
                </span>
              </div>
            </div>
            
            <h2 className="text-xl font-bold text-gray-800">
              {questions[currentQuestion].text}
            </h2>
          </div>

          {/* Answer Options */}
          <div className="space-y-3 mb-8">
            {questions[currentQuestion].options.map((option, idx) => {
              const isSelected = selectedAnswers[currentQuestion] === option;
              const isCorrect = option === questions[currentQuestion].correctAnswer;
              const showCorrect = showFeedback && isCorrect;
              const showWrong = showFeedback && isSelected && !isCorrect;

              return (
                <button
                  key={idx}
                  onClick={() => handleAnswerSelect(option)}
                  disabled={isAnswered}
                  className={`w-full text-left p-4 rounded-lg border-2 transition-all ${
                    showCorrect ? 'border-green-500 bg-green-50' :
                    showWrong ? 'border-red-500 bg-red-50' :
                    isSelected ? 'border-blue-500 bg-blue-50' :
                    'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                  } ${isAnswered ? 'cursor-not-allowed' : 'cursor-pointer'}`}
                >
                  <div className="flex items-center gap-3">
                    <div className={`w-10 h-10 rounded-full border-2 flex items-center justify-center font-bold ${
                      showCorrect ? 'border-green-500 bg-green-500 text-white' :
                      showWrong ? 'border-red-500 bg-red-500 text-white' :
                      isSelected ? 'border-blue-500 bg-blue-500 text-white' :
                      'border-gray-300 text-gray-600'
                    }`}>
                      {String.fromCharCode(65 + idx)}
                    </div>
                    <span className={`text-lg ${
                      showCorrect ? 'text-green-700 font-medium' :
                      showWrong ? 'text-red-700' :
                      isSelected ? 'text-blue-700 font-medium' :
                      'text-gray-700'
                    }`}>
                      {option}
                    </span>
                    {showCorrect && <span className="ml-auto text-green-600 font-bold">✓ Doğru</span>}
                    {showWrong && <span className="ml-auto text-red-600 font-bold">✗ Yanlış</span>}
                  </div>
                </button>
              );
            })}
          </div>

          {/* Navigation Buttons */}
          <div className="flex justify-between">
            <button
              onClick={handlePreviousQuestion}
              disabled={currentQuestion === 0}
              className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                currentQuestion === 0 
                  ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
                  : 'bg-gray-200 text-gray-700 hover:bg-gray-300'
              }`}
            >
              ← Önceki Soru
            </button>

            {currentQuestion === questions.length - 1 ? (
              <button
                onClick={handleQuizComplete}
                className="px-8 py-3 bg-green-500 text-white rounded-lg font-medium hover:bg-green-600 transition-colors"
              >
                Quiz'i Bitir
              </button>
            ) : (
              <button
                onClick={handleNextQuestion}
                disabled={!isAnswered}
                className={`px-6 py-3 rounded-lg font-medium transition-colors ${
                  !isAnswered 
                    ? 'bg-gray-100 text-gray-400 cursor-not-allowed' 
                    : 'bg-blue-500 text-white hover:bg-blue-600'
                }`}
              >
                Sonraki Soru →
              </button>
            )}
          </div>
        </div>

        {/* Question Navigator */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mt-6">
          <h3 className="text-lg font-bold text-gray-800 mb-4">Soru Navigasyonu</h3>
          <div className="flex flex-wrap gap-2">
            {questions.map((_, idx) => (
              <button
                key={idx}
                onClick={() => setCurrentQuestion(idx)}
                className={`w-12 h-12 rounded-lg font-medium transition-all ${
                  idx === currentQuestion ? 'bg-blue-500 text-white' :
                  selectedAnswers[idx] ? 'bg-green-100 text-green-700' :
                  'bg-gray-100 text-gray-600 hover:bg-gray-200'
                }`}
              >
                {idx + 1}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default QuizInterface;