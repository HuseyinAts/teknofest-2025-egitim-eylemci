'use client';

import { useState } from 'react';
import { useSearchParams } from 'next/navigation';
import QuizInterface from '@/components/quiz/QuizInterface';
import { Box, Typography, Paper, Button, Grid, Card, CardContent } from '@mui/material';
import { School, Timer, EmojiEvents, Psychology } from '@mui/icons-material';
import Link from 'next/link';

export default function QuizPage() {
  const searchParams = useSearchParams();
  const [activeQuiz, setActiveQuiz] = useState(false);
  const [selectedSubject, setSelectedSubject] = useState(searchParams.get('subject') || 'Matematik');
  const [selectedGrade, setSelectedGrade] = useState(Number(searchParams.get('grade')) || 10);
  
  const subjects = [
    { name: 'Matematik', icon: '📐', color: '#3b82f6' },
    { name: 'Fizik', icon: '⚡', color: '#8b5cf6' },
    { name: 'Kimya', icon: '🧪', color: '#10b981' },
    { name: 'Biyoloji', icon: '🧬', color: '#f59e0b' },
    { name: 'Türkçe', icon: '📖', color: '#ef4444' },
    { name: 'Tarih', icon: '📜', color: '#6366f1' },
    { name: 'Coğrafya', icon: '🌍', color: '#06b6d4' },
    { name: 'İngilizce', icon: '🇬🇧', color: '#ec4899' }
  ];

  const handleQuizComplete = (score: number, answers: any[]) => {
    console.log('Quiz completed with score:', score);
    console.log('Answers:', answers);
    setActiveQuiz(false);
  };

  if (activeQuiz) {
    return (
      <QuizInterface 
        topic={selectedSubject}
        grade={selectedGrade}
        onComplete={handleQuizComplete}
      />
    );
  }

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Quiz Merkezi
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Bilgini test et, öğrenme seviyeni ölç
        </Typography>
      </Box>

      {/* Quick Stats */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <School sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="body2" color="text.secondary">
                  Toplam Quiz
                </Typography>
              </Box>
              <Typography variant="h5">24</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <EmojiEvents sx={{ mr: 1, color: 'warning.main' }} />
                <Typography variant="body2" color="text.secondary">
                  Ortalama Skor
                </Typography>
              </Box>
              <Typography variant="h5">85%</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Timer sx={{ mr: 1, color: 'success.main' }} />
                <Typography variant="body2" color="text.secondary">
                  Toplam Süre
                </Typography>
              </Box>
              <Typography variant="h5">6.5 saat</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Psychology sx={{ mr: 1, color: 'error.main' }} />
                <Typography variant="body2" color="text.secondary">
                  Zorluk
                </Typography>
              </Box>
              <Typography variant="h5">Adaptif</Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Subject Selection */}
      <Paper sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          Konu Seç
        </Typography>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          {subjects.map((subject) => (
            <Grid item xs={6} sm={4} md={3} key={subject.name}>
              <Card
                sx={{
                  cursor: 'pointer',
                  transition: 'all 0.3s',
                  border: selectedSubject === subject.name ? `2px solid ${subject.color}` : '2px solid transparent',
                  '&:hover': {
                    transform: 'translateY(-4px)',
                    boxShadow: 3
                  }
                }}
                onClick={() => setSelectedSubject(subject.name)}
              >
                <CardContent sx={{ textAlign: 'center' }}>
                  <Typography variant="h2" sx={{ mb: 1 }}>
                    {subject.icon}
                  </Typography>
                  <Typography variant="body1" fontWeight={selectedSubject === subject.name ? 'bold' : 'normal'}>
                    {subject.name}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Grade Selection */}
      <Paper sx={{ p: 3, mb: 4 }}>
        <Typography variant="h6" gutterBottom>
          Sınıf Seç
        </Typography>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          {[9, 10, 11, 12].map((grade) => (
            <Grid item xs={3} key={grade}>
              <Button
                variant={selectedGrade === grade ? 'contained' : 'outlined'}
                fullWidth
                onClick={() => setSelectedGrade(grade)}
                sx={{ py: 2 }}
              >
                {grade}. Sınıf
              </Button>
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Action Buttons */}
      <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
        <Button
          variant="contained"
          size="large"
          onClick={() => setActiveQuiz(true)}
          sx={{ px: 6, py: 2 }}
        >
          Quiz'e Başla
        </Button>
        <Link href="/dashboard" passHref>
          <Button variant="outlined" size="large" sx={{ px: 6, py: 2 }}>
            Dashboard'a Dön
          </Button>
        </Link>
      </Box>

      {/* Recent Quizzes */}
      <Paper sx={{ p: 3, mt: 4 }}>
        <Typography variant="h6" gutterBottom>
          Son Quiz Sonuçları
        </Typography>
        <Grid container spacing={2} sx={{ mt: 1 }}>
          {[
            { subject: 'Matematik', score: 90, date: '2 saat önce', difficulty: 'Orta' },
            { subject: 'Fizik', score: 75, date: 'Dün', difficulty: 'Zor' },
            { subject: 'Kimya', score: 85, date: '2 gün önce', difficulty: 'Kolay' }
          ].map((quiz, index) => (
            <Grid item xs={12} key={index}>
              <Card variant="outlined">
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Box>
                      <Typography variant="h6">{quiz.subject}</Typography>
                      <Typography variant="body2" color="text.secondary">
                        {quiz.date} • {quiz.difficulty}
                      </Typography>
                    </Box>
                    <Box sx={{ textAlign: 'right' }}>
                      <Typography 
                        variant="h4" 
                        sx={{ 
                          color: quiz.score >= 80 ? 'success.main' : quiz.score >= 60 ? 'warning.main' : 'error.main' 
                        }}
                      >
                        %{quiz.score}
                      </Typography>
                    </Box>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Paper>
    </Box>
  );
}