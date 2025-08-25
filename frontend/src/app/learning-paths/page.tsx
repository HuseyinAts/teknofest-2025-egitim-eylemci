'use client';

import { useState } from 'react';
import LearningPathVisualization from '@/components/learning/LearningPathVisualization';
import { 
  Box, 
  Typography, 
  Paper, 
  Grid, 
  Card, 
  CardContent, 
  Button,
  Chip,
  LinearProgress,
  IconButton,
  Tabs,
  Tab
} from '@mui/material';
import { 
  TrendingUp, 
  School, 
  Timer, 
  EmojiEvents,
  ArrowForward,
  FilterList,
  Add
} from '@mui/icons-material';
import Link from 'next/link';

interface TabPanelProps {
  children?: React.ReactNode;
  index: number;
  value: number;
}

function TabPanel(props: TabPanelProps) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`learning-path-tabpanel-${index}`}
      aria-labelledby={`learning-path-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

export default function LearningPathsPage() {
  const [selectedSubject, setSelectedSubject] = useState('Matematik');
  const [tabValue, setTabValue] = useState(0);

  const paths = [
    {
      id: 1,
      subject: 'Matematik',
      title: 'Analiz ve Kalkülüs Yolu',
      description: 'Limit, türev ve integral konularını kapsayan ileri matematik',
      progress: 65,
      totalTopics: 7,
      completedTopics: 4,
      estimatedTime: '3 ay',
      difficulty: 'Zor',
      xp: 1500,
      status: 'active'
    },
    {
      id: 2,
      subject: 'Fizik',
      title: 'Mekanik ve Hareket',
      description: 'Newton yasaları, enerji ve momentum konuları',
      progress: 45,
      totalTopics: 6,
      completedTopics: 3,
      estimatedTime: '2.5 ay',
      difficulty: 'Orta',
      xp: 1200,
      status: 'active'
    },
    {
      id: 3,
      subject: 'Kimya',
      title: 'Organik Kimya Temelleri',
      description: 'Karbon bileşikleri ve organik reaksiyonlar',
      progress: 30,
      totalTopics: 8,
      completedTopics: 2,
      estimatedTime: '4 ay',
      difficulty: 'Zor',
      xp: 1800,
      status: 'active'
    },
    {
      id: 4,
      subject: 'Biyoloji',
      title: 'Hücre ve Genetik',
      description: 'Hücre yapısı, DNA ve kalıtım konuları',
      progress: 80,
      totalTopics: 5,
      completedTopics: 4,
      estimatedTime: '2 ay',
      difficulty: 'Orta',
      xp: 1000,
      status: 'active'
    },
    {
      id: 5,
      subject: 'Matematik',
      title: 'Geometri ve Trigonometri',
      description: 'Üçgenler, çember ve trigonometrik fonksiyonlar',
      progress: 100,
      totalTopics: 6,
      completedTopics: 6,
      estimatedTime: '2 ay',
      difficulty: 'Kolay',
      xp: 800,
      status: 'completed'
    }
  ];

  const activePaths = paths.filter(p => p.status === 'active');
  const completedPaths = paths.filter(p => p.status === 'completed');

  const handleTabChange = (event: React.SyntheticEvent, newValue: number) => {
    setTabValue(newValue);
  };

  const getDifficultyColor = (difficulty: string) => {
    switch(difficulty) {
      case 'Kolay': return 'success';
      case 'Orta': return 'warning';
      case 'Zor': return 'error';
      default: return 'default';
    }
  };

  return (
    <Box sx={{ p: 3 }}>
      {/* Header */}
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Öğrenme Yolları
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Kişiselleştirilmiş öğrenme rotalarınızı keşfedin
        </Typography>
      </Box>

      {/* Stats Overview */}
      <Grid container spacing={3} sx={{ mb: 4 }}>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <School sx={{ mr: 1, color: 'primary.main' }} />
                <Typography variant="body2" color="text.secondary">
                  Aktif Yollar
                </Typography>
              </Box>
              <Typography variant="h5">{activePaths.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <EmojiEvents sx={{ mr: 1, color: 'success.main' }} />
                <Typography variant="body2" color="text.secondary">
                  Tamamlanan
                </Typography>
              </Box>
              <Typography variant="h5">{completedPaths.length}</Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <TrendingUp sx={{ mr: 1, color: 'warning.main' }} />
                <Typography variant="body2" color="text.secondary">
                  Toplam İlerleme
                </Typography>
              </Box>
              <Typography variant="h5">
                {Math.round(paths.reduce((sum, p) => sum + p.progress, 0) / paths.length)}%
              </Typography>
            </CardContent>
          </Card>
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <Card>
            <CardContent>
              <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                <Timer sx={{ mr: 1, color: 'info.main' }} />
                <Typography variant="body2" color="text.secondary">
                  Toplam XP
                </Typography>
              </Box>
              <Typography variant="h5">
                {paths.reduce((sum, p) => sum + (p.progress / 100 * p.xp), 0).toFixed(0)}
              </Typography>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Tabs */}
      <Paper sx={{ mb: 3 }}>
        <Tabs value={tabValue} onChange={handleTabChange} aria-label="learning paths tabs">
          <Tab label="Aktif Yollar" />
          <Tab label="Tamamlanan" />
          <Tab label="Görselleştirme" />
          <Tab label="Önerilen" />
        </Tabs>
      </Paper>

      {/* Active Paths */}
      <TabPanel value={tabValue} index={0}>
        <Grid container spacing={3}>
          {activePaths.map((path) => (
            <Grid item xs={12} md={6} key={path.id}>
              <Card sx={{ height: '100%' }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                    <Box>
                      <Typography variant="h6" gutterBottom>
                        {path.title}
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        {path.description}
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 1, flexWrap: 'wrap' }}>
                        <Chip label={path.subject} size="small" color="primary" />
                        <Chip 
                          label={path.difficulty} 
                          size="small" 
                          color={getDifficultyColor(path.difficulty) as any}
                        />
                        <Chip label={`${path.estimatedTime}`} size="small" variant="outlined" />
                      </Box>
                    </Box>
                    <IconButton color="primary">
                      <ArrowForward />
                    </IconButton>
                  </Box>

                  <Box sx={{ mb: 2 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 1 }}>
                      <Typography variant="body2" color="text.secondary">
                        İlerleme
                      </Typography>
                      <Typography variant="body2" fontWeight="bold">
                        {path.completedTopics}/{path.totalTopics} konu
                      </Typography>
                    </Box>
                    <LinearProgress 
                      variant="determinate" 
                      value={path.progress} 
                      sx={{ height: 8, borderRadius: 4 }}
                    />
                    <Typography variant="caption" color="text.secondary">
                      %{path.progress} tamamlandı
                    </Typography>
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="body2" color="primary">
                      +{Math.round(path.xp * path.progress / 100)} / {path.xp} XP
                    </Typography>
                    <Button variant="outlined" size="small">
                      Devam Et
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>

        <Box sx={{ mt: 3, textAlign: 'center' }}>
          <Button variant="contained" startIcon={<Add />}>
            Yeni Öğrenme Yolu Oluştur
          </Button>
        </Box>
      </TabPanel>

      {/* Completed Paths */}
      <TabPanel value={tabValue} index={1}>
        <Grid container spacing={3}>
          {completedPaths.map((path) => (
            <Grid item xs={12} md={6} key={path.id}>
              <Card sx={{ opacity: 0.9 }}>
                <CardContent>
                  <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                    <Box>
                      <Typography variant="h6" gutterBottom>
                        {path.title} ✓
                      </Typography>
                      <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                        {path.description}
                      </Typography>
                      <Box sx={{ display: 'flex', gap: 1 }}>
                        <Chip label={path.subject} size="small" color="success" />
                        <Chip label="Tamamlandı" size="small" color="success" variant="outlined" />
                        <Chip label={`+${path.xp} XP`} size="small" color="primary" />
                      </Box>
                    </Box>
                  </Box>

                  <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <Typography variant="body2" color="success.main">
                      Tüm konular başarıyla tamamlandı!
                    </Typography>
                    <Button variant="text" size="small">
                      Tekrar Et
                    </Button>
                  </Box>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </TabPanel>

      {/* Visualization */}
      <TabPanel value={tabValue} index={2}>
        <LearningPathVisualization 
          subject={selectedSubject}
          grade={10}
          studentLevel={0.65}
        />
      </TabPanel>

      {/* Recommended */}
      <TabPanel value={tabValue} index={3}>
        <Grid container spacing={3}>
          <Grid item xs={12}>
            <Paper sx={{ p: 3, bgcolor: 'primary.light', color: 'white' }}>
              <Typography variant="h6" gutterBottom>
                AI Önerileri
              </Typography>
              <Typography variant="body1">
                Mevcut performansınıza göre size özel önerilen öğrenme yolları
              </Typography>
            </Paper>
          </Grid>
          
          {[
            {
              title: 'İleri Matematik: Diferansiyel Denklemler',
              reason: 'Türev konusundaki başarınıza dayanarak',
              difficulty: 'Zor',
              duration: '3 ay'
            },
            {
              title: 'Uygulamalı Fizik: Elektrik ve Manyetizma',
              reason: 'Mekanik konusundaki ilerlemenize göre',
              difficulty: 'Orta',
              duration: '2.5 ay'
            }
          ].map((rec, index) => (
            <Grid item xs={12} md={6} key={index}>
              <Card>
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    {rec.title}
                  </Typography>
                  <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    {rec.reason}
                  </Typography>
                  <Box sx={{ display: 'flex', gap: 1, mb: 2 }}>
                    <Chip label={rec.difficulty} size="small" color={getDifficultyColor(rec.difficulty) as any} />
                    <Chip label={rec.duration} size="small" variant="outlined" />
                  </Box>
                  <Button variant="contained" fullWidth>
                    Bu Yolu Başlat
                  </Button>
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </TabPanel>
    </Box>
  );
}