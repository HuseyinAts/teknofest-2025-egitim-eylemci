'use client';

import { useEffect, useState } from 'react';
import { useSelector, useDispatch } from 'react-redux';
import { RootState, AppDispatch } from '@/store';
import { fetchLearningPaths } from '@/store/slices/learningSlice';
import {
  Box,
  Grid,
  Card,
  CardContent,
  Typography,
  Button,
  LinearProgress,
  Paper,
  IconButton,
} from '@mui/material';
import {
  TrendingUp,
  AccessTime,
  Assignment,
  School,
  ArrowForward,
  Refresh,
} from '@mui/icons-material';
import { useRouter } from 'next/navigation';
import Link from 'next/link';

export default function DashboardPage() {
  const router = useRouter();
  const dispatch = useDispatch<AppDispatch>();
  const { user } = useSelector((state: RootState) => state.auth);
  const { paths, loading } = useSelector((state: RootState) => state.learning);
  const [stats, setStats] = useState({
    totalHours: 24,
    completedLessons: 12,
    averageScore: 85,
    streak: 7,
  });

  useEffect(() => {
    dispatch(fetchLearningPaths());
  }, [dispatch]);

  const StatCard = ({ icon, title, value, color }: any) => (
    <Card>
      <CardContent>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Box sx={{ color, mr: 2 }}>{icon}</Box>
          <Typography variant="body2" color="text.secondary">
            {title}
          </Typography>
        </Box>
        <Typography variant="h4">{value}</Typography>
      </CardContent>
    </Card>
  );

  return (
    <Box sx={{ p: 3 }}>
      <Box sx={{ mb: 4 }}>
        <Typography variant="h4" gutterBottom>
          Hoş Geldin, {user?.name || 'Öğrenci'}!
        </Typography>
        <Typography variant="body1" color="text.secondary">
          Öğrenme yolculuğuna devam et
        </Typography>
      </Box>

      <Grid container spacing={3}>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            icon={<AccessTime />}
            title="Toplam Çalışma"
            value={`${stats.totalHours} saat`}
            color="primary.main"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            icon={<Assignment />}
            title="Tamamlanan Dersler"
            value={stats.completedLessons}
            color="success.main"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            icon={<TrendingUp />}
            title="Ortalama Puan"
            value={`%${stats.averageScore}`}
            color="warning.main"
          />
        </Grid>
        <Grid item xs={12} sm={6} md={3}>
          <StatCard
            icon={<School />}
            title="Çalışma Serisi"
            value={`${stats.streak} gün`}
            color="error.main"
          />
        </Grid>

        <Grid item xs={12} md={8}>
          <Paper sx={{ p: 3 }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
              <Typography variant="h6">Devam Eden Öğrenme Yolları</Typography>
              <IconButton size="small" onClick={() => dispatch(fetchLearningPaths())}>
                <Refresh />
              </IconButton>
            </Box>
            
            {loading ? (
              <LinearProgress />
            ) : (
              <Box>
                {paths.slice(0, 3).map((path) => (
                  <Card key={path.id} sx={{ mb: 2 }}>
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                        <Box sx={{ flex: 1 }}>
                          <Typography variant="h6">{path.title}</Typography>
                          <Typography variant="body2" color="text.secondary" sx={{ mb: 1 }}>
                            {path.description}
                          </Typography>
                          <LinearProgress
                            variant="determinate"
                            value={path.progress || 0}
                            sx={{ height: 8, borderRadius: 4 }}
                          />
                          <Typography variant="caption" color="text.secondary" sx={{ mt: 1 }}>
                            %{path.progress || 0} tamamlandı
                          </Typography>
                        </Box>
                        <IconButton
                          color="primary"
                          onClick={() => router.push(`/learning-paths/${path.id}`)}
                        >
                          <ArrowForward />
                        </IconButton>
                      </Box>
                    </CardContent>
                  </Card>
                ))}
                
                <Link href="/learning-paths" passHref>
                  <Button fullWidth variant="outlined" sx={{ mt: 2 }}>
                    Tüm Öğrenme Yollarını Gör
                  </Button>
                </Link>
              </Box>
            )}
          </Paper>
        </Grid>

        <Grid item xs={12} md={4}>
          <Paper sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Hızlı Erişim
            </Typography>
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
              <Link href="/study-session" passHref>
                <Button fullWidth variant="contained" startIcon={<School />}>
                  Çalışmaya Başla
                </Button>
              </Link>
              <Link href="/assessments" passHref>
                <Button fullWidth variant="outlined" startIcon={<Assignment />}>
                  Değerlendirmeler
                </Button>
              </Link>
              <Link href="/profile" passHref>
                <Button fullWidth variant="outlined">
                  Profil Ayarları
                </Button>
              </Link>
            </Box>
          </Paper>

          <Paper sx={{ p: 3, mt: 3 }}>
            <Typography variant="h6" gutterBottom>
              Günün Önerisi
            </Typography>
            <Card sx={{ bgcolor: 'primary.light', color: 'white' }}>
              <CardContent>
                <Typography variant="subtitle1" gutterBottom>
                  Matematik Temelleri
                </Typography>
                <Typography variant="body2">
                  Bugün cebirsel ifadeler konusunu çalışmayı deneyin
                </Typography>
                <Button
                  variant="contained"
                  size="small"
                  sx={{ mt: 2, bgcolor: 'white', color: 'primary.main' }}
                >
                  Başla
                </Button>
              </CardContent>
            </Card>
          </Paper>
        </Grid>
      </Grid>
    </Box>
  );
}