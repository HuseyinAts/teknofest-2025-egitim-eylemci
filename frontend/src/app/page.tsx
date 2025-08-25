'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';
import { useSelector } from 'react-redux';
import { RootState } from '@/store';
import { Button, Card, CardContent, Grid, Typography, Box } from '@mui/material';
import { School, Assessment, TrendingUp, Group } from '@mui/icons-material';
import Link from 'next/link';

export default function Home() {
  const router = useRouter();
  const { isAuthenticated } = useSelector((state: RootState) => state.auth);

  useEffect(() => {
    if (isAuthenticated) {
      router.push('/dashboard');
    }
  }, [isAuthenticated, router]);

  const features = [
    {
      icon: <School fontSize="large" />,
      title: 'Kişiselleştirilmiş Öğrenme',
      description: 'Yapay zeka destekli öğrenme yolları ile kendi hızınızda ilerleyin',
    },
    {
      icon: <Assessment fontSize="large" />,
      title: 'Akıllı Değerlendirme',
      description: 'Performansınızı analiz eden ve gelişim önerileri sunan değerlendirmeler',
    },
    {
      icon: <TrendingUp fontSize="large" />,
      title: 'İlerleme Takibi',
      description: 'Detaylı raporlar ve analizlerle öğrenme sürecinizi takip edin',
    },
    {
      icon: <Group fontSize="large" />,
      title: 'İşbirlikçi Öğrenme',
      description: 'Diğer öğrencilerle etkileşim kurarak birlikte öğrenin',
    },
  ];

  return (
    <Box sx={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Box sx={{ bgcolor: 'primary.main', color: 'white', py: 8 }}>
        <Box sx={{ maxWidth: 'lg', mx: 'auto', px: 3 }}>
          <Typography variant="h2" component="h1" gutterBottom align="center">
            Teknofest 2025 Eğitim Platformu
          </Typography>
          <Typography variant="h5" align="center" sx={{ mb: 4 }}>
            Yapay Zeka Destekli Kişiselleştirilmiş Eğitim Deneyimi
          </Typography>
          <Box sx={{ display: 'flex', gap: 2, justifyContent: 'center' }}>
            <Link href="/login" passHref>
              <Button variant="contained" size="large" sx={{ bgcolor: 'white', color: 'primary.main' }}>
                Giriş Yap
              </Button>
            </Link>
            <Link href="/register" passHref>
              <Button variant="outlined" size="large" sx={{ borderColor: 'white', color: 'white' }}>
                Kayıt Ol
              </Button>
            </Link>
          </Box>
        </Box>
      </Box>

      <Box sx={{ flexGrow: 1, py: 8 }}>
        <Box sx={{ maxWidth: 'lg', mx: 'auto', px: 3 }}>
          <Typography variant="h4" align="center" gutterBottom sx={{ mb: 6 }}>
            Platform Özellikleri
          </Typography>
          <Grid container spacing={4}>
            {features.map((feature, index) => (
              <Grid item xs={12} sm={6} md={3} key={index}>
                <Card sx={{ height: '100%', textAlign: 'center' }}>
                  <CardContent>
                    <Box sx={{ color: 'primary.main', mb: 2 }}>
                      {feature.icon}
                    </Box>
                    <Typography variant="h6" gutterBottom>
                      {feature.title}
                    </Typography>
                    <Typography variant="body2" color="text.secondary">
                      {feature.description}
                    </Typography>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Box>
      </Box>

      <Box component="footer" sx={{ bgcolor: 'background.paper', py: 3, mt: 'auto' }}>
        <Typography variant="body2" color="text.secondary" align="center">
          © 2025 Teknofest Eğitim Platformu. Tüm hakları saklıdır.
        </Typography>
      </Box>
    </Box>
  );
}