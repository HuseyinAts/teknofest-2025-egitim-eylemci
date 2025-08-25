'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useDispatch } from 'react-redux';
import { AppDispatch } from '@/store';
import { login } from '@/store/slices/authSlice';
import { useForm } from 'react-hook-form';
import {
  Box,
  Button,
  Card,
  CardContent,
  TextField,
  Typography,
  Link as MuiLink,
  CircularProgress,
  Alert,
} from '@mui/material';
import Link from 'next/link';
import { toast } from 'react-hot-toast';

interface LoginForm {
  email: string;
  password: string;
}

export default function LoginPage() {
  const router = useRouter();
  const dispatch = useDispatch<AppDispatch>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const {
    register,
    handleSubmit,
    formState: { errors },
  } = useForm<LoginForm>();

  const onSubmit = async (data: LoginForm) => {
    setLoading(true);
    setError('');
    
    try {
      await dispatch(login(data)).unwrap();
      toast.success('Giriş başarılı!');
      router.push('/dashboard');
    } catch (err: any) {
      setError(err.message || 'Giriş yapılamadı');
      toast.error('Giriş başarısız!');
    } finally {
      setLoading(false);
    }
  };

  return (
    <Box
      sx={{
        minHeight: '100vh',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: 'background.default',
      }}
    >
      <Card sx={{ maxWidth: 400, width: '100%', mx: 2 }}>
        <CardContent sx={{ p: 4 }}>
          <Typography variant="h4" align="center" gutterBottom>
            Giriş Yap
          </Typography>
          
          <Typography variant="body2" align="center" color="text.secondary" sx={{ mb: 3 }}>
            Eğitim platformuna hoş geldiniz
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <form onSubmit={handleSubmit(onSubmit)}>
            <TextField
              fullWidth
              label="E-posta"
              type="email"
              margin="normal"
              error={!!errors.email}
              helperText={errors.email?.message}
              {...register('email', {
                required: 'E-posta zorunludur',
                pattern: {
                  value: /^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$/i,
                  message: 'Geçersiz e-posta adresi',
                },
              })}
            />

            <TextField
              fullWidth
              label="Şifre"
              type="password"
              margin="normal"
              error={!!errors.password}
              helperText={errors.password?.message}
              {...register('password', {
                required: 'Şifre zorunludur',
                minLength: {
                  value: 6,
                  message: 'Şifre en az 6 karakter olmalıdır',
                },
              })}
            />

            <Button
              fullWidth
              type="submit"
              variant="contained"
              size="large"
              disabled={loading}
              sx={{ mt: 3, mb: 2 }}
            >
              {loading ? <CircularProgress size={24} /> : 'Giriş Yap'}
            </Button>
          </form>

          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="body2">
              Hesabınız yok mu?{' '}
              <Link href="/register" passHref>
                <MuiLink component="span">Kayıt Ol</MuiLink>
              </Link>
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}