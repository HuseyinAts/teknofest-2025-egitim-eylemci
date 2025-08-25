'use client';

import { useState } from 'react';
import { useRouter } from 'next/navigation';
import { useDispatch } from 'react-redux';
import { AppDispatch } from '@/store';
import { register as registerUser } from '@/store/slices/authSlice';
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

interface RegisterForm {
  name: string;
  email: string;
  password: string;
  confirmPassword: string;
}

export default function RegisterPage() {
  const router = useRouter();
  const dispatch = useDispatch<AppDispatch>();
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const {
    register,
    handleSubmit,
    watch,
    formState: { errors },
  } = useForm<RegisterForm>();

  const password = watch('password');

  const onSubmit = async (data: RegisterForm) => {
    setLoading(true);
    setError('');
    
    try {
      const { confirmPassword, ...registerData } = data;
      await dispatch(registerUser(registerData)).unwrap();
      toast.success('Kayıt başarılı!');
      router.push('/dashboard');
    } catch (err: any) {
      setError(err.message || 'Kayıt olunamadı');
      toast.error('Kayıt başarısız!');
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
            Kayıt Ol
          </Typography>
          
          <Typography variant="body2" align="center" color="text.secondary" sx={{ mb: 3 }}>
            Yeni hesap oluşturun
          </Typography>

          {error && (
            <Alert severity="error" sx={{ mb: 2 }}>
              {error}
            </Alert>
          )}

          <form onSubmit={handleSubmit(onSubmit)}>
            <TextField
              fullWidth
              label="Ad Soyad"
              margin="normal"
              error={!!errors.name}
              helperText={errors.name?.message}
              {...register('name', {
                required: 'Ad soyad zorunludur',
                minLength: {
                  value: 3,
                  message: 'Ad soyad en az 3 karakter olmalıdır',
                },
              })}
            />

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
                  value: 8,
                  message: 'Şifre en az 8 karakter olmalıdır',
                },
              })}
            />

            <TextField
              fullWidth
              label="Şifre Tekrar"
              type="password"
              margin="normal"
              error={!!errors.confirmPassword}
              helperText={errors.confirmPassword?.message}
              {...register('confirmPassword', {
                required: 'Şifre tekrarı zorunludur',
                validate: value => value === password || 'Şifreler eşleşmiyor',
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
              {loading ? <CircularProgress size={24} /> : 'Kayıt Ol'}
            </Button>
          </form>

          <Box sx={{ textAlign: 'center' }}>
            <Typography variant="body2">
              Zaten hesabınız var mı?{' '}
              <Link href="/login" passHref>
                <MuiLink component="span">Giriş Yap</MuiLink>
              </Link>
            </Typography>
          </Box>
        </CardContent>
      </Card>
    </Box>
  );
}