'use client';

import React, { useState, useCallback, useMemo } from 'react';
import {
  AppBar,
  Box,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Toolbar,
  Typography,
  Avatar,
  Menu,
  MenuItem,
  Divider,
  Badge,
  Tooltip,
  useTheme,
  useMediaQuery,
  Switch,
  FormControlLabel,
} from '@mui/material';
import {
  Menu as MenuIcon,
  Dashboard,
  School,
  Assessment,
  Person,
  Settings,
  Logout,
  Notifications,
  Brightness4,
  Brightness7,
  Language,
  ChevronLeft,
} from '@mui/icons-material';
import { useRouter, usePathname } from 'next/navigation';
import { useSelector, useDispatch } from 'react-redux';
import { RootState, AppDispatch } from '@/store';
import { logout } from '@/store/slices/authSlice';
import { toggleTheme } from '@/store/slices/uiSlice';
import { useTranslation } from 'react-i18next';
import Link from 'next/link';

const drawerWidth = 280;

interface AppLayoutProps {
  children: React.ReactNode;
}

const AppLayout: React.FC<AppLayoutProps> = ({ children }) => {
  const theme = useTheme();
  const isMobile = useMediaQuery(theme.breakpoints.down('md'));
  const router = useRouter();
  const pathname = usePathname();
  const dispatch = useDispatch<AppDispatch>();
  const { t, i18n } = useTranslation();
  
  const { user } = useSelector((state: RootState) => state.auth);
  const { isDarkMode } = useSelector((state: RootState) => state.ui);
  
  const [mobileOpen, setMobileOpen] = useState(false);
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null);
  const [notificationAnchor, setNotificationAnchor] = useState<null | HTMLElement>(null);

  const handleDrawerToggle = useCallback(() => {
    setMobileOpen(prev => !prev);
  }, []);

  const handleProfileMenuOpen = useCallback((event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget);
  }, []);

  const handleProfileMenuClose = useCallback(() => {
    setAnchorEl(null);
  }, []);

  const handleNotificationOpen = useCallback((event: React.MouseEvent<HTMLElement>) => {
    setNotificationAnchor(event.currentTarget);
  }, []);

  const handleNotificationClose = useCallback(() => {
    setNotificationAnchor(null);
  }, []);

  const handleLogout = useCallback(async () => {
    await dispatch(logout());
    router.push('/login');
  }, [dispatch, router]);

  const handleThemeToggle = useCallback(() => {
    dispatch(toggleTheme());
  }, [dispatch]);

  const handleLanguageChange = useCallback(() => {
    const newLang = i18n.language === 'tr' ? 'en' : 'tr';
    i18n.changeLanguage(newLang);
    localStorage.setItem('language', newLang);
  }, [i18n]);

  const menuItems = useMemo(() => [
    {
      text: t('menu.dashboard'),
      icon: <Dashboard />,
      path: '/dashboard',
      ariaLabel: 'Ana panel',
    },
    {
      text: t('menu.learningPaths'),
      icon: <School />,
      path: '/learning-paths',
      ariaLabel: 'Öğrenme yolları',
    },
    {
      text: t('menu.assessments'),
      icon: <Assessment />,
      path: '/assessments',
      ariaLabel: 'Değerlendirmeler',
    },
    {
      text: t('menu.profile'),
      icon: <Person />,
      path: '/profile',
      ariaLabel: 'Profil',
    },
    {
      text: t('menu.settings'),
      icon: <Settings />,
      path: '/settings',
      ariaLabel: 'Ayarlar',
    },
  ], [t]);

  const drawer = (
    <Box
      sx={{
        height: '100%',
        display: 'flex',
        flexDirection: 'column',
      }}
    >
      <Toolbar sx={{ px: 2 }}>
        <Typography variant="h6" noWrap sx={{ flexGrow: 1 }}>
          Teknofest 2025
        </Typography>
        {isMobile && (
          <IconButton
            onClick={handleDrawerToggle}
            aria-label="Menüyü kapat"
          >
            <ChevronLeft />
          </IconButton>
        )}
      </Toolbar>
      <Divider />
      
      <List sx={{ flexGrow: 1, px: 1 }}>
        {menuItems.map((item) => (
          <ListItem key={item.path} disablePadding sx={{ my: 0.5 }}>
            <ListItemButton
              component={Link}
              href={item.path}
              selected={pathname === item.path}
              onClick={() => isMobile && handleDrawerToggle()}
              sx={{
                borderRadius: 2,
                '&.Mui-selected': {
                  bgcolor: 'primary.main',
                  color: 'white',
                  '& .MuiListItemIcon-root': {
                    color: 'white',
                  },
                  '&:hover': {
                    bgcolor: 'primary.dark',
                  },
                },
              }}
              aria-label={item.ariaLabel}
              aria-current={pathname === item.path ? 'page' : undefined}
            >
              <ListItemIcon>{item.icon}</ListItemIcon>
              <ListItemText primary={item.text} />
            </ListItemButton>
          </ListItem>
        ))}
      </List>

      <Divider />
      
      <Box sx={{ p: 2 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', mb: 2 }}>
          <Avatar
            src={user?.avatar}
            alt={user?.name}
            sx={{ width: 40, height: 40, mr: 2 }}
          />
          <Box sx={{ flexGrow: 1 }}>
            <Typography variant="body2" fontWeight="bold">
              {user?.name || 'Kullanıcı'}
            </Typography>
            <Typography variant="caption" color="text.secondary">
              {user?.email}
            </Typography>
          </Box>
        </Box>
        
        <FormControlLabel
          control={
            <Switch
              checked={isDarkMode}
              onChange={handleThemeToggle}
              size="small"
              aria-label="Karanlık mod"
            />
          }
          label={
            <Typography variant="body2">
              {isDarkMode ? 'Karanlık' : 'Açık'} Mod
            </Typography>
          }
        />
      </Box>
    </Box>
  );

  return (
    <Box sx={{ display: 'flex', minHeight: '100vh' }}>
      <AppBar
        position="fixed"
        sx={{
          width: { md: `calc(100% - ${drawerWidth}px)` },
          ml: { md: `${drawerWidth}px` },
          bgcolor: 'background.paper',
          color: 'text.primary',
          boxShadow: 1,
        }}
      >
        <Toolbar>
          <IconButton
            color="inherit"
            aria-label="Menüyü aç"
            edge="start"
            onClick={handleDrawerToggle}
            sx={{ mr: 2, display: { md: 'none' } }}
          >
            <MenuIcon />
          </IconButton>
          
          <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1 }}>
            {menuItems.find(item => item.path === pathname)?.text || 'Sayfa'}
          </Typography>

          <Tooltip title="Dil değiştir">
            <IconButton
              onClick={handleLanguageChange}
              color="inherit"
              aria-label="Dil değiştir"
            >
              <Language />
            </IconButton>
          </Tooltip>

          <Tooltip title="Tema değiştir">
            <IconButton
              onClick={handleThemeToggle}
              color="inherit"
              aria-label="Tema değiştir"
            >
              {isDarkMode ? <Brightness7 /> : <Brightness4 />}
            </IconButton>
          </Tooltip>

          <Tooltip title="Bildirimler">
            <IconButton
              onClick={handleNotificationOpen}
              color="inherit"
              aria-label="Bildirimler"
            >
              <Badge badgeContent={3} color="error">
                <Notifications />
              </Badge>
            </IconButton>
          </Tooltip>

          <Tooltip title="Profil">
            <IconButton
              onClick={handleProfileMenuOpen}
              sx={{ ml: 1 }}
              aria-label="Profil menüsü"
            >
              <Avatar
                src={user?.avatar}
                alt={user?.name}
                sx={{ width: 32, height: 32 }}
              />
            </IconButton>
          </Tooltip>
        </Toolbar>
      </AppBar>

      <Box
        component="nav"
        sx={{ width: { md: drawerWidth }, flexShrink: { md: 0 } }}
      >
        <Drawer
          variant={isMobile ? 'temporary' : 'permanent'}
          open={isMobile ? mobileOpen : true}
          onClose={handleDrawerToggle}
          ModalProps={{
            keepMounted: true, // Better open performance on mobile
          }}
          sx={{
            '& .MuiDrawer-paper': {
              boxSizing: 'border-box',
              width: drawerWidth,
              borderRight: '1px solid',
              borderColor: 'divider',
            },
          }}
        >
          {drawer}
        </Drawer>
      </Box>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          p: { xs: 2, sm: 3 },
          width: { md: `calc(100% - ${drawerWidth}px)` },
          mt: 8,
          minHeight: 'calc(100vh - 64px)',
          bgcolor: 'background.default',
        }}
      >
        {children}
      </Box>

      <Menu
        anchorEl={anchorEl}
        open={Boolean(anchorEl)}
        onClose={handleProfileMenuClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
      >
        <MenuItem onClick={() => { router.push('/profile'); handleProfileMenuClose(); }}>
          <ListItemIcon>
            <Person fontSize="small" />
          </ListItemIcon>
          <ListItemText>Profil</ListItemText>
        </MenuItem>
        <MenuItem onClick={() => { router.push('/settings'); handleProfileMenuClose(); }}>
          <ListItemIcon>
            <Settings fontSize="small" />
          </ListItemIcon>
          <ListItemText>Ayarlar</ListItemText>
        </MenuItem>
        <Divider />
        <MenuItem onClick={handleLogout}>
          <ListItemIcon>
            <Logout fontSize="small" />
          </ListItemIcon>
          <ListItemText>Çıkış Yap</ListItemText>
        </MenuItem>
      </Menu>

      <Menu
        anchorEl={notificationAnchor}
        open={Boolean(notificationAnchor)}
        onClose={handleNotificationClose}
        transformOrigin={{ horizontal: 'right', vertical: 'top' }}
        anchorOrigin={{ horizontal: 'right', vertical: 'bottom' }}
        PaperProps={{
          sx: { width: 320, maxHeight: 400 }
        }}
      >
        <Box sx={{ p: 2 }}>
          <Typography variant="h6">Bildirimler</Typography>
        </Box>
        <Divider />
        <MenuItem onClick={handleNotificationClose}>
          <Typography variant="body2">
            Yeni ders içeriği eklendi
          </Typography>
        </MenuItem>
        <MenuItem onClick={handleNotificationClose}>
          <Typography variant="body2">
            Değerlendirme sonuçlarınız hazır
          </Typography>
        </MenuItem>
        <MenuItem onClick={handleNotificationClose}>
          <Typography variant="body2">
            7 günlük çalışma serinizi tamamladınız!
          </Typography>
        </MenuItem>
      </Menu>
    </Box>
  );
};

export default AppLayout;