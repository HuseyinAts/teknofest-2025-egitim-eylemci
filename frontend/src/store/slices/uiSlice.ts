import { createSlice, PayloadAction } from '@reduxjs/toolkit';

interface UIState {
  theme: 'light' | 'dark';
  isDarkMode: boolean;
  sidebarOpen: boolean;
  language: 'tr' | 'en';
  notifications: Notification[];
  isLoading: boolean;
  loadingMessage: string;
  mobileMenuOpen: boolean;
  keyboardShortcutsEnabled: boolean;
  reducedMotion: boolean;
  fontSize: 'small' | 'medium' | 'large';
  highContrast: boolean;
}

interface Notification {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  message: string;
  timestamp: number;
  read: boolean;
}

const getInitialState = (): UIState => {
  if (typeof window !== 'undefined') {
    const savedTheme = localStorage.getItem('theme') as 'light' | 'dark' | null;
    const savedLanguage = localStorage.getItem('language') as 'tr' | 'en' | null;
    const savedFontSize = localStorage.getItem('fontSize') as UIState['fontSize'] | null;
    const savedHighContrast = localStorage.getItem('highContrast');
    const savedReducedMotion = localStorage.getItem('reducedMotion');
    
    return {
      theme: savedTheme || 'light',
      isDarkMode: savedTheme === 'dark',
      sidebarOpen: window.innerWidth >= 1024,
      language: savedLanguage || 'tr',
      notifications: [],
      isLoading: false,
      loadingMessage: '',
      mobileMenuOpen: false,
      keyboardShortcutsEnabled: true,
      reducedMotion: savedReducedMotion === 'true',
      fontSize: savedFontSize || 'medium',
      highContrast: savedHighContrast === 'true',
    };
  }
  
  return {
    theme: 'light',
    isDarkMode: false,
    sidebarOpen: true,
    language: 'tr',
    notifications: [],
    isLoading: false,
    loadingMessage: '',
    mobileMenuOpen: false,
    keyboardShortcutsEnabled: true,
    reducedMotion: false,
    fontSize: 'medium',
    highContrast: false,
  };
};

const initialState: UIState = getInitialState();

const uiSlice = createSlice({
  name: 'ui',
  initialState,
  reducers: {
    toggleTheme: (state) => {
      state.theme = state.theme === 'light' ? 'dark' : 'light';
      state.isDarkMode = state.theme === 'dark';
      if (typeof window !== 'undefined') {
        localStorage.setItem('theme', state.theme);
        document.documentElement.classList.toggle('dark', state.isDarkMode);
      }
    },
    setTheme: (state, action: PayloadAction<'light' | 'dark'>) => {
      state.theme = action.payload;
      state.isDarkMode = action.payload === 'dark';
      if (typeof window !== 'undefined') {
        localStorage.setItem('theme', action.payload);
        document.documentElement.classList.toggle('dark', state.isDarkMode);
      }
    },
    toggleSidebar: (state) => {
      state.sidebarOpen = !state.sidebarOpen;
    },
    setSidebarOpen: (state, action: PayloadAction<boolean>) => {
      state.sidebarOpen = action.payload;
    },
    setLanguage: (state, action: PayloadAction<'tr' | 'en'>) => {
      state.language = action.payload;
      if (typeof window !== 'undefined') {
        localStorage.setItem('language', action.payload);
      }
    },
    addNotification: (state, action: PayloadAction<Omit<Notification, 'id' | 'timestamp' | 'read'>>) => {
      const notification: Notification = {
        ...action.payload,
        id: `notif-${Date.now()}-${Math.random()}`,
        timestamp: Date.now(),
        read: false,
      };
      state.notifications.unshift(notification);
      
      if (state.notifications.length > 50) {
        state.notifications = state.notifications.slice(0, 50);
      }
    },
    markNotificationAsRead: (state, action: PayloadAction<string>) => {
      const notification = state.notifications.find(n => n.id === action.payload);
      if (notification) {
        notification.read = true;
      }
    },
    markAllNotificationsAsRead: (state) => {
      state.notifications.forEach(n => { n.read = true; });
    },
    removeNotification: (state, action: PayloadAction<string>) => {
      state.notifications = state.notifications.filter(n => n.id !== action.payload);
    },
    clearNotifications: (state) => {
      state.notifications = [];
    },
    setLoading: (state, action: PayloadAction<{ isLoading: boolean; message?: string }>) => {
      state.isLoading = action.payload.isLoading;
      state.loadingMessage = action.payload.message || '';
    },
    toggleMobileMenu: (state) => {
      state.mobileMenuOpen = !state.mobileMenuOpen;
    },
    setMobileMenuOpen: (state, action: PayloadAction<boolean>) => {
      state.mobileMenuOpen = action.payload;
    },
    toggleKeyboardShortcuts: (state) => {
      state.keyboardShortcutsEnabled = !state.keyboardShortcutsEnabled;
    },
    setReducedMotion: (state, action: PayloadAction<boolean>) => {
      state.reducedMotion = action.payload;
      if (typeof window !== 'undefined') {
        localStorage.setItem('reducedMotion', String(action.payload));
      }
    },
    setFontSize: (state, action: PayloadAction<UIState['fontSize']>) => {
      state.fontSize = action.payload;
      if (typeof window !== 'undefined') {
        localStorage.setItem('fontSize', action.payload);
        document.documentElement.setAttribute('data-font-size', action.payload);
      }
    },
    setHighContrast: (state, action: PayloadAction<boolean>) => {
      state.highContrast = action.payload;
      if (typeof window !== 'undefined') {
        localStorage.setItem('highContrast', String(action.payload));
        document.documentElement.classList.toggle('high-contrast', action.payload);
      }
    },
  },
});

export const {
  toggleTheme,
  setTheme,
  toggleSidebar,
  setSidebarOpen,
  setLanguage,
  addNotification,
  markNotificationAsRead,
  markAllNotificationsAsRead,
  removeNotification,
  clearNotifications,
  setLoading,
  toggleMobileMenu,
  setMobileMenuOpen,
  toggleKeyboardShortcuts,
  setReducedMotion,
  setFontSize,
  setHighContrast,
} = uiSlice.actions;

export default uiSlice.reducer;