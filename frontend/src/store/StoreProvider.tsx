'use client';

import { useRef } from 'react';
import { Provider } from 'react-redux';
import { makeStore, AppStore } from './index';
import { PersistGate } from 'redux-persist/integration/react';
import { persistStore } from 'redux-persist';

export default function StoreProvider({
  children,
}: {
  children: React.ReactNode;
}) {
  const storeRef = useRef<AppStore | null>(null);
  const persistorRef = useRef<any>(null);
  
  if (!storeRef.current) {
    storeRef.current = makeStore();
    
    if (typeof window !== 'undefined') {
      persistorRef.current = persistStore(storeRef.current);
    }
  }

  if (persistorRef.current) {
    return (
      <Provider store={storeRef.current}>
        <PersistGate loading={null} persistor={persistorRef.current}>
          {children}
        </PersistGate>
      </Provider>
    );
  }

  return <Provider store={storeRef.current}>{children}</Provider>;
}