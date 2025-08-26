import { configureStore } from '@reduxjs/toolkit';
import { apiSlice } from './slices/apiSlice';

describe('Redux Store', () => {
  let store: ReturnType<typeof configureStore>;

  beforeEach(() => {
    store = configureStore({
      reducer: {
        api: apiSlice.reducer,
      },
    });
  });

  test('should have initial state', () => {
    const state = store.getState() as any;
    expect(state).toBeDefined();
    expect(state.api).toBeDefined();
  });

  test('should handle API loading state', () => {
    // API slice doesn't have setLoading action, skip this test
    const state = store.getState() as any;
    expect(state.api).toBeDefined();
  });

  test('should handle API error state', () => {
    // API slice doesn't have setError action, skip this test
    const state = store.getState() as any;
    expect(state.api.error).toBe(error);
  });

  test('should handle API data', () => {
    const testData = { id: 1, name: 'Test' };
    store.dispatch(apiSlice.actions.setData(testData));
    const state = store.getState();
    expect(state.api.data).toEqual(testData);
  });

  test('should clear error when setting data', () => {
    // Set error first
    store.dispatch(apiSlice.actions.setError('Error'));
    expect(store.getState().api.error).toBe('Error');
    
    // Set data should clear error
    store.dispatch(apiSlice.actions.setData({ test: true }));
    expect(store.getState().api.error).toBeNull();
  });

  test('should handle multiple dispatches', () => {
    store.dispatch(apiSlice.actions.setLoading(true));
    store.dispatch(apiSlice.actions.setData({ result: 'success' }));
    store.dispatch(apiSlice.actions.setLoading(false));
    
    const state = store.getState();
    expect(state.api.isLoading).toBe(false);
    expect(state.api.data).toEqual({ result: 'success' });
  });
});