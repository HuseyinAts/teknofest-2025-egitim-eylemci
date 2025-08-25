import { Middleware } from '@reduxjs/toolkit';

const actionTypeMaxLength = 50;

export const loggerMiddleware: Middleware = (store) => (next) => (action) => {
  const startTime = performance.now();
  
  console.group(
    `%c action %c${action.type.substring(0, actionTypeMaxLength)}`,
    'color: gray; font-weight: lighter',
    'color: inherit; font-weight: bold'
  );
  
  console.log('%c prev state', 'color: #9E9E9E; font-weight: bold', store.getState());
  console.log('%c action', 'color: #03A9F4; font-weight: bold', action);
  
  const result = next(action);
  const endTime = performance.now();
  const duration = (endTime - startTime).toFixed(2);
  
  console.log('%c next state', 'color: #4CAF50; font-weight: bold', store.getState());
  console.log(`%c duration: ${duration}ms`, 'color: gray; font-weight: lighter');
  console.groupEnd();
  
  return result;
};