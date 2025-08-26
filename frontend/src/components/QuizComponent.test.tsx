import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import '@testing-library/jest-dom';
import { Provider } from 'react-redux';
import { configureStore } from '@reduxjs/toolkit';
import QuizComponent from './quiz/QuizInterface';
import { apiSlice } from '../store/slices/apiSlice';

// Mock store
const createMockStore = () => {
  return configureStore({
    reducer: {
      api: apiSlice.reducer,
    },
  });
};

describe('QuizComponent', () => {
  let store: ReturnType<typeof createMockStore>;

  beforeEach(() => {
    store = createMockStore();
  });

  test('renders quiz component', () => {
    render(
      <Provider store={store}>
        <QuizComponent />
      </Provider>
    );
    
    expect(screen.getByText(/Quiz/i)).toBeInTheDocument();
  });

  test('displays loading state', () => {
    render(
      <Provider store={store}>
        <QuizComponent isLoading={true} />
      </Provider>
    );
    
    expect(screen.getByText(/Loading/i)).toBeInTheDocument();
  });

  test('displays questions when loaded', async () => {
    const mockQuestions = [
      {
        id: 1,
        text: 'What is 2+2?',
        options: ['3', '4', '5', '6'],
        correctAnswer: 1,
      },
    ];

    render(
      <Provider store={store}>
        <QuizComponent questions={mockQuestions} />
      </Provider>
    );
    
    await waitFor(() => {
      expect(screen.getByText('What is 2+2?')).toBeInTheDocument();
    });
  });

  test('handles answer selection', async () => {
    const mockQuestions = [
      {
        id: 1,
        text: 'What is 2+2?',
        options: ['3', '4', '5', '6'],
        correctAnswer: 1,
      },
    ];
    
    const onAnswer = jest.fn();

    render(
      <Provider store={store}>
        <QuizComponent questions={mockQuestions} onAnswer={onAnswer} />
      </Provider>
    );
    
    const option = screen.getByText('4');
    fireEvent.click(option);
    
    await waitFor(() => {
      expect(onAnswer).toHaveBeenCalledWith(1, 1);
    });
  });

  test('shows score after completion', async () => {
    const mockQuestions = [
      {
        id: 1,
        text: 'Question 1',
        options: ['A', 'B', 'C', 'D'],
        correctAnswer: 0,
      },
    ];

    render(
      <Provider store={store}>
        <QuizComponent 
          questions={mockQuestions}
          showScore={true}
          score={80}
        />
      </Provider>
    );
    
    await waitFor(() => {
      expect(screen.getByText(/Score: 80%/i)).toBeInTheDocument();
    });
  });

  test('handles quiz submission', async () => {
    const onSubmit = jest.fn();
    
    render(
      <Provider store={store}>
        <QuizComponent onSubmit={onSubmit} />
      </Provider>
    );
    
    const submitButton = screen.getByRole('button', { name: /Submit/i });
    fireEvent.click(submitButton);
    
    await waitFor(() => {
      expect(onSubmit).toHaveBeenCalled();
    });
  });

  test('displays error state', () => {
    const error = 'Failed to load quiz';
    
    render(
      <Provider store={store}>
        <QuizComponent error={error} />
      </Provider>
    );
    
    expect(screen.getByText(error)).toBeInTheDocument();
  });

  test('handles navigation between questions', async () => {
    const mockQuestions = [
      {
        id: 1,
        text: 'Question 1',
        options: ['A', 'B', 'C', 'D'],
        correctAnswer: 0,
      },
      {
        id: 2,
        text: 'Question 2',
        options: ['E', 'F', 'G', 'H'],
        correctAnswer: 1,
      },
    ];

    render(
      <Provider store={store}>
        <QuizComponent questions={mockQuestions} />
      </Provider>
    );
    
    // Initially shows first question
    expect(screen.getByText('Question 1')).toBeInTheDocument();
    
    // Click next
    const nextButton = screen.getByRole('button', { name: /Next/i });
    fireEvent.click(nextButton);
    
    // Should show second question
    await waitFor(() => {
      expect(screen.getByText('Question 2')).toBeInTheDocument();
    });
    
    // Click previous
    const prevButton = screen.getByRole('button', { name: /Previous/i });
    fireEvent.click(prevButton);
    
    // Should show first question again
    await waitFor(() => {
      expect(screen.getByText('Question 1')).toBeInTheDocument();
    });
  });

  test('disables submit when no answers selected', () => {
    const mockQuestions = [
      {
        id: 1,
        text: 'Question 1',
        options: ['A', 'B', 'C', 'D'],
        correctAnswer: 0,
      },
    ];

    render(
      <Provider store={store}>
        <QuizComponent questions={mockQuestions} />
      </Provider>
    );
    
    const submitButton = screen.getByRole('button', { name: /Submit/i });
    expect(submitButton).toBeDisabled();
  });

  test('shows progress indicator', () => {
    const mockQuestions = Array(10).fill(null).map((_, i) => ({
      id: i + 1,
      text: `Question ${i + 1}`,
      options: ['A', 'B', 'C', 'D'],
      correctAnswer: 0,
    }));

    render(
      <Provider store={store}>
        <QuizComponent questions={mockQuestions} currentQuestion={3} />
      </Provider>
    );
    
    expect(screen.getByText(/Question 4 of 10/i)).toBeInTheDocument();
  });
});