import { useEffect, useRef, useState, RefObject } from 'react';

interface UseIntersectionObserverProps {
  threshold?: number | number[];
  root?: Element | null;
  rootMargin?: string;
  triggerOnce?: boolean;
}

/**
 * Intersection Observer hook for lazy loading and animations
 */
export function useIntersectionObserver<T extends Element>(
  options: UseIntersectionObserverProps = {}
): [RefObject<T | null>, boolean] {
  const {
    threshold = 0,
    root = null,
    rootMargin = '0px',
    triggerOnce = false,
  } = options;

  const targetRef = useRef<T | null>(null);
  const [isIntersecting, setIsIntersecting] = useState(false);
  const [hasIntersected, setHasIntersected] = useState(false);

  useEffect(() => {
    const target = targetRef.current;

    if (!target || (triggerOnce && hasIntersected)) {
      return;
    }

    const observer = new IntersectionObserver(
      ([entry]) => {
        const intersecting = entry.isIntersecting;
        setIsIntersecting(intersecting);

        if (intersecting && !hasIntersected) {
          setHasIntersected(true);
        }
      },
      {
        threshold,
        root,
        rootMargin,
      }
    );

    observer.observe(target);

    return () => {
      observer.disconnect();
    };
  }, [threshold, root, rootMargin, triggerOnce, hasIntersected]);

  return [targetRef, triggerOnce ? hasIntersected : isIntersecting];
}

/**
 * Hook for implementing infinite scroll
 */
export function useInfiniteScroll(
  callback: () => void,
  options: UseIntersectionObserverProps = {}
) {
  const [ref, isIntersecting] = useIntersectionObserver<HTMLDivElement>(options);

  useEffect(() => {
    if (isIntersecting) {
      callback();
    }
  }, [isIntersecting, callback]);

  return ref;
}