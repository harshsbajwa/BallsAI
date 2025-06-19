import React from 'react';

import { cn } from './utils';

interface LoadingSpinnerProps extends React.HTMLAttributes<HTMLDivElement> {
  size?: 'sm' | 'md' | 'lg';
}

const LoadingSpinner = React.forwardRef<HTMLDivElement, LoadingSpinnerProps>(
  ({ className, size = 'md', ...props }, ref) => {
    const sizeClasses = {
      sm: 'w-4 h-4',
      md: 'w-6 h-6',
      lg: 'w-8 h-8',
    };

    return (
      <div
        ref={ref}
        className={cn(
          'animate-spin rounded-full border-2 border-current border-t-transparent',
          sizeClasses[size],
          className
        )}
        {...props}
      />
    );
  }
);
LoadingSpinner.displayName = "LoadingSpinner";

interface LoadingProps {
  text?: string;
  size?: 'sm' | 'md' | 'lg';
}

const Loading: React.FC<LoadingProps> = ({ text = 'Loading...', size = 'md' }) => {
  return (
    <div className="flex items-center justify-center space-x-2 p-4">
      <LoadingSpinner size={size} />
      <span className="text-muted-foreground">{text}</span>
    </div>
  );
};

export { LoadingSpinner, Loading };