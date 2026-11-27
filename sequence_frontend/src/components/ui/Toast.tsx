import React from 'react';
import { AlertCircle, AlertTriangle, CheckCircle2, Info, X } from 'lucide-react';

export type ToastType = 'error' | 'success' | 'info' | 'warning';

export interface ToastProps {
  type: ToastType;
  message: string;
  onClose: () => void;
}

export const Toast: React.FC<ToastProps> = ({ type, message, onClose }) => {
  const typeStyles: Record<ToastType, string> = {
    error: 'bg-rose-50 dark:bg-rose-950/90 text-rose-900 dark:text-rose-100 border-rose-300 dark:border-rose-800 border-l-4 border-l-rose-600 shadow-rose-500/20',
    warning: 'bg-amber-50 dark:bg-amber-950/90 text-amber-900 dark:text-amber-100 border-amber-300 dark:border-amber-800 border-l-4 border-l-amber-500 shadow-amber-500/20',
    success: 'bg-emerald-50 dark:bg-emerald-950/90 text-emerald-900 dark:text-emerald-100 border-emerald-300 dark:border-emerald-800 border-l-4 border-l-emerald-600 shadow-emerald-500/20',
    info: 'bg-[var(--bg-card)] text-[var(--text-primary)] border-[var(--border-color)] border-l-4 shadow-lg shadow-black/5',
  };

  const icons: Record<ToastType, React.ReactNode> = {
    error: <AlertCircle className="w-4 h-4 shrink-0 text-rose-600 dark:text-rose-400" />,
    warning: <AlertTriangle className="w-4 h-4 shrink-0 text-amber-600 dark:text-amber-400" />,
    success: <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-600 dark:text-emerald-400" />,
    info: <Info className="w-4 h-4 shrink-0" style={{ color: 'var(--accent-main)' }} />,
  };

  return (
    <div className="fixed bottom-4 right-4 z-50 animate-in fade-in slide-in-from-bottom-3 duration-200 pointer-events-auto max-w-md">
      <div
        className={`flex items-center gap-3 px-4 py-3 rounded-xl shadow-xl border backdrop-blur-md ${typeStyles[type]}`}
        style={type === 'info' ? { borderLeftColor: 'var(--accent-main)' } : undefined}
      >
        {icons[type]}
        <span className="text-xs font-bold tracking-wide flex-1">{message}</span>
        <button
          onClick={onClose}
          aria-label="Dismiss notification"
          className="ml-2 opacity-70 hover:opacity-100 transition-opacity p-1 rounded-md hover:bg-black/5 dark:hover:bg-white/10 cursor-pointer"
        >
          <X className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};
