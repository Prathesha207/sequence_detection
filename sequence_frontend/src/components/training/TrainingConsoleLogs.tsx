import React, { useRef, useEffect, useState, useCallback } from 'react';
import { Terminal, Copy, Check, ArrowDown } from 'lucide-react';

interface TrainingConsoleLogsProps {
  logs: string[];
  isTrainingActive?: boolean;
  fillHeight?: boolean;
}

export const TrainingConsoleLogs: React.FC<TrainingConsoleLogsProps> = ({
  logs,
  isTrainingActive = false,
  fillHeight = false,
}) => {
  const containerRef = useRef<HTMLDivElement>(null);
  const isAtBottomRef = useRef<boolean>(true);
  const [showScrollDown, setShowScrollDown] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleScroll = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    const nearBottom = distanceFromBottom <= 32;
    isAtBottomRef.current = nearBottom;
    setShowScrollDown(!nearBottom);
  }, []);

  const scrollToBottom = useCallback(() => {
    const el = containerRef.current;
    if (!el) return;
    el.scrollTop = el.scrollHeight;
    isAtBottomRef.current = true;
    setShowScrollDown(false);
  }, []);

  // Standard terminal auto-scroll: follow new logs only if user hasn't scrolled up
  useEffect(() => {
    if (isAtBottomRef.current && containerRef.current) {
      containerRef.current.scrollTop = containerRef.current.scrollHeight;
    }
  }, [logs]);

  const handleCopy = () => {
    navigator.clipboard.writeText(logs.join('\n')).catch(() => {});
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const getLogStyle = (text: string) => {
    const lower = text.toLowerCase();
    if (lower.includes('error') || lower.includes('failed') || lower.includes('halted')) {
      return 'text-[var(--status-bad-text,#DC2626)] font-semibold';
    }
    if (
      lower.includes('done in') ||
      lower.includes('training completed') ||
      lower.includes('normal')
    ) {
      return 'text-[var(--status-normal-text,#059669)] font-semibold';
    }
    if (/^\s*\[(\d+)\s+[^\]]+\]/.test(text) || text.includes('[train]')) {
      return 'text-[var(--accent-main)] font-semibold';
    }
    if (text.includes('[progress]') || text.includes('[System]')) {
      return 'text-[var(--text-primary)]';
    }
    return 'text-[var(--text-secondary)]';
  };

  return (
    <div
      className={`relative flex flex-col w-full rounded-xl bg-[var(--terminal-bg)] border border-[var(--terminal-border)] text-[var(--text-primary)] overflow-hidden ${
        fillHeight ? 'flex-1 min-h-0' : ''
      }`}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 bg-[var(--bg-card-subtle)] border-b border-[var(--terminal-border)] shrink-0">
        <div className="flex items-center gap-2 min-w-0">
          <Terminal className="w-3.5 h-3.5 text-[var(--accent-main)] shrink-0" />
          <span className="text-[11px] font-bold text-[var(--text-primary)] tracking-wide uppercase truncate">
            Live Terminal
          </span>
          {isTrainingActive && (
            <span className="flex items-center gap-1.5 text-[10px] font-bold text-[var(--text-primary)] bg-[var(--bg-card)] px-2 py-0.5 rounded-lg border border-[var(--border-color)] shrink-0">
              <span className="w-1.5 h-1.5 rounded-full bg-[var(--status-normal-text,#10B981)] animate-pulse" />
              Live
            </span>
          )}
        </div>

        <button
          type="button"
          onClick={handleCopy}
          className="flex items-center gap-1 px-2.5 py-1 text-[10px] font-semibold rounded-lg bg-[var(--bg-card)] hover:bg-[var(--btn-secondary-hover)] border border-[var(--border-color)] text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer transition-colors shrink-0"
          title="Copy logs to clipboard"
        >
          {copied ? (
            <Check className="w-3 h-3 text-[var(--status-normal-text,#10B981)]" />
          ) : (
            <Copy className="w-3 h-3" />
          )}
          <span>{copied ? 'Copied' : 'Copy'}</span>
        </button>
      </div>

      {/* Log lines container with standard scroll behavior */}
      <div
        ref={containerRef}
        onScroll={handleScroll}
        className={`${
          fillHeight ? 'flex-1 min-h-[220px]' : 'h-44 sm:h-52'
        } p-3 overflow-y-auto custom-scrollbar font-mono text-[11px] leading-relaxed flex flex-col gap-1 select-text`}
      >
        {logs.length === 0 ? (
          <span className="text-[var(--text-muted)] italic">Waiting for ML output...</span>
        ) : (
          logs.map((log, index) => (
            <div key={index} className={`flex items-start gap-1.5 ${getLogStyle(log)}`}>
              <span className="opacity-45 select-none shrink-0">&gt;</span>
              <span className="break-words min-w-0 flex-1">{log}</span>
            </div>
          ))
        )}
      </div>

      {/* Jump-to-latest pill shown when user scrolls up */}
      {showScrollDown && (
        <button
          type="button"
          onClick={scrollToBottom}
          className="absolute bottom-3 right-3 flex items-center gap-1 px-2.5 py-1 rounded-full bg-[var(--btn-primary-bg)] hover:bg-[var(--btn-primary-hover)] text-[var(--btn-primary-text)] text-[10px] font-bold shadow-md cursor-pointer transition-all"
          title="Resume auto-scroll"
        >
          <ArrowDown className="w-3 h-3" />
          <span>Latest</span>
        </button>
      )}
    </div>
  );
};
