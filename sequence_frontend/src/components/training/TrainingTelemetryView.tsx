import React from 'react';
import {
  Clock,
  RotateCcw,
  Square,
  AlertTriangle,
  Plus,
  Hourglass,
  Cpu,
  Activity,
  TrendingDown,
} from 'lucide-react';
import type { TrainingJobState, TrainingStageInfo } from '../../types/trainingTypes';
import { TrainingCompletedView } from './TrainingCompletedView';

interface TrainingTelemetryViewProps {
  job: TrainingJobState;
  fallbackModelName: string;
  isRunning: boolean;
  onStop: () => void;
  onReset: () => void;
  onNewTraining?: () => void;
  onDeploy: () => void;
}

const formatDurationClock = (totalSeconds?: number | null): string => {
  if (totalSeconds == null || !Number.isFinite(totalSeconds) || totalSeconds <= 0) return '00:00';
  const safe = Math.max(0, Math.round(totalSeconds));
  const hrs = Math.floor(safe / 3600);
  const mins = Math.floor((safe % 3600) / 60);
  const secs = safe % 60;
  if (hrs > 0) {
    return `${hrs}:${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
  }
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

const formatEtaDisplay = (etaSeconds?: number | null): string => {
  if (etaSeconds == null || !Number.isFinite(etaSeconds) || etaSeconds <= 0) return '--';
  const safe = Math.round(etaSeconds);
  if (safe < 60) return `~${safe}s`;
  return `~${formatDurationClock(safe)}`;
};

export const TrainingTelemetryView: React.FC<TrainingTelemetryViewProps> = ({
  job,
  fallbackModelName,
  isRunning,
  onStop,
  onReset,
  onNewTraining,
  onDeploy,
}) => {
  if (job.status === 'completed') {
    return (
      <TrainingCompletedView
        job={job}
        fallbackModelName={fallbackModelName}
        onStartInference={onDeploy}
        onNewTraining={onNewTraining || onReset}
      />
    );
  }

  const stages: TrainingStageInfo[] = job.stages && job.stages.length > 0 ? job.stages : [];
  const totalStages = stages.length || job.total_epochs || 9;
  const activeStageObj = stages.find((s) => s.status === 'active') || stages.find((s) => s.status === 'failed');
  const displayStageNum = activeStageObj
    ? activeStageObj.id + 1
    : Math.min(totalStages, (job.current_epoch || 0) + 1);

  const pct = Math.max(0, Math.min(100, Number(job.progress_percent) || 0));
  const ringSize = 200;
  const ringCenter = ringSize / 2;
  const ringRadius = 74;
  const ringCircumference = 2 * Math.PI * ringRadius;
  const ringOffset = ringCircumference * (1 - pct / 100);
  const gradId = `train-ring-grad-${job.job_id || 'active'}`;
  const glowId = `train-ring-glow-${job.job_id || 'active'}`;
  const trackGlowId = `train-track-glow-${job.job_id || 'active'}`;

  const displayModelName = job.model_name || fallbackModelName || 'SOP Model';
  const isFailed = job.status === 'failed';
  const stageTitle = job.stage_name || activeStageObj?.title || 'Preparing Training';
  const stageDetail = job.stage_detail || activeStageObj?.description || 'Initializing ML training pipeline...';

  return (
    <div className="relative w-full flex-1 h-full min-h-[380px] lg:min-h-0 flex flex-col rounded-2xl bg-[var(--bg-card)] border border-[var(--border-color)] shadow-sm overflow-hidden select-none">
      {/* Compact Top Action Bar */}
      <div className="flex flex-wrap items-center justify-between gap-2 px-4 sm:px-5 py-3 bg-[var(--bg-card-subtle)] border-b border-[var(--border-color)] shrink-0">
        <div className="flex items-center gap-2.5 min-w-0">
          <span className="text-xs sm:text-sm font-extrabold text-[var(--text-primary)] truncate">
            {displayModelName}
          </span>
          <span
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-[10.5px] font-bold uppercase tracking-wider border shrink-0 ${isRunning
                ? 'bg-[var(--btn-secondary-bg)] text-[var(--btn-secondary-text)] border-[var(--btn-secondary-border)]'
                : isFailed
                  ? 'bg-[var(--status-bad-bg)] text-[var(--status-bad-text)] border-[var(--status-bad-border)]'
                  : 'bg-[var(--status-normal-bg)] text-[var(--status-normal-text)] border-[var(--status-normal-border)]'
              }`}
          >
            {isRunning && <span className="w-2 h-2 rounded-full bg-[var(--accent-main)] animate-pulse" />}
            <span>{isRunning ? 'Training in Progress' : job.status}</span>
          </span>
        </div>

        <div className="flex items-center gap-2 shrink-0 whitespace-nowrap">
          {isRunning && (
            <button
              type="button"
              onClick={onStop}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[var(--status-bad,#EF4444)] hover:opacity-90 text-white text-xs font-bold cursor-pointer transition-opacity"
            >
              <Square className="w-3 h-3 fill-current" />
              <span>Stop Training</span>
            </button>
          )}
          {!isRunning && onNewTraining && (
            <button
              type="button"
              onClick={onNewTraining}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[var(--btn-primary-bg)] hover:bg-[var(--btn-primary-hover)] text-[var(--btn-primary-text)] text-xs font-bold shadow-xs cursor-pointer transition-colors"
            >
              <Plus className="w-3.5 h-3.5" />
              <span>New Training</span>
            </button>
          )}
          {!isRunning && (
            <button
              type="button"
              onClick={onReset}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-[var(--bg-card)] border border-[var(--border-color)] text-xs font-semibold text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer transition-colors"
            >
              <RotateCcw className="w-3 h-3" />
              <span>{isFailed ? 'Edit & Retry' : 'Back'}</span>
            </button>
          )}
        </div>
      </div>

      {/* Centered Body */}
      <div className="flex-1 min-h-0 p-5 sm:p-8 overflow-y-auto custom-scrollbar flex flex-col items-center justify-center">
        {isFailed && (
          <div className="w-full max-w-lg mb-5 p-3.5 rounded-2xl bg-[var(--status-bad-bg)] border border-[var(--status-bad-border)] flex items-start gap-3 text-xs text-[var(--status-bad-text)]">
            <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
            <div className="min-w-0 flex-1">
              <p className="font-bold">Training Failed</p>
              <p className="mt-0.5 font-medium break-words">
                {job.error || 'Training pipeline encountered an error. Check the Live Terminal.'}
              </p>
            </div>
          </div>
        )}

        <div className="w-full max-w-md flex flex-col items-center text-center gap-5">
          {/* Top Icon & Headline */}
          <div className="flex flex-col items-center gap-1.5">
            <div className="w-11 h-11 rounded-2xl bg-[var(--accent-pond-subtle)] border border-[var(--border-color)] flex items-center justify-center text-[var(--accent-main)] shadow-xs mb-1">
              <Cpu className={`w-5 h-5 ${isRunning ? 'animate-pulse' : ''}`} />
            </div>
            <h2 className="text-lg sm:text-xl font-extrabold text-[var(--text-primary)] tracking-tight">
              {isFailed ? 'Training Stopped with Error' : `Training ${displayModelName}`}
            </h2>
            {job.selected_video && (
              <p className="text-xs text-[var(--text-muted)] font-medium truncate max-w-sm">
                {job.selected_video}
                {job.model_kind ? ` · ${job.model_kind}` : ''}
              </p>
            )}
          </div>

          {/* Sleek Fluid Energy-Ring Progress Gauge */}
          <div className="relative flex items-center justify-center my-1" style={{ width: ringSize, height: ringSize }}>
            {/* Soft ambient radial halo */}
            <div
              className={`absolute inset-6 rounded-full blur-2xl transition-opacity duration-500 ${isFailed
                  ? 'bg-rose-500/20 opacity-90'
                  : isRunning
                    ? 'bg-[var(--accent-main)]/20 opacity-100 animate-pulse'
                    : 'bg-[var(--accent-main)]/10 opacity-70'
                }`}
            />

            <svg
              width={ringSize}
              height={ringSize}
              viewBox={`0 0 ${ringSize} ${ringSize}`}
              className="absolute inset-0 z-10 overflow-visible"
            >
              <defs>
                <linearGradient id={gradId} x1="0%" y1="0%" x2="100%" y2="100%">
                  <stop
                    offset="0%"
                    stopColor={isFailed ? '#FB7185' : 'var(--accent-main, #7886C7)'}
                  />
                  <stop
                    offset="60%"
                    stopColor={isFailed ? '#F43F5E' : 'var(--accent-pond-dark, #5E6EB9)'}
                  />
                  <stop
                    offset="100%"
                    stopColor={isFailed ? '#DC2626' : 'var(--accent-main, #7886C7)'}
                  />
                </linearGradient>
                <filter id={glowId} x="-35%" y="-35%" width="170%" height="170%">
                  <feGaussianBlur stdDeviation="3.5" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
                <filter id={trackGlowId} x="-35%" y="-35%" width="170%" height="170%">
                  <feGaussianBlur stdDeviation="2.5" result="blur" />
                  <feMerge>
                    <feMergeNode in="blur" />
                    <feMergeNode in="SourceGraphic" />
                  </feMerge>
                </filter>
              </defs>

              {/* Enhanced Flowing Organic Outer Aura Waves */}
              {!isFailed && (
                <g filter={`url(#${trackGlowId})`}>
                  <ellipse
                    cx={ringCenter}
                    cy={ringCenter}
                    rx={ringRadius + 13}
                    ry={ringRadius + 5}
                    fill="none"
                    stroke="var(--accent-main)"
                    strokeWidth="1.6"
                    opacity={isRunning ? 0.34 : 0.16}
                    className={isRunning ? 'animate-spin' : ''}
                    style={{
                      transformOrigin: `${ringCenter}px ${ringCenter}px`,
                      animationDuration: '11s',
                    }}
                  />
                  <ellipse
                    cx={ringCenter}
                    cy={ringCenter}
                    rx={ringRadius + 6}
                    ry={ringRadius + 14}
                    fill="none"
                    stroke="var(--accent-pond-dark, var(--accent-main))"
                    strokeWidth="1.35"
                    opacity={isRunning ? 0.26 : 0.12}
                    className={isRunning ? 'animate-spin' : ''}
                    style={{
                      transformOrigin: `${ringCenter}px ${ringCenter}px`,
                      animationDuration: '15s',
                      animationDirection: 'reverse',
                    }}
                  />
                  <ellipse
                    cx={ringCenter}
                    cy={ringCenter}
                    rx={ringRadius + 16}
                    ry={ringRadius + 9}
                    fill="none"
                    stroke="var(--accent-main)"
                    strokeWidth="1.1"
                    opacity={isRunning ? 0.2 : 0.08}
                    className={isRunning ? 'animate-spin' : ''}
                    style={{
                      transformOrigin: `${ringCenter}px ${ringCenter}px`,
                      animationDuration: '20s',
                    }}
                  />
                </g>
              )}

              {/* Slimmer Glowing Track Ring */}
              <circle
                cx={ringCenter}
                cy={ringCenter}
                r={ringRadius}
                fill="none"
                stroke={isFailed ? '#F43F5E' : 'var(--accent-main)'}
                strokeWidth="8"
                opacity="0.14"
              />
              <circle
                cx={ringCenter}
                cy={ringCenter}
                r={ringRadius}
                fill="none"
                stroke={isFailed ? '#F43F5E' : 'var(--accent-main)'}
                strokeWidth="2"
                opacity="0.35"
                filter={`url(#${trackGlowId})`}
              />

              {/* Slimmer, Sleek Gradient Progress Arc */}
              <circle
                cx={ringCenter}
                cy={ringCenter}
                r={ringRadius}
                fill="none"
                stroke={`url(#${gradId})`}
                strokeWidth="8"
                strokeLinecap="round"
                strokeDasharray={ringCircumference}
                strokeDashoffset={ringOffset}
                transform={`rotate(-90 ${ringCenter} ${ringCenter})`}
                filter={pct > 0 ? `url(#${glowId})` : undefined}
                className="transition-all duration-500 ease-out"
              />
            </svg>

            {/* Clean Center Typography */}
            <div className="absolute inset-0 z-20 flex flex-col items-center justify-center pointer-events-none">
              <span className="text-4xl font-extrabold tracking-tight text-[var(--text-primary)] tabular-nums leading-none">
                {pct}%
              </span>
              <span className="mt-1.5 text-xs font-medium text-[var(--text-secondary)]">
                {isFailed ? 'Failed' : isRunning ? 'Training...' : 'Ready'}
              </span>
            </div>
          </div>

          {/* Active Stage Pill Card */}
          <div className="w-full rounded-2xl bg-[var(--bg-card-subtle)] border border-[var(--border-color)] px-4 py-3.5 flex items-center gap-3.5 text-left shadow-xs">
            <div
              className={`w-10 h-10 rounded-full flex items-center justify-center shrink-0 border ${isFailed
                  ? 'bg-[var(--status-bad-bg)] border-[var(--status-bad-border)] text-[var(--status-bad-text)]'
                  : 'bg-[var(--accent-pond-subtle)] border-[var(--border-color)] text-[var(--accent-main)]'
                }`}
            >
              <Activity className={`w-5 h-5 ${isRunning ? 'animate-pulse' : ''}`} />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center justify-between gap-2">
                <p className="text-xs sm:text-sm font-extrabold text-[var(--text-primary)] truncate">
                  Stage {displayStageNum} / {totalStages} · {stageTitle}
                </p>
                {job.train_loss > 0 && (
                  <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-md bg-[var(--bg-card)] border border-[var(--border-color)] text-[10px] font-mono font-bold text-[var(--text-primary)] shrink-0">
                    <TrendingDown className="w-3 h-3 text-[var(--accent-main)]" />
                    {job.train_loss.toFixed(3)}
                  </span>
                )}
              </div>
              <p className="text-[11px] sm:text-xs text-[var(--text-secondary)] font-medium mt-0.5 truncate" title={stageDetail}>
                {stageDetail}
              </p>
            </div>
          </div>

          {/* Elapsed | ETA Footer */}
          <div className="flex items-center justify-center gap-5 sm:gap-7 text-xs pt-1">
            <div className="flex items-center gap-1.5 text-[var(--text-secondary)]">
              <Clock className="w-3.5 h-3.5 text-[var(--accent-main)]" />
              <span className="font-medium text-[var(--text-muted)]">Elapsed</span>
              <span className="font-extrabold font-mono text-[var(--text-primary)] tabular-nums">
                {formatDurationClock(job.elapsed_seconds)}
              </span>
            </div>

            <span className="h-3.5 w-px bg-[var(--border-color)]" />

            <div className="flex items-center gap-1.5 text-[var(--text-secondary)]">
              <Hourglass className="w-3.5 h-3.5 text-[var(--accent-main)]" />
              <span className="font-medium text-[var(--text-muted)]">ETA</span>
              <span className="font-extrabold font-mono text-[var(--text-primary)] tabular-nums">
                {formatEtaDisplay(job.eta_seconds)}
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};
