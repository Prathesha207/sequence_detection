import React from 'react';
import { useInferenceStore } from '../../store/inferenceStore';
import type { AnomalyStatus, DetectionBox } from '../../types';
import { Activity } from 'lucide-react';

interface StatusBarProps {
  anomalyStatus?: AnomalyStatus;
  fps?: number;
  backendStatus?: string;
  detections?: DetectionBox[];
}

export const StatusBar: React.FC<StatusBarProps> = ({
  fps = 0,
}) => {
  const stats = useInferenceStore((state) => state.stats);

  const displayFps = fps > 0 ? fps : (stats.fps || 0);
  const sequence = stats.sequence || [];
  const doing = stats.doing;
  const steps = stats.steps || [];
  const totalSteps = stats.total_steps || (steps.length > 0 ? steps.length : 5);
  const doneCount = sequence.length > 0 ? sequence.length : stats.steps_done_count || 0;
  const progressPercent = Math.min(100, Math.round((doneCount / totalSteps) * 100));

  const isAnomaly = stats.is_anomaly_frame || stats.status === 'ANOMALY';
  const isWarming = stats.status === 'WARMING';
  const doingStepObj = steps.find((s) => s.id === doing);
  const doingLabel = doingStepObj ? doingStepObj.label : doing ? `Step ${doing}` : 'Awaiting start';

  return (
    <div className="absolute bottom-3 left-1/2 transform -translate-x-1/2 z-30 pointer-events-auto animate-in fade-in slide-in-from-bottom-2 duration-300 max-w-[calc(100%-24px)] w-auto">
      <div className="flex items-center gap-3 sm:gap-5 px-3.5 sm:px-5 py-2 rounded-xl bg-[var(--bg-card)]/95 backdrop-blur-md border border-[var(--border-color)] shadow-xl text-[var(--text-primary)]">

        {/* Currently Doing Step */}
        <div className="flex items-center gap-2">
          <div className="w-6 h-6 rounded-lg bg-[var(--accent-soft)] border border-[var(--accent-main)]/30 flex items-center justify-center text-[var(--accent-main)] font-mono font-bold text-xs shrink-0">
            {doing ? `#${doing}` : '—'}
          </div>
          <div className="flex flex-col">
            <span className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider font-bold">
              Doing
            </span>
            <span className="text-xs sm:text-sm font-bold text-[var(--text-primary)] truncate max-w-[140px] sm:max-w-[200px]">
              {doingLabel}
            </span>
          </div>
        </div>

        <div className="h-6 w-[1px] bg-[var(--border-color)]" />

        {/* Steps Progress */}
        <div className="flex flex-col items-center">
          <span className="text-[9px] text-[var(--text-muted)] uppercase tracking-wider font-bold">
            Progress
          </span>
          <div className="flex items-center gap-1 mt-0.5 font-mono">
            <span className="text-xs sm:text-sm font-extrabold text-[var(--accent-main)]">
              {doneCount}
            </span>
            <span className="text-xs text-[var(--text-muted)]">/</span>
            <span className="text-xs sm:text-sm font-extrabold text-emerald-600 dark:text-emerald-400">
              {totalSteps}
            </span>
            <span className="text-[10px] font-bold text-[var(--accent-main)] ml-1">
              ({progressPercent}%)
            </span>
          </div>
        </div>

        <div className="h-6 w-[1px] bg-[var(--border-color)]" />

        {/* Status Badge */}
        <div className="flex items-center gap-1.5">
          <span
            className={`px-2.5 py-1 rounded-lg text-xs font-black tracking-wide uppercase ${
              isAnomaly
                ? 'bg-[var(--status-bad-bg)] text-[var(--status-bad-text)] border border-[var(--status-bad)]'
                : isWarming
                ? 'bg-[var(--status-warn-bg)] text-[var(--status-warn-text)] border border-[var(--status-warn)]'
                : 'bg-[var(--status-ok-bg)] text-[var(--status-ok-text)] border border-[var(--status-ok)]'
            }`}
          >
            {isAnomaly ? 'ANOMALY' : isWarming ? 'WARMING' : 'NORMAL'}
          </span>
        </div>

        {/* Telemetry Metrics: FPS in theme color, Latency, Frames */}
        <div className="h-6 w-[1px] bg-[var(--border-color)]" />
        <div className="flex items-center gap-3 text-xs font-mono">
          <div className="flex items-center gap-1 text-[var(--text-secondary)]">
            <Activity className="w-3.5 h-3.5 text-[var(--accent-main)]" />
            <span className="font-bold text-[var(--accent-main)]">{displayFps > 0 ? displayFps.toFixed(1) : (stats.fps || 0).toFixed(1)}</span>
            <span className="text-[10px] text-[var(--text-muted)]">FPS</span>
          </div>

          <div className="flex items-center gap-1 text-[var(--text-secondary)]">
            <span className="font-bold">{Math.round(stats.latency_ms || stats.metrics?.latency_ms || (displayFps > 0 ? 1000 / displayFps : 32))}</span>
            <span className="text-[10px] text-[var(--text-muted)]">ms</span>
          </div>

          <div className="hidden sm:flex items-center gap-1 text-[var(--text-secondary)]">
            <span className="font-bold">#{stats.frames_processed || 0}</span>
            <span className="text-[10px] text-[var(--text-muted)]">frames</span>
          </div>
        </div>

      </div>
    </div>
  );
};
