import React from 'react';
import { Check, Box, Tag, Hash, Clock, CheckCircle2, Play, Plus, FolderOpen } from 'lucide-react';
import type { TrainingJobState } from '../../types/trainingTypes';

interface TrainingCompletedViewProps {
  job: TrainingJobState;
  fallbackModelName: string;
  onStartInference: () => void;
  onNewTraining: () => void;
}

const formatElapsedDuration = (totalSeconds: number): string => {
  const safe = Math.max(0, Math.round(totalSeconds || 0));
  const mins = Math.floor(safe / 60);
  const secs = safe % 60;
  return `${mins.toString().padStart(2, '0')} min ${secs.toString().padStart(2, '0')} sec`;
};

export const TrainingCompletedView: React.FC<TrainingCompletedViewProps> = ({
  job,
  fallbackModelName,
  onStartInference,
  onNewTraining,
}) => {
  const displayModelName = job.model_name || fallbackModelName || 'Custom SOP Model';
  const displayJobId = job.job_id || 'TR-001';
  const displayDuration = formatElapsedDuration(job.elapsed_seconds);

  return (
    <div className="relative w-full flex-1 h-full min-h-[350px] lg:min-h-0 flex flex-col items-center justify-center rounded-3xl bg-[var(--bg-card)] border border-[var(--border-color)] shadow-sm p-6 sm:p-8 overflow-y-auto custom-scrollbar select-none">
      <div className="w-full max-w-lg flex flex-col items-center text-center">
        {/* Celebration Checkmark with Subtle Theme Accents */}
        <div className="relative flex items-center justify-center mb-4">
          <span className="absolute -left-14 -top-2 w-2.5 h-1 rounded-full bg-[var(--accent-main)]/70 -rotate-45" />
          <span className="absolute -left-10 top-8 w-2 h-2 rounded-full bg-[var(--status-warn)]/70" />
          <span className="absolute -right-14 -top-1 w-2.5 h-1 rounded-full bg-[var(--status-warn)]/70 rotate-45" />
          <span className="absolute -right-10 top-7 w-2 h-2 rounded-full bg-[var(--accent-main)]/70" />

          <div className="w-14 h-14 rounded-full bg-[var(--status-normal-bg)] border border-[var(--status-normal-border)] flex items-center justify-center text-[var(--status-normal-text)] shadow-xs">
            <Check className="w-7 h-7 stroke-[2.75]" />
          </div>
        </div>

        {/* Heading & Subtitle */}
        <h2 className="text-xl sm:text-2xl font-extrabold text-[var(--text-primary)] tracking-tight">
          Training Completed!
        </h2>
        <p className="text-xs sm:text-sm text-[var(--text-secondary)] mt-1 font-medium">
          Your model has been successfully trained and saved.
        </p>

        {/* Trained Model Summary Card */}
        <div className="w-full mt-6 rounded-2xl bg-[var(--bg-card)] border border-[var(--border-color)] shadow-xs overflow-hidden text-left">
          <div className="flex items-center gap-2 px-4 py-3 bg-[var(--bg-card-subtle)] border-b border-[var(--border-color)]">
            <Box className="w-4 h-4 text-[var(--accent-main)]" />
            <span className="text-xs sm:text-sm font-bold text-[var(--text-primary)]">
              Trained Model
            </span>
          </div>

          <div className="divide-y divide-[var(--border-color)] text-xs sm:text-[13px]">
            <div className="flex items-center justify-between gap-4 px-4 py-3">
              <span className="flex items-center gap-2 text-[var(--text-secondary)] font-medium">
                <Tag className="w-3.5 h-3.5 text-[var(--text-muted)] shrink-0" />
                <span>Model Name</span>
              </span>
              <span className="font-bold text-[var(--text-primary)] truncate" title={displayModelName}>
                {displayModelName}
              </span>
            </div>

            <div className="flex items-center justify-between gap-4 px-4 py-3">
              <span className="flex items-center gap-2 text-[var(--text-secondary)] font-medium">
                <Hash className="w-3.5 h-3.5 text-[var(--text-muted)] shrink-0" />
                <span>Training ID</span>
              </span>
              <span className="font-mono font-semibold text-[var(--text-primary)]">
                {displayJobId}
              </span>
            </div>

            <div className="flex items-center justify-between gap-4 px-4 py-3">
              <span className="flex items-center gap-2 text-[var(--text-secondary)] font-medium">
                <Clock className="w-3.5 h-3.5 text-[var(--text-muted)] shrink-0" />
                <span>Training Duration</span>
              </span>
              <span className="font-semibold text-[var(--text-primary)] tabular-nums">
                {displayDuration}
              </span>
            </div>

            <div className="flex items-center justify-between gap-4 px-4 py-3">
              <span className="flex items-center gap-2 text-[var(--text-secondary)] font-medium">
                <CheckCircle2 className="w-3.5 h-3.5 text-[var(--text-muted)] shrink-0" />
                <span>Status</span>
              </span>
              <span className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-lg bg-[var(--status-normal-bg)] text-[var(--status-normal-text)] border border-[var(--status-normal-border)] text-xs font-bold">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--status-normal-text)]" />
                <span>Ready</span>
              </span>
            </div>

            {job.output_dir && (
              <div className="flex items-center justify-between gap-4 px-4 py-3">
                <span className="flex items-center gap-2 text-[var(--text-secondary)] font-medium shrink-0">
                  <FolderOpen className="w-3.5 h-3.5 text-[var(--text-muted)] shrink-0" />
                  <span>Output Folder</span>
                </span>
                <span
                  className="font-mono text-[11px] font-medium text-[var(--text-primary)] truncate max-w-[260px]"
                  title={job.output_dir}
                >
                  {job.output_dir}
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Action Buttons Row */}
        <div className="w-full mt-6 grid grid-cols-1 sm:grid-cols-2 gap-3">
          <button
            type="button"
            onClick={onStartInference}
            className="flex items-center justify-center gap-2.5 px-4 py-3 rounded-xl bg-[var(--btn-primary-bg)] hover:bg-[var(--btn-primary-hover)] text-[var(--btn-primary-text)] shadow-sm cursor-pointer active:scale-[0.99] transition-all text-left"
          >
            <Play className="w-4 h-4 fill-current shrink-0" />
            <div className="flex flex-col leading-tight">
              <span className="text-xs sm:text-sm font-bold">Start Inference</span>
              <span className="text-[10.5px] opacity-85 font-medium">Use this trained model</span>
            </div>
          </button>

          <button
            type="button"
            onClick={onNewTraining}
            className="flex items-center justify-center gap-2.5 px-4 py-3 rounded-xl bg-[var(--bg-card)] hover:bg-[var(--btn-secondary-hover)] text-[var(--text-primary)] border border-[var(--border-color)] shadow-xs cursor-pointer active:scale-[0.99] transition-all text-left"
          >
            <Plus className="w-4 h-4 stroke-[2.5] text-[var(--accent-main)] shrink-0" />
            <div className="flex flex-col leading-tight">
              <span className="text-xs sm:text-sm font-bold">New Training</span>
              <span className="text-[10.5px] text-[var(--text-muted)] font-medium">Train another model</span>
            </div>
          </button>
        </div>
      </div>
    </div>
  );
};
