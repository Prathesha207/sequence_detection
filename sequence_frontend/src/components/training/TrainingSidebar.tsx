import React from 'react';
import type { StreamSourceType, CameraConfig } from '../../types';
import { useTrainingStore } from '../../store/trainingStore';
import { CameraImageAdjustmentsCard } from '../CameraImageAdjustmentsCard';
import { TrainingConsoleLogs } from './TrainingConsoleLogs';
import {
  Sparkles,
  CheckCircle2,
  Loader2,
  AlertTriangle,
} from 'lucide-react';

interface TrainingSidebarProps {
  sourceType: StreamSourceType;
  isCameraConnected?: boolean;
  cameraConfig?: CameraConfig;
  onUpdateCameraConfig?: (patch: any) => void;
  isRecording?: boolean;
  onShowToast?: (type: 'success' | 'error' | 'info' | 'warning', message: string) => void;
}

export const TrainingSidebar: React.FC<TrainingSidebarProps> = ({
  sourceType,
  isCameraConnected = false,
  cameraConfig,
  onUpdateCameraConfig,
  isRecording = false,
  onShowToast,
}) => {
  const {
    job,
    hyperparams,
    selectedDatasetVideoName,
  } = useTrainingStore();

  const isRunning =
    job.status === 'training' || job.status === 'preparing' || job.status === 'evaluating';
  const showActiveSidebar = isRunning || job.status === 'completed' || job.status === 'failed';

  const isCameraSource = sourceType === 'oak-camera' || sourceType === 'webcam';

  // Idle / Standby state: clean placeholder card matching SequenceSidebar
  if (!showActiveSidebar) {
    return (
      <aside className="w-full lg:w-[350px] xl:w-[360px] flex-shrink-0 h-full max-h-full flex flex-col gap-3 overflow-y-auto pr-1 custom-scrollbar">
        <div className="bg-[var(--bg-card)] border border-[var(--border-color)] rounded-xl p-4 shadow-sm text-[var(--text-primary)] select-none">
          <div className="flex flex-col items-center text-center py-5">
            <div className="w-10 h-10 rounded-xl bg-[var(--accent-main)]/10 border border-[var(--accent-main)]/30 flex items-center justify-center text-[var(--accent-main)] mb-2.5">
              <Sparkles className="w-5 h-5 stroke-[1.75]" />
            </div>
            <h3 className="text-xs font-extrabold text-[var(--text-primary)] mb-0.5">
              {selectedDatasetVideoName
                ? 'Training Video Ready'
                : isCameraSource
                  ? (isCameraConnected ? 'Camera Training Mode' : 'Camera Offline')
                  : 'No Training Video Selected'}
            </h3>
            <p className="text-[11px] text-[var(--text-muted)] max-w-[230px] leading-relaxed">
              {selectedDatasetVideoName
                ? `${selectedDatasetVideoName} · Enter Training Name & Prompt, then click Start Training.`
                : isCameraSource
                  ? (isCameraConnected
                      ? 'Start the camera stream and record a demonstration video. After stopping, the training card will open automatically.'
                      : 'Connect an OAK camera or switch to Video mode to begin training.')
                  : 'Upload a training video to begin training.'}
            </p>
          </div>
        </div>

        {/* Camera Adjustments Card: shown in camera mode when connected and not recording */}
        {isCameraSource && isCameraConnected && !isRecording && (
          <CameraImageAdjustmentsCard
            cameraId={cameraConfig?.id}
            config={cameraConfig}
            onUpdateConfig={onUpdateCameraConfig}
            isLive={isCameraConnected}
            isRecording={isRecording}
            onShowToast={onShowToast}
          />
        )}
      </aside>
    );
  }

  // Active / Completed / Failed state: compact status header + live themed terminal
  const displayName = job.model_name || hyperparams.model_name || 'SOP Model';
  return (
    <aside className="w-full lg:w-[350px] xl:w-[370px] flex-shrink-0 h-full max-h-full bg-[var(--bg-card)] border border-[var(--border-color)] rounded-xl p-3.5 shadow-sm text-[var(--text-primary)] flex flex-col gap-3 overflow-hidden">
      {/* Header */}
      <header className="flex justify-between items-baseline gap-2 shrink-0">
        <div className="min-w-0 flex-1">
          <h2 className="text-base font-extrabold tracking-tight">Training Console</h2>
          <p className="text-[11px] text-[var(--text-muted)] truncate block font-medium" title={displayName}>
            {displayName} · {job.stage_name || `Stage ${job.current_epoch}/${job.total_epochs}`}
          </p>
        </div>
        <span className="text-xs font-extrabold text-[var(--accent-main)] tabular-nums shrink-0">
          {job.progress_percent}%
        </span>
      </header>

      {/* Progress Bar */}
      <div className="h-2 w-full bg-[var(--bg-card-subtle)] rounded-full overflow-hidden border border-[var(--border-color)] shrink-0">
        <div
          className="h-full transition-all duration-200"
          style={{
            width: `${job.progress_percent}%`,
            backgroundColor:
              job.status === 'completed'
                ? 'var(--status-ok,#10B981)'
                : job.status === 'failed'
                  ? 'var(--status-bad,#EF4444)'
                  : 'var(--accent-main)',
          }}
        />
      </div>

      {/* Compact Status Banner */}
      <div
        className={`flex items-center justify-between px-3 py-2 rounded-xl border text-xs font-bold shrink-0 ${
          job.status === 'completed'
            ? 'bg-[var(--status-normal-bg)] border-[var(--status-normal-border)] text-[var(--status-normal-text)]'
            : job.status === 'failed'
              ? 'bg-[var(--status-bad-bg)] border-[var(--status-bad-border)] text-[var(--status-bad-text)]'
              : 'bg-[var(--accent-pond-subtle)] border-[var(--border-color)] text-[var(--text-primary)]'
        }`}
      >
        <span className="flex items-center gap-1.5">
          {isRunning ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin text-[var(--accent-main)]" />
          ) : job.status === 'failed' ? (
            <AlertTriangle className="w-3.5 h-3.5 text-[var(--status-bad-text)]" />
          ) : (
            <CheckCircle2 className="w-3.5 h-3.5 text-[var(--status-normal-text,#10B981)]" />
          )}
          <span className="capitalize">{job.status}</span>
        </span>
        <span className="text-[11px] font-mono">
          {job.status === 'completed'
            ? (job.accuracy != null && job.accuracy > 0 ? `Acc: ${job.accuracy.toFixed(1)}%` : job.self_check || 'Ready')
            : job.train_loss > 0
              ? `Loss: ${job.train_loss.toFixed(3)}`
              : `Stage ${Math.min(job.total_epochs || 9, job.current_epoch + 1)}/${job.total_epochs || 9}`}
        </span>
      </div>

      {/* Live Terminal occupying right sidebar */}
      <TrainingConsoleLogs logs={job.logs} isTrainingActive={isRunning} fillHeight />
    </aside>
  );
};
