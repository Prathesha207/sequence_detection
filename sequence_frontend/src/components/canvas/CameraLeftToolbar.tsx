import React from 'react';
import { Video, Disc, Square, Loader2, Power } from 'lucide-react';

interface CameraLeftToolbarProps {
  isStreaming: boolean;
  isRecording?: boolean;
  isSavingRecording?: boolean;
  recordingDuration?: number;
  onStartStream?: () => void;
  onStopStream?: () => void;
  onToggleRecording?: () => void;
  recordTooltip?: string;
}

const formatTime = (totalSeconds: number = 0) => {
  const mins = Math.floor(totalSeconds / 60);
  const secs = totalSeconds % 60;
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

export const CameraLeftToolbar: React.FC<CameraLeftToolbarProps> = ({
  isStreaming,
  isRecording = false,
  isSavingRecording = false,
  recordingDuration = 0,
  onStartStream,
  onStopStream,
  onToggleRecording,
  recordTooltip = 'Record Video',
}) => {
  return (
    <div
      onClick={(e) => e.stopPropagation()}
      className="absolute left-3 sm:left-4 top-1/2 -translate-y-1/2 z-30 flex flex-col items-center gap-1.5 p-1.5 rounded-xl bg-[var(--bg-card)]/95 backdrop-blur-md border border-[var(--border-color)] shadow-md select-none"
    >
      {/* Stream Status / Start Stream */}
      <button
        type="button"
        onClick={() => {
          if (!isStreaming) onStartStream?.();
        }}
        title={isStreaming ? 'Camera Stream Active' : 'Start Camera Stream'}
        aria-label={isStreaming ? 'Camera Stream Active' : 'Start Camera Stream'}
        className={`w-9 h-9 rounded-lg flex items-center justify-center border transition-all ${
          isStreaming
            ? 'bg-[var(--accent-pond-subtle)] text-[var(--accent-main)] border-[var(--accent-main)]/35 shadow-xs cursor-default'
            : 'bg-[var(--bg-card-subtle)] text-[var(--text-secondary)] border-[var(--border-color)] hover:bg-[var(--btn-secondary-hover)] hover:text-[var(--text-primary)] cursor-pointer active:scale-95'
        }`}
      >
        <Video className="w-4 h-4 stroke-[2]" />
      </button>

      {/* Record / Stop Recording */}
      {onToggleRecording && (
        <button
          type="button"
          disabled={!isStreaming || isSavingRecording}
          onClick={onToggleRecording}
          title={
            isSavingRecording
              ? 'Saving recording...'
              : isRecording
                ? `Stop Recording (${formatTime(recordingDuration)})`
                : recordTooltip
          }
          aria-label={isRecording ? 'Stop Recording' : recordTooltip}
          className={`w-9 h-9 rounded-lg flex items-center justify-center border transition-all ${
            !isStreaming || isSavingRecording
              ? 'bg-[var(--bg-card-subtle)] text-[var(--text-muted)] border-[var(--border-color)] cursor-not-allowed opacity-50'
              : isRecording
                ? 'bg-[var(--status-bad)] text-white border-[var(--status-bad)] shadow-sm animate-pulse cursor-pointer active:scale-95'
                : 'bg-[var(--status-normal-bg)] text-[var(--status-normal-text)] border-[var(--status-normal-border)] hover:bg-[var(--status-ok)] hover:text-white cursor-pointer active:scale-95'
          }`}
        >
          {isSavingRecording ? (
            <Loader2 className="w-4 h-4 animate-spin" />
          ) : isRecording ? (
            <Square className="w-3.5 h-3.5 fill-current" />
          ) : (
            <Disc className="w-4 h-4 stroke-[2]" />
          )}
        </button>
      )}

      {/* Divider */}
      <div className="w-6 h-px bg-[var(--border-color)] my-0.5" />

      {/* Stop Stream */}
      {onStopStream && (
        <button
          type="button"
          disabled={!isStreaming || isRecording || isSavingRecording}
          onClick={onStopStream}
          title={
            isRecording
              ? 'Stop recording before stopping stream'
              : 'Stop Camera Stream'
          }
          aria-label="Stop Camera Stream"
          className={`w-9 h-9 rounded-lg flex items-center justify-center border transition-all ${
            !isStreaming || isRecording || isSavingRecording
              ? 'bg-[var(--status-bad-bg)] text-[var(--status-bad-text)] border-[var(--status-bad-border)] opacity-40 cursor-not-allowed'
              : 'bg-[var(--status-bad-bg)] text-[var(--status-bad-text)] border-[var(--status-bad-border)] hover:bg-[var(--status-bad)] hover:text-white cursor-pointer active:scale-95'
          }`}
        >
          <Power className="w-4 h-4 stroke-[2]" />
        </button>
      )}
    </div>
  );
};
