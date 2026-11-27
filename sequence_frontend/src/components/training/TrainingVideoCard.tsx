import React from 'react';
import { CheckCircle2, RefreshCw } from 'lucide-react';
import type { TrainingVideoMetadata } from '../../types/trainingTypes';

interface TrainingVideoCardProps {
  videoUrl: string;
  videoName: string | null;
  metadata: TrainingVideoMetadata | null;
  onMetadataLoaded: (e: React.SyntheticEvent<HTMLVideoElement>) => void;
  onReplaceVideo: () => void;
  replaceLabel?: string;
}

export const TrainingVideoCard: React.FC<TrainingVideoCardProps> = ({
  videoUrl,
  videoName,
  metadata,
  onMetadataLoaded,
  onReplaceVideo,
  replaceLabel = 'Replace Video',
}) => {
  return (
    <div className="flex-1 min-w-0 rounded-2xl bg-[var(--bg-card-subtle)] border border-[var(--border-color)] p-4 sm:p-5 flex flex-col justify-between">
      <div className="flex flex-col flex-1 min-h-0">
        <div className="flex items-center justify-between mb-3 shrink-0">
          <h4 className="text-sm font-extrabold text-[var(--text-primary)]">
            Selected Video
          </h4>
          <div className="inline-flex items-center gap-1.5 text-[11px] font-bold text-emerald-600 bg-emerald-500/10 px-2.5 py-0.5 rounded-full border border-emerald-500/20">
            <CheckCircle2 className="w-3.5 h-3.5" />
            <span>Ready</span>
          </div>
        </div>

        <div className="relative flex-1 min-h-[200px] rounded-xl overflow-hidden bg-black mb-3.5 border border-[var(--border-color)] flex items-center justify-center">
          <video
            src={videoUrl}
            controls
            onLoadedMetadata={onMetadataLoaded}
            className="w-full h-full max-h-[380px] object-contain"
            playsInline
          />
        </div>

        <p
          className="text-xs sm:text-sm font-bold text-[var(--text-primary)] truncate"
          title={videoName || ''}
        >
          {videoName}
        </p>
        <p className="text-xs text-[var(--text-muted)] mt-0.5 font-medium">
          {metadata && metadata.width > 0
            ? `${metadata.width} × ${metadata.height} • `
            : ''}
          {metadata?.sizeFormatted || ''}
          {metadata?.durationFormatted && metadata.durationFormatted !== '--:--'
            ? ` • ${metadata.durationFormatted}`
            : ''}
        </p>
      </div>

      <div className="mt-4 pt-3 border-t border-[var(--border-color)] shrink-0">
        <button
          type="button"
          onClick={onReplaceVideo}
          className="inline-flex items-center gap-1.5 text-xs font-bold text-[var(--accent-main)] hover:opacity-80 transition-opacity cursor-pointer"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>{replaceLabel}</span>
        </button>
      </div>
    </div>
  );
};
