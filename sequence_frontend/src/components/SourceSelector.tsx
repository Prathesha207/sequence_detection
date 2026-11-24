import React, { useState } from 'react';
import { StreamSourceType } from '../types';
import { Video, Camera, Play, Square, Loader2, Disc, Trash2 } from 'lucide-react';

import { useInferenceStore } from '../store/inferenceStore';
import { ModelChooserModal } from './sequence/ModelChooserModal';
import { selectAndLoadModel } from '../utils/modelService';

const formatTime = (totalSeconds: number = 0) => {
  const mins = Math.floor(totalSeconds / 60);
  const secs = totalSeconds % 60;
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
};

interface SourceSelectorProps {
  sourceType: StreamSourceType;
  onSourceChange: (type: StreamSourceType) => void;

  onOpenSettings?: () => void;
  onCustomVideoUploaded?: (videoUrl: string, fileName: string, sessionId?: string) => void;
  customVideoName?: string;
  customVideoUrl?: string;
  videoSessionId?: string | null;
  hasActiveVideo?: boolean;
  onClearCustomVideo?: () => void;

  isRunning?: boolean;
  isStarting?: boolean;
  isRecording?: boolean;
  isSavingRecording?: boolean;
  recordingDuration?: number;
  onToggleRecording?: () => void;
  onToggleRunning?: () => void;
  onStopInference?: () => void;
  onResumeInference?: () => void;
  onRequestSwitchMode?: (type: StreamSourceType) => void;
  isStreaming?: boolean;
  onStartStream?: () => void;
  onStopStream?: () => void;
  isCameraConnected?: boolean;
  cameraStartingState?: 'idle' | 'waking_camera' | 'waiting_frame' | 'ready';
  cameraRecordSessionId?: string | null;
  onClearCameraRecord?: () => void;
}

export const SourceSelector: React.FC<SourceSelectorProps> = ({
  sourceType,
  onSourceChange,

  onOpenSettings: _onOpenSettings,
  customVideoName: _customVideoName,
  customVideoUrl: _customVideoUrl,
  hasActiveVideo = false,
  onClearCustomVideo,
  isRunning = false,
  isStarting = false,
  isRecording = false,
  isSavingRecording = false,
  recordingDuration = 0,
  onToggleRecording,
  onToggleRunning,
  onStopInference,
  onResumeInference,
  onRequestSwitchMode,
  isStreaming = false,
  onStartStream,
  onStopStream,
  isCameraConnected = true,
  cameraStartingState = 'ready',
  cameraRecordSessionId,
  onClearCameraRecord: _onClearCameraRecord,
}) => {
  const isVideoLoading = useInferenceStore((state) => state.isVideoLoading);
  const activeModel = useInferenceStore((state) => state.activeModel);
  const setActiveModel = useInferenceStore((state) => state.setActiveModel);

  const isModelModalOpen = useInferenceStore((state) => state.isModelModalOpen);
  const setIsModelModalOpen = useInferenceStore((state) => state.setIsModelModalOpen);

  const handleSourceClick = (targetType: StreamSourceType) => {
    if (isRunning) return;

    if (onRequestSwitchMode) {
      onRequestSwitchMode(targetType);
    } else {
      onSourceChange(targetType);
    }
  };

  const isVideoMode = sourceType === 'sample-pond' || sourceType === 'uploaded-video';
  const isCameraMode = sourceType === 'oak-camera' || sourceType === 'webcam';
  const hasMediaToPlay = (isCameraMode && isCameraConnected) || isVideoMode;

  return (
    <div className="w-full flex-shrink-0">
      <div className="flex flex-row items-center justify-between gap-2 sm:gap-4 p-2 sm:p-3 rounded-2xl bg-[var(--bg-card)] border border-[var(--border-color)] shadow-sm flex-wrap">

        {/* Source Toggle Tabs */}
        <div className="flex bg-[var(--bg-card-subtle)] p-1 rounded-xl border border-[var(--border-color)] flex-shrink-0">
          <button
            disabled={isRunning}
            onClick={() => handleSourceClick('uploaded-video')}
            title={isRunning ? 'Stop inference before switching source' : 'Switch to Video File mode'}
            className={`flex items-center justify-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-1.5 sm:py-2 rounded-lg text-xs font-bold transition-all ${isRunning ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
              } ${sourceType === 'uploaded-video'
                ? 'bg-[var(--bg-card)] text-[var(--text-primary)] shadow-xs'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
          >
            <Video className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-[var(--accent-main)]" />
            <span>Video<span className="hidden sm:inline"> File</span></span>
          </button>

          <button
            disabled={isRunning}
            onClick={() => handleSourceClick('oak-camera')}
            title={isRunning ? 'Stop inference before switching source' : 'Switch to OAK Camera mode'}
            className={`flex items-center justify-center gap-1.5 sm:gap-2 px-3 sm:px-4 py-1.5 sm:py-2 rounded-lg text-xs font-bold transition-all ${isRunning ? 'opacity-50 cursor-not-allowed' : 'cursor-pointer'
              } ${sourceType === 'oak-camera'
                ? 'bg-[var(--bg-card)] text-[var(--text-primary)] shadow-xs'
                : 'text-[var(--text-secondary)] hover:text-[var(--text-primary)]'
              }`}
          >
            <Camera className="w-3.5 h-3.5 sm:w-4 sm:h-4 text-[var(--accent-main)]" />
            <span>OAK<span className="hidden sm:inline"> Camera</span></span>
          </button>
        </div>

        {/* Stream Play/Stop Controls */}
        <div className="flex items-center justify-end gap-1.5 sm:gap-3 flex-shrink-0 ml-auto">
          {hasMediaToPlay && (
            <div className="flex items-center gap-1 sm:gap-2 pl-2 sm:pl-3 border-l border-[var(--border-color)]">
              {isCameraMode && cameraStartingState !== 'ready' ? (
                <button
                  disabled
                  className="h-8 sm:h-9 flex items-center gap-1.5 px-3.5 sm:px-4 rounded-xl bg-amber-600/80 text-white font-bold text-[11px] sm:text-xs shadow-xs cursor-wait opacity-90 transition-all shrink-0"
                >
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>{cameraStartingState === 'waking_camera' ? 'WAKING CAMERA...' : 'WARMING UP...'}</span>
                </button>
              ) : isStarting ? (
                <button
                  disabled
                  className="h-8 sm:h-9 flex items-center gap-1.5 px-3.5 sm:px-4 rounded-xl bg-emerald-600/80 text-white font-bold text-[11px] sm:text-xs shadow-xs cursor-wait opacity-90 transition-all shrink-0"
                >
                  <Loader2 className="w-3.5 h-3.5 animate-spin" />
                  <span>STARTING...</span>
                </button>
              ) : isRunning ? (
                <button
                  onClick={() => {

                    if (onStopInference) onStopInference();
                    else if (onToggleRunning) onToggleRunning();
                  }}
                  title="Stop AI inference"
                  className="h-8 sm:h-9 flex items-center gap-1.5 px-3 sm:px-4 rounded-xl bg-[var(--status-bad)] hover:opacity-90 text-white font-bold text-[11px] sm:text-xs shadow-xs active:scale-95 cursor-pointer transition-all"
                >
                  <Square className="w-3.5 h-3.5 fill-current" />
                  <span>{isCameraMode ? 'FINISH INSPECTION' : 'STOP'}</span>
                </button>
              ) : (
                <div className="flex items-center gap-1.5 sm:gap-2">
                  <button
                    disabled={isStarting || isVideoLoading}
                    onClick={() => {
                      if (isStarting || isVideoLoading) return;

                      // In video mode: when no video is uploaded/running, always open the modal so user can choose model
                      if (isVideoMode && !hasActiveVideo) {
                        setIsModelModalOpen(true);
                        return;
                      }

                      // In camera mode: if no model selected yet, open modal so user can choose model
                      if (isCameraMode && !activeModel) {
                        setIsModelModalOpen(true);
                        return;
                      }

                      // If video is loaded or camera is active, start/resume inference
                      if (onResumeInference) onResumeInference();
                      else if (onToggleRunning) onToggleRunning();
                    }}
                    title="Start SOP sequence inference"
                    className={`h-8 sm:h-9 flex items-center gap-1.5 px-3 sm:px-5 rounded-xl font-bold text-[11px] sm:text-xs shadow-sm transition-all shrink-0 ${isStarting || isVideoLoading
                      ? 'bg-emerald-950/80 text-emerald-400/80 border border-emerald-700/50 cursor-not-allowed opacity-80'
                      : 'bg-[var(--accent-main)] text-white hover:opacity-90 active:scale-95 cursor-pointer'
                      }`}
                  >
                    {isStarting || isVideoLoading ? (
                      <Loader2 className="w-3.5 h-3.5 animate-spin" />
                    ) : (
                      <Play className="w-3.5 h-3.5 fill-current" />
                    )}
                    <span className="whitespace-nowrap">START INFERENCE</span>
                  </button>

                  {/* Clear Video button */}
                  {isVideoMode && hasActiveVideo && !isRunning && (
                    <button
                      disabled={isStarting || isVideoLoading}
                      onClick={() => {
                        onClearCustomVideo?.();
                      }}
                      title="Clear video and results"
                      className="h-8 sm:h-9 flex items-center gap-1.5 px-3 sm:px-4 rounded-xl bg-rose-600 hover:bg-rose-700 active:bg-rose-800 text-white font-bold text-[11px] sm:text-xs shadow-md shadow-rose-600/30 active:scale-95 cursor-pointer transition-all shrink-0 border border-rose-500"
                    >
                      <Trash2 className="w-3.5 h-3.5" />
                      <span>CLEAR</span>
                    </button>
                  )}

                  {/* Camera Record Button */}
                  {isCameraMode && isStreaming && !isRunning && !cameraRecordSessionId && (
                    <button
                      disabled={isSavingRecording}
                      onClick={() => {

                        onToggleRecording?.();
                      }}
                      title={isRecording ? 'Stop camera recording' : 'Start recording camera stream'}
                      className={`h-8 sm:h-9 flex items-center gap-1.5 px-3 rounded-xl font-bold text-[11px] sm:text-xs shadow-xs transition-all shrink-0 cursor-pointer active:scale-95 ${isRecording
                        ? 'bg-[var(--status-bad)] text-white animate-pulse'
                        : 'bg-[var(--status-bad-bg)] border border-[var(--status-bad)]/40 text-[var(--status-bad-text)] hover:bg-[var(--status-bad)] hover:text-white'
                        }`}
                    >
                      {isRecording ? (
                        <>
                          <Disc className="w-3.5 h-3.5 animate-spin" />
                          <span>REC {formatTime(recordingDuration)}</span>
                        </>
                      ) : (
                        <>
                          <Video className="w-3.5 h-3.5" />
                          <span>REC</span>
                        </>
                      )}
                    </button>
                  )}

                  {/* Camera Stream Only */}
                  {isCameraMode && !cameraRecordSessionId && (
                    !isStreaming ? (
                      <button
                        onClick={() => {

                          onStartStream?.();
                        }}
                        title="Start camera stream without inference"
                        className="h-8 sm:h-9 flex items-center gap-1.5 px-3 rounded-xl bg-[var(--bg-card-subtle)] hover:border-[var(--accent-main)] border border-[var(--border-color)] text-[var(--text-primary)] font-bold text-[11px] sm:text-xs active:scale-95 cursor-pointer transition-all shrink-0"
                      >
                        <Play className="w-3 h-3 fill-current" />
                        <span>STREAM</span>
                      </button>
                    ) : (
                      <button
                        onClick={() => {

                          onStopStream?.();
                        }}
                        title="Stop camera stream"
                        className="h-8 sm:h-9 flex items-center gap-1.5 px-3 rounded-xl bg-[var(--bg-card-subtle)] hover:bg-[var(--status-bad)] hover:text-white border border-[var(--border-color)] text-[var(--text-primary)] font-bold text-[11px] sm:text-xs active:scale-95 cursor-pointer transition-all shrink-0"
                      >
                        <Square className="w-3 h-3 fill-current" />
                        <span>STOP</span>
                      </button>
                    )
                  )}
                </div>
              )}
            </div>
          )}
        </div>

      </div>

      {/* Model Chooser Dialog Modal */}
      <ModelChooserModal
        isOpen={isModelModalOpen}
        onClose={() => setIsModelModalOpen(false)}
        activeModel={activeModel}
        onSelectModel={(modelName) => {
          selectAndLoadModel(modelName, !isCameraMode, isCameraMode);
        }}
      />
    </div>
  );
};
