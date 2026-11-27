import React, { useEffect, useRef, useState } from 'react';
import {
  Settings,
  Lightbulb,
  Play,
  Loader2,
  Trash2,
  Cpu,
  FolderOpen,
  ChevronDown,
  Check,
} from 'lucide-react';
import type { TrainingDetectorOption } from '../../types/trainingTypes';
import { FolderBrowserModal } from './FolderBrowserModal';

interface TrainingDetailsCardProps {
  modelName: string;
  prompt: string;
  baseDetector: string;
  outputDir: string;
  savedOutputDir: string;
  workspaceDir: string;
  defaultOutputDir: string;
  availableDetectors: TrainingDetectorOption[];
  isStarting: boolean;
  onChangeModelName: (value: string) => void;
  onChangePrompt: (value: string) => void;
  onChangeDetector: (value: string) => void;
  onChangeOutputDir: (value: string) => void;
  onSaveOutputDir: (value: string) => void;
  onStartTraining: () => void;
  onClearVideo: () => void;
}

export const TrainingDetailsCard: React.FC<TrainingDetailsCardProps> = ({
  modelName,
  prompt,
  baseDetector,
  outputDir,
  savedOutputDir,
  workspaceDir,
  defaultOutputDir,
  availableDetectors,
  isStarting,
  onChangeModelName,
  onChangePrompt,
  onChangeDetector,
  onChangeOutputDir,
  onSaveOutputDir,
  onStartTraining,
  onClearVideo,
}) => {
  const [detectorOpen, setDetectorOpen] = useState(false);
  const [folderModalOpen, setFolderModalOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const selectedDetectorObj = availableDetectors.find((d) => d.name === baseDetector);
  const isDirtyOutput = outputDir.trim() !== savedOutputDir;
  void workspaceDir;

  useEffect(() => {
    if (!detectorOpen) return;
    const handleOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDetectorOpen(false);
      }
    };
    const handleEsc = (e: KeyboardEvent) => {
      if (e.key === 'Escape') setDetectorOpen(false);
    };
    document.addEventListener('mousedown', handleOutside);
    document.addEventListener('keydown', handleEsc);
    return () => {
      document.removeEventListener('mousedown', handleOutside);
      document.removeEventListener('keydown', handleEsc);
    };
  }, [detectorOpen]);

  const handleBrowseFolder = async () => {
    const electronApi = (window as any).electronAPI;
    if (electronApi && typeof electronApi.selectFolder === 'function') {
      try {
        const picked = await electronApi.selectFolder();
        if (picked) {
          onChangeOutputDir(picked);
          onSaveOutputDir(picked);
        }
        return;
      } catch {
        // Fall back to cross-platform backend folder browser modal
      }
    }
    setFolderModalOpen(true);
  };

  return (
    <div className="w-full lg:w-[320px] xl:w-[350px] shrink-0 rounded-2xl bg-[var(--bg-card-subtle)] border border-[var(--border-color)] p-4 flex flex-col justify-between overflow-y-auto custom-scrollbar">
      <div className="flex-1 flex flex-col gap-3 min-h-0">
        <div className="flex items-center gap-2 shrink-0">
          <Settings className="w-4 h-4 text-[var(--accent-main)]" />
          <h4 className="text-sm font-extrabold text-[var(--text-primary)]">
            Training Details
          </h4>
        </div>

        {/* Training Name */}
        <div className="flex flex-col gap-1 shrink-0">
          <label className="text-xs font-bold text-[var(--text-primary)]">
            Training Name <span className="text-rose-500">*</span>
          </label>
          <input
            type="text"
            value={modelName}
            onChange={(e) => onChangeModelName(e.target.value)}
            placeholder="e.g. Assembling Box Model"
            className="w-full px-3 py-2 rounded-lg bg-[var(--bg-card)] border border-[var(--border-color)] text-xs text-[var(--text-primary)] font-medium placeholder:text-[var(--text-muted)] select-text focus:outline-none focus:border-[var(--accent-main)] transition-colors"
          />
        </div>

        {/* Custom Detection Model Dropdown */}
        <div className="flex flex-col gap-1 shrink-0" ref={dropdownRef}>
          <label className="text-xs font-bold text-[var(--text-primary)] flex items-center gap-1.5">
            <Cpu className="w-3.5 h-3.5 text-[var(--accent-main)]" />
            <span>Detection Model</span>
          </label>
          <div className="relative">
            <button
              type="button"
              onClick={() => setDetectorOpen((o) => !o)}
              aria-expanded={detectorOpen}
              className={`w-full flex items-center justify-between gap-2 px-3 py-2 rounded-lg bg-[var(--bg-card)] border text-left text-xs transition-colors cursor-pointer ${
                detectorOpen
                  ? 'border-[var(--accent-main)] ring-2 ring-[var(--accent-main)]/15'
                  : 'border-[var(--border-color)] hover:border-[var(--accent-main)]/60'
              }`}
            >
              <span className="truncate font-medium text-[var(--text-primary)]">
                {selectedDetectorObj
                  ? selectedDetectorObj.name
                  : 'Auto (Use wheel enabled models)'}
              </span>
              <div className="flex items-center gap-1.5 shrink-0">
                {selectedDetectorObj && (
                  <span className="px-1.5 py-0.5 rounded bg-[var(--accent-pond-subtle)] text-[var(--accent-main)] text-[10px] font-bold">
                    {selectedDetectorObj.classes?.length || 0} classes
                  </span>
                )}
                <ChevronDown
                  className={`w-3.5 h-3.5 text-[var(--text-muted)] transition-transform duration-150 ${
                    detectorOpen ? 'rotate-180 text-[var(--accent-main)]' : ''
                  }`}
                />
              </div>
            </button>

            {detectorOpen && (
              <div className="absolute left-0 right-0 top-full mt-1.5 z-50 rounded-xl bg-[var(--bg-card)] border border-[var(--border-color)] shadow-xl py-1 max-h-56 overflow-y-auto custom-scrollbar">
                {/* Auto option */}
                <button
                  type="button"
                  onClick={() => {
                    onChangeDetector('');
                    setDetectorOpen(false);
                  }}
                  className={`w-full flex items-start justify-between gap-2 px-3 py-2 text-left transition-colors cursor-pointer ${
                    !baseDetector
                      ? 'bg-[var(--accent-pond-subtle)]'
                      : 'hover:bg-[var(--bg-card-subtle)]'
                  }`}
                >
                  <div className="min-w-0 flex-1">
                    <div className="text-xs font-bold text-[var(--text-primary)]">
                      Auto (Use wheel enabled models)
                    </div>
                    <div className="text-[10.5px] text-[var(--text-muted)] mt-0.5">
                      Matches prompt against wheel models.json automatically
                    </div>
                  </div>
                  {!baseDetector && (
                    <Check className="w-3.5 h-3.5 text-[var(--accent-main)] shrink-0 mt-0.5" />
                  )}
                </button>

                {/* Individual detector entries */}
                {availableDetectors.map((det) => {
                  const isSelected = baseDetector === det.name;
                  return (
                    <button
                      key={det.name}
                      type="button"
                      onClick={() => {
                        onChangeDetector(det.name);
                        setDetectorOpen(false);
                      }}
                      className={`w-full flex items-start justify-between gap-2 px-3 py-2 text-left border-t border-[var(--border-color)]/60 transition-colors cursor-pointer ${
                        isSelected
                          ? 'bg-[var(--accent-pond-subtle)]'
                          : 'hover:bg-[var(--bg-card-subtle)]'
                      }`}
                    >
                      <div className="min-w-0 flex-1">
                        <div className="flex items-center gap-1.5">
                          <span className="text-xs font-bold text-[var(--text-primary)] truncate">
                            {det.name}
                          </span>
                          <span className="px-1.5 py-0.5 rounded bg-[var(--bg-card-subtle)] border border-[var(--border-color)] text-[9.5px] font-semibold text-[var(--text-secondary)] shrink-0">
                            {det.classes?.length || 0} classes
                          </span>
                        </div>
                        {det.classes?.length > 0 && (
                          <div className="text-[10.5px] text-[var(--text-muted)] mt-0.5 line-clamp-2">
                            {det.classes.map((c) => c.replace(/_/g, ' ')).join(', ')}
                          </div>
                        )}
                      </div>
                      {isSelected && (
                        <Check className="w-3.5 h-3.5 text-[var(--accent-main)] shrink-0 mt-0.5" />
                      )}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {selectedDetectorObj?.classes?.length ? (
            <p className="text-[10.5px] text-[var(--text-muted)] leading-snug">
              <span className="font-semibold text-[var(--text-secondary)]">Classes: </span>
              {selectedDetectorObj.classes.map((c) => c.replace(/_/g, ' ')).join(', ')}
            </p>
          ) : (
            <p className="text-[10.5px] text-[var(--text-muted)] leading-snug">
              Auto selects matching enabled detector(s) from the ML wheel.
            </p>
          )}
        </div>

        {/* Output Folder (saved in DB, shared by Training & Inference under sequence_train/) */}
        <div className="flex flex-col gap-1 shrink-0">
          <div className="flex items-center justify-between gap-2">
            <label className="text-xs font-bold text-[var(--text-primary)] flex items-center gap-1.5">
              <FolderOpen className="w-3.5 h-3.5 text-[var(--accent-main)]" />
              <span>Output Folder</span>
            </label>
            {savedOutputDir && !isDirtyOutput && (
              <span className="text-[10px] font-bold text-[var(--status-normal-text,#059669)]">
                Saved in DB
              </span>
            )}
          </div>
          <div className="flex items-center gap-1.5 select-text">
            <input
              type="text"
              value={outputDir}
              onChange={(e) => onChangeOutputDir(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter' && outputDir.trim()) {
                  onSaveOutputDir(outputDir.trim());
                }
              }}
              placeholder={savedOutputDir || defaultOutputDir || 'Select folder (saved in DB)'}
              title={outputDir || savedOutputDir || defaultOutputDir}
              className="flex-1 min-w-0 px-3 py-2 rounded-lg bg-[var(--bg-card)] border border-[var(--border-color)] text-xs text-[var(--text-primary)] font-mono placeholder:text-[var(--text-muted)] select-text focus:outline-none focus:border-[var(--accent-main)] transition-colors"
            />
            {isDirtyOutput && (
              <button
                type="button"
                onClick={() => onSaveOutputDir(outputDir.trim())}
                className="h-[34px] px-2.5 rounded-lg bg-[var(--btn-primary-bg)] hover:bg-[var(--btn-primary-hover)] text-[var(--btn-primary-text)] text-xs font-bold shrink-0 cursor-pointer transition-colors"
                title="Save output folder in DB for training & inference"
              >
                Save
              </button>
            )}
            <button
              type="button"
              onClick={handleBrowseFolder}
              className="h-[34px] px-3 rounded-lg bg-[var(--bg-card)] hover:bg-[var(--btn-secondary-hover)] border border-[var(--border-color)] text-xs font-bold text-[var(--text-primary)] flex items-center gap-1.5 shrink-0 cursor-pointer transition-colors"
              title="Browse folders (Windows & Linux)"
            >
              <FolderOpen className="w-3.5 h-3.5 text-[var(--accent-main)]" />
              <span>Browse</span>
            </button>
          </div>
        </div>

        {/* Training Prompt */}
        <div className="flex-1 flex flex-col gap-1 min-h-[120px]">
          <label className="text-xs font-bold text-[var(--text-primary)] shrink-0">
            Training Prompt <span className="text-rose-500">*</span>
          </label>
          <div className="relative flex-1 flex flex-col min-h-0">
            <textarea
              maxLength={500}
              value={prompt}
              onChange={(e) => onChangePrompt(e.target.value)}
              placeholder={"1. open box\n2. mouse\n3. instruction card..."}
              className="w-full flex-1 min-h-[105px] px-3 py-2 rounded-lg bg-[var(--bg-card)] border border-[var(--border-color)] text-xs text-[var(--text-primary)] font-medium placeholder:text-[var(--text-muted)] focus:outline-none focus:border-[var(--accent-main)] transition-colors resize-none pb-6"
            />
            <span className="absolute bottom-2 right-3 text-[10px] font-medium text-[var(--text-muted)]">
              {prompt.length} / 500
            </span>
          </div>
        </div>

        {/* Tip Banner */}
        <div className="p-2.5 rounded-lg bg-[var(--bg-card)] border border-[var(--border-color)] flex items-start gap-2 text-[11px] text-[var(--accent-main)] leading-relaxed shrink-0">
          <Lightbulb className="w-3.5 h-3.5 shrink-0 mt-0.5" />
          <span>
            <strong>Tip:</strong> Number each step (<code>1. ...</code>) and include the detector class name.
          </span>
        </div>
      </div>

      {/* Action Row: Start Training + Clear */}
      <div className="mt-4 flex items-center gap-2 shrink-0">
        <button
          type="button"
          disabled={isStarting}
          onClick={onStartTraining}
          className="flex-1 h-9 rounded-lg bg-[var(--btn-primary-bg)] hover:bg-[var(--btn-primary-hover)] text-[var(--btn-primary-text)] font-bold text-xs flex items-center justify-center gap-1.5 shadow-sm cursor-pointer active:scale-98 transition-all disabled:opacity-60"
        >
          {isStarting ? (
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
          ) : (
            <Play className="w-3.5 h-3.5 fill-current" />
          )}
          <span>Start Training</span>
        </button>

        <button
          type="button"
          disabled={isStarting}
          onClick={onClearVideo}
          title="Clear selected video"
          className="h-9 px-3.5 rounded-lg bg-rose-600 hover:bg-rose-700 active:bg-rose-800 text-white font-bold text-xs flex items-center justify-center gap-1.5 shadow-sm shadow-rose-600/20 cursor-pointer active:scale-95 transition-all border border-rose-500 shrink-0 disabled:opacity-60"
        >
          <Trash2 className="w-3.5 h-3.5" />
          <span>Clear</span>
        </button>
      </div>

      {/* Cross-platform folder picker modal (used when running in web browser on Linux/Windows) */}
      <FolderBrowserModal
        isOpen={folderModalOpen}
        initialPath={outputDir || savedOutputDir}
        defaultPath={defaultOutputDir}
        onSelect={(picked) => {
          onChangeOutputDir(picked);
          onSaveOutputDir(picked);
        }}
        onClose={() => setFolderModalOpen(false)}
      />
    </div>
  );
};
