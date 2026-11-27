import React, { useEffect, useState } from 'react';
import { getApiBaseUrl } from '../../lib/api';
import { useInferenceStore } from '../../store/inferenceStore';
import { SOPModel } from '../../types';
import { Check, Loader2, X, Cpu } from 'lucide-react';

interface ModelChooserModalProps {
  isOpen: boolean;
  onClose: () => void;
  activeModel: string;
  onSelectModel: (modelName: string) => void;
}

export const ModelChooserModal: React.FC<ModelChooserModalProps> = ({
  isOpen,
  onClose,
  activeModel,
  onSelectModel,
}) => {
  const [models, setModels] = useState<SOPModel[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [selected, setSelected] = useState<string>(activeModel || '');

  useEffect(() => {
    if (!isOpen) return;

    setLoading(true);
    fetch(`${getApiBaseUrl()}/models`)
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json();
      })
      .then((data) => {
        if (Array.isArray(data)) {
          setModels(data);
          useInferenceStore.getState().setAvailableModels(data);
          const matching = data.find((m: any) => m.name === activeModel);
          if (matching) {
            setSelected(matching.name);
          } else {
            setSelected('');
          }
        } else {
          setModels([]);
          setSelected('');
        }
      })
      .catch((err) => {
        console.warn('Failed to fetch /models from backend:', err);
        setModels([]);
        setSelected('');
      })
      .finally(() => setLoading(false));
  }, [isOpen, activeModel]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape' && isOpen) onClose();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const handleConfirm = (modelToSelect?: string) => {
    const target = modelToSelect || selected;
    if (target) {
      onSelectModel(target);
      onClose();
    }
  };

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-xs grid place-items-center p-3 select-none"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="w-full max-w-[420px] rounded-xl bg-[var(--bg-card)] border border-[var(--border-color)] shadow-2xl p-4 text-[var(--text-primary)] animate-in fade-in zoom-in-95 duration-150">
        
        {/* Compact Header */}
        <div className="flex items-center justify-between pb-2 border-b border-[var(--border-color)]">
          <div className="flex items-center gap-2">
            <Cpu className="w-4 h-4 text-[var(--accent-main)] shrink-0" />
            <h2 className="text-sm font-extrabold tracking-tight">Select SOP Model</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-[var(--text-muted)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-card-subtle)] transition cursor-pointer"
          >
            <X className="w-3.5 h-3.5" />
          </button>
        </div>

        {/* Dynamic Model List (Purely fetched from GET /models API) */}
        <div className="my-2.5 max-h-[45vh] overflow-y-auto space-y-1.5 pr-0.5">
          {loading ? (
            <div className="py-8 flex flex-col items-center justify-center gap-2 text-xs text-[var(--text-muted)]">
              <Loader2 className="w-5 h-5 animate-spin text-[var(--accent-main)]" />
              <span>Loading trained models...</span>
            </div>
          ) : models.length === 0 ? (
            <div className="py-8 flex flex-col items-center justify-center gap-1.5 text-center text-xs text-[var(--text-muted)]">
              <Cpu className="w-6 h-6 opacity-40 text-[var(--text-muted)]" />
              <span className="font-semibold text-[var(--text-primary)]">No Trained Models Found</span>
              <span className="text-[11px] max-w-[240px]">No sequence models available on backend. Check your ML model installation.</span>
            </div>
          ) : (
            models.map((m) => {
              const isChosen = selected === m.name;
              const stepCount = m.num_steps || (m.labels ? m.labels.length : ((m as any).steps || 0));
              return (
                <div
                  key={m.name}
                  onClick={() => setSelected(m.name)}
                  onDoubleClick={() => handleConfirm(m.name)}
                  className={`flex items-center gap-2.5 px-3 py-2 rounded-lg border cursor-pointer transition-all select-none ${
                    isChosen
                      ? 'border-[var(--accent-main)] bg-[var(--accent-main)]/10 shadow-xs'
                      : 'border-[var(--border-color)] bg-[var(--bg-card-subtle)] hover:border-[var(--accent-main)]/50'
                  }`}
                >
                  <div
                    className={`w-4 h-4 rounded-full border flex items-center justify-center shrink-0 transition-all ${
                      isChosen
                        ? 'border-[var(--accent-main)] bg-[var(--accent-main)] text-white'
                        : 'border-slate-300 dark:border-slate-600 bg-[var(--bg-card)]'
                    }`}
                  >
                    {isChosen && <Check className="w-2.5 h-2.5 stroke-[3]" />}
                  </div>

                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-1.5">
                      <span className="text-xs font-bold truncate block">{m.name}</span>
                      {(m.is_active || (m as any).active) && (
                        <span className="text-[9px] font-bold px-1.5 py-0.2 rounded bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/30 shrink-0">
                          Active
                        </span>
                      )}
                    </div>
                    <span className="text-[10px] text-[var(--text-muted)] block mt-0.5 font-mono">
                      {stepCount > 0 ? `${stepCount} steps · trained sequence` : 'trained sequence model'}
                    </span>
                  </div>
                </div>
              );
            })
          )}
        </div>

        {/* Actions */}
        <div className="flex justify-end items-center gap-2 pt-2.5 border-t border-[var(--border-color)]">
          <button
            onClick={onClose}
            className="px-3 py-1.5 rounded-lg text-xs font-semibold text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-card-subtle)] transition cursor-pointer"
          >
            Cancel
          </button>
          <button
            onClick={() => handleConfirm()}
            disabled={!selected || models.length === 0}
            className="px-3.5 py-1.5 rounded-lg text-xs font-bold bg-[var(--accent-main)] text-white hover:opacity-90 active:scale-95 transition disabled:opacity-50 cursor-pointer shadow-xs"
          >
            Select & Load Model
          </button>
        </div>

      </div>
    </div>
  );
};
