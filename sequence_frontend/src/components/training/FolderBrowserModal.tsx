import React, { useEffect, useState, useCallback } from 'react';
import axios from 'axios';
import {
  Folder,
  FolderPlus,
  ArrowUp,
  Home,
  HardDrive,
  Loader2,
  Check,
  AlertCircle,
  Copy,
  CornerDownLeft,
} from 'lucide-react';
import { Modal } from '../ui/Modal';
import { getApiBaseUrl } from '../../lib/api';

interface FolderEntry {
  name: string;
  path: string;
}

interface FolderBrowserModalProps {
  isOpen: boolean;
  initialPath: string;
  defaultPath: string;
  onSelect: (selectedPath: string) => void;
  onClose: () => void;
}

export const FolderBrowserModal: React.FC<FolderBrowserModalProps> = ({
  isOpen,
  initialPath,
  defaultPath,
  onSelect,
  onClose,
}) => {
  const [currentPath, setCurrentPath] = useState<string>('');
  const [pathInput, setPathInput] = useState<string>('');
  const [copied, setCopied] = useState(false);
  const [parentPath, setParentPath] = useState<string | null>(null);
  const [roots, setRoots] = useState<string[]>([]);
  const [homePath, setHomePath] = useState<string>('');
  const [folders, setFolders] = useState<FolderEntry[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [creatingFolder, setCreatingFolder] = useState(false);
  const [newFolderName, setNewFolderName] = useState('');

  const loadFolder = useCallback(async (targetPath?: string) => {
    setLoading(true);
    setError(null);
    try {
      const baseUrl = getApiBaseUrl();
      const q = targetPath ? `?path=${encodeURIComponent(targetPath)}` : '';
      const res = await axios.get(`${baseUrl}/training/browse-folders${q}`);
      if (res.data) {
        const nextPath = res.data.current_path || '';
        setCurrentPath(nextPath);
        setPathInput(nextPath);
        setParentPath(res.data.parent_path ?? null);
        setRoots(Array.isArray(res.data.roots) ? res.data.roots : []);
        setHomePath(res.data.home_path || '');
        setFolders(Array.isArray(res.data.folders) ? res.data.folders : []);
      }
    } catch (err: any) {
      setError(err?.response?.data?.detail || err?.message || 'Could not browse directories');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    if (isOpen) {
      setCreatingFolder(false);
      setNewFolderName('');
      loadFolder(initialPath || defaultPath || '');
    }
  }, [isOpen, initialPath, defaultPath, loadFolder]);

  const handleCreateFolder = async () => {
    const trimmed = newFolderName.trim();
    if (!trimmed || !currentPath) return;
    setLoading(true);
    setError(null);
    try {
      const baseUrl = getApiBaseUrl();
      const res = await axios.post(`${baseUrl}/training/mkdir`, {
        parent_path: currentPath,
        folder_name: trimmed,
      });
      setCreatingFolder(false);
      setNewFolderName('');
      if (res.data?.path) {
        await loadFolder(res.data.path);
      } else {
        await loadFolder(currentPath);
      }
    } catch (err: any) {
      setError(err?.response?.data?.detail || 'Failed to create folder');
      setLoading(false);
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onClose={onClose}
      maxWidth="lg"
      icon={<Folder className="w-4 h-4" />}
      title="Choose Output Folder"
      description="Browse directories on Windows or Linux and select where trained model packages are saved."
      footer={
        <div className="flex items-center justify-between gap-2 w-full">
          <span
            className="text-[11px] font-mono text-[var(--text-muted)] truncate flex-1 text-left"
            title={currentPath}
          >
            {currentPath || 'Select a drive or folder'}
          </span>
          <div className="flex items-center gap-2 shrink-0">
            <button
              type="button"
              onClick={onClose}
              className="px-3.5 py-2 rounded-lg bg-[var(--bg-card)] border border-[var(--border-color)] text-xs font-semibold text-[var(--text-secondary)] hover:text-[var(--text-primary)] cursor-pointer transition-colors"
            >
              Cancel
            </button>
            <button
              type="button"
              disabled={!(pathInput.trim() || currentPath)}
              onClick={() => {
                const chosen = pathInput.trim() || currentPath;
                if (chosen) {
                  onSelect(chosen);
                  onClose();
                }
              }}
              className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-[var(--btn-primary-bg)] hover:bg-[var(--btn-primary-hover)] text-[var(--btn-primary-text)] text-xs font-bold shadow-xs cursor-pointer disabled:opacity-45 transition-colors"
            >
              <Check className="w-3.5 h-3.5" />
              <span>Select Folder</span>
            </button>
          </div>
        </div>
      }
    >
      <div className="flex flex-col gap-3">
        {/* Quick Navigation Bar */}
        <div className="flex flex-wrap items-center gap-1.5">
          {parentPath && (
            <button
              type="button"
              onClick={() => loadFolder(parentPath)}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-[var(--bg-card-subtle)] hover:bg-[var(--btn-secondary-hover)] border border-[var(--border-color)] text-xs font-semibold text-[var(--text-primary)] cursor-pointer transition-colors"
              title="Go up one folder"
            >
              <ArrowUp className="w-3.5 h-3.5" />
              <span>Up</span>
            </button>
          )}
          {homePath && (
            <button
              type="button"
              onClick={() => loadFolder(homePath)}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-[var(--bg-card-subtle)] hover:bg-[var(--btn-secondary-hover)] border border-[var(--border-color)] text-xs font-semibold text-[var(--text-primary)] cursor-pointer transition-colors"
            >
              <Home className="w-3.5 h-3.5 text-[var(--accent-main)]" />
              <span>Home</span>
            </button>
          )}
          {defaultPath && (
            <button
              type="button"
              onClick={() => loadFolder(defaultPath)}
              className="flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-[var(--bg-card-subtle)] hover:bg-[var(--btn-secondary-hover)] border border-[var(--border-color)] text-xs font-semibold text-[var(--text-primary)] cursor-pointer transition-colors"
            >
              <Folder className="w-3.5 h-3.5 text-[var(--accent-main)]" />
              <span>Default</span>
            </button>
          )}
          {roots.map((r) => (
            <button
              key={r}
              type="button"
              onClick={() => loadFolder(r)}
              className="flex items-center gap-1 px-2 py-1.5 rounded-lg bg-[var(--bg-card-subtle)] hover:bg-[var(--btn-secondary-hover)] border border-[var(--border-color)] text-[11px] font-mono font-semibold text-[var(--text-secondary)] cursor-pointer transition-colors"
            >
              <HardDrive className="w-3 h-3 text-[var(--text-muted)]" />
              <span>{r}</span>
            </button>
          ))}
          {currentPath && (
            <button
              type="button"
              onClick={() => setCreatingFolder((v) => !v)}
              className="ml-auto flex items-center gap-1 px-2.5 py-1.5 rounded-lg bg-[var(--accent-pond-subtle)] border border-[var(--border-color)] text-xs font-bold text-[var(--accent-main)] hover:opacity-90 cursor-pointer transition-opacity"
            >
              <FolderPlus className="w-3.5 h-3.5" />
              <span>New Folder</span>
            </button>
          )}
        </div>

        {/* Inline New Folder Input */}
        {creatingFolder && (
          <div className="flex items-center gap-2 p-2 rounded-lg bg-[var(--bg-card-subtle)] border border-[var(--border-color)]">
            <input
              type="text"
              value={newFolderName}
              onChange={(e) => setNewFolderName(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === 'Enter') handleCreateFolder();
                if (e.key === 'Escape') setCreatingFolder(false);
              }}
              placeholder="Folder name (e.g. sop_models)"
              autoFocus
              className="flex-1 px-2.5 py-1.5 rounded-md bg-[var(--bg-card)] border border-[var(--border-color)] text-xs text-[var(--text-primary)] focus:outline-none focus:border-[var(--accent-main)]"
            />
            <button
              type="button"
              onClick={handleCreateFolder}
              className="px-3 py-1.5 rounded-md bg-[var(--btn-primary-bg)] text-[var(--btn-primary-text)] text-xs font-bold cursor-pointer"
            >
              Create
            </button>
          </div>
        )}

        {/* Editable & Copyable Path Box */}
        <div className="flex items-center gap-1.5 select-text">
          <input
            type="text"
            value={pathInput}
            onChange={(e) => setPathInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' && pathInput.trim()) {
                loadFolder(pathInput.trim());
              }
            }}
            placeholder="Paste or type a Windows / Linux path (e.g. D:\models or /home/user/models)"
            className="flex-1 min-w-0 px-3 py-2 rounded-lg bg-[var(--bg-card)] border border-[var(--border-color)] text-xs font-mono text-[var(--text-primary)] select-text focus:outline-none focus:border-[var(--accent-main)] transition-colors"
          />
          {pathInput.trim() && pathInput.trim() !== currentPath && (
            <button
              type="button"
              onClick={() => loadFolder(pathInput.trim())}
              className="h-[34px] px-2.5 rounded-lg bg-[var(--bg-card-subtle)] hover:bg-[var(--btn-secondary-hover)] border border-[var(--border-color)] text-xs font-bold text-[var(--text-primary)] flex items-center gap-1 shrink-0 cursor-pointer"
              title="Open pasted path"
            >
              <CornerDownLeft className="w-3.5 h-3.5 text-[var(--accent-main)]" />
              <span>Go</span>
            </button>
          )}
          <button
            type="button"
            onClick={() => {
              const txt = pathInput.trim() || currentPath;
              if (!txt) return;
              navigator.clipboard?.writeText(txt).then(() => {
                setCopied(true);
                setTimeout(() => setCopied(false), 1500);
              }).catch(() => {});
            }}
            disabled={!(pathInput.trim() || currentPath)}
            className="h-[34px] px-2.5 rounded-lg bg-[var(--bg-card)] hover:bg-[var(--btn-secondary-hover)] border border-[var(--border-color)] text-xs font-semibold text-[var(--text-secondary)] flex items-center gap-1 shrink-0 cursor-pointer disabled:opacity-40"
            title="Copy current path"
          >
            {copied ? (
              <Check className="w-3.5 h-3.5 text-emerald-500" />
            ) : (
              <Copy className="w-3.5 h-3.5" />
            )}
          </button>
        </div>

        {error && (
          <div className="flex items-center gap-2 p-2.5 rounded-lg bg-[var(--status-bad-bg)] border border-[var(--status-bad-border)] text-xs text-[var(--status-bad-text)]">
            <AlertCircle className="w-4 h-4 shrink-0" />
            <span>{error}</span>
          </div>
        )}

        {/* Directory List */}
        <div className="h-60 overflow-y-auto custom-scrollbar rounded-xl border border-[var(--border-color)] bg-[var(--bg-card)] divide-y divide-[var(--border-color)]">
          {loading ? (
            <div className="h-full flex items-center justify-center gap-2 text-xs text-[var(--text-muted)]">
              <Loader2 className="w-4 h-4 animate-spin text-[var(--accent-main)]" />
              <span>Loading folders...</span>
            </div>
          ) : folders.length === 0 ? (
            <div className="h-full flex items-center justify-center text-xs text-[var(--text-muted)]">
              No subfolders inside this directory.
            </div>
          ) : (
            folders.map((f) => (
              <button
                key={f.path}
                type="button"
                onClick={() => loadFolder(f.path)}
                className="w-full flex items-center gap-2.5 px-3.5 py-2.5 text-left hover:bg-[var(--bg-card-subtle)] text-xs font-medium text-[var(--text-primary)] cursor-pointer transition-colors"
              >
                <Folder className="w-4 h-4 text-[var(--accent-main)] shrink-0" />
                <span className="truncate">{f.name}</span>
              </button>
            ))
          )}
        </div>
      </div>
    </Modal>
  );
};
