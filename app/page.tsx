'use client';

import { useEffect, useState } from 'react';
import { Sidebar, type View } from '@/components/sidebar';
import { WorkspaceView } from '@/components/workspace-view';
import { HistoryView } from '@/components/history-view';
import { DatasetsView } from '@/components/datasets-view';
import { ExperimentConfig } from '@/components/experiment-config';
import { ExperimentLibrary } from '@/components/experiment-library';
import { SettingsView } from '@/components/settings-view';
import { MetricsView } from '@/components/metrics-view';
import { getWorkspaceSettings } from '@/lib/workspace-settings';
import type { Settings } from '@/lib/types';

const VIEW_COPY: Record<View, { title: string; subtitle: string }> = {
  library: {
    title: 'Experiment Library',
    subtitle: 'Plan experiments, track status, and continue into analysis when ready',
  },
  overview: {
    title: 'Overview',
    subtitle: 'Upload data, configure the analysis, and review the AI-generated report',
  },
  experiments: {
    title: 'Experiments',
    subtitle: 'Browse past experiment sessions, reopen a report, and continue its follow-up chat',
  },
  datasets: {
    title: 'Datasets',
    subtitle: 'Every dataset you\u2019ve worked with, grouped from your experiment history',
  },
  metrics: { title: 'Metrics', subtitle: 'A reference guide to common experimentation metrics' },
  settings: {
    title: 'Settings',
    subtitle: 'Configure default analysis behavior for your experiments',
  },
};

export default function Home() {
  const [view, setView] = useState<View>('library');
  // Bumped whenever a new experiment is saved, so Experiments/Datasets
  // know to refetch even if the user doesn't manually revisit them.
  const [historyVersion, setHistoryVersion] = useState(0);
  // Set when the user jumps from a Datasets card to a specific run.
  const [pendingExperimentId, setPendingExperimentId] = useState<string | undefined>(undefined);

  // Experiment-level configuration (CUPED / bootstrap). Lives here, not
  // inside WorkspaceView, so it's not reset every time the user loads a
  // new dataset — it persists across dataset switches within a session.
  const [settings, setSettings] = useState<Settings>({ cuped: false, bootstrap: false });

  // Seed the very first session's settings from the user's saved
  // workspace defaults (Settings screen / lib/workspace-settings.ts).
  // Runs once, on mount, before the user has had a chance to touch
  // anything in ExperimentConfig — so merging saved defaults on top of
  // the hardcoded fallback here can never clobber a real user choice.
  useEffect(() => {
    getWorkspaceSettings()
      .then((saved) => {
        if (saved) setSettings((prev) => ({ ...prev, ...saved }));
      })
      .catch(() => {
        // No saved defaults yet (or not signed in) — the hardcoded
        // fallback above is exactly the app's pre-existing behavior.
      });
  }, []);

  const handleSessionSaved = () => {
    setHistoryVersion((v) => v + 1);
  };

  const goToExperiment = (experimentId?: string) => {
    setPendingExperimentId(experimentId);
    setView('experiments');
  };

  const copy = VIEW_COPY[view];

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-surface md:flex-row">
      <Sidebar view={view} onViewChange={setView} />
      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <header className="hidden h-14 shrink-0 items-center justify-between border-b border-border bg-surface-raised px-6 md:flex">
          <div>
            <h2 className="text-[14px] font-semibold tracking-tight text-foreground">{copy.title}</h2>
            <p className="text-[12px] text-muted-foreground">{copy.subtitle}</p>
          </div>
        </header>
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 2xl:px-10">
          <div className="mx-auto w-full max-w-[1600px]">
          <div className="mb-4 md:hidden">
            <h2 className="text-[15px] font-semibold tracking-tight text-foreground">{copy.title}</h2>
            <p className="text-xs text-muted-foreground">{copy.subtitle}</p>
          </div>
          {view === 'library' && (
            <ExperimentLibrary
              refreshKey={historyVersion}
              settings={settings}
              onContinueToAnalysis={() => setView('overview')}
            />
          )}
          {view === 'overview' && (
            <div className="space-y-4">
              <div className="mx-auto max-w-3xl">
                <ExperimentConfig settings={settings} onChange={setSettings} />
              </div>
              <WorkspaceView
                onSessionSaved={handleSessionSaved}
                settings={settings}
                onSettingsChange={setSettings}
              />
            </div>
          )}
          {view === 'experiments' && (
            <HistoryView refreshKey={historyVersion} initialExperimentId={pendingExperimentId} settings={settings} />
          )}
          {view === 'datasets' && (
            <DatasetsView refreshKey={historyVersion} onViewExperiments={goToExperiment} />
          )}
          {view === 'metrics' && <MetricsView />}
          {view === 'settings' && <SettingsView />}
          </div>
        </div>
      </main>
    </div>
  );
}
