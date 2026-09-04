'use client';

import { useState } from 'react';
import { Sidebar, type View } from '@/components/sidebar';
import { WorkspaceView } from '@/components/workspace-view';
import { HistoryView } from '@/components/history-view';
import { DatasetsView } from '@/components/datasets-view';
import { ExperimentConfig } from '@/components/experiment-config';
import type { Settings } from '@/lib/types';

const VIEW_COPY: Record<View, { title: string; subtitle: string }> = {
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
  metrics: { title: 'Metrics', subtitle: 'Coming soon' },
  settings: { title: 'Settings', subtitle: 'Coming soon' },
};

export default function Home() {
  const [view, setView] = useState<View>('overview');
  // Bumped whenever a new experiment is saved, so Experiments/Datasets
  // know to refetch even if the user doesn't manually revisit them.
  const [historyVersion, setHistoryVersion] = useState(0);
  // Set when the user jumps from a Datasets card to a specific run.
  const [pendingExperimentId, setPendingExperimentId] = useState<string | undefined>(undefined);

  // Experiment-level configuration (CUPED / bootstrap). Lives here, not
  // inside WorkspaceView, so it's not reset every time the user loads a
  // new dataset — it persists across dataset switches within a session.
  const [settings, setSettings] = useState<Settings>({ cuped: false, bootstrap: false });

  const handleSessionSaved = () => {
    setHistoryVersion((v) => v + 1);
  };

  const goToExperiment = (experimentId?: string) => {
    setPendingExperimentId(experimentId);
    setView('experiments');
  };

  const copy = VIEW_COPY[view];

  return (
    <div className="flex h-screen flex-col overflow-hidden bg-white md:flex-row">
      <Sidebar view={view} onViewChange={setView} />
      <main className="flex min-w-0 flex-1 flex-col overflow-hidden">
        <header className="hidden h-14 shrink-0 items-center justify-between border-b border-black/10 bg-white px-6 md:flex">
          <div>
            <h2 className="text-[13px] font-semibold tracking-tight text-black">{copy.title}</h2>
            <p className="text-xs text-neutral-400">{copy.subtitle}</p>
          </div>
        </header>
        <div className="flex-1 overflow-y-auto p-4 sm:p-6">
          <div className="mb-4 md:hidden">
            <h2 className="text-[15px] font-semibold tracking-tight text-black">{copy.title}</h2>
            <p className="text-xs text-neutral-400">{copy.subtitle}</p>
          </div>
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
            <HistoryView refreshKey={historyVersion} initialExperimentId={pendingExperimentId} />
          )}
          {view === 'datasets' && (
            <DatasetsView refreshKey={historyVersion} onViewExperiments={goToExperiment} />
          )}
        </div>
      </main>
    </div>
  );
}
