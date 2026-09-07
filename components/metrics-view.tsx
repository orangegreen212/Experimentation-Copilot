'use client';

import { useState } from 'react';
import { BarChart3, ChevronDown } from 'lucide-react';
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card';
import { cn } from '@/lib/utils';

interface MetricEntry {
  name: string;
  summary: string;
  useWhen: string;
  definition: string;
  formula: string;
  example: string;
  direction: 'Higher is better' | 'Lower is better' | 'Depends on context';
  commonUseCases: string[];
}

// Purely a static reference guide — deliberately NOT wired to
// ExperimentConfig or AnalysisSettings (see the Settings/Metrics split
// discussion): this screen never selects a metric for an experiment,
// it only explains the metrics an analyst is likely to see in a
// report. No backend call needed.
const METRICS: MetricEntry[] = [
  {
    name: 'Conversion Rate',
    summary: 'Share of users who complete a target action',
    useWhen: 'Purchases, signups, activation',
    definition:
      'The proportion of users in a group who complete a defined action (e.g. checkout, signup) out of everyone exposed to that group.',
    formula: 'Conversions / Total users exposed',
    example: '420 purchases out of 8,000 visitors = 5.25% conversion rate.',
    direction: 'Higher is better',
    commonUseCases: ['Checkout funnels', 'Signup flows', 'Feature activation'],
  },
  {
    name: 'Revenue per User',
    summary: 'Average revenue generated per user',
    useWhen: 'Monetization',
    definition:
      'Total revenue attributed to a group divided by the number of users in that group, including users who spent nothing.',
    formula: 'Total revenue / Total users',
    example: '$18,400 revenue across 4,000 users = $4.60 revenue per user.',
    direction: 'Higher is better',
    commonUseCases: ['Pricing experiments', 'Paywall placement', 'Upsell flows'],
  },
  {
    name: 'Average Order Value (AOV)',
    summary: 'Average amount spent per completed order',
    useWhen: 'E-commerce',
    definition: 'Total revenue divided by the number of orders placed — unlike revenue per user, non-buyers are excluded.',
    formula: 'Total revenue / Number of orders',
    example: '$18,400 revenue across 920 orders = $20.00 AOV.',
    direction: 'Higher is better',
    commonUseCases: ['Cross-sell/upsell', 'Bundle pricing', 'Minimum-order thresholds'],
  },
  {
    name: 'Retention Rate',
    summary: 'Share of users who come back after a given period',
    useWhen: 'Retention',
    definition:
      'The proportion of users active at the start of a period who are still active at a later checkpoint (e.g. Day 7, Day 30).',
    formula: 'Users retained at checkpoint / Users active at period start',
    example: '620 of 1,000 Day-0 users returned by Day 7 = 62% D7 retention.',
    direction: 'Higher is better',
    commonUseCases: ['Onboarding changes', 'Push notification strategy', 'Habit-forming features'],
  },
  {
    name: 'Click-through Rate (CTR)',
    summary: 'Share of viewers who clicked an element',
    useWhen: 'UI / content experiments',
    definition: 'The proportion of users who saw an element (banner, button, link) and clicked it.',
    formula: 'Clicks / Impressions',
    example: '312 clicks on 6,000 impressions = 5.2% CTR.',
    direction: 'Higher is better',
    commonUseCases: ['Button copy/placement', 'Homepage banners', 'Recommendation widgets'],
  },
  {
    name: 'Session Duration',
    summary: 'Average length of a user session',
    useWhen: 'Engagement',
    definition: 'The average amount of time users spend actively engaged in a single session.',
    formula: 'Sum of session lengths / Number of sessions',
    example: 'Average session length of 4m 12s across all sessions in the group.',
    direction: 'Depends on context',
    commonUseCases: ['Content/feed changes', 'Video/media features', 'Navigation redesigns'],
  },
  {
    name: 'Bounce Rate',
    summary: 'Share of users who leave without further action',
    useWhen: 'Landing pages',
    definition: 'The proportion of visits where a user leaves after viewing only one page, with no further interaction.',
    formula: 'Single-page sessions / Total sessions',
    example: '2,100 single-page sessions out of 6,000 total sessions = 35% bounce rate.',
    direction: 'Lower is better',
    commonUseCases: ['Landing page copy/layout', 'Page load speed', 'Ad-to-page relevance'],
  },
  {
    name: 'Churn Rate',
    summary: 'Share of users who stop using the product',
    useWhen: 'Retention',
    definition: 'The proportion of users (or subscribers) active at the start of a period who are no longer active by its end.',
    formula: 'Users lost during period / Users active at period start',
    example: '48 of 1,200 subscribers cancelled this month = 4% monthly churn.',
    direction: 'Lower is better',
    commonUseCases: ['Subscription pricing', 'Cancellation flow changes', 'Win-back campaigns'],
  },
];

export function MetricsView() {
  return (
    <div className="mx-auto max-w-3xl space-y-4">
      <Card className="border-black/10 shadow-none">
        <CardHeader>
          <div className="flex items-center gap-2">
            <BarChart3 className="h-4 w-4 text-black" />
            <CardTitle className="text-[15px] tracking-tight">Metrics</CardTitle>
          </div>
          <CardDescription>A reference guide to common experimentation metrics.</CardDescription>
        </CardHeader>
        <CardContent className="space-y-0">
          {METRICS.map((metric, i) => (
            <div key={metric.name}>
              <MetricRow metric={metric} />
              {i < METRICS.length - 1 && <div className="h-px bg-black/10" />}
            </div>
          ))}
        </CardContent>
      </Card>
    </div>
  );
}

function MetricRow({ metric }: { metric: MetricEntry }) {
  const [open, setOpen] = useState(false);

  return (
    <div className="py-3.5">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start justify-between gap-4 text-left"
      >
        <div className="min-w-0">
          <p className="text-[13px] font-medium text-black">{metric.name}</p>
          <p className="text-xs text-neutral-400">
            {metric.summary} &middot; {metric.useWhen}
          </p>
        </div>
        <ChevronDown
          className={cn('mt-0.5 h-4 w-4 shrink-0 text-neutral-400 transition-transform', open && 'rotate-180')}
        />
      </button>

      {open && (
        <dl className="mt-3 space-y-2.5 rounded-md bg-neutral-50 p-3.5 text-xs">
          <MetricField label="Definition" value={metric.definition} />
          <MetricField label="Formula" value={metric.formula} mono />
          <MetricField label="Example" value={metric.example} />
          <MetricField label="Direction" value={metric.direction} />
          <div>
            <dt className="font-medium text-black">Common use cases</dt>
            <dd className="mt-1 flex flex-wrap gap-1.5 text-neutral-500">
              {metric.commonUseCases.map((useCase) => (
                <span
                  key={useCase}
                  className="rounded border border-black/10 bg-white px-1.5 py-0.5 text-[11px]"
                >
                  {useCase}
                </span>
              ))}
            </dd>
          </div>
        </dl>
      )}
    </div>
  );
}

function MetricField({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div>
      <dt className="font-medium text-black">{label}</dt>
      <dd className={cn('mt-0.5 text-neutral-500', mono && 'font-mono text-[11px]')}>{value}</dd>
    </div>
  );
}
