'use client';

import { useEffect, useState } from 'react';
import {
  FolderKanban,
  History,
  Database,
  BarChart3,
  Settings,
  FlaskConical,
  Menu,
  X,
  LogOut,
} from 'lucide-react';
import { cn } from '@/lib/utils';
import { createClient } from '@/lib/supabase/client';

export type View = 'library' | 'overview' | 'experiments' | 'datasets' | 'metrics' | 'settings';

interface SidebarProps {
  view: View;
  onViewChange: (v: View) => void;
}

interface NavItem {
  id: View;
  label: string;
  icon: typeof FolderKanban;
  /** Not wired up to a real screen yet — shown but not clickable. */
  comingSoon?: boolean;
}

const NAV: NavItem[] = [
  { id: 'library', label: 'Library', icon: FolderKanban },
  { id: 'overview', label: 'New Analysis', icon: FlaskConical },
  { id: 'experiments', label: 'Experiments', icon: History },
  { id: 'datasets', label: 'Datasets', icon: Database },
  { id: 'metrics', label: 'Metrics', icon: BarChart3, comingSoon: true },
  { id: 'settings', label: 'Settings', icon: Settings, comingSoon: true },
];

function Brand() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="flex h-8 w-8 items-center justify-center rounded-md bg-indigo-600 text-white">
        <FlaskConical className="h-4 w-4" />
      </div>
      <div className="min-w-0">
        <h1 className="truncate text-[13px] font-semibold leading-tight tracking-tight text-black">
          Experiment Review Copilot
        </h1>
      </div>
    </div>
  );
}

function NavList({ view, onSelect }: { view: View; onSelect: (v: View) => void }) {
  return (
    <nav className="flex flex-col gap-0.5 px-3 py-4">
      {NAV.map((item) => {
        const Icon = item.icon;
        const active = view === item.id;
        return (
          <button
            key={item.id}
            onClick={() => !item.comingSoon && onSelect(item.id)}
            disabled={item.comingSoon}
            className={cn(
              'group flex items-center gap-2.5 rounded-md px-2.5 py-2 text-[13px] font-medium transition-colors',
              active && !item.comingSoon
                ? 'bg-indigo-600 text-white'
                : item.comingSoon
                ? 'cursor-not-allowed text-neutral-300'
                : 'text-neutral-600 hover:bg-neutral-100 hover:text-black'
            )}
          >
            <Icon className="h-4 w-4 shrink-0" />
            <span className="flex-1 text-left">{item.label}</span>
            {item.comingSoon && (
              <span className="rounded border border-neutral-200 px-1 py-0.5 text-[9px] font-semibold uppercase tracking-wide text-neutral-400">
                Soon
              </span>
            )}
          </button>
        );
      })}
    </nav>
  );
}

function initialsFor(label: string): string {
  const parts = label.trim().split(/\s+/);
  const chars = parts.length > 1 ? [parts[0][0], parts[1][0]] : [label.slice(0, 2)];
  return chars.join('').toUpperCase();
}

/**
 * Signed-in user block — email/avatar + sign-out. Fetches the session
 * client-side so Sidebar doesn't need the whole app tree converted to a
 * Server Component just to know who's signed in; middleware.ts is what
 * actually enforces the auth gate, this is display-only.
 */
function UserBlock() {
  const [email, setEmail] = useState<string | null>(null);
  const [avatarUrl, setAvatarUrl] = useState<string | null>(null);
  const [name, setName] = useState<string | null>(null);

  useEffect(() => {
    const supabase = createClient();
    supabase.auth.getUser().then(({ data }) => {
      setEmail(data.user?.email ?? null);
      setAvatarUrl((data.user?.user_metadata?.avatar_url as string | undefined) ?? null);
      setName((data.user?.user_metadata?.full_name as string | undefined) ?? null);
    });
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setEmail(session?.user?.email ?? null);
      setAvatarUrl((session?.user?.user_metadata?.avatar_url as string | undefined) ?? null);
      setName((session?.user?.user_metadata?.full_name as string | undefined) ?? null);
    });
    return () => subscription.unsubscribe();
  }, []);

  if (!email) return null;

  const label = name || email;

  return (
    <div className="mt-auto flex items-center gap-2.5 border-t border-black/10 px-5 py-3.5">
      {avatarUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={avatarUrl} alt="" className="h-8 w-8 shrink-0 rounded-full" />
      ) : (
        <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-indigo-100 text-[11px] font-semibold text-indigo-700">
          {initialsFor(label)}
        </div>
      )}
      <div className="min-w-0 flex-1">
        <p className="truncate text-[12px] font-medium text-black">{label}</p>
        <p className="truncate text-[10px] text-neutral-400">{email}</p>
      </div>
      <form action="/auth/signout" method="post">
        <button
          type="submit"
          title="Sign out"
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-neutral-400 hover:bg-neutral-100 hover:text-black"
        >
          <LogOut className="h-3.5 w-3.5" />
        </button>
      </form>
    </div>
  );
}

export function Sidebar({ view, onViewChange }: SidebarProps) {
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleSelect = (v: View) => {
    onViewChange(v);
    setMobileOpen(false);
  };

  return (
    <>
      {/* Mobile top bar — replaces the sidebar below the md breakpoint */}
      <div className="flex h-14 shrink-0 items-center justify-between border-b border-black/10 bg-white px-4 md:hidden">
        <Brand />
        <button
          onClick={() => setMobileOpen(true)}
          aria-label="Open navigation"
          className="flex h-8 w-8 items-center justify-center rounded-md text-neutral-500 hover:bg-neutral-100"
        >
          <Menu className="h-5 w-5" />
        </button>
      </div>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div
            className="absolute inset-0 bg-black/30"
            onClick={() => setMobileOpen(false)}
          />
          <aside className="absolute left-0 top-0 flex h-full w-72 max-w-[80vw] flex-col bg-white shadow-xl animate-fade-in">
            <div className="flex items-center justify-between border-b border-black/10 px-5 py-5">
              <Brand />
              <button
                onClick={() => setMobileOpen(false)}
                aria-label="Close navigation"
                className="flex h-8 w-8 items-center justify-center rounded-md text-neutral-500 hover:bg-neutral-100"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <NavList view={view} onSelect={handleSelect} />
            <UserBlock />
            <div className="border-t border-black/10 px-5 py-4">
              <p className="text-[10px] leading-relaxed text-neutral-400">
                Plan-and-Execute agent. LLM plans &amp; interprets; stats executed in
                Python nodes.
              </p>
            </div>
          </aside>
        </div>
      )}

      {/* Desktop sidebar */}
      <aside className="hidden h-full w-60 shrink-0 flex-col border-r border-black/10 bg-white md:flex">
        <div className="flex flex-col gap-2.5 border-b border-black/10 px-5 py-5">
          <Brand />
          <span className="inline-flex w-fit items-center rounded border border-black/10 bg-neutral-50 px-1.5 py-0.5 text-[10px] font-medium uppercase tracking-wide text-neutral-500">
            AI Decision Support System
          </span>
        </div>

        <NavList view={view} onSelect={handleSelect} />
        <UserBlock />
        <div className="border-t border-black/10 px-5 py-4">
          <p className="text-[10px] leading-relaxed text-neutral-400">
            Plan-and-Execute agent. LLM plans &amp; interprets; stats executed in
            Python nodes.
          </p>
        </div>
      </aside>
    </>
  );
}
