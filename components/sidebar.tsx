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
  { id: 'overview', label: 'New analysis', icon: FlaskConical },
  { id: 'experiments', label: 'Experiments', icon: History },
  { id: 'datasets', label: 'Datasets', icon: Database },
  { id: 'metrics', label: 'Metrics', icon: BarChart3 },
  { id: 'settings', label: 'Settings', icon: Settings },
];

function Brand() {
  return (
    <div className="flex items-center gap-2.5">
      <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-[7px] bg-primary text-primary-foreground">
        <FlaskConical className="h-3.5 w-3.5" />
      </div>
      <div className="min-w-0">
        <h1 className="truncate text-[13px] font-semibold leading-tight tracking-tight text-foreground">
          Experiment Review Copilot
        </h1>
      </div>
    </div>
  );
}

function NavList({ view, onSelect }: { view: View; onSelect: (v: View) => void }) {
  return (
    <nav className="flex flex-col gap-0.5 px-3 py-3">
      {NAV.map((item) => {
        const Icon = item.icon;
        const active = view === item.id;
        return (
          <button
            key={item.id}
            onClick={() => !item.comingSoon && onSelect(item.id)}
            disabled={item.comingSoon}
            aria-current={active ? 'page' : undefined}
            className={cn(
              'group relative flex items-center gap-2.5 rounded-md px-2.5 py-[7px] text-[13px] font-medium transition-colors',
              active && !item.comingSoon
                ? 'bg-accent text-accent-foreground'
                : item.comingSoon
                ? 'cursor-not-allowed text-muted-foreground/50'
                : 'text-muted-foreground hover:bg-secondary hover:text-foreground'
            )}
          >
            {active && !item.comingSoon && (
              <span className="absolute left-0 top-1/2 h-4 w-[3px] -translate-y-1/2 rounded-full bg-primary" />
            )}
            <Icon className="h-[15px] w-[15px] shrink-0" strokeWidth={2} />
            <span className="flex-1 text-left">{item.label}</span>
            {item.comingSoon && (
              <span className="rounded border border-border px-1 py-0.5 text-[10px] font-medium text-muted-foreground/70">
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
    <div className="mt-auto flex items-center gap-2.5 border-t border-border px-4 py-3">
      {avatarUrl ? (
        // eslint-disable-next-line @next/next/no-img-element
        <img src={avatarUrl} alt="" className="h-7 w-7 shrink-0 rounded-full" />
      ) : (
        <div className="flex h-7 w-7 shrink-0 items-center justify-center rounded-full bg-accent text-[11px] font-semibold text-accent-foreground">
          {initialsFor(label)}
        </div>
      )}
      <div className="min-w-0 flex-1">
        <p className="truncate text-[12.5px] font-medium text-foreground">{label}</p>
        <p className="truncate text-[11px] text-muted-foreground">{email}</p>
      </div>
      <form action="/auth/signout" method="post">
        <button
          type="submit"
          title="Sign out"
          className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md text-muted-foreground hover:bg-secondary hover:text-foreground"
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
      <div className="flex h-14 shrink-0 items-center justify-between border-b border-border bg-surface px-4 md:hidden">
        <Brand />
        <button
          onClick={() => setMobileOpen(true)}
          aria-label="Open navigation"
          className="flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground hover:bg-secondary"
        >
          <Menu className="h-5 w-5" />
        </button>
      </div>

      {/* Mobile drawer */}
      {mobileOpen && (
        <div className="fixed inset-0 z-50 md:hidden">
          <div
            className="absolute inset-0 bg-foreground/30"
            onClick={() => setMobileOpen(false)}
          />
          <aside className="absolute left-0 top-0 flex h-full w-72 max-w-[80vw] flex-col bg-surface shadow-xl animate-fade-in">
            <div className="flex items-center justify-between border-b border-border px-4 py-4">
              <Brand />
              <button
                onClick={() => setMobileOpen(false)}
                aria-label="Close navigation"
                className="flex h-8 w-8 items-center justify-center rounded-md text-muted-foreground hover:bg-secondary"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
            <NavList view={view} onSelect={handleSelect} />
            <UserBlock />
            <div className="border-t border-border px-4 py-3">
              <p className="text-[11px] leading-relaxed text-muted-foreground">
                Plan-and-Execute agent. The model plans &amp; interprets; statistics run in
                Python nodes.
              </p>
            </div>
          </aside>
        </div>
      )}

      {/* Desktop sidebar */}
      <aside className="hidden h-full w-[220px] shrink-0 flex-col border-r border-border bg-surface md:flex">
        <div className="flex flex-col gap-2 border-b border-border px-4 py-4">
          <Brand />
          <span className="inline-flex w-fit items-center rounded-full border border-border bg-secondary px-2 py-0.5 text-[11px] font-medium text-muted-foreground">
            AI decision support
          </span>
        </div>

        <NavList view={view} onSelect={handleSelect} />
        <UserBlock />
        <div className="border-t border-border px-4 py-3">
          <p className="text-[11px] leading-relaxed text-muted-foreground">
            Plan-and-Execute agent. The model plans &amp; interprets; statistics run in
            Python nodes.
          </p>
        </div>
      </aside>
    </>
  );
}
