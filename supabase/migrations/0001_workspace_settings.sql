-- Workspace-level analysis defaults, one row per user.
--
-- This is DELIBERATELY separate from the per-run `AnalysisSettings`
-- baked into each `ExperimentReport` (backend/app/schemas/settings.py)
-- and from experiment_store.py's SQLAlchemy-managed tables. Those are
-- the FastAPI backend's own persistence (SQLite locally / Postgres in
-- prod, reached only from the backend). This table is read/written
-- directly by the Next.js frontend via the Supabase browser client
-- (lib/supabase/client.ts) — the FastAPI backend never touches it and
-- has no user_id concept at all today, so routing "save my defaults"
-- through the backend would require adding auth/user-scoping there
-- first. Writing straight to Supabase (RLS-scoped to auth.uid()) gets
-- persistence without that larger change.
create table if not exists public.user_workspace_settings (
  user_id uuid primary key references auth.users (id) on delete cascade,
  -- Free-form JSON blob mirroring a subset of lib/types.ts `Settings`
  -- (model / confidenceLevel / statisticalPower / cuped / bootstrap
  -- today). Kept as jsonb rather than one column per field so adding a
  -- new default later is a frontend-only change, no migration needed.
  settings jsonb not null default '{}'::jsonb,
  updated_at timestamptz not null default now()
);

alter table public.user_workspace_settings enable row level security;

-- Each user can only ever see/edit their own row.
create policy "Users can view their own workspace settings"
  on public.user_workspace_settings for select
  using (auth.uid() = user_id);

create policy "Users can upsert their own workspace settings"
  on public.user_workspace_settings for insert
  with check (auth.uid() = user_id);

create policy "Users can update their own workspace settings"
  on public.user_workspace_settings for update
  using (auth.uid() = user_id)
  with check (auth.uid() = user_id);
