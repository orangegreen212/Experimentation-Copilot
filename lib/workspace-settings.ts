import { createClient } from '@/lib/supabase/client';
import type { Settings } from '@/lib/types';

/**
 * Workspace-level analysis defaults, editable from the Settings screen
 * (components/settings-view.tsx). A DELIBERATE subset of `Settings`
 * (lib/types.ts): only the fields that already have a real,
 * implemented effect at analyze-time (model / confidenceLevel /
 * statisticalPower / cuped / bootstrap). Fields like guardrailMetrics
 * or analysisMode are inherently per-experiment (they name specific
 * columns from a specific dataset) and don't make sense as a global
 * default, so they're intentionally excluded here rather than copied
 * in wholesale.
 */
export type WorkspaceDefaults = Pick<
  Settings,
  'model' | 'confidenceLevel' | 'statisticalPower' | 'cuped' | 'bootstrap'
>;

const TABLE = 'user_workspace_settings';

/**
 * Loads the signed-in user's saved defaults, or `null` if they've never
 * saved any yet (brand-new user / row doesn't exist) — callers should
 * treat `null` the same as "use the app's built-in defaults", not as
 * an error.
 */
export async function getWorkspaceSettings(): Promise<WorkspaceDefaults | null> {
  const supabase = createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) return null;

  // `createClient()` isn't instantiated with a generated `Database`
  // type (see lib/supabase/client.ts), so `.from()` has no schema to
  // look up `user_workspace_settings` against and infers `never` for
  // the row shape. Casting the builder here (not the client itself)
  // keeps that untyped-ness local to this one table instead of
  // spreading `any` through every other Supabase call in the app.
  const { data, error } = await (supabase.from(TABLE) as any)
    .select('settings')
    .eq('user_id', user.id)
    .maybeSingle();

  // PGRST116 / no-row-yet cases are already handled by maybeSingle()
  // returning `data: null` rather than throwing — a real `error` here
  // means something else went wrong (network, RLS misconfigured), and
  // callers should fall back to built-in defaults rather than crash
  // the page over a settings-load hiccup.
  if (error) {
    console.error('Failed to load workspace settings:', error);
    return null;
  }

  return (data?.settings as WorkspaceDefaults | undefined) ?? null;
}

/**
 * Upserts the signed-in user's defaults. Throws on failure so the
 * Settings screen can show a real error instead of silently pretending
 * to save.
 */
export async function saveWorkspaceSettings(defaults: WorkspaceDefaults): Promise<void> {
  const supabase = createClient();
  const {
    data: { user },
  } = await supabase.auth.getUser();
  if (!user) {
    throw new Error('You must be signed in to save workspace settings.');
  }

  const { error } = await (supabase.from(TABLE) as any)
    .upsert(
      { user_id: user.id, settings: defaults, updated_at: new Date().toISOString() },
      { onConflict: 'user_id' }
    );

  if (error) {
    throw new Error(error.message);
  }
}
