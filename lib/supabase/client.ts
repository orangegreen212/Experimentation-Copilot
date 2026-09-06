import { createBrowserClient } from '@supabase/ssr';

// Derived directly from createBrowserClient's own return type instead of a
// hand-imported `SupabaseClient` from @supabase/supabase-js — that separate
// import can resolve to a different generic instantiation (schema type
// param) than what @supabase/ssr's createBrowserClient actually returns,
// which is what caused the "GenericSchema is not assignable to 'public'"
// build error. Always sourcing the type from the function itself keeps
// the two in sync automatically, including across future version bumps.
type BrowserSupabaseClient = ReturnType<typeof createBrowserClient>;

/**
 * Browser-side Supabase client. Safe to call from any Client Component —
 * reads/writes the auth session via cookies so it stays in sync with the
 * server-side client in lib/supabase/server.ts and middleware.ts.
 *
 * Requires NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY —
 * see .env.local.example. The anon key is safe to expose to the browser
 * by design (Supabase enforces access via Row Level Security, not by
 * hiding this key).
 *
 * SINGLETON, deliberately — unlike lib/supabase/server.ts (which must
 * stay request-scoped), this one must NOT create a fresh client per
 * call. Every browser client internally runs a GoTrueClient that
 * coordinates auth-token refresh across tabs via a named
 * `navigator.locks` lock derived from the storage key (surfaces in
 * DevTools as "Acquiring an exclusive Navigator LockManager lock
 * lock:sb-<project>-auth-token"). That's expected and handled
 * internally BETWEEN separate browser tabs — but calling this factory
 * more than once within the SAME tab (e.g. sidebar.tsx's UserBlock
 * calling it fresh on every mount, plus the login page calling it
 * again independently) spins up multiple independent GoTrueClient
 * instances all racing for that one lock inside a single tab, and one
 * of them loses as an unhandled rejection. Caching the instance here
 * means every call site in this tab shares the one client, so there's
 * only ever one lock holder to begin with.
 */
let browserClient: BrowserSupabaseClient | undefined;

export function createClient(): BrowserSupabaseClient {
  let client = browserClient;
  if (!client) {
    client = createBrowserClient(
      process.env.NEXT_PUBLIC_SUPABASE_URL!,
      process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
    );
    browserClient = client;
  }
  return client;
}
