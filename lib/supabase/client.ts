import { createBrowserClient } from '@supabase/ssr';

/**
 * Browser-side Supabase client. Safe to call from any Client Component —
 * reads/writes the auth session via cookies so it stays in sync with the
 * server-side client in lib/supabase/server.ts and middleware.ts.
 *
 * Requires NEXT_PUBLIC_SUPABASE_URL and NEXT_PUBLIC_SUPABASE_ANON_KEY —
 * see .env.local.example. The anon key is safe to expose to the browser
 * by design (Supabase enforces access via Row Level Security, not by
 * hiding this key).
 */
export function createClient() {
  return createBrowserClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
  );
}
