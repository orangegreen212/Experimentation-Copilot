import { createServerClient } from '@supabase/ssr';
import { cookies } from 'next/headers';

/**
 * Server-side Supabase client — for Server Components, Route Handlers,
 * and Server Actions. Must be created fresh per request (it closes over
 * that request's cookie jar), so never hoist this to a module-level
 * singleton.
 *
 * In a Server Component, `cookies().set()` is a no-op (Next.js can't set
 * cookies from a component render) — that's expected here, session
 * *refresh* writes happen in middleware.ts instead. This client's job in
 * Server Components is reading the already-refreshed session.
 */
export async function createClient() {
  const cookieStore = await cookies();

  return createServerClient(
    process.env.NEXT_PUBLIC_SUPABASE_URL!,
    process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!,
    {
      cookies: {
        getAll() {
          return cookieStore.getAll();
        },
        setAll(cookiesToSet) {
          try {
            cookiesToSet.forEach(({ name, value, options }) =>
              cookieStore.set(name, value, options)
            );
          } catch {
            // Called from a Server Component render — safe to ignore,
            // middleware.ts is responsible for session refresh there.
          }
        },
      },
    }
  );
}
