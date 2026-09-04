'use client';

import { useState, Suspense } from 'react';
import { useSearchParams } from 'next/navigation';
import { FlaskConical, Mail, Loader2, CheckCircle2 } from 'lucide-react';
import { createClient } from '@/lib/supabase/client';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent } from '@/components/ui/card';

function GoogleIcon() {
  // Inline, license-free "G" mark (standard 4-color Google logo paths) —
  // avoids pulling in an extra icon package for a single icon.
  return (
    <svg viewBox="0 0 24 24" className="h-4 w-4" aria-hidden="true">
      <path
        fill="#4285F4"
        d="M23.52 12.27c0-.85-.08-1.66-.22-2.45H12v4.64h6.47a5.53 5.53 0 0 1-2.4 3.63v3h3.87c2.27-2.09 3.58-5.17 3.58-8.82Z"
      />
      <path
        fill="#34A853"
        d="M12 24c3.24 0 5.96-1.08 7.94-2.91l-3.87-3a7.4 7.4 0 0 1-11.02-3.89H1.06v3.09A12 12 0 0 0 12 24Z"
      />
      <path
        fill="#FBBC05"
        d="M5.05 14.2a7.2 7.2 0 0 1 0-4.4V6.71H1.06a12 12 0 0 0 0 10.58l3.99-3.09Z"
      />
      <path
        fill="#EA4335"
        d="M12 4.75c1.76 0 3.34.6 4.58 1.79l3.44-3.44C17.95 1.19 15.24 0 12 0A12 12 0 0 0 1.06 6.71l3.99 3.09A7.16 7.16 0 0 1 12 4.75Z"
      />
    </svg>
  );
}

function LoginForm() {
  const searchParams = useSearchParams();
  const next = searchParams.get('next') || '/';

  const [email, setEmail] = useState('');
  const [magicLinkSent, setMagicLinkSent] = useState(false);
  const [isGoogleLoading, setIsGoogleLoading] = useState(false);
  const [isEmailLoading, setIsEmailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleGoogleSignIn = async () => {
    setError(null);
    setIsGoogleLoading(true);
    const supabase = createClient();
    const { error: authError } = await supabase.auth.signInWithOAuth({
      provider: 'google',
      options: {
        redirectTo: `${window.location.origin}/auth/callback?next=${encodeURIComponent(next)}`,
      },
    });
    if (authError) {
      setError(authError.message);
      setIsGoogleLoading(false);
    }
    // On success the browser is redirected to Google — nothing else to do here.
  };

  const handleEmailSignIn = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setIsEmailLoading(true);
    const supabase = createClient();
    const { error: authError } = await supabase.auth.signInWithOtp({
      email,
      options: {
        emailRedirectTo: `${window.location.origin}/auth/callback?next=${encodeURIComponent(next)}`,
      },
    });
    setIsEmailLoading(false);
    if (authError) {
      setError(authError.message);
    } else {
      setMagicLinkSent(true);
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-neutral-50 px-4">
      <div className="w-full max-w-sm space-y-6">
        <div className="flex flex-col items-center gap-2 text-center">
          <div className="flex h-10 w-10 items-center justify-center rounded-md bg-indigo-600 text-white">
            <FlaskConical className="h-5 w-5" />
          </div>
          <h1 className="text-lg font-semibold tracking-tight text-black">
            Experiment Review Copilot
          </h1>
          <p className="text-[13px] text-neutral-500">Sign in to view and run experiments</p>
        </div>

        <Card className="border-black/10 shadow-none">
          <CardContent className="space-y-4 p-5">
            {magicLinkSent ? (
              <div className="flex flex-col items-center gap-2 py-4 text-center">
                <CheckCircle2 className="h-6 w-6 text-green-600" />
                <p className="text-[13px] font-medium text-black">Check your inbox</p>
                <p className="text-[12px] text-neutral-500">
                  We sent a sign-in link to <span className="font-medium text-black">{email}</span>.
                </p>
              </div>
            ) : (
              <>
                <Button
                  variant="outline"
                  className="w-full gap-2 border-black/15"
                  onClick={handleGoogleSignIn}
                  disabled={isGoogleLoading || isEmailLoading}
                >
                  {isGoogleLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <GoogleIcon />
                  )}
                  Continue with Google
                </Button>

                <div className="flex items-center gap-3">
                  <div className="h-px flex-1 bg-black/10" />
                  <span className="text-[11px] uppercase tracking-wide text-neutral-400">or</span>
                  <div className="h-px flex-1 bg-black/10" />
                </div>

                <form onSubmit={handleEmailSignIn} className="space-y-2">
                  <Input
                    type="email"
                    required
                    placeholder="you@company.com"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    disabled={isEmailLoading || isGoogleLoading}
                    className="border-black/10"
                  />
                  <Button
                    type="submit"
                    variant="outline"
                    className="w-full gap-2 border-black/15"
                    disabled={isEmailLoading || isGoogleLoading || !email}
                  >
                    {isEmailLoading ? (
                      <Loader2 className="h-4 w-4 animate-spin" />
                    ) : (
                      <Mail className="h-4 w-4" />
                    )}
                    Send magic link
                  </Button>
                </form>
              </>
            )}

            {error && <p className="text-[12px] text-red-600">{error}</p>}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

export default function LoginPage() {
  // useSearchParams needs a Suspense boundary in the App Router.
  return (
    <Suspense fallback={null}>
      <LoginForm />
    </Suspense>
  );
}
