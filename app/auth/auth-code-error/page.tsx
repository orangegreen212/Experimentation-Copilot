import Link from 'next/link';
import { AlertTriangle } from 'lucide-react';

export default function AuthCodeErrorPage() {
  return (
    <div className="flex min-h-screen items-center justify-center bg-neutral-50 px-4">
      <div className="flex w-full max-w-sm flex-col items-center gap-3 rounded-lg border border-black/10 bg-white p-6 text-center">
        <AlertTriangle className="h-6 w-6 text-amber-600" />
        <p className="text-[13px] font-medium text-black">Sign-in link didn&apos;t work</p>
        <p className="text-[12px] text-neutral-500">
          It may have expired or already been used. Please try signing in again.
        </p>
        <Link
          href="/login"
          className="mt-2 inline-flex rounded-md bg-indigo-600 px-3 py-1.5 text-[12px] font-medium text-white hover:bg-indigo-700"
        >
          Back to login
        </Link>
      </div>
    </div>
  );
}
