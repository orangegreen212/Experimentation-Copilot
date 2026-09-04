import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Experiment Review Copilot — AI Decision Support System',
  description:
    'An AI Decision Support System for Product Experimentation. Plan-and-Execute agent architecture for rigorous A/B test review.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="h-full">
      <body className="h-full font-sans">{children}</body>
    </html>
  );
}
