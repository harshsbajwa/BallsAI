import { Toaster } from "@nba/ui/toaster";
import type { Metadata } from "next";
import { Inter } from "next/font/google";

import "./globals.css";
import { Providers } from "@/components/providers";
import { SiteHeader } from "@/components/site-header";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "BallsAI - NBA Analytics",
  description: "Real-time NBA stats, predictions, and analytics.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={inter.className}>
        <Providers>
          <div className="relative flex min-h-screen flex-col">
            <SiteHeader />
            <div className="flex-1">{children}</div>
          </div>
          <Toaster />
        </Providers>
      </body>
    </html>
  );
}