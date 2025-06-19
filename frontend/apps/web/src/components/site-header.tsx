'use client';

import Link from "next/link";
import { MainNav } from "./main-nav";
import { UserNav } from "./user-nav";
import { Button } from "@nba/ui/button";
import { Moon, Sun, Home } from "lucide-react";
import { useTheme } from "next-themes";

export function SiteHeader() {
  const { theme, setTheme } = useTheme();

  return (
    <header className="sticky top-0 z-40 w-full border-b bg-background">
      <div className="container flex h-16 items-center space-x-4 sm:justify-between sm:space-x-0">
        <div className="flex gap-6 md:gap-10">
          <Link href="/" className="flex items-center space-x-2">
            <Home className="h-6 w-6" />
            <span className="inline-block font-bold">BallsAI</span>
          </Link>
          <MainNav />
        </div>
        <div className="flex flex-1 items-center justify-end space-x-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
          >
            <Sun className="h-[1.2rem] w-[1.2rem] rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
            <Moon className="absolute h-[1.2rem] w-[1.2rem] rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
            <span className="sr-only">Toggle theme</span>
          </Button>
          <UserNav />
        </div>
      </div>
    </header>
  );
}