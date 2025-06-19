"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@nba/ui/utils";

export function MainNav() {
  const pathname = usePathname();
  const routes = [
    { href: "/", label: "Home" },
    { href: "/players", label: "Players" },
    { href: "/teams", label: "Teams" },
    { href: "/games", label: "Games" },
  ];

  return (
    <nav className="hidden md:flex items-center space-x-4 lg:space-x-6">
      {routes.map((route) => (
        <Link
          key={route.href}
          href={route.href}
          className={cn(
            "text-sm font-medium transition-colors hover:text-primary",
            pathname === route.href
              ? "text-primary"
              : "text-muted-foreground",
          )}
        >
          {route.label}
        </Link>
      ))}
    </nav>
  );
}