"use client";

import { Player } from "@nba/api/types";
import {
  Card,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@nba/ui/card";
import { Input } from "@nba/ui/input";
import { useQuery } from "@tanstack/react-query";
import Link from "next/link";
import { useState } from "react";

import { PageHeader } from "@/components/page-header";
import { apiClient } from "@/lib/api-client";


export default function PlayersPage() {
  const [query, setQuery] = useState("");

  const { data, isLoading } = useQuery({
    queryKey: ["players", "search", query],
    queryFn: () => apiClient.searchPlayers(query, 50),
    enabled: query.length >= 2,
  });

  return (
    <main className="container mx-auto px-4 py-8">
      <PageHeader
        title="Find Players"
        description="Search for any active or historical NBA player."
      />
      <div className="mt-8">
        <Input
          placeholder="Search by player name..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="max-w-lg"
        />
      </div>

      <div className="mt-8">
        {query.length < 2 && (
          <div className="text-center py-16">
            <p className="text-lg text-muted-foreground">
              Enter at least 2 characters to start searching.
            </p>
          </div>
        )}
        {isLoading && query.length >= 2 && <p>Searching...</p>}
        {data && data.items.length > 0 && (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {data.items.map((player: Player) => (
              <Link key={player.person_id} href={`/players/${player.person_id}`}>
                <Card className="hover:bg-accent transition-colors">
                  <CardHeader>
                    <CardTitle>{player.full_name}</CardTitle>
                    <CardDescription>
                      {player.primary_position}
                      {player.draft_year ? ` • Draft ${player.draft_year}` : ""}
                    </CardDescription>
                  </CardHeader>
                </Card>
              </Link>
            ))}
          </div>
        )}
        {data && data.items.length === 0 && query.length >= 2 && (
          <div className="text-center py-16">
            <p className="text-lg text-muted-foreground">
              No players found for `&quot;`{query}`&quot;`.
            </p>
          </div>
        )}
      </div>
    </main>
  );
}